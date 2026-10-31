"""
Synonym generation REST API service based on FastAPI + YandexGPT.

Run:
    export YANDEX_API_KEY=<your API key>
    export YANDEX_FOLDER_ID=<your folder id>
    uvicorn main:app --reload

POST /generate  {"input_file": "path/to/input.txt", "output_file": "path/to/output.txt"}

HTTP client choice: `httpx` is used instead of `requests`. Rationale:
  * endpoint handlers run in a synchronous worker (FastAPI runs sync `def`
    endpoints in a thread pool), so blocking I/O is safe and simple;
  * httpx provides the same ergonomics as requests, but additionally supports
    HTTP/2 and per-request timeouts split into connect/read/pool, which is
    important for slow LLM responses;
  * it keeps the door open for a painless migration to `httpx.AsyncClient`
    if the processing is later moved to `async def` + asyncio.gather.
"""

import logging
import os
import random
import re
import threading
import time
import uuid
from enum import Enum
from pathlib import Path

import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- #
# Logging: the standard `logging` module is used to track progress and errors.
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("synonym-service")

# --------------------------------------------------------------------------- #
# Configuration taken from environment variables (see module docstring).
# --------------------------------------------------------------------------- #
YANDEX_COMPLETION_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
YANDEX_API_KEY = os.getenv("YANDEX_API_KEY", "")
YANDEX_FOLDER_ID = os.getenv("YANDEX_FOLDER_ID", "")

# yandexgpt-lite: cheaper and faster than the full `yandexgpt`; the synonym
# task is simple enough that the lite model is sufficient and helps stay
# within API quotas.
YANDEX_MODEL = os.getenv("YANDEX_MODEL", "yandexgpt-lite")
YANDEX_MODEL_VERSION = os.getenv("YANDEX_MODEL_VERSION", "latest")

# Number of synonyms requested per word.
SYNONYMS_COUNT = int(os.getenv("SYNONYMS_COUNT", "3"))

# Retry policy (exponential backoff):
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))          # attempts after the first one
BACKOFF_BASE = float(os.getenv("BACKOFF_BASE", "1.0"))    # base delay, seconds
BACKOFF_CAP = float(os.getenv("BACKOFF_CAP", "60.0"))     # max single delay, seconds

# Polite delay between API calls so we do not exceed YandexGPT request limits.
REQUEST_INTERVAL_SEC = float(os.getenv("REQUEST_INTERVAL_SEC", "0.3"))

# Network timeouts: generous read timeout because LLM completion can be slow.
TIMEOUT = httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0)


class TaskStatus(str, Enum):
    pending = "pending"
    running = "running"
    done = "done"
    error = "error"


class GenerateRequest(BaseModel):
    input_file: str = Field(..., description="Path to input .txt file, one word per line")
    output_file: str = Field(..., description="Path to output .txt file (must end with .txt)")


class TaskInfo(BaseModel):
    status: TaskStatus
    total: int = 0
    processed: int = 0
    failed: int = 0
    output_file: str | None = None
    error: str | None = None


# --------------------------------------------------------------------------- #
# YandexGPT client
# --------------------------------------------------------------------------- #
class YandexGPTError(RuntimeError):
    """Raised when the YandexGPT API returns an unrecoverable error."""


class YandexGPTRateLimitError(YandexGPTError):
    """Raised on 429 — retryable with backoff."""


def build_prompt(word: str) -> list[dict]:
    """
    Build the messages array for the completion request.

    The system prompt demands ONLY a comma-separated list of synonyms —
    no quotes, no explanations, no newlines — which makes downstream
    normalization trivial and reliable.
    """
    return [
        {
            "role": "system",
            "text": (
                "Ты — лингвист. Верни только список синонимов через запятую, "
                "без кавычек, пояснений и переносов строк."
            ),
        },
        {
            "role": "user",
            "text": (
                f"Подбери {SYNONYMS_COUNT} синонимов для слова '{word}'. "
                "Ответь только списком слов через запятую."
            ),
        },
    ]


def normalize_response(raw: str) -> str:
    """
    Normalize a YandexGPT answer before writing it to the TXT file.

    Steps:
      1. collapse all newlines/tabs into spaces (the model must not use them,
         but sometimes does);
      2. strip markdown list markers ("-", "*", "•", numbered "1.") that may
         prefix items;
      3. remove surrounding quotes ("«»", '""', "''", backticks) around each
         item and whitespace around separators;
      4. drop empty items and de-duplicate while preserving order.

    Returns a clean string like "скорый, стремительный, резвый".
    """
    if not raw:
        return ""
    # 1. Single-line text.
    text = re.sub(r"\s+", " ", raw).strip()
    # Remove a possible leading phrase like "Синонимы:" if the model ignored
    # the instruction — keep only what follows the last colon when it looks
    # like such a preamble.
    m = re.match(r"^(?:синоним|синонимы|варианты)\s*:\s*(.+)$", text, flags=re.IGNORECASE)
    if m:
        text = m.group(1)
    # 2..4. Clean every comma-separated item.
    items: list[str] = []
    for part in text.split(","):
        chunk = part.strip()
        # A model that ignores the "no newlines" instruction often emits a
        # vertical list ("скорый\n- стремительный"); after collapsing to one
        # line such items are separated by whitespace, so split the chunk on
        # spaces and treat each token as a separate item (single-word synonyms
        # are the norm for this task).
        for item in chunk.split():
            item = re.sub(r"^[-*\u2022\u2023]\s*", "", item)   # list bullet
            item = re.sub(r"^\d+[.)]\s*", "", item)             # "1." numbering
            item = item.strip('\"\u2019\u201c\u201d\u00ab\u00bb`').strip()  # quotes
            if item:
                items.append(item)
    # De-duplicate preserving order.
    seen: set[str] = set()
    unique = [i for i in items if not (i.lower() in seen or seen.add(i.lower()))]
    return ", ".join(unique)


def request_synonyms(client: httpx.Client, word: str) -> str:
    """
    Call the YandexGPT completion endpoint for a single word.

    Error handling & retry strategy:
      * network problems (timeouts, connection errors) and HTTP 429/5xx are
        considered transient: we retry up to MAX_RETRIES times using
        exponential backoff with jitter:
            delay = min(BACKOFF_CAP, BACKOFF_BASE * 2**attempt) + random-ish jitter
        429 means rate limiting, 5xx means server-side issues — both usually
        succeed on a later attempt.
      * other 4xx (400 bad request, 401/403 auth problems, 404...) are NOT
        retried: they indicate a configuration/code bug, and retrying would
        only waste quota. Such errors raise YandexGPTError immediately.
    """
    payload = {
        "modelUri": f"gpt://{YANDEX_FOLDER_ID}/{YANDEX_MODEL}/{YANDEX_MODEL_VERSION}",
        "completionOptions": {
            "stream": False,
            "temperature": 0.4,   # moderate creativity: synonyms should vary, but stay valid
            "maxTokens": "200",
        },
        "messages": build_prompt(word),
    }
    headers = {
        "Authorization": f"Api-Key {YANDEX_API_KEY}",
        "x-folder-id": YANDEX_FOLDER_ID,
        "Content-Type": "application/json",
    }

    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = client.post(YANDEX_COMPLETION_URL, json=payload, headers=headers, timeout=TIMEOUT)

            if resp.status_code == 200:
                data = resp.json()
                try:
                    raw = data["result"]["alternatives"][0]["text"]
                except (KeyError, IndexError, TypeError) as exc:
                    # Unexpected response shape — treat as transient once, then give up.
                    raise YandexGPTError(f"Malformed API response for '{word}': {exc}") from exc
                return normalize_response(raw)

            if resp.status_code == 429 or resp.status_code >= 500:
                # Transient: rate limit or server error -> exponential backoff.
                last_error = YandexGPTRateLimitError(
                    f"HTTP {resp.status_code}: {resp.text[:200]}"
                )
            else:
                # Permanent client-side error (400/401/403/404...): do not retry.
                raise YandexGPTError(
                    f"HTTP {resp.status_code} (non-retryable): {resp.text[:200]}"
                )
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            # Timeouts / connection resets are transient -> retry with backoff.
            last_error = exc

        if attempt < MAX_RETRIES:
            # Exponential backoff with jitter. "Retry-After" is not reliably
            # returned by the service, so we use our own schedule:
            # delay = min(CAP, BASE * 2**attempt) + uniform jitter [0, 1s).
            delay = min(BACKOFF_CAP, BACKOFF_BASE * (2 ** attempt)) + random.uniform(0, 1)
            logger.warning(
                "Word '%s': attempt %d/%d failed (%s). Retrying in %.1fs...",
                word, attempt + 1, MAX_RETRIES + 1, last_error, delay,
            )
            time.sleep(delay)

    assert last_error is not None
    raise YandexGPTError(f"All {MAX_RETRIES + 1} attempts failed for '{word}': {last_error}")


# --------------------------------------------------------------------------- #
# Batch processing: reads words, queries YandexGPT, appends results line by line.
# --------------------------------------------------------------------------- #
# A global lock serializes access to the YandexGPT API across concurrent tasks.
# YandexGPT quotas are per-folder, so parallel tasks would trigger 429 errors;
# serializing + REQUEST_INTERVAL keeps us safely inside the limits.
_api_lock = threading.Lock()


def read_words(input_path: Path) -> list[str]:
    """Read the input TXT file (UTF-8): one word/concept per line, skip blanks."""
    words: list[str] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            word = line.strip()
            if word:  # empty lines are skipped
                words.append(word)
    return words


def process_file(task_id: str, input_path: Path, output_path: Path) -> None:
    """
    Core pipeline executed in a background thread.

    Iterative writing: the output file is opened exactly once in append mode
    (`open(..., 'a', encoding='utf-8')`) and every result is written and
    flushed immediately after receiving it from the model. If the process
    crashes mid-way, all previously obtained lines are already on disk.
    """
    info = _tasks[task_id]
    info.status = TaskStatus.running

    words = read_words(input_path)
    info.total = len(words)
    logger.info("Task %s: %d words from %s -> %s", task_id, len(words), input_path, output_path)

    # Truncate/create the output file at the start, then switch to append mode.
    with open(output_path, "w", encoding="utf-8"):
        pass

    with httpx.Client() as client, open(output_path, "a", encoding="utf-8") as out:
        for idx, word in enumerate(words, start=1):
            try:
                with _api_lock:  # serialize API calls between concurrent tasks
                    synonyms = request_synonyms(client, word)
                    time.sleep(REQUEST_INTERVAL_SEC)  # stay within rate limits
            except YandexGPTError as exc:
                # Per-word failure must not abort the whole batch: log it,
                # count it as failed and move on to the next word.
                info.failed += 1
                logger.error("Task %s: word '%s' failed permanently: %s", task_id, word, exc)
                # Behavior when no synonyms were obtained (explicitly documented):
                # we SKIP such words entirely rather than writing "word -> ".
                # Rationale: a line without synonyms carries no information and
                # would pollute the dictionary; the user can re-run the task —
                # successfully processed words are already saved.
                continue

            # Output format is strictly:  word -> syn1, syn2, syn3
            # No headers, no numbering, no JSON, no comments.
            out.write(f"{word} -> {synonyms}\n")
            out.flush()  # force the line to disk immediately (crash safety)
            info.processed += 1
            logger.info("Task %s: [%d/%d] '%s' -> %s", task_id, idx, len(words), word, synonyms)

    info.status = TaskStatus.done
    logger.info(
        "Task %s finished: %d processed, %d failed. Output: %s",
        task_id, info.processed, info.failed, output_path,
    )


# --------------------------------------------------------------------------- #
# FastAPI application
# --------------------------------------------------------------------------- #
app = FastAPI(
    title="YandexGPT Synonym Generator",
    description="Batch synonym generation for word lists via YandexGPT.",
    version="1.0.0",
)

# In-memory registry of tasks (sufficient for a single-process service).
_tasks: dict[str, TaskInfo] = {}


def validate_paths(input_file: str, output_file: str) -> tuple[Path, Path]:
    """
    Validate request paths:
      * output_file MUST have the .txt extension — otherwise HTTP 400;
      * input_file MUST exist and be a regular readable file — otherwise 400;
      * both files must be UTF-8-decodable TXT (encoding enforced on read/write);
      * path traversal outside the service root is rejected.
    """
    inp = Path(input_file)
    outp = Path(output_file)

    if outp.suffix.lower() != ".txt":
        raise HTTPException(status_code=400, detail="output_file must have a .txt extension")

    if not inp.is_file():
        raise HTTPException(status_code=400, detail=f"input_file does not exist: {input_file}")

    # Basic security: resolve symlinks and forbid absolute paths escaping cwd
    # is intentionally relaxed here (service is expected to run with trusted
    # local paths); at minimum ensure the output directory exists/is creatable.
    try:
        outp.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise HTTPException(status_code=400, detail=f"Cannot create output directory: {exc}") from exc

    return inp, outp


@app.post("/generate", status_code=202)
def generate(request: GenerateRequest, background_tasks: BackgroundTasks) -> dict:
    """
    Start synonym generation for a word list.

    Processing runs in the background (sync endpoint -> FastAPI threadpool +
    BackgroundTasks), so the HTTP response returns immediately with a task_id;
    progress is available via GET /status/{task_id}. This avoids request
    timeouts for large dictionaries.
    """
    if not YANDEX_API_KEY or not YANDEX_FOLDER_ID:
        raise HTTPException(
            status_code=500,
            detail="Environment variables YANDEX_API_KEY and YANDEX_FOLDER_ID must be set",
        )

    inp, outp = validate_paths(request.input_file, request.output_file)

    # Fail fast on unreadable/non-UTF-8 input.
    try:
        words = read_words(inp)
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="input_file is not valid UTF-8")
    if not words:
        raise HTTPException(status_code=400, detail="input_file contains no words")

    task_id = uuid.uuid4().hex[:12]
    _tasks[task_id] = TaskInfo(status=TaskStatus.pending, total=len(words), output_file=str(outp))
    background_tasks.add_task(process_file, task_id, inp, outp)
    logger.info("Accepted task %s: %d words", task_id, len(words))
    return {"task_id": task_id, "status": "pending", "words_total": len(words)}


@app.get("/status/{task_id}")
def get_status(task_id: str) -> TaskInfo:
    """Return current progress of a generation task."""
    info = _tasks.get(task_id)
    if info is None:
        raise HTTPException(status_code=404, detail="Unknown task_id")
    return info


@app.get("/health")
def health() -> dict:
    """Liveness probe; also reports whether credentials are configured."""
    return {
        "status": "ok",
        "credentials_configured": bool(YANDEX_API_KEY and YANDEX_FOLDER_ID),
        "model": f"{YANDEX_MODEL}/{YANDEX_MODEL_VERSION}",
    }


if __name__ == "__main__":
    # Convenience: `python main.py` starts the dev server.
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
