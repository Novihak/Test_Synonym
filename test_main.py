"""Offline tests for main.py: the YandexGPT endpoint is mocked via httpx transport."""
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

import main


# --------------------------------------------------------------------------- #
# normalize_response
# --------------------------------------------------------------------------- #
def test_normalize_strips_newlines_bullets_quotes():
    raw = 'Синонимы:\n1. "скорый"\n- стремительный\n* «резвый»,\n'
    assert main.normalize_response(raw) == "скорый, стремительный, резвый"


def test_normalize_dedupes_and_handles_empty():
    assert main.normalize_response("быстрый, Быстрый, , быстрый") == "быстрый"
    assert main.normalize_response("") == ""


# --------------------------------------------------------------------------- #
# request_synonyms with a mocked transport
# --------------------------------------------------------------------------- #
def make_client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def ok_response(text):
    return httpx.Response(200, json={"result": {"alternatives": [{"text": text}]}})


def test_request_success(monkeypatch):
    monkeypatch.setattr(main, "YANDEX_API_KEY", "k")
    monkeypatch.setattr(main, "YANDEX_FOLDER_ID", "f")

    def handler(request):
        assert request.headers["Authorization"] == "Api-Key k"
        assert request.headers["x-folder-id"] == "f"
        return ok_response("  скорый, стремительный,\nрезвый ")

    assert main.request_synonyms(make_client(handler), "быстрый") == \
        "скорый, стремительный, резвый"


def test_retry_on_429_then_success(monkeypatch):
    monkeypatch.setattr(main, "BACKOFF_BASE", 0.01)
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(429, text="rate limit")
        return ok_response("жилище, здание, кров")

    t0 = time.monotonic()
    out = main.request_synonyms(make_client(handler), "дом")
    assert out == "жилище, здание, кров"
    assert calls["n"] == 3
    assert time.monotonic() - t0 >= 0.02  # backoff actually slept


def test_timeout_is_retried(monkeypatch):
    monkeypatch.setattr(main, "BACKOFF_BASE", 0.01)
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ReadTimeout("boom", request=request)
        return ok_response("светило, луч, блеск")

    assert main.request_synonyms(make_client(handler), "свет") == "светило, луч, блеск"


def test_permanent_4xx_not_retried(monkeypatch):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(403, text="forbidden")

    with pytest.raises(main.YandexGPTError):
        main.request_synonyms(make_client(handler), "word")
    assert calls["n"] == 1  # no retries for non-retryable errors


def test_all_retries_exhausted(monkeypatch):
    monkeypatch.setattr(main, "MAX_RETRIES", 1)
    monkeypatch.setattr(main, "BACKOFF_BASE", 0.01)

    def handler(request):
        return httpx.Response(500, text="oops")

    with pytest.raises(main.YandexGPTError):
        main.request_synonyms(make_client(handler), "word")


# --------------------------------------------------------------------------- #
# FastAPI endpoints (generation pipeline end-to-end with mock transport)
# --------------------------------------------------------------------------- #
@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(main, "YANDEX_API_KEY", "test-key")
    monkeypatch.setattr(main, "YANDEX_FOLDER_ID", "test-folder")
    monkeypatch.setattr(main, "REQUEST_INTERVAL_SEC", 0.0)
    # Patch httpx.Client used inside process_file to use MockTransport.
    real_client = httpx.Client

    class PatchedClient(real_client):
        def __init__(self, *a, **kw):
            kw["transport"] = httpx.MockTransport(
                lambda req: ok_response("синонимА, синонимБ, синонимВ")
            )
            super().__init__(*a, **kw)

    monkeypatch.setattr(main.httpx, "Client", PatchedClient)
    return TestClient(main.app)


def write_input(tmp_path: Path) -> Path:
    p = tmp_path / "input.txt"
    p.write_text("быстрый\nдом\n\nсвет\n", encoding="utf-8")
    return p


def wait_done(tc, task_id, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        r = tc.get(f"/status/{task_id}")
        if r.json()["status"] in ("done", "error"):
            return r.json()
        time.sleep(0.05)
    raise AssertionError("task did not finish in time")


def test_generate_end_to_end(client, tmp_path):
    inp = write_input(tmp_path)
    out = tmp_path / "out" / "synonyms.txt"
    r = client.post("/generate", json={"input_file": str(inp), "output_file": str(out)})
    assert r.status_code == 202
    status = wait_done(client, r.json()["task_id"])
    assert status["status"] == "done" and status["processed"] == 3 and status["failed"] == 0

    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines == [
        "быстрый -> синонимА, синонимБ, синонимВ",
        "дом -> синонимА, синонимБ, синонимВ",
        "свет -> синонимА, синонимБ, синонимВ",
    ]


def test_output_must_be_txt(client, tmp_path):
    inp = write_input(tmp_path)
    r = client.post("/generate", json={"input_file": str(inp), "output_file": "out.json"})
    assert r.status_code == 400


def test_missing_input_file(client, tmp_path):
    r = client.post("/generate", json={"input_file": "/nope/missing.txt", "output_file": "o.txt"})
    assert r.status_code == 400


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["credentials_configured"] is True
