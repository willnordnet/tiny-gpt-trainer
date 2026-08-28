"""Smoke tests for the viewer's HTTP layer, per DESIGN.md section 6.1.

These exist because of a bug that reached master. `_checked_tokenizer` grew an
argument, the three call sites inside `web/introspect.py` were updated, and the
fourth -- in `server.py`'s generation endpoint -- was not. Every unit test still
passed, because nothing anywhere exercised `server.py`. `POST /api/generate`
raised TypeError for anyone who pressed Generate.

So the bar here is deliberately low and deliberately wide: call every route
once against a real server on a real socket, and assert the shape of what comes
back. Not what the panels do with it -- `test_introspect.py` covers that -- just
that the wiring between the HTTP layer and the code beneath it still connects.
A signature that drifts again fails here in milliseconds.
"""

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from tinygpt.config import ModelConfig
from tinygpt.model import TinyGPT
from tinygpt.tokenizer.tokenizer import BPETokenizer
from tinygpt.train import save_checkpoint
from web import server as server_module


@pytest.fixture
def tokenizer() -> BPETokenizer:
    """vocab_size 258: the 256 byte tokens plus two merges, small enough that a
    checkpoint built against it costs nothing to write."""
    return BPETokenizer(merges=[(116, 104), (256, 101)])


@pytest.fixture
def live_server(tmp_path, tokenizer, monkeypatch):
    """A real server on a real port, over a repo rooted at tmp_path.

    Port 0 lets the OS pick a free one, so a developer running the viewer while
    the suite runs does not collide with it.
    """
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    cfg = ModelConfig(vocab_size=tokenizer.vocab_size, n_layers=1, d_model=32,
                      n_heads=4, context_len=16)
    save_checkpoint(checkpoints / "tiny-step10.safetensors", TinyGPT(cfg),
                    "tiny", 10, 1.5, tokenizer=tokenizer)

    # The handler resolves checkpoints through introspect, which reads its own
    # REPO_ROOT, and writes uploads through the server's UPLOAD_DIR.
    monkeypatch.setattr(server_module.introspect, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(server_module, "UPLOAD_DIR", tmp_path / "data" / "raw")
    server_module.introspect._model_cache.clear()

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server_module.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def get(base: str, route: str):
    with urllib.request.urlopen(f"{base}{route}", timeout=30) as response:
        return response.status, response.read()


def post(base: str, route: str, payload: dict):
    request = urllib.request.Request(
        f"{base}{route}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:      # the server's own 500s
        return error.code, error.read()


# --- the page itself -------------------------------------------------------


def test_the_page_and_its_two_static_files_are_served(live_server):
    """index.html references exactly these; a rename that broke one would
    otherwise only show up as an unstyled or inert page in a browser."""
    status, body = get(live_server, "/")
    assert status == 200
    assert b"<title>tiny-GPT-trainer</title>" in body

    for route in ["/static/style.css", "/static/app.js"]:
        status, body = get(live_server, route)
        assert status == 200, route
        assert body, route


def test_a_path_escape_is_refused(live_server):
    """The server binds to localhost with no auth, so this is not the last line
    of defence -- but static file serving that follows .. is a bug regardless."""
    with pytest.raises(urllib.error.HTTPError) as raised:
        get(live_server, "/static/../../../etc/passwd")
    assert raised.value.code in (403, 404)


# --- read routes -----------------------------------------------------------


def test_status_reports_an_idle_registry(live_server):
    status, body = get(live_server, "/api/status")
    assert status == 200
    payload = json.loads(body)
    assert payload["running"] is False
    assert payload["stages"] == []
    # The page reads the cap from here rather than hardcoding it.
    assert payload["max_upload_bytes"] == server_module.MAX_UPLOAD_BYTES


def test_checkpoints_lists_the_file_with_its_metadata_and_vocab_verdict(live_server):
    status, body = get(live_server, "/api/checkpoints")
    assert status == 200
    entries = json.loads(body)["checkpoints"]

    assert len(entries) == 1
    entry = entries[0]
    assert entry["preset"] == "tiny"
    assert entry["step"] == 10
    assert entry["val_loss"] == pytest.approx(1.5)
    # A checkpoint carrying its own vocabulary cannot be mispaired.
    assert entry["vocab"] == "verified"


# --- the panel routes, which is where the signature drift happened ---------


def test_next_token_returns_a_distribution_before_and_after_the_knobs(live_server):
    status, body = post(live_server, "/api/next-token", {
        "checkpoint": "checkpoints/tiny-step10.safetensors",
        "prompt": "the", "temperature": 0.8, "top_k": 5, "top_p": 1.0, "top_n": 5,
    })
    assert status == 200, body
    data = json.loads(body)
    assert len(data["candidates"]) == 5
    assert {"token", "prob", "prob_after", "eliminated"} <= set(data["candidates"][0])
    assert data["entropy_raw"] > 0


def test_attention_returns_a_causal_grid(live_server):
    status, body = post(live_server, "/api/attention", {
        "checkpoint": "checkpoints/tiny-step10.safetensors",
        "prompt": "the the", "layer": 0, "head": 0,
    })
    assert status == 200, body
    data = json.loads(body)
    weights = data["weights"]
    assert len(weights) == len(data["tokens"])
    # Nothing above the diagonal: a token cannot attend to its own future.
    for row, attention in enumerate(weights):
        assert all(w == 0 for w in attention[row + 1:])


def test_generate_streams_ndjson_deltas_then_a_done_summary(live_server):
    """The regression that prompted this file: this route pre-resolves the
    model to keep a vocabulary mismatch a clean error, and that call went stale."""
    status, body = post(live_server, "/api/generate", {
        "checkpoint": "checkpoints/tiny-step10.safetensors",
        "prompt": "the", "max_tokens": 5,
    })
    assert status == 200, body

    lines = [json.loads(line) for line in body.decode().splitlines() if line]
    assert lines, "no NDJSON at all"

    *deltas, final = lines
    assert all("delta" in line for line in deltas)
    assert final["done"] is True
    assert final["step"] == 10
    assert final["tokens_per_second"] > 0


# --- errors ----------------------------------------------------------------


def test_an_upload_over_the_cap_is_a_json_error_not_a_traceback(live_server):
    request = urllib.request.Request(
        f"{live_server}/api/upload",
        data=b"x" * (server_module.MAX_UPLOAD_BYTES + 1),
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as raised:
        urllib.request.urlopen(request, timeout=60)

    assert raised.value.code == 500
    assert "limit" in json.loads(raised.value.read())["error"]


def test_a_missing_checkpoint_is_a_json_error_not_a_traceback(live_server):
    """Every handler is wrapped so the page gets {"error": ...} to show in its
    banner. A bare traceback would close the connection with nothing to read."""
    status, body = post(live_server, "/api/next-token", {
        "checkpoint": "checkpoints/does-not-exist.safetensors", "prompt": "the",
    })
    assert status == 500
    assert "error" in json.loads(body)


def test_an_unknown_route_is_404(live_server):
    with pytest.raises(urllib.error.HTTPError) as raised:
        get(live_server, "/api/nope")
    assert raised.value.code == 404
