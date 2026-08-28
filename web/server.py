"""A dependency-free HTTP server for watching a training run.

Written on Python's stdlib http.server rather than Flask or FastAPI, for the
same reason the transformer in tinygpt/model.py is written out by hand: this
project's premise is that the interesting machinery should be visible. Server
-sent events in particular are worth seeing implemented -- they are just a
long-lived response with a particular content type, and every line is here.

    python -m web.server --port 8000

Bound to 127.0.0.1 and nothing else. There is no authentication, no CSRF
protection, and endpoints that start subprocesses and write files. That is
fine for a tool you run on your own machine to look at your own training run,
and it is emphatically not fine on a network, which is why the bind address
is not configurable.
"""

import argparse
import json
import mimetypes
import queue
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from web import introspect
from web.runner import REPO_ROOT, Job, build_stages

STATIC_DIR = Path(__file__).resolve().parent / "static"
UPLOAD_DIR = REPO_ROOT / "data" / "raw"

# The cost of an upload is dominated by learning a BPE vocab from it, and the
# hand-rolled trainer in tinygpt/ is O(merges x *distinct words*), not
# O(merges x corpus): it collapses the text to a unique-word frequency table
# before it merges anything. English prose runs out of new words long before it
# runs out of bytes, so the cost grows far more slowly than the file does.
# Measured here: 3.5 MB of Conan Doyle is 3.2x the bytes of 1.1 MB of
# Shakespeare but only 1.9x the distinct words, and learns a 4096-token vocab
# in 117s against 57s. The trainer prints merge progress and an ETA throughout,
# so a wait that long is visible rather than mysterious.
#
# The cap that remains is a memory guard, not a time one: _upload reads the
# whole body with rfile.read() and the adapter then does path.read_text(), so
# an unbounded upload would be a way to run this machine out of memory by
# mistyping a filename.
MAX_UPLOAD_BYTES = 16 * 1024 * 1024

# How long an idle SSE connection waits before sending a keep-alive comment.
# Without one, a proxy or a laptop sleeping can drop a quiet connection and
# the page would stop updating with no visible error.
SSE_KEEPALIVE_SECONDS = 15.0


class JobRegistry:
    """Holds the one job that may run at a time.

    One at a time is a real constraint, not laziness: two training processes
    on one GPU make both slower and the numbers meaningless to compare.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._job: Job | None = None

    @property
    def current(self) -> Job | None:
        with self._lock:
            return self._job

    def start(self, stages) -> Job:
        with self._lock:
            if self._job is not None and self._job.running:
                raise RuntimeError(
                    "a run is already in progress; stop it before starting another"
                )
            self._job = Job(stages)
            self._job.start()
            return self._job

    def stop(self) -> bool:
        job = self.current
        if job is None or not job.running:
            return False
        job.stop()
        return True


REGISTRY = JobRegistry()


class Handler(BaseHTTPRequestHandler):
    server_version = "tinygpt-viewer"

    # --- small helpers -----------------------------------------------------

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def log_message(self, fmt: str, *args) -> None:
        # The default logs every request including the SSE stream, which
        # drowns out anything useful. Keep errors, drop the rest.
        if not str(args[1] if len(args) > 1 else "").startswith(("2", "3")):
            super().log_message(fmt, *args)

    # --- routing -----------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - name fixed by BaseHTTPRequestHandler
        route = urlparse(self.path).path
        try:
            if route in ("/", "/index.html"):
                self._serve_static("index.html")
            elif route.startswith("/static/"):
                self._serve_static(route[len("/static/"):])
            elif route == "/api/events":
                self._stream_events()
            elif route == "/api/checkpoints":
                self._send_json({"checkpoints": introspect.list_checkpoints()})
            elif route == "/api/status":
                self._send_json(self._status())
            else:
                self._send_json({"error": f"no route {route}"}, status=404)
        except BrokenPipeError:
            pass  # the browser navigated away mid-response; nothing to do
        except Exception as error:  # noqa: BLE001
            self._fail(error)

    def do_POST(self) -> None:  # noqa: N802
        route = urlparse(self.path).path
        try:
            if route == "/api/upload":
                self._send_json(self._upload())
            elif route == "/api/run":
                self._send_json(self._run(self._read_json()))
            elif route == "/api/stop":
                self._send_json({"stopped": REGISTRY.stop()})
            elif route == "/api/next-token":
                body = self._read_json()
                self._send_json(introspect.next_token_distribution(
                    checkpoint=body["checkpoint"],
                    prompt=body.get("prompt", ""),
                    temperature=float(body.get("temperature", 1.0)),
                    top_k=int(body.get("top_k", 0)),
                    top_p=float(body.get("top_p", 1.0)),
                    top_n=int(body.get("top_n", 20)),
                ))
            elif route == "/api/attention":
                body = self._read_json()
                self._send_json(introspect.attention_grid(
                    checkpoint=body["checkpoint"],
                    prompt=body.get("prompt", ""),
                    layer=int(body.get("layer", 0)),
                    head=int(body.get("head", 0)),
                ))
            elif route == "/api/generate":
                self._stream_generation(self._read_json())
            else:
                self._send_json({"error": f"no route {route}"}, status=404)
        except BrokenPipeError:
            pass
        except Exception as error:  # noqa: BLE001
            self._fail(error)

    def _fail(self, error: Exception) -> None:
        """Return the actual reason, with a traceback in the server console.

        A local tool that swallows its own errors is worse than useless when
        the thing being debugged is a training run.
        """
        traceback.print_exc()
        self._send_json({"error": f"{type(error).__name__}: {error}"}, status=500)

    # --- static files ------------------------------------------------------

    def _serve_static(self, relative: str) -> None:
        path = (STATIC_DIR / relative).resolve()
        # Refuse anything that escapes static/, including via "..".
        if not path.is_file() or STATIC_DIR not in path.parents:
            self._send_json({"error": f"no such file {relative}"}, status=404)
            return

        body = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")  # so an edit shows on reload
        self.end_headers()
        self.wfile.write(body)

    # --- the event stream --------------------------------------------------

    def _stream_events(self) -> None:
        """Server-sent events: one long response, one JSON object per event.

        SSE rather than a WebSocket because the traffic here is entirely one
        directional -- the server talks, the browser listens -- and SSE is
        plain HTTP with a reconnect built into the browser. A WebSocket would
        mean implementing a framing protocol by hand for no gain.

        The backlog replays first, so a browser that reloads mid-run redraws
        the complete chart rather than resuming from a blank one. subscribe()
        hands back the backlog and a private queue together under one lock, so
        the join between "what already happened" and "what happens next" is
        exact: nothing is sent twice and nothing falls through the gap.
        """
        job = REGISTRY.current
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        if job is None:
            self._sse({"type": "idle"})
            return

        backlog, stream = job.subscribe()
        try:
            for event in backlog:
                self._sse(event)
            self._sse({"type": "replayed", "count": len(backlog)})

            last_keepalive = time.time()
            while True:
                try:
                    event = stream.get(timeout=1.0)
                except queue.Empty:
                    if not job.running:
                        break
                    if time.time() - last_keepalive > SSE_KEEPALIVE_SECONDS:
                        self.wfile.write(b": keep-alive\n\n")
                        self.wfile.flush()
                        last_keepalive = time.time()
                    continue

                if event.get("type") == "eof":
                    break
                self._sse(event)
                last_keepalive = time.time()

            self._sse({"type": "eof"})
        finally:
            # Whether the loop ended or the browser hung up mid-stream, stop
            # filling a queue nobody is reading.
            job.unsubscribe(stream)

    def _sse(self, event: dict) -> None:
        """One SSE frame. The blank line is what ends it; the flush is what
        makes it arrive now rather than whenever the buffer happens to fill."""
        self.wfile.write(f"data: {json.dumps(event)}\n\n".encode("utf-8"))
        self.wfile.flush()

    # --- actions -----------------------------------------------------------

    def _status(self) -> dict:
        job = REGISTRY.current
        # max_upload_bytes travels with the status so the page can refuse an
        # oversized file before sending it, rather than duplicating the number
        # in JavaScript and letting the two drift.
        if job is None:
            return {"running": False, "stages": [],
                    "max_upload_bytes": MAX_UPLOAD_BYTES}
        return {
            "running": job.running,
            "stages": [stage.name for stage in job.stages],
            "started_at": job.started_at,
            "events": len(job.snapshot()),
            "max_upload_bytes": MAX_UPLOAD_BYTES,
        }

    def _drain(self, remaining: int) -> None:
        """Read and discard a request body, in constant memory."""
        while remaining > 0:
            chunk = self.rfile.read(min(64 * 1024, remaining))
            if not chunk:
                break
            remaining -= len(chunk)

    def _upload(self) -> dict:
        """Store an uploaded corpus under data/raw/.

        The body is the raw file bytes, not multipart: the browser sends it
        with fetch(file), which avoids parsing multipart by hand for the sake
        of one field.
        """
        length = int(self.headers.get("Content-Length") or 0)
        if length == 0:
            raise ValueError("empty upload")
        if length > MAX_UPLOAD_BYTES:
            # Drain before answering. A client that is still writing its body
            # when the response arrives gets a connection reset instead of the
            # response, so refusing early without draining means the caller
            # sees a generic network error rather than the sentence below --
            # which is the one thing that would have told them what to do.
            # Discarded in chunks, so the memory this limit protects is never
            # allocated, and bounded so an absurd upload cannot hold the
            # handler open indefinitely.
            self._drain(min(length, MAX_UPLOAD_BYTES))
            raise ValueError(
                f"{length:,} bytes exceeds the {MAX_UPLOAD_BYTES:,} byte limit. "
                "The limit is memory, not time: the upload is read into memory "
                "whole, and so is the corpus behind it."
            )

        raw = self.rfile.read(length)
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(
                "upload is not valid UTF-8 text; this trainer has one adapter "
                "and it reads .txt"
            ) from error

        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        name = f"upload-{time.strftime('%Y%m%d-%H%M%S')}.txt"
        path = UPLOAD_DIR / name
        path.write_text(text, encoding="utf-8")

        return {
            "source": f"data/raw/{name}",
            "bytes": len(raw),
            "characters": len(text),
            "lines": text.count("\n") + 1,
        }

    def _run(self, body: dict) -> dict:
        stages = build_stages(
            source=body.get("source") or None,
            preset=body.get("preset", "tiny"),
            steps=int(body["steps"]) if body.get("steps") else None,
            prompt=body.get("prompt", "ROMEO:"),
            overfit_gate=bool(body.get("overfit_gate", True)),
            new_tokenizer=bool(body.get("new_tokenizer", False)),
        )
        job = REGISTRY.start(stages)
        return {"started": True, "stages": [stage.name for stage in job.stages]}

    def _stream_generation(self, body: dict) -> None:
        """Continue a prompt, writing each new piece of text as it is produced.

        Newline-delimited JSON rather than server-sent events. SSE would be the
        obvious choice, but the browser's EventSource can only issue GET
        requests and a prompt belongs in a request body, so the page reads this
        with fetch() and a stream reader -- and NDJSON is less to parse than
        SSE framing once EventSource is not involved.

        No Content-Length is sent. BaseHTTPRequestHandler speaks HTTP/1.0, so
        the response ends when the connection closes, which is also how
        _stream_events above gets away with it.

        Stopping is free and needs no endpoint: if the browser aborts the
        fetch, the write below raises BrokenPipeError inside the sink, which
        propagates out through generate() and ends the loop. do_POST already
        swallows that exception.
        """
        checkpoint = body["checkpoint"]

        # Resolve the model before writing any headers, so a bad checkpoint or
        # a vocabulary mismatch is still a clean 500 with a JSON error rather
        # than a half-written stream the page has to guess about.
        model, metadata = introspect._load(checkpoint)
        # None, not "vocab.json": let the checkpoint's own embedded vocabulary
        # win where it has one. introspect._resolve_tokenizer owns that order.
        introspect._checked_tokenizer(model, metadata, checkpoint, None)

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        def write(payload: dict) -> None:
            self.wfile.write(f"{json.dumps(payload)}\n".encode("utf-8"))
            self.wfile.flush()  # per line, or the deltas sit in a buffer

        summary = introspect.generate_completion(
            checkpoint=checkpoint,
            prompt=body.get("prompt", ""),
            max_tokens=int(body.get("max_tokens", 200)),
            temperature=float(body.get("temperature", 0.8)),
            top_k=int(body.get("top_k", 0)),
            top_p=float(body.get("top_p", 1.0)),
            on_text=lambda delta: write({"delta": delta}),
        )

        summary.pop("continuation")  # already sent, delta by delta
        write({"done": True, **summary})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    # ThreadingHTTPServer, not HTTPServer: an open /api/events response lives
    # for the length of a training run, and a single-threaded server would
    # spend that whole run unable to answer anything else.
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"[server] tiny-gpt-trainer viewer on http://127.0.0.1:{args.port}")
    print(f"[server] serving {STATIC_DIR}")
    print("[server] local only; ctrl-c to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[server] stopping")
        REGISTRY.stop()
        server.shutdown()


if __name__ == "__main__":
    main()
