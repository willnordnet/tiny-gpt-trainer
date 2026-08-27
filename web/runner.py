"""Run the training pipeline as subprocesses and stream their output.

The viewer never imports the training loop. It shells out to the same
`python -m tinygpt.*` commands a person would type, and reads stdout. That
buys three things worth more than the convenience of an in-process call:

- `train.py` needs no changes at all, and cannot be broken by the viewer.
- A run that crashes, hangs, or exhausts memory takes down a child process,
  not the server, so the page stays up to show you what happened.
- Stop actually stops. Killing a process is reliable in a way that asking a
  tight MLX loop in a thread to please stop is not.

The cost is that the viewer can only see what the trainer prints, which is
what web/logparse.py is for.

A Job is up to four stages run in sequence; each stage is one subprocess:

    train tokenizer  ->  prepare tokens  ->  overfit gate  ->  train

The first is skipped when reusing an existing vocab, the second when reusing
existing token shards, and the third unless it is asked for. A failing gate
stops the pipeline before the real run starts, which is the whole point of
having a gate (DESIGN.md section 6.2).
"""

import json
import queue
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

from web.logparse import parse_line

REPO_ROOT = Path(__file__).resolve().parent.parent

# How long to let a stage exit politely after SIGTERM before SIGKILL. MLX
# needs a moment to release Metal buffers; a second is plenty and a hung
# process should not be able to wedge the server for longer than that.
TERMINATE_GRACE_SECONDS = 2.0


class Stage:
    """One subprocess in the pipeline, with a label the UI can show."""

    def __init__(self, name: str, argv: list[str]) -> None:
        self.name = name
        self.argv = argv
        # Set on the train stage: its --vocab-size cannot be decided until an
        # earlier stage has written the token shards. See resolve_vocab_size.
        self.needs_vocab_size = False
        self.tokens_dir = ""

    def __repr__(self) -> str:
        return f"Stage({self.name!r}, {shlex.join(self.argv)!r})"


def build_stages(
    *,
    source: str | None,
    preset: str,
    steps: int | None,
    prompt: str,
    overfit_gate: bool,
    new_tokenizer: bool,
    vocab_size: int = 4096,
    vocab_path: str = "vocab.json",
    tokens_dir: str = "data/tokens",
    out_dir: str = "checkpoints",
) -> list[Stage]:
    """Assemble the argv for each stage. Pure: builds lists, runs nothing.

    `source` is the corpus to (re)tokenize and prepare. Pass None to train on
    whatever is already in `tokens_dir`, which is the fast path for demoing
    against a corpus that has been prepared once already.

    Every command is spelled with `-u`. Without it Python block-buffers stdout
    when it is a pipe rather than a terminal, and the entire run would arrive
    in the browser as one burst at the end. RunLogger's flush-per-line covers
    the log *file*, not stdout, so this is not optional.
    """
    python = sys.executable
    stages: list[Stage] = []

    if source is not None and new_tokenizer:
        stages.append(Stage("tokenizer", [
            python, "-u", "-m", "tinygpt.tokenizer.train_tokenizer",
            "--input", source,
            "--vocab-size", str(vocab_size),
            "--out", vocab_path,
        ]))

    if source is not None:
        stages.append(Stage("prepare", [
            python, "-u", "-m", "tinygpt.data.prepare",
            "--input", source,
            "--vocab", vocab_path,
            "--out-dir", tokens_dir,
        ]))

    train_base = [
        python, "-u", "-m", "tinygpt.train",
        "--preset", preset,
        "--data", tokens_dir,
        "--vocab", vocab_path,
        "--prompt", prompt,
    ]

    if overfit_gate:
        # The gate deliberately gets no --out: it memorises one batch, and a
        # checkpoint of that is worth nothing.
        stages.append(Stage("gate", train_base + ["--overfit-one-batch"]))

    train_argv = train_base + ["--out", out_dir]
    if steps is not None:
        train_argv += ["--steps", str(steps)]

    train_stage = Stage("train", train_argv)
    # A preset fixes vocab_size at 4096, but a freshly-trained BPE vocab is
    # whatever the corpus supported -- a small file exhausts its pairs early.
    # The real number is not known until prepare has written meta.json, so the
    # train stage is marked to look it up just before it runs.
    train_stage.needs_vocab_size = True
    train_stage.tokens_dir = tokens_dir
    stages.append(train_stage)

    return stages


def resolve_vocab_size(stage: Stage, cwd: Path = REPO_ROOT) -> list[str]:
    """Append --vocab-size to the train stage, read from the prepared shards.

    prepare.py writes the tokenizer's real vocab_size into meta.json, and the
    embedding table has to be that size or the ids index the wrong rows.
    Reading it here rather than guessing means an uploaded corpus trains at
    whatever vocabulary its own text could support.

    Returns argv unchanged when there is no meta.json to read: train.py has
    its own check and a better error message than anything invented here.
    """
    if not stage.needs_vocab_size or "--vocab-size" in stage.argv:
        return stage.argv

    meta_path = cwd / stage.tokens_dir / "meta.json"
    try:
        vocab_size = json.loads(meta_path.read_text())["vocab_size"]
    except (OSError, ValueError, KeyError):
        return stage.argv

    return stage.argv + ["--vocab-size", str(int(vocab_size))]


class Job:
    """A running pipeline. One at a time; the server enforces that.

    Each listener (an open SSE connection) gets its OWN queue, handed out by
    subscribe() together with the history so far, both taken under one lock.
    That atomicity is the whole point: a listener must see every event exactly
    once, and an event arriving between "copy the history" and "start
    listening" would otherwise be delivered twice or not at all.

    A single shared queue would also mean two browser tabs stealing events
    from each other, since a queue delivers each item to exactly one consumer.

    The queues are unbounded on purpose. Dropping events to bound memory would
    mean the raw-log pane silently losing lines, and even a 2000-step run
    produces only a few thousand.
    """

    def __init__(self, stages: list[Stage], cwd: Path = REPO_ROOT) -> None:
        self.stages = stages
        self.cwd = cwd
        self.history: list[dict] = []
        self._subscribers: list[queue.Queue] = []
        self._history_lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        self._process_lock = threading.Lock()
        self._stop_requested = False
        self.finished = threading.Event()
        self.started_at = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)

    # --- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        """Ask the current stage to exit, then insist.

        Sets the stop flag first so _run does not helpfully start the next
        stage the moment this one dies.
        """
        self._stop_requested = True
        with self._process_lock:
            process = self._process
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=TERMINATE_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()

    @property
    def running(self) -> bool:
        return not self.finished.is_set()

    # --- event plumbing ----------------------------------------------------

    def _emit(self, event: dict) -> None:
        with self._history_lock:
            self.history.append(event)
            for subscriber in self._subscribers:
                subscriber.put(event)

    def subscribe(self) -> tuple[list[dict], queue.Queue]:
        """Everything so far, plus a queue receiving everything from now on.

        Both under one lock, so the seam between them is exact: no event is
        delivered twice, and none slips through the gap.
        """
        stream: queue.Queue = queue.Queue()
        with self._history_lock:
            backlog = list(self.history)
            self._subscribers.append(stream)
        return backlog, stream

    def unsubscribe(self, stream: queue.Queue) -> None:
        with self._history_lock:
            if stream in self._subscribers:
                self._subscribers.remove(stream)

    def snapshot(self) -> list[dict]:
        """Every event so far, for callers that only want a count."""
        with self._history_lock:
            return list(self.history)

    # --- the worker --------------------------------------------------------

    def _run(self) -> None:
        try:
            for index, stage in enumerate(self.stages):
                if self._stop_requested:
                    self._emit({"type": "stage_skipped", "stage": stage.name})
                    continue

                # Late-bind anything an earlier stage had to produce first.
                stage.argv = resolve_vocab_size(stage, self.cwd)

                self._emit({
                    "type": "stage_start",
                    "stage": stage.name,
                    "index": index,
                    "total": len(self.stages),
                    "command": shlex.join(stage.argv),
                })

                code = self._run_stage(stage)

                self._emit({
                    "type": "stage_end",
                    "stage": stage.name,
                    "returncode": code,
                    "stopped": self._stop_requested,
                })

                if code != 0:
                    # A non-zero exit ends the pipeline. Most importantly this
                    # is how a failed overfit gate prevents the real run: the
                    # gate exits non-zero precisely so it can be used this way.
                    self._emit({
                        "type": "pipeline_failed",
                        "stage": stage.name,
                        "returncode": code,
                    })
                    break
            else:
                self._emit({"type": "pipeline_done"})
        except Exception as error:  # noqa: BLE001 - the browser deserves the reason
            self._emit({"type": "pipeline_error", "error": f"{type(error).__name__}: {error}"})
        finally:
            self.finished.set()
            # Wakes every SSE reader blocked on an empty queue so each can
            # notice the job is over rather than sitting on a get() forever.
            self._emit({"type": "eof"})

    def _run_stage(self, stage: Stage) -> int:
        process = subprocess.Popen(
            stage.argv,
            cwd=str(self.cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # a traceback is output worth seeing too
            text=True,
            bufsize=1,  # line buffered on our side; -u handles the child's
            encoding="utf-8",
            errors="replace",
        )
        with self._process_lock:
            self._process = process

        pending = None
        assert process.stdout is not None
        for line in process.stdout:
            event, pending = parse_line(line, pending)
            event["stage"] = stage.name
            self._emit(event)

        process.wait()
        with self._process_lock:
            self._process = None
        return process.returncode


if __name__ == "__main__":
    # A standalone demo: run the overfit gate and print the parsed event
    # stream, proving the subprocess plumbing works with no server involved.
    job = Job(build_stages(
        source=None,
        preset="tiny",
        steps=120,
        prompt="ROMEO:",
        overfit_gate=True,
        new_tokenizer=False,
    ))
    print("[runner] stages:", [s.name for s in job.stages])
    backlog, stream = job.subscribe()
    job.start()
    while True:
        event = stream.get()
        if event["type"] == "eof":
            break
        if event["type"] in {"stage_start", "stage_end", "gate", "pipeline_done",
                             "pipeline_failed", "overfit_step", "step"}:
            print(f"[runner] {event}")
    print("[runner] finished")
