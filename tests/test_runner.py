"""Plumbing tests for web/runner.py, per DESIGN.md section 6.1.

Two things here are worth pinning down. The first is that the argv built for
each pipeline stage is the same command a person would type, because that
equivalence is the entire reason the viewer needs no changes to train.py. The
second is event fan-out, which had a real bug: a single shared queue delivered
each event to exactly one listener, so the history replay and the live stream
double-sent, and two browser tabs stole events from each other.

Neither test starts a subprocess, so the whole file runs in milliseconds.
"""

import queue
import threading

import pytest

from web.runner import Job, Stage, build_stages


# --- building the commands -------------------------------------------------


def stage_names(**kwargs) -> list[str]:
    defaults = dict(source=None, preset="tiny", steps=None, prompt="ROMEO:",
                    overfit_gate=False, new_tokenizer=False)
    return [stage.name for stage in build_stages(**{**defaults, **kwargs})]


def test_training_on_existing_tokens_is_a_single_stage():
    assert stage_names() == ["train"]


def test_a_corpus_adds_a_prepare_stage():
    assert stage_names(source="data/raw/x.txt") == ["prepare", "train"]


def test_asking_for_a_new_vocab_adds_the_tokenizer_stage_first():
    assert stage_names(source="data/raw/x.txt", new_tokenizer=True) == [
        "tokenizer", "prepare", "train"
    ]


def test_a_new_vocab_without_a_corpus_has_nothing_to_train_on():
    """new_tokenizer is meaningless with no source, and must not produce a
    tokenizer stage with no --input to give it."""
    assert stage_names(new_tokenizer=True) == ["train"]


def test_the_gate_runs_before_training_and_writes_no_checkpoint():
    stages = build_stages(source=None, preset="tiny", steps=100, prompt="ROMEO:",
                          overfit_gate=True, new_tokenizer=False)
    assert [stage.name for stage in stages] == ["gate", "train"]

    gate, train = stages
    assert "--overfit-one-batch" in gate.argv
    # A checkpoint of a deliberately memorised batch is worth nothing.
    assert "--out" not in gate.argv
    assert "--out" in train.argv


def test_steps_applies_to_the_real_run_and_not_to_the_gate():
    """--steps means max_steps for a run but the gate's own step count, so
    passing it to both would silently shorten the gate as well."""
    gate, train = build_stages(source=None, preset="tiny", steps=50, prompt="p",
                               overfit_gate=True, new_tokenizer=False)
    assert "--steps" not in gate.argv
    assert train.argv[train.argv.index("--steps") + 1] == "50"


@pytest.mark.parametrize("kwargs", [
    {"source": None},
    {"source": "data/raw/x.txt", "new_tokenizer": True},
])
def test_every_stage_is_unbuffered(kwargs):
    """Without -u, Python block-buffers stdout to a pipe and the entire run
    would reach the browser in one burst at the end. RunLogger's flush-per-line
    covers the log file, not stdout."""
    defaults = dict(source=None, preset="tiny", steps=None, prompt="p",
                    overfit_gate=True, new_tokenizer=False)
    for stage in build_stages(**{**defaults, **kwargs}):
        assert "-u" in stage.argv, f"{stage.name} is buffered"
        assert stage.argv[2] == "-m", f"{stage.name} does not run a module"


def test_stages_invoke_the_tinygpt_package_entry_points():
    stages = build_stages(source="data/raw/x.txt", preset="small", steps=None,
                          prompt="p", overfit_gate=False, new_tokenizer=True)
    modules = [stage.argv[3] for stage in stages]
    assert modules == [
        "tinygpt.tokenizer.train_tokenizer",
        "tinygpt.data.prepare",
        "tinygpt.train",
    ]


# --- event fan-out ---------------------------------------------------------


def idle_job() -> Job:
    """A Job that is never started, so nothing races with the assertions."""
    return Job([Stage("noop", ["true"])])


def drain(stream: queue.Queue) -> list[dict]:
    events = []
    while True:
        try:
            events.append(stream.get_nowait())
        except queue.Empty:
            return events


def test_a_subscriber_gets_the_backlog_once_and_new_events_once():
    """The regression that prompted this file: replay plus live stream used to
    deliver everything that happened before the connection twice."""
    job = idle_job()
    job._emit({"type": "log", "text": "before"})

    backlog, stream = job.subscribe()
    job._emit({"type": "log", "text": "after"})

    assert [event["text"] for event in backlog] == ["before"]
    assert [event["text"] for event in drain(stream)] == ["after"]


def test_two_subscribers_each_get_every_event():
    """One shared queue would hand each event to whichever tab asked first."""
    job = idle_job()
    _, first = job.subscribe()
    _, second = job.subscribe()

    job._emit({"type": "log", "text": "one"})
    job._emit({"type": "log", "text": "two"})

    assert [e["text"] for e in drain(first)] == ["one", "two"]
    assert [e["text"] for e in drain(second)] == ["one", "two"]


def test_unsubscribing_stops_delivery():
    job = idle_job()
    _, stream = job.subscribe()
    job.unsubscribe(stream)
    job._emit({"type": "log", "text": "ignored"})
    assert drain(stream) == []


def test_unsubscribing_twice_is_harmless():
    """The server unsubscribes in a finally block, which can run after the
    loop already cleaned up."""
    job = idle_job()
    _, stream = job.subscribe()
    job.unsubscribe(stream)
    job.unsubscribe(stream)


def test_history_keeps_every_event_for_a_late_subscriber():
    job = idle_job()
    for index in range(5):
        job._emit({"type": "log", "text": str(index)})

    backlog, _ = job.subscribe()
    assert [event["text"] for event in backlog] == ["0", "1", "2", "3", "4"]
    assert len(job.snapshot()) == 5


def test_subscribing_while_events_are_being_emitted_loses_nothing():
    """The lock in subscribe() is what makes the backlog/stream seam exact.

    A writer thread emits continuously while this subscribes; backlog plus
    stream must reconstruct an unbroken prefix of the sequence with no gap
    and no duplicate.
    """
    job = idle_job()
    total = 300
    done = threading.Event()

    def writer():
        for index in range(total):
            job._emit({"type": "log", "text": str(index)})
        done.set()

    thread = threading.Thread(target=writer)
    thread.start()
    backlog, stream = job.subscribe()
    thread.join()
    done.wait(timeout=5)

    seen = [int(event["text"]) for event in backlog] + \
           [int(event["text"]) for event in drain(stream)]
    # Contiguous from zero, in order, no repeats.
    assert seen == list(range(len(seen)))
    assert len(seen) == total


# --- late-bound vocab size -------------------------------------------------


def test_the_train_stage_takes_its_vocab_size_from_the_prepared_shards(tmp_path):
    """A preset fixes vocab_size at 4096, but a freshly-trained BPE vocab is
    whatever the corpus supported. The real number only exists once prepare
    has written meta.json, so it is resolved when the stage starts."""
    from web.runner import resolve_vocab_size

    (tmp_path / "data" / "tokens").mkdir(parents=True)
    (tmp_path / "data" / "tokens" / "meta.json").write_text('{"vocab_size": 372}')

    *_, train = build_stages(source="data/raw/x.txt", preset="tiny", steps=None,
                             prompt="p", overfit_gate=False, new_tokenizer=True)
    argv = resolve_vocab_size(train, tmp_path)
    assert argv[argv.index("--vocab-size") + 1] == "372"


def test_only_the_train_stage_is_rewritten(tmp_path):
    """Note the tokenizer stage carries its own --vocab-size: that is
    train_tokenizer.py's target merge count, a different flag on a different
    command. The check is that resolve_vocab_size leaves every non-train
    stage's argv exactly as built."""
    from web.runner import resolve_vocab_size

    (tmp_path / "data" / "tokens").mkdir(parents=True)
    (tmp_path / "data" / "tokens" / "meta.json").write_text('{"vocab_size": 372}')

    stages = build_stages(source="data/raw/x.txt", preset="tiny", steps=None,
                          prompt="p", overfit_gate=True, new_tokenizer=True)
    for stage in stages[:-1]:
        assert resolve_vocab_size(stage, tmp_path) == stage.argv


def test_a_missing_meta_json_leaves_argv_alone(tmp_path):
    """train.py has its own vocab check with a better message than anything
    this could invent, so an unreadable meta.json defers to it."""
    from web.runner import resolve_vocab_size

    *_, train = build_stages(source="data/raw/x.txt", preset="tiny", steps=None,
                             prompt="p", overfit_gate=False, new_tokenizer=True)
    assert "--vocab-size" not in resolve_vocab_size(train, tmp_path)


def test_resolving_twice_does_not_duplicate_the_flag(tmp_path):
    """stage.argv is reassigned in place when a stage starts; a retry or a
    resumed job must not append a second --vocab-size."""
    from web.runner import resolve_vocab_size

    (tmp_path / "data" / "tokens").mkdir(parents=True)
    (tmp_path / "data" / "tokens" / "meta.json").write_text('{"vocab_size": 372}')

    *_, train = build_stages(source="data/raw/x.txt", preset="tiny", steps=None,
                             prompt="p", overfit_gate=False, new_tokenizer=True)
    train.argv = resolve_vocab_size(train, tmp_path)
    train.argv = resolve_vocab_size(train, tmp_path)
    assert train.argv.count("--vocab-size") == 1
