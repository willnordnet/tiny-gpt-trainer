"""Plumbing tests for web/logparse.py, per DESIGN.md section 6.1.

The viewer reads the trainer's stdout rather than being wired into the
training loop, which keeps train.py untouched but couples the two through a
log *format*. These tests are what makes that coupling safe: they feed the
parser real lines, copied verbatim from logs/, so a format string changing in
train.py breaks a test here in milliseconds instead of silently emptying a
chart in the browser.

No MLX, no subprocess, no server - just strings in and dicts out.
"""

import math

import pytest

from web.logparse import parse_line, parse_lines

# Lines copied verbatim from logs/run-20260827-091048-tiny.log. Do not tidy
# the whitespace: the column alignment is part of what is being tested.
STEP_LINE = "step    10/2000  loss 7.8886  lr 3.00e-05  grad_norm  2.120    37,695 tok/s"
EVAL_LINE = "  [eval] val loss 4.9348  perplexity    139.0  (uniform guess would be 4096)"
CHECKPOINT_LINE = "  [checkpoint] checkpoints/tiny-step500.safetensors (val loss 4.4882)"
BASELINE_LINE = (
    "a model that has learned nothing scores ln(4096) = 8.3178; "
    "everything below that is real learning"
)
SAMPLE_HEADER = "  [sample @ step 250] prompt='ROMEO:' temperature=0.8"
SAMPLE_TEXT = '''  "ROMEO:\\nSong, down, we'll reck our traitor.\\n\\nJOHN OF YORK:\\nAnd such."'''
OVERFIT_LINE = "  step  270  loss  0.3790  grad_norm  4.3219"
GATE_LINE = "PASS: final loss is 1.6% of the uniform-guess loss."
BPE_LINE = (
    "[bpe] merge   250/3840  count=  1,234  -> 'th'  (120 merges/s, eta 30s)"
)


def one(line: str) -> dict:
    """Parse a single self-contained line."""
    event, pending = parse_line(line)
    assert pending is None, "this line should not expect a continuation"
    return event


# --- the training step line ------------------------------------------------


def test_step_line_yields_every_charted_number():
    event = one(STEP_LINE)
    assert event["type"] == "step"
    assert event["step"] == 10
    assert event["total"] == 2000
    assert event["loss"] == pytest.approx(7.8886)
    assert event["lr"] == pytest.approx(3.00e-05)
    assert event["grad_norm"] == pytest.approx(2.120)
    # Printed with a thousands separator; a chart needs the number.
    assert event["tok_s"] == pytest.approx(37695.0)


@pytest.mark.parametrize("bad", ["nan", "inf", "-inf"])
def test_step_line_survives_a_diverged_loss(bad):
    """A run that blows up must still chart, or the viewer goes blank exactly
    when it is most worth watching."""
    line = f"step    10/2000  loss {bad}  lr 3.00e-05  grad_norm  2.120    37,695 tok/s"
    event = one(line)
    assert event["type"] == "step"
    assert math.isnan(event["loss"]) or math.isinf(event["loss"])


# --- eval, checkpoint, baseline --------------------------------------------


def test_eval_line_yields_val_loss_and_perplexity():
    event = one(EVAL_LINE)
    assert event["type"] == "eval"
    assert event["val_loss"] == pytest.approx(4.9348)
    assert event["perplexity"] == pytest.approx(139.0)
    assert event["vocab_size"] == 4096


def test_checkpoint_line_yields_a_loadable_path():
    event = one(CHECKPOINT_LINE)
    assert event["type"] == "checkpoint"
    assert event["path"] == "checkpoints/tiny-step500.safetensors"
    assert event["val_loss"] == pytest.approx(4.4882)


def test_baseline_line_supplies_the_learned_nothing_reference():
    """ln(vocab_size) is the horizontal line on the loss chart. Reading it out
    of the log means the chart never has to guess the vocabulary size."""
    event = one(BASELINE_LINE)
    assert event["type"] == "baseline"
    assert event["vocab_size"] == 4096
    assert event["baseline"] == pytest.approx(math.log(4096), abs=1e-4)


# --- the two-line sample ---------------------------------------------------


def test_a_sample_spans_two_lines_and_is_un_repr_ed():
    header_event, pending = parse_line(SAMPLE_HEADER)
    assert header_event["type"] == "log"
    assert pending is not None
    assert pending["step"] == 250
    assert pending["prompt"] == "ROMEO:"
    assert pending["temperature"] == pytest.approx(0.8)

    sample_event, still_pending = parse_line(SAMPLE_TEXT, pending)
    assert still_pending is None
    assert sample_event["type"] == "sample"
    assert sample_event["step"] == 250
    # The repr is decoded, so the newlines are real newlines again.
    assert sample_event["text"].startswith("ROMEO:\n")
    assert "\\n" not in sample_event["text"]
    assert "JOHN OF YORK:" in sample_event["text"]


def test_a_pending_sample_claims_the_next_line_whatever_it_holds():
    """The text line is arbitrary content and must never be re-parsed as
    something else - a generated sample could contain anything, including a
    string that looks like a step line."""
    _, pending = parse_line(SAMPLE_HEADER)
    event, _ = parse_line("  'step    10/2000  loss 1.0'", pending)
    assert event["type"] == "sample"


# --- the overfit gate ------------------------------------------------------


def test_overfit_step_is_not_confused_with_a_training_step():
    """The gate prints no '/total', which is the only thing separating the two
    shapes. If this ever regressed, gate steps would pollute the loss chart."""
    event = one(OVERFIT_LINE)
    assert event["type"] == "overfit_step"
    assert event["step"] == 270
    assert event["loss"] == pytest.approx(0.3790)
    assert event["grad_norm"] == pytest.approx(4.3219)


@pytest.mark.parametrize(
    "line, expected",
    [(GATE_LINE, True), ("FAIL: final loss is 91.0% of the uniform-guess loss.", False)],
)
def test_gate_verdict_is_captured_both_ways(line, expected):
    event = one(line)
    assert event["type"] == "gate"
    assert event["passed"] is expected


# --- tokenizer training ----------------------------------------------------


def test_bpe_merge_line_yields_progress():
    event = one(BPE_LINE)
    assert event["type"] == "bpe"
    assert event["done"] == 250
    assert event["total"] == 3840
    assert event["count"] == 1234
    assert event["token"] == "th"


# --- the catch-all ---------------------------------------------------------


@pytest.mark.parametrize(
    "line",
    ["", "  layers      6", "=" * 72, "[prepare] encoding 1,115,394 bytes"],
)
def test_unrecognised_lines_pass_through_rather_than_vanishing(line):
    """The raw-log pane must stay a faithful copy of what the process printed."""
    event = one(line)
    assert event["type"] == "log"
    assert event["text"] == line


def test_every_line_produces_exactly_one_event():
    lines = [STEP_LINE, EVAL_LINE, SAMPLE_HEADER, SAMPLE_TEXT, CHECKPOINT_LINE, ""]
    assert len(parse_lines(lines)) == len(lines)


def test_a_real_log_file_parses_into_the_expected_event_counts():
    """The end-to-end check: replay the recorded 2000-step run.

    200 step lines (every 10 steps), 20 evals (every 100), 8 samples (every
    250), and 4 checkpoints (every 500). If any of these drift, the format
    changed underneath the parser.
    """
    from pathlib import Path

    log = Path(__file__).resolve().parent.parent / "logs" / "run-20260827-091048-tiny.log"
    if not log.exists():
        pytest.skip("logs/ is gitignored; this check only runs where the log exists")

    with log.open(encoding="utf-8") as handle:
        events = parse_lines(handle)

    counted = {}
    for event in events:
        counted[event["type"]] = counted.get(event["type"], 0) + 1

    assert counted["step"] == 200
    assert counted["eval"] == 20
    assert counted["sample"] == 8
    assert counted["checkpoint"] == 4
    assert counted["baseline"] == 1
