"""Turn the trainer's stdout into structured events.

This module is the whole reason the viewer needs no changes to the training
loop. `train.py` already prints everything worth charting -- loss, learning
rate, gradient norm, throughput, validation loss, periodic samples, and
checkpoint paths -- in a handful of fixed line shapes. Reading those lines
back is cheaper, and far less invasive, than threading a metrics callback
through a loop whose whole design goal is to fit on one screen.

The tradeoff, stated plainly: this couples the viewer to the trainer's *log
format*. If a format string in `train.py` changes, a chart here goes quiet.
That is the price of the zero-touch boundary, and it is why every parser
below is a pure string-in / dict-out function with a test that feeds it real
lines lifted from `logs/`. A format change should break a fast unit test, not
be discovered as a mysteriously empty chart.

Anything unrecognised comes back as {"type": "log"} rather than being
dropped, so the raw-log pane in the browser stays a faithful copy of what
the process actually printed.

Run this file directly to replay a real log file through the parser:

    python -m web.logparse logs/run-20260827-091048-tiny.log
"""

import ast
import re

# --- the line shapes -------------------------------------------------------
#
# Each pattern is written against the f-string in the trainer that produces
# it, and the comment names that source. Keep the two in sync.
#
# A number here may be "nan" or "inf": a diverging run prints those rather
# than crashing, and a viewer that refuses to parse them would go blank at
# exactly the moment it is most worth watching.
_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?(?:[eE][-+]?\d+)?|nan|inf)"

# train.py: "step {n:5d}/{total}  loss {loss:6.4f}  lr {lr:.2e}
#            grad_norm {g:6.3f}  {tok_s:8,.0f} tok/s"
_STEP = re.compile(
    rf"^step\s+(?P<step>\d+)/(?P<total>\d+)\s+"
    rf"loss\s+(?P<loss>{_NUMBER})\s+"
    rf"lr\s+(?P<lr>{_NUMBER})\s+"
    rf"grad_norm\s+(?P<grad_norm>{_NUMBER})\s+"
    rf"(?P<tok_s>[\d,]+)\s+tok/s\s*$"
)

# train.py: "  [eval] val loss {v:6.4f}  perplexity {p:8.1f}
#             (uniform guess would be {vocab})"
_EVAL = re.compile(
    rf"^\s*\[eval\]\s+val loss\s+(?P<val_loss>{_NUMBER})\s+"
    rf"perplexity\s+(?P<perplexity>{_NUMBER})\s+"
    rf"\(uniform guess would be (?P<vocab_size>\d+)\)\s*$"
)

# train.py: "  [sample @ step {n}] prompt={prompt!r} temperature=0.8"
# The generated text itself is on the FOLLOWING line, as a Python repr.
_SAMPLE_HEADER = re.compile(
    rf"^\s*\[sample @ step (?P<step>\d+)\]\s+"
    rf"prompt=(?P<prompt>.+?)\s+temperature=(?P<temperature>{_NUMBER})\s*$"
)

# train.py: "  [checkpoint] {path} (val loss {v:.4f})"
_CHECKPOINT = re.compile(
    rf"^\s*\[checkpoint\]\s+(?P<path>\S+)\s+\(val loss (?P<val_loss>{_NUMBER})\)\s*$"
)

# train.py header: "a model that has learned nothing scores ln(4096) = 8.3178; ..."
# Worth capturing: it is the horizontal reference line on the loss chart, and
# taking it from the log means the chart never has to guess the vocab size.
_BASELINE = re.compile(
    rf"^a model that has learned nothing scores "
    rf"ln\((?P<vocab_size>\d+)\) = (?P<baseline>{_NUMBER})"
)

# train.py, overfit_one_batch: "  step {n:4d}  loss {loss:7.4f}  grad_norm {g:7.4f}"
# Note the absence of "/total": that is what distinguishes an overfit-gate
# step from a real training step, and why _STEP above anchors on the slash.
_OVERFIT_STEP = re.compile(
    rf"^\s*step\s+(?P<step>\d+)\s+loss\s+(?P<loss>{_NUMBER})\s+"
    rf"grad_norm\s+(?P<grad_norm>{_NUMBER})\s*$"
)

# train.py, overfit_one_batch verdict: "PASS: ..." / "FAIL: ..."
_GATE_VERDICT = re.compile(r"^(?P<verdict>PASS|FAIL):\s*(?P<detail>.*)$")

# train_tokenizer.py: "[bpe] merge {n:>5}/{target}  count={c:>7,}
#                      -> {spelled!r}  ({rate:.0f} merges/s, eta {eta:.0f}s)"
_BPE_MERGE = re.compile(
    rf"^\[bpe\]\s+merge\s+(?P<done>\d+)/(?P<total>\d+)\s+"
    rf"count=\s*(?P<count>[\d,]+)\s+->\s+(?P<token>.+?)\s+"
    rf"\((?P<rate>{_NUMBER}) merges/s, eta (?P<eta>{_NUMBER})s\)\s*$"
)


def _to_float(text: str) -> float:
    """float() that also accepts the comma-grouped integers the trainer prints."""
    return float(text.replace(",", ""))


def _unrepr(text: str) -> str:
    """Decode a Python repr back to the string it came from.

    The trainer logs generated samples with !r on purpose: a sample contains
    newlines, and printing it raw would smear one event across a dozen log
    lines with no way to tell where it ended. The repr keeps it on one line.
    Undoing that needs a real Python literal parser, not a strip('"'), because
    the text is full of escaped quotes and \\n. ast.literal_eval only
    evaluates literals -- it will not execute anything -- which is what makes
    it safe to point at process output.
    """
    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text
    return value if isinstance(value, str) else text


def parse_line(line: str, pending_sample: dict | None = None) -> tuple[dict, dict | None]:
    """Classify one line of trainer output.

    Returns (event, pending_sample). The second value exists because a sample
    spans two lines: the header says which step it belongs to, and the text
    arrives on the next line. When this function returns a non-None
    pending_sample, the caller must hand it straight back on the next call.

    Every line produces exactly one event, so a caller that forwards them all
    reproduces the log verbatim in order.
    """
    line = line.rstrip("\n")

    # A pending sample header claims the very next line, whatever it holds.
    if pending_sample is not None:
        event = dict(pending_sample, type="sample", text=_unrepr(line.strip()))
        return event, None

    match = _STEP.match(line)
    if match:
        return {
            "type": "step",
            "step": int(match["step"]),
            "total": int(match["total"]),
            "loss": _to_float(match["loss"]),
            "lr": _to_float(match["lr"]),
            "grad_norm": _to_float(match["grad_norm"]),
            "tok_s": _to_float(match["tok_s"]),
            "text": line,
        }, None

    match = _EVAL.match(line)
    if match:
        return {
            "type": "eval",
            "val_loss": _to_float(match["val_loss"]),
            "perplexity": _to_float(match["perplexity"]),
            "vocab_size": int(match["vocab_size"]),
            "text": line,
        }, None

    match = _SAMPLE_HEADER.match(line)
    if match:
        # Emit the header as a plain log line and remember what it announced;
        # the text on the next line completes the event.
        pending = {
            "step": int(match["step"]),
            "prompt": _unrepr(match["prompt"]),
            "temperature": _to_float(match["temperature"]),
        }
        return {"type": "log", "text": line}, pending

    match = _CHECKPOINT.match(line)
    if match:
        return {
            "type": "checkpoint",
            "path": match["path"],
            "val_loss": _to_float(match["val_loss"]),
            "text": line,
        }, None

    match = _BASELINE.match(line)
    if match:
        return {
            "type": "baseline",
            "vocab_size": int(match["vocab_size"]),
            "baseline": _to_float(match["baseline"]),
            "text": line,
        }, None

    match = _OVERFIT_STEP.match(line)
    if match:
        return {
            "type": "overfit_step",
            "step": int(match["step"]),
            "loss": _to_float(match["loss"]),
            "grad_norm": _to_float(match["grad_norm"]),
            "text": line,
        }, None

    match = _GATE_VERDICT.match(line)
    if match:
        return {
            "type": "gate",
            "passed": match["verdict"] == "PASS",
            "detail": match["detail"],
            "text": line,
        }, None

    match = _BPE_MERGE.match(line)
    if match:
        return {
            "type": "bpe",
            "done": int(match["done"]),
            "total": int(match["total"]),
            "count": int(match["count"].replace(",", "")),
            "token": _unrepr(match["token"]),
            "text": line,
        }, None

    return {"type": "log", "text": line}, None


def parse_lines(lines) -> list[dict]:
    """Run parse_line over an iterable, carrying the two-line sample state."""
    events = []
    pending = None
    for line in lines:
        event, pending = parse_line(line, pending)
        events.append(event)
    return events


if __name__ == "__main__":
    import collections
    import sys

    if len(sys.argv) != 2:
        print(__doc__.strip().splitlines()[-1].strip())
        raise SystemExit(2)

    with open(sys.argv[1], encoding="utf-8") as handle:
        events = parse_lines(handle)

    counts = collections.Counter(event["type"] for event in events)
    print(f"[logparse] {sys.argv[1]}: {len(events)} lines")
    for kind, count in counts.most_common():
        print(f"[logparse]   {kind:<14} {count:>5}")

    samples = [e for e in events if e["type"] == "sample"]
    if samples:
        first, last = samples[0], samples[-1]
        print()
        print(f"[logparse] first sample, step {first['step']}:")
        print(f"[logparse]   {first['text'][:70]!r}")
        print(f"[logparse] last sample, step {last['step']}:")
        print(f"[logparse]   {last['text'][:70]!r}")
