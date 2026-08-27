# web/ — watching a training run in a browser

An optional viewer. The trainer does not need it and does not know it exists:
everything under `tinygpt/` runs exactly the same whether this is here or not.

```bash
python -m web.server --port 8000
# then open http://127.0.0.1:8000
```

## What it is not

Not an inference server, and not a deployment path. It binds to `127.0.0.1`
only, has no authentication, and exposes endpoints that start subprocesses and
write files. That is fine for a tool you run on your own machine to look at
your own training run, and it is emphatically not fine on a network — which is
why the bind address is not a flag.

## How it works

```
browser ──POST /api/run──> server.py ──spawn──> python -u -m tinygpt.train
        <──SSE /api/events──          <─stdout─  "step 120/2000 loss 4.71 ..."
                                       logparse.py: line -> {step, loss, lr, ...}
```

Training runs as a **subprocess**, and the viewer reads its stdout. Nothing is
wired into the training loop. Three things follow from that:

- `train.py` needs no changes and cannot be broken by the viewer.
- A run that crashes or exhausts memory takes down a child process, not the
  server, so the page stays up to show you what happened.
- Stop actually stops, because killing a process is reliable in a way that
  asking a tight MLX loop to please stop is not.

The cost is that the viewer only sees what the trainer prints, so it is coupled
to the trainer's *log format*. `tests/test_logparse.py` feeds the parser real
lines from `logs/` precisely so a format change breaks a fast test rather than
silently emptying a chart.

| File | Role |
|---|---|
| `server.py` | routes, static files, and hand-rolled server-sent events |
| `runner.py` | subprocess lifecycle for the four pipeline stages |
| `logparse.py` | stdout lines to structured events — pure functions |
| `introspect.py` | loads a checkpoint for the next-token, generate and attention panels |
| `static/` | one HTML page, one stylesheet, one script. No build step. |

Two of these run standalone, which is the fastest way to see what they do:

```bash
python -m web.logparse logs/run-20260827-091048-tiny.log   # replay a real log
python -m web.runner                                       # gate + short run
python -m web.introspect                                   # newest checkpoint
```

## The panels

**Run.** Preset, steps, sample prompt, and an optional `.txt` upload (2 MB cap
— the BPE trainer is hand-rolled and `O(merges × corpus)`, so a bigger file
would look like a hang). The overfit gate is a checkbox and its verdict is a
badge, because `DESIGN.md` §6.2 treats it as the check you do not train
without; a green light makes that discipline visible instead of buried in a
flag. A failing gate exits non-zero and the pipeline stops there.

**Loss.** Train and validation loss, with a dashed line at `ln(vocab_size)` —
the loss of a model that has learned nothing, so everything below it is real
learning. When validation loss turns back up while training loss keeps
falling, the chart marks where that started. On this corpus it happens early
and hard (the recorded run bottoms out near step 300 at val 4.36 and ends at
6.40 while train loss reaches 0.47), which makes overfitting something you
watch rather than read about.

**Sample timeline.** The trainer already generates from the same prompt every
`sample_interval` steps. Drag the scrubber to watch noise become words become
something with the shape of dialogue.

**Next-token lab.** One forward pass, one distribution. The pale bar is what
the model predicted; the solid bar is what survives temperature, top-k and
top-p; struck-out rows are candidates the knobs eliminated. Showing the cut
rather than hiding it is the point. The reshaping happens server-side through
the very same `sample.reshape_logits` that `generate()` uses — a JavaScript
reimplementation would be a second copy of the exact mechanics this project
exists to explain, sitting where it could silently drift.

**Generate.** Type a prompt, get a continuation, streamed token by token.
Worth being blunt about what this is: a **completion playground, not a chat
window**. The model is a base LM trained only to predict the next token in
TinyShakespeare -- no instruction tuning, no chat template, and per
`DESIGN.md` no special tokens at all. It continues a prefix in the style of
its corpus; it cannot answer a question. A chat UI would imply a capability it
does not have. The prompt is echoed dim and the continuation written bright,
so the two read as one passage while it stays obvious where the model took
over. Stop aborts the fetch, which is the whole mechanism -- the server's next
write fails and generation unwinds.

The readout shows the prompt's token count against the model's context length,
because `generate()` re-slices `ids[-context_len:]` on every step in silence:
a long prompt plus a long generation loses its own beginning partway through
with no error. On `tiny` that limit is 256 tokens, a few hundred words.

**Attention.** A layer/head grid over a prompt. The empty upper triangle is
the causal mask. Cell shading is `sqrt(weight)`, for display only: attention
is spiky, and on a linear ramp everything but the peak reads as black.

**Raw log.** Every line the child processes printed, parsed or not. The
dashboard should never be the only way to see what happened.

## Things worth knowing

- **Uploads get their own vocabulary size.** A preset fixes `vocab_size` at
  4096, but a BPE vocab trained on your upload is whatever that corpus could
  support. After the prepare stage the runner reads the real number out of
  `data/tokens/meta.json` and passes it as `--vocab-size`, so any corpus
  trains. Note that training a new vocab **overwrites `vocab.json` in place**,
  which orphans every checkpoint from before that run; the panels refuse to
  decode one rather than silently mislabelling its tokens.
- **GPU contention.** The server and the training subprocess both use Metal.
  Prompting a checkpoint mid-run works and slows training while it does.
- **Checkpoints, not live weights.** The panels read `.safetensors` files, so
  the newest thing they can show is the last checkpoint, not the current step.
  Lower `checkpoint_interval` in `tinygpt/config.py` for a finer-grained demo.
- **No KV cache** (a deliberate choice, see `tinygpt/sample.py`), so generation
  is `O(n²)`. Fine for short samples; keep lab prefixes short. Streaming makes
  this visible rather than hiding it behind a spinner -- text visibly slows as
  the sequence grows.
- **Generation is not seeded.** Two identical requests give different text.
  `python -m tinygpt.sample --seed N` is the reproducible path.
- **One run at a time.** Two training processes on one GPU make both slower
  and the numbers meaningless to compare.
