# web/ — watching a training run in a browser

An optional viewer. The trainer does not need it and does not know it exists:
everything under `tinygpt/` runs exactly the same whether this is here or not.

```bash
python -m web.server --port 8000
# then open http://127.0.0.1:8000
```

![The viewer: the loss curve against the ln(vocab) baseline, the sample
scrubber, and the next-token distribution under the sampling knobs.](../docs/viewer.png)

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

## The masthead

Above the panels, the four stages `runner.py` can build --
`tokenizer → prepare → gate → train` -- shown in the order they run and lit as
the run reaches each one. The first three are conditional (`build_stages`
includes them only for an upload or a ticked gate), so stages this particular
job skipped are struck through rather than hidden: which steps did *not* run is
part of reading a run. It is driven by the same `stage_start` / `stage_end`
events that drive the status badges, so it needs nothing from the server that
the page was not already receiving.

## The panels

In page order: **Run** and **Loss** side by side, then the **raw log** across
the full width, then **Sample timeline** / **Next-token lab** / **Generate**,
then **Attention**. The log sits third rather than last because it is the
ground truth the panels above it are summarising, and reading it should not
mean scrolling past everything else.

Each panel carries a short `<dl>` defining its own terms -- `train` versus
`val`, `grad norm`, `top-k` versus `top-p`, `layer` versus `head` -- beside the
control or readout in question rather than in one glossary at the bottom.

**Run.** Preset, steps, sample prompt, and an optional `.txt` upload (16 MB
cap, a memory guard rather than a time one -- the whole body is read into
memory, and so is the corpus behind it). BPE training is the slow part but it
scales with *distinct words*, not bytes. 3.5 MB of Conan Doyle is 3.2x the
bytes of 1.1 MB of Shakespeare but only 1.9x the distinct words, and learns a
4096-token vocab in ~117s against ~57s, with merge progress and an ETA in the
raw log throughout. The overfit gate is a checkbox and its verdict is a
badge, because `DESIGN.md` §6.2 treats it as the check you do not train
without; a green light makes that discipline visible instead of buried in a
flag. A failing gate exits non-zero and the pipeline stops there.

**Loss.** Both numbers are cross-entropy in *nats*, and **lower is better**:
zero is perfect prediction, `ln(vocab_size)` = 8.32 is a model that has learned
nothing. `train` is measured on the batches being learned from right now, so it
falls almost by construction; `val` (validation) is measured on the 10% of the
corpus held back in `val.npy`, which the model never trains on, and is the only
one of the two that can tell learning from memorising. Perplexity, printed
beside each eval, is just `exp(loss)` -- roughly how many tokens the model is
choosing between at each position. `val` is also what the checkpoint dropdowns
show, so the lowest one there is the best model you have; it is comparable only
across runs sharing a corpus and a vocabulary.

The chart draws both curves against that `ln(vocab_size)` ceiling as a dashed
line, so everything below it is real learning, and marks the step where
validation loss turned back up while training loss kept falling. On
TinyShakespeare that happens early and hard -- the recorded run bottoms out
near step 300 at val 4.36 and ends at 6.40 while train loss reaches 0.47 --
which makes overfitting something you watch rather than read about.

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
dashboard should never be the only way to see what happened. A trainer prints
plenty of blank lines to group its output, and one `<div>` per line at a full
line-height meant those blanks pushed the interesting lines off the pane; they
are kept in the DOM but collapsed to a sliver, so the pane stays a faithful
copy without being mostly whitespace.

## Things worth knowing

- **Checkpoints name the vocabulary they were trained with.** A checkpoint
  stores `vocab_size` but not the vocabulary, and every preset targets 4096, so
  two vocabularies trained on different corpora are indistinguishable by size
  alone. Reading a checkpoint with the wrong one of them decodes every id into
  different text without a single error. Checkpoints therefore record the
  tokenizer's `fingerprint` in their metadata and the panels refuse a
  mismatch. `list_checkpoints` carries the same verdict, so a stale checkpoint
  is marked in the dropdown rather than only discovered after picking it. Files
  written before that field existed report as *unverifiable* rather than being
  rejected: that is a third state, and calling it "fine" would be exactly the
  kind of silent wrong answer the guard exists to prevent.
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
