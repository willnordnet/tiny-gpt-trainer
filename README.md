# tiny-gpt-trainer

A small, from-scratch, decoder-only transformer trainer that runs locally on
Apple Silicon via [MLX](https://github.com/ml-explore/mlx). Every core piece
(BPE tokenizer training, RoPE attention, SwiGLU feed-forward, the training
loop, the sampler) is written out by hand rather than imported, so that every
line maps to a concept you can explain.

This is a **learning project**. It is optimized for a human reading the code
and coming away understanding how a transformer is tokenized, trained, and
sampled from. See [`DESIGN.md`](DESIGN.md) for the architecture and the
reasoning behind each choice, and [`CLAUDE.md`](CLAUDE.md) for the conventions
the code follows.

## What this is not

Stated up front so it is not discovered by disappointment later:

- **Not** an attempt to build a good language model. A ~6M parameter model
  trained on ~1MB of text will not be reliably coherent. The honest bar is
  that generated samples drift from noise toward the *style* of the training
  corpus. Nothing more.
- **Not** optimized for training speed, or for data larger than fits
  comfortably in memory on one machine.
- **Not** a serving or deployment project. There is no inference server.
- **Not** multi-domain. There is exactly one data adapter (plain text) on
  purpose. See `DESIGN.md` §7 for why the scope is deliberately this small.

## Setup

**Python 3.13 is required, not merely preferred.** MLX does not publish wheels
for Python 3.14, so a 3.14 interpreter will fail at `pip install` with no
obvious explanation. On macOS with Homebrew:

```bash
/opt/homebrew/bin/python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Verify MLX sees the GPU:

```bash
python -c "import mlx.core as mx; print(mx.array([1, 2, 3]).sum())"
```

Then fetch the corpus (TinyShakespeare, ~1.1MB, stdlib download only):

```bash
python scripts/get_tinyshakespeare.py
```

## The pipeline

Four commands, in order. Each one logs what it is doing, because this project
is meant to be watched rather than run silently to completion.

### 1. Train the tokenizer

```bash
python -m tokenizer.train_tokenizer \
  --input data/raw/tinyshakespeare.txt \
  --vocab-size 4096 \
  --out vocab.json
```

A byte-level BPE tokenizer trained from scratch on the corpus. It starts from
the 256 single-byte tokens, which means any input is encodable and there is no
UNK token to reason about, then repeatedly merges the most frequent adjacent
pair until the vocab is full.

Text is pre-tokenized into words (GPT-2's regex) before merging, so merges are
only ever learned *within* a word and never across a word boundary. Without
that restriction BPE would learn `" the cat"` as one token, spending vocabulary
on phrases instead of on reusable sub-word pieces. It is also what makes the
pure-Python trainer fast enough to be practical: the corpus collapses to 15,056
distinct words, so each of the 3,840 merge passes does ~20x less work than it
would over the raw byte sequence.

Vocab size 4096 is chosen so the embedding table stays a modest fraction of a
tiny model. There are no special tokens: paragraph boundaries survive as a
literal blank line in the token stream, so the model learns the boundary from
data instead of from a reserved id.

Expect to see the merges scroll past, which is the interesting part:

```
[bpe] corpus: 1,100,949 bytes, 283,390 word occurrences, 15,056 distinct words
[bpe] learning 3,840 merges -> vocab_size=4096
[bpe] merge     1/3840  count= 23,837  -> ' t'      (45 merges/s, eta 85s)
[bpe] merge  1000/3840  count=     90  -> 'TRANIO'  (59 merges/s, eta 48s)
[bpe] merge  3500/3840  count=     14  -> ' Jove'   (66 merges/s, eta 5s)
[bpe] learned 3,840 merges in 57.2s
[tokenizer] saved vocab.json (vocab_size=4096, fingerprint=f12702e48a5a)
[bpe] round trip: ok
[bpe] compression: 1,100,949 bytes -> 329,647 tokens (3.34 bytes/token)
```

### 2. Prepare token shards

```bash
python -m data.prepare \
  --input data/raw/tinyshakespeare.txt \
  --vocab vocab.json \
  --out-dir data/tokens
```

Runs the adapter, encodes everything, and writes one flat `uint16` token
stream per split to `data/tokens/train.npy` and `val.npy`. The split is
contiguous (the last 10% of tokens is held out) rather than random, because
training windows overlap and a shuffled split would leak validation tokens
into training.

Note the compression ratio here (3.24 bytes/token) is slightly worse than the
tokenizer reported (3.34), and the difference is entirely accounted for: the
7,221 paragraph separators cost 2 tokens each, because GPT-2's pre-tokenization
regex splits `"\n\n"` into two separate whitespace words. 329,647 + 14,442 =
344,089 exactly. That is the price of using a literal blank line as a boundary
instead of a special token, and it is a cheap price.

Windows are sliced at batch time rather than baked into the file. Pre-packing
at stride 1 would store every token `context_len` times over, and pre-packing
at stride `context_len` would throw away most of the 309,425 available windows.

A `meta.json` alongside the shards records which tokenizer built them. Training
and sampling check it, because a vocabulary mismatch does not crash: ids stay
in range, every shape stays valid, and the model just reads and writes
gibberish. Gibberish being the *expected* output of a model this size, there
would otherwise be no symptom to notice.

```
[adapter] total 7,222 chunks, mean 152.4 chars/chunk
[prepare] 1,115,391 bytes -> 344,089 tokens (3.24 bytes/token)
[prepare] train 309,681 tokens, val 34,408 tokens (10% held out, contiguous tail)
[prepare] 309,425 training windows and 34,152 validation windows at context_len=256
[prepare] first 48 training tokens decode to:
[prepare]   'First Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:\nSp'
```

### 3. Train

```bash
python train.py --preset tiny --data data/tokens --out checkpoints/
```

AdamW, cosine learning-rate decay with linear warmup, gradient clipping.
Logs step, loss, tokens/sec, periodic held-out validation loss, and a short
sample generation every N steps, so improvement is something you watch rather
than infer from a number. Output goes to stdout and `logs/run-<timestamp>.log`.

Every loss is printed against `ln(4096) = 8.318`, which is what a model that has
learned nothing scores by spreading its probability evenly over the vocabulary.
That is the honest zero point: anything below it is real learning, and the first
few steps sitting just under it is expected rather than alarming.

Weight decay is applied to matrices but not to the 1-D RMSNorm scales, since
shrinking a normalisation scale toward zero fights the thing the layer exists to
do. That split is why the optimiser is a `MultiOptimizer` rather than a plain
`AdamW`.

```
step    10/60  loss 7.8886  lr 3.00e-05  grad_norm  2.120    35,376 tok/s
step    30/60  loss 6.4557  lr 9.00e-05  grad_norm  0.991    35,424 tok/s
step    60/60  loss 5.5241  lr 1.80e-04  grad_norm  1.068    37,515 tok/s
  [eval] val loss 5.4396  perplexity    230.3  (uniform guess would be 4096)

  [sample @ step 60] prompt='ROMEO:' temperature=0.8
  'ROMEO:\n\nSecond, never say we own, be a traitor on.\nPOMPEYed too the shame...'

  [checkpoint] checkpoints/tiny-step60.safetensors (val loss 5.4524)
```

Checkpoints are `.safetensors` rather than `.npz`, because safetensors files
carry a string metadata dict alongside the arrays. The `ModelConfig` and step
number travel *inside* the checkpoint, so `sample.py` rebuilds the right
architecture from the file alone rather than from a path plus a promise that you
remembered which preset it was.

**Before any real run, run the overfit check first:**

```bash
python train.py --preset tiny --data data/tokens --overfit-one-batch
```

This trains on a single fixed batch for 500 steps, letting the model memorise it
outright. Loss should collapse toward zero. If it does not, something is broken
in `model.py` or `train.py`, and a real dataset will never train correctly. This
is the single most useful sanity check in the project (`DESIGN.md` §6.2) and it
costs about two minutes on `tiny`.

```
  step    0  loss  8.2938  grad_norm  7.0610
  step  200  loss  1.4585  grad_norm  2.1044
  step  499  loss  0.0077  grad_norm  0.0208

PASS: final loss is 0.1% of the uniform-guess loss.
```

The check distinguishes two failures that wear the same number. A loss still
falling steeply at the last step is reported as *inconclusive* (too few steps),
not as a failure. A loss that has gone flat well above zero is the real failure
signature, and the one worth stopping for. The default of 500 steps is measured
rather than guessed: at 200 steps `tiny` is only down to ~1.67 and still
descending fast, which looks like a broken model but is not one.

### 4. Sample

```bash
python sample.py \
  --checkpoint checkpoints/tiny-step400.safetensors \
  --prompt "ROMEO:" \
  --max-tokens 120 \
  --temperature 0.8 \
  --top-k 40
```

Temperature, top-k, and top-p are implemented directly rather than hidden
behind a library call, so the mechanics of autoregressive generation are
visible and tweakable. The parameters are echoed above every generation, so it
is always clear *why* a given output looks the way it does:

```
sampling parameters
  checkpoint    checkpoints/tiny-step400.safetensors
  preset        tiny (5.87M params, context 256)
  trained to    step 400, val loss 4.400514
  prompt        'ROMEO:'
  max_tokens    120
  temperature   0.8  (sharpened, more confident and more repetitive than trained)
  top_k         40  (only the top 40 tokens eligible)
  top_p         1.0  (off, full tail eligible)
  seed          0

--- sample 1 of 1 ----------------------------------------
ROMEO:
Song, and Dis.

KING RICHARD III:
If it be well, sir, that are but a son.

QUEEN:
Bring, gentle sir: you may I be a cause,
And I do so; this is not so.
--- 120 tokens in 0.34s (351.4 tok/s, no KV cache)
```

Note the throughput. Training pushes ~36,000 tok/s; sampling manages ~400.
Generation is batch-of-one and there is no KV cache, so every new token re-runs
the whole forward pass over the entire prefix. That is O(n²) work for an
n-token sample, and it is a deliberate omission: a cache is a performance
optimisation with no effect on what the model says, and adding one would put a
second, subtly different attention path in the codebase.

**What the knobs actually do.** Each reshapes the next-token distribution
before a token is drawn from it:

| Flag | Effect | Reach for it when |
|---|---|---|
| `--temperature` | Divides the logits. Below 1 sharpens the distribution, above 1 flattens it. Never reorders tokens. | Trading coherence against variety |
| `--top-k` | Keeps the k highest-scoring tokens, rest get zero probability. Reads only the ranking, so temperature does not change what it keeps. | You want a hard cap on how weird the choice can get |
| `--top-p` | Keeps the smallest set of tokens covering p of the probability mass. The *count* varies with the shape of the distribution. | You want the cap to relax when the model is genuinely torn and tighten when it is confident |

`--temperature 0` is greedy decoding (always the argmax) and is exactly
reproducible. It is also the clearest demonstration of why nobody ships greedy
decoding:

```
ROMEO:
I'll not, sir, I'll not be so.

KING RICHARD III:
I am a man of a man: I'll not be a man.

KING RICHARD III:
I am a man of a man: I'll not be a man.
```

That loop is not a bug. Greedy decoding is deterministic given the prefix, so
the moment the model produces a state it has been in before, it must produce
the same continuation, forever. Randomness is not decoration here, it is what
breaks the cycle.

Top-p on the same checkpoint, at the temperature the model was actually trained
at:

```bash
python sample.py --checkpoint checkpoints/tiny-step400.safetensors \
  --prompt "QUEEN:" --temperature 1.0 --top-k 0 --top-p 0.9 --num-samples 2
```

```
QUEEN:
Svail me not! the nature I know the crown,
I would can seek thee: hath he been stines
In our till he stands 't and open this.

JOHN OF YORK:
What is thy action: here I came in love,
And ill fruit that do so stands at all,
```

Run `python sample.py` with no `--checkpoint` and it demonstrates the knobs on
two hand-written distributions instead, no model required. This is the fastest
way to see the difference between top-k and top-p:

```
A CONFIDENT distribution (one clear winner)
  top_k=3                        0.598  0.269  0.133    -      -      -     [3 eligible]
  top_p=0.90                     0.598  0.269  0.133    -      -      -     [3 eligible]

A TORN distribution (no clear winner), same settings
  top_k=3                        0.367  0.332  0.301    -      -      -     [3 eligible]
  top_p=0.90                     0.214  0.193  0.175  0.158  0.143  0.117   [6 eligible]

    top_k=3    kept 3 tokens when confident, 3 when torn.  Always 3.
    top_p=0.90 kept 3 tokens when confident, 6 when torn.  It follows the shape.
```

## What a real run looks like

Everything below is from one 2000-step `tiny` run on TinyShakespeare, about
eight minutes on an M-series Mac, logged to `logs/run-20260827-091048-tiny.log`.
The numbers are copied out of that log rather than reconstructed, including the
part that does not flatter the model.

### The loss curve

| Step | Train | Val | |
|---|---|---|---|
| 100 | 4.832 | 4.935 | |
| 200 | 4.248 | 4.546 | |
| 300 | 3.820 | **4.364** | best held-out loss |
| 400 | 3.373 | 4.367 | |
| 500 | 2.957 | 4.504 | val starts climbing |
| 700 | 2.218 | 4.917 | |
| 1000 | 1.287 | 5.599 | |
| 1500 | 0.644 | 6.114 | |
| 2000 | **0.466** | **6.399** | worse than step 100 |

Two things happen here and only one of them is the thing you were hoping for.

Training loss falls from 4.83 to 0.47, smoothly, with no spikes and no NaN.
That part is the pipeline working: gradients reach the weights, the optimiser
is stable, the targets line up.

Validation loss bottoms out around step 300 and then climbs for the remaining
1,700 steps, finishing *worse than it was at step 100*. The model is not
learning Shakespeare any more. It is memorising this particular copy of it.

That is not a bug, it is arithmetic. 5.87M parameters against 310k training
tokens, and 2000 steps x 8,192 tokens is 16.4M tokens seen, roughly 53 passes
over the corpus. The usual compute-optimal rule of thumb wants something like
20 tokens per parameter; this run has about 0.05. There is nothing for that
much capacity to do with that little text except remember it.

**So the `tiny` preset's default of 2000 steps is too many for this corpus.**
The useful checkpoint is somewhere around step 300 to 400. That default is left
alone and documented here rather than quietly tuned away, because the shape of
this curve is one of the more instructive things in the repo. If you want a
checkpoint to actually sample from, stop early:

```bash
python train.py --preset tiny --data data/tokens --out checkpoints/ --steps 400
```

The honest fixes for the underlying problem are more data, a smaller model, or
regularisation. Not a different learning rate.

### Why the samples do not tell you any of this

This is the part worth sitting with. Both of these are real output from the run
above, same prompt, same temperature. One is from near the validation minimum,
the other from the end, by which point held-out loss is about 40% worse.
Both are shown in full, cut off only where the 80-token preview limit cut
them off:

```
[sample @ step 500]  val loss 4.50
ROMEO:
Ay, but not a kind.

Nurse:
Do you speak from me? What are you so?

ROMEO:
Romeo, but she is a happy foes is dead.

Nurse:
A old thing, a very friend, and false, she's,
alter'd, a tender man, and I know, be king,
Is all a hundred
```

```
[sample @ step 2000]  val loss 6.40
ROMEO:
Go to, go to; I say to Henry,
Which, how thou kill'd it, still you for his son.

ROMEO:
Who dost weeping tell thee by the world which 'twas
For waterspose to me.

JULIET:
O all this fouloundothege to curse
The safe of thy mouth, which is my breast.

ROMEO
```

Read those cold and you would not reliably pick the loser. Both have speaker
labels, line breaks, roughly plausible syntax, and a scattering of invented
words. The second one is arguably the more fun read.

Validation loss says the second model is substantially worse at predicting
held-out Shakespeare, and it is right. What that model gained instead was the
training split, memorised, and memorisation does not announce itself in a
60-token generation.

So: reading generations tells you whether the model has picked up the *shape*
of the data. It cannot tell you whether the model is *generalising*. Those are
two different questions and only the first one is answerable by eye. This is
why the validation curve exists, and why `DESIGN.md` §6.3 sets the acceptance
bar on loss behaviour rather than on how good the text looks.

### Where it started

For contrast, the same prompt through an untrained model, before any gradient
step at all:

```
ROMEO: morning most whiincentio raumb birthharusion Kate yield traitor
almostLLookPOMPEYed tooissionlandorryitedellersetYourMISTRESSStandBYrahLUCIO
div safonounceing
```

That is the tokenizer's vocabulary sampled at near-random: real word fragments,
because byte-level BPE learned them from the corpus, in no order, with no line
structure and no notion that a speaker label is followed by a colon and a
newline. By step 250 the structure is already there:

```
ROMEO:
Song, down, we'll reck our traitor on a warrant.

JOHN OF YORK:
And such a blances, and safening
Be fead'd, that I am not.

BUCKINGHAM:
I do beseech you, my liege; for I will be gone.

NORFROU:
A partial dayly or by me
```

Speaker labels, colons, blank lines between speeches, dialogue that starts with
a verb of address. Still nonsense sentence by sentence, and that is the correct
outcome for 5.9M parameters and eight minutes. The acceptance bar is that the
model moved from the first block to the second, not that it wrote a good play.

## Model presets

Set with `--preset`. Defined in `config.py`, which prints the exact parameter
breakdown when run directly (`python config.py`).

| Preset | Layers | d_model | Heads | Context | Params | Use |
|---|---|---|---|---|---|---|
| `tiny` | 6 | 256 | 4 | 256 | ~5.9M | Fast iteration, prove the pipeline end to end |
| `small` | 8 | 512 | 8 | 512 | ~28M | The "real" run once `tiny` is verified |

Note: `DESIGN.md` §3.3 estimates these at ~15M and ~50M. Those figures do not
match the stated dimensions once tied embeddings and a 4096 vocab are accounted
for. The table above is the arithmetic; `config.py` is the authority.

The architecture is the modern small-LM recipe rather than the original GPT-2
one: RoPE instead of learned position embeddings, RMSNorm instead of LayerNorm,
SwiGLU instead of a GELU MLP, and tied input/output embeddings. `DESIGN.md`
§3.3 explains each trade-off.

## Running any piece on its own

Every module with meaningfully independent behavior is runnable alone, so you
can understand one file without running the whole pipeline:

```bash
python -m adapters.plain_text      # chunking, on an inline sample string
python -m tokenizer.tokenizer      # trains a tiny vocab, shows a round trip
python model.py                    # guided tour of RMSNorm/RoPE/SwiGLU/causality
python config.py                   # parameter count breakdown per preset
python sample.py                   # temperature/top-k/top-p demo, no model needed
```

`python model.py` is worth running before anything else. It does not just prove
the architecture is wired correctly, it demonstrates each of the four ideas that
make this a modern transformer rather than a 2019 one, numerically:

- **RMSNorm** normalising vectors from RMS 123 down to RMS 1.0, and eps visibly
  taking over when the input is near zero
- **RoPE** giving byte-for-byte identical attention scores for the same content
  at positions (2,0), (7,5), (11,9) and (15,13), because all four are an offset
  of 2 apart, and different scores once the offset changes
- **SwiGLU**'s gate closing a unit to ~0 or opening it linearly, per input
- **Causal masking**, by editing the last token of a sequence and showing the
  logit change at every earlier position is exactly `0.000000`

It closes by counting the built model's parameters and checking that number
against `config.py`'s independent hand-derived arithmetic (both say 5,868,800
for `tiny`), and by confirming the untrained loss lands near `ln(4096) = 8.318`.

And the fast tests, which do no real training and finish in seconds:

```bash
pytest
```

These cover plumbing correctness: adapter chunk counts, tokenizer round trips,
token-id ranges and dtypes, forward-pass shapes, determinism, that targets are
inputs shifted by exactly one, that the learning-rate schedule has the right
shape, that weight decay reaches matrices but not RMSNorm scales, and that a
checkpoint round-trips to bit-identical logits. Run them after any change to the
tokenizer, the data pipeline, `model.py`, or `train.py`.

## What "it worked" means

The acceptance bar, per `DESIGN.md` §6.3, stated so this is judged on its own
terms:

- [x] All `pytest` tests pass. (151 of them, in 1.5s.)
- [x] `--overfit-one-batch` drives loss to near zero. (8.294 -> 0.044, which is
      0.5% of the uniform-guess loss.)
- [x] Loss on the real dataset decreases, without spiking or going NaN. Train
      loss falls 4.83 -> 0.47 over 2000 steps, smoothly. Validation loss does
      *not* plateau: it bottoms out near step 300 and then climbs, for the
      reasons in [What a real run looks like](#what-a-real-run-looks-like).
      That is a property of this corpus being far too small for this model,
      not of the training loop misbehaving.
- [x] Sample generations visibly shift from noise toward corpus-like style over
      the course of a run. The shift happens early, inside the first few
      hundred steps, and then samples stop visibly changing even as held-out
      loss gets substantially worse.

That is the definition of success here. Not "the model produces impressive
text."

## Repo map

```
tiny-gpt-trainer/
├── adapters/
│   ├── base.py              # Adapter interface: raw source -> iterator of text
│   └── plain_text.py        # Reads .txt file(s), yields paragraph-ish chunks
├── tokenizer/
│   ├── train_tokenizer.py   # Trains a byte-level BPE vocab
│   └── tokenizer.py         # Load / encode / decode
├── data/
│   ├── prepare.py           # adapter -> tokenizer -> uint16 token streams
│   ├── raw/                 # downloaded corpora (gitignored)
│   └── tokens/              # train.npy / val.npy (gitignored)
├── model.py                 # RoPE attention + SwiGLU + RMSNorm, in MLX
├── train.py                 # training loop, checkpoints, logging
├── sample.py                # temperature / top-k / top-p generation
├── config.py                # model + training size presets
├── scripts/
│   └── get_tinyshakespeare.py
├── tests/                   # fast plumbing tests, see DESIGN.md §6.1
└── logs/                    # training logs + periodic samples (gitignored)
```

The one abstraction that matters is the adapter interface: a single `read()`
method that yields text. Everything downstream of it never learns what domain
the text came from. Adding a new domain later should mean writing one new file
in `adapters/` and re-running steps 1 and 2, with no change to `model.py`,
`train.py`, or `sample.py`. If it ever requires more than that, the interface
is wrong (`DESIGN.md` §5).

## Build status

This repo is being built step by step as a readable sequence, not dropped in
one commit. Current state:

- [x] `DESIGN.md`, `CLAUDE.md`, README
- [x] Step 0: environment, skeleton, corpus download script
- [x] Step 1: `config.py`
- [x] Step 2: `adapters/`
- [x] Step 3: `tokenizer/`
- [x] Step 4: `data/prepare.py`
- [x] Step 5: `model.py`
- [x] Step 6: `train.py`
- [x] Step 7: `sample.py`
- [x] Step 8: first real run, honest samples pasted here

Commands documented above describe the target pipeline. Anything not ticked
off does not exist yet.

## License

See [LICENSE](LICENSE).
