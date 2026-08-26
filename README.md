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

**Before any real run, run the overfit check first:**

```bash
python train.py --preset tiny --data data/tokens --overfit-one-batch
```

This trains on a single fixed batch for many steps. Loss should collapse toward
zero. If it does not, something is broken in `model.py` or `train.py`, and a
real dataset will never train correctly. This is the single most useful sanity
check in the project (`DESIGN.md` §6.2) and it costs seconds.

### 4. Sample

```bash
python sample.py \
  --checkpoint checkpoints/tiny-step1000.npz \
  --prompt "ROMEO:" \
  --max-tokens 200 \
  --temperature 0.8 \
  --top-k 40
```

Temperature, top-k, and top-p are implemented directly rather than hidden
behind a library call, so the mechanics of autoregressive generation are
visible and tweakable. The sampling parameters are logged above the generation,
so it is always clear *why* a given output looks the way it does.

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
python model.py                    # builds an untrained model, one forward pass
python config.py                   # parameter count breakdown per preset
```

And the fast tests, which do no real training and finish in seconds:

```bash
pytest
```

These cover plumbing correctness: adapter chunk counts, tokenizer round trips,
token-id ranges and dtypes, forward-pass shapes, determinism, and the check
that a gradient step actually moves the weights. Run them after any change to
the tokenizer, the data pipeline, or `model.py`.

## What "it worked" means

The acceptance bar, per `DESIGN.md` §6.3, stated so this is judged on its own
terms:

- [ ] All `pytest` tests pass.
- [ ] `--overfit-one-batch` drives loss to near zero.
- [ ] Loss on the real dataset decreases and roughly plateaus, without spiking
      or going NaN.
- [ ] Sample generations visibly shift from noise toward corpus-like style over
      the course of a run.

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
- [ ] Step 5: `model.py`
- [ ] Step 6: `train.py`
- [ ] Step 7: `sample.py`
- [ ] Step 8: first real run, honest samples pasted here

Commands documented above describe the target pipeline. Anything not ticked
off does not exist yet.

## License

See [LICENSE](LICENSE).
