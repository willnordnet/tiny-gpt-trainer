# DESIGN.md — tiny-GPT-trainer

## 1. What this project is

**tiny-GPT-trainer** is a small, from-scratch, nanoGPT-style transformer
trainer built for **learning**, not production use. It trains a decoder-only language model on plain text, using a
from-scratch implementation of every core piece (tokenizer training, attention,
feed-forward blocks, training loop, sampling) so that every line of code maps to a
concept someone building this to learn transformers should be able to explain.

It runs locally on Apple Silicon via **MLX**.

## 2. Scope for this version

**In scope:**
- One data adapter: **plain text**. Point it at a `.txt` file (or a folder of them),
  it trains a model that generates more text in that style.
- A from-scratch decoder-only transformer (RoPE attention, SwiGLU feed-forward,
  RMSNorm, tied embeddings — the modern small-LM recipe, not the original
  GPT-2 recipe).
- A from-scratch BPE tokenizer trainer (small vocab, since a tiny model doesn't
  need or benefit from a large one).
- A training loop with visible, readable logging (loss curve, tokens/sec, sample
  generations at checkpoints) so training is something you can *watch and
  understand*, not a black box you wait on.
- A sampling script with temperature / top-k / top-p, so generation is also
  something you can experiment with, not just a fixed demo.

**Explicitly out of scope for this version (see §7 for why):**
- Any domain other than plain text (FIX messages, MIDI, SMILES, chess, etc.).
  These were explored conversationally as future directions but are **not**
  built here. Adding one should mean writing one new adapter file, nothing else.
- Multi-domain training / domain-tag conditioning.
- Distributed or multi-GPU training.
- Any inference-serving concerns (this is a training project, not a deployment
  one).

Keeping the scope this small is deliberate: a general-purpose *interface* is
easiest to design correctly when you've only proven it against one concrete
case. Building three adapters at once tends to produce an interface shaped
around guesswork instead of one shaped around a real, working example.

## 3. Architecture

```
tiny-gpt-trainer/
├── adapters/
│   ├── base.py          # Adapter interface: raw source -> iterator of text examples
│   └── plain_text.py     # Reads .txt file(s), yields chunks of text
├── tokenizer/
│   ├── train_tokenizer.py   # Trains a small BPE vocab from adapter output
│   └── tokenizer.py          # Load/encode/decode wrapper around the trained vocab
├── data/
│   └── prepare.py            # adapter -> tokenizer -> packed token shards (.npy)
├── model.py                   # RoPE attention + SwiGLU block + full model, in MLX
├── train.py                    # training loop: loss, backward, optimizer, checkpoints
├── sample.py                    # generation: temperature / top-k / top-p sampling
├── config.py                     # model + training size presets (tiny / small)
├── tests/                          # unit tests — see §6.1
│   ├── test_adapters.py
│   ├── test_tokenizer.py
│   ├── test_data_prepare.py
│   └── test_model.py               # includes overfit-one-batch check, §6.2
└── logs/                             # training logs + periodic sample generations
```

### 3.1 The adapter interface (the one abstraction that matters)

```python
class Adapter:
    def read(self, source_path: str) -> Iterator[str]:
        """Yield raw text examples from a data source.

        Every domain (plain text today; FIX/MIDI/SMILES/etc. later, if ever)
        implements this one method. Everything downstream of it — tokenizer,
        model, training loop — never needs to know or care what domain the
        text came from. That's the whole point of the interface: it's the
        *only* place domain-specific logic is allowed to live.
        """
```

`plain_text.py` is the only implementation right now. It reads one or more
`.txt` files and yields chunks of text (e.g. paragraph-sized, or fixed-length
windows — kept simple). A future adapter (say, FIX messages) would implement
the same method, turning structured records into strings, and nothing else
in the codebase would change.

### 3.2 Tokenizer

A small BPE tokenizer trained from scratch on whatever the adapter yields.
Not a hardcoded/pretrained tokenizer (like tiktoken) — training your own is
part of the point of this project, and a tiny model wants a small vocabulary
(low thousands of tokens, not GPT-4's ~100k) so the embedding table doesn't
dominate total parameter count.

### 3.3 Model

Decoder-only transformer, the modern small-LM recipe:

| Component | Choice | Why (not the classic GPT-2 choice) |
|---|---|---|
| Position encoding | RoPE (rotary) | No fixed-length position table; generalizes better; what current models (Llama/Qwen/Gemma) actually use |
| Normalization | RMSNorm, pre-norm | Simpler than LayerNorm (no mean-centering/bias), slightly cheaper, current standard |
| Feed-forward | SwiGLU | Consistently outperforms plain GELU-MLP at any scale; worth understanding since it's everywhere now |
| Embedding | Tied input/output | Meaningful parameter savings at small scale, where the embedding table is a large fraction of the model |

Two size presets to start (`config.py`):

| Preset | Params | Layers | d_model | Heads | Context | Use |
|---|---|---|---|---|---|---|
| `tiny` | ~15M | 6 | 256 | 4 | 256 | Fast iteration — prove the pipeline works end to end |
| `small` | ~50M | 8 | 512 | 8 | 512 | The "real" run once `tiny` is verified |

### 3.4 Training loop

Standard MLX pattern: `mx.value_and_grad` on next-token cross-entropy,
AdamW, cosine LR schedule with warmup. Nothing exotic — the point of this
project is to see the standard recipe implemented plainly, not to explore
training-loop research.

### 3.5 Sampling

Temperature, top-k, top-p, all implemented directly (not hidden behind a
library call) so the mechanics of autoregressive generation are visible and
tweakable.

## 4. Data flow, end to end

```
.txt file(s)
   │  adapters/plain_text.py: read()
   ▼
raw text chunks
   │  tokenizer/train_tokenizer.py (once) → vocab.json
   │  tokenizer/tokenizer.py: encode()
   ▼
token ID sequences
   │  data/prepare.py: pack into fixed-length training windows
   ▼
.npy shards
   │  train.py: load shards, train model.py, checkpoint + log
   ▼
checkpoints/ + logs/
   │  sample.py: load checkpoint, generate
   ▼
generated text
```

## 5. How to extend later (not built now, but designed for)

Adding a new domain later should require:
1. One new file in `adapters/` implementing `read()`.
2. Re-running `train_tokenizer.py` and `prepare.py` pointed at the new source.
3. No changes to `model.py`, `train.py`, or `sample.py`.

If adding a domain ever *does* require touching the model or training loop,
that's a signal the adapter interface was designed wrong — worth stopping and
fixing the interface rather than special-casing the new domain.

Multi-domain training (mixing plain text with something else, with a
`<|domain|>` conditioning tag) is a natural next step but is intentionally
**not** designed in detail here — it should be designed once there are two
real adapters to design it against, not speculatively now.

## 6. How to test this

This project mixes two kinds of correctness that need different testing
strategies: **plumbing correctness** (did I wire the shapes/types together
right — this has a definite right answer and should be tested like normal
software) and **learning correctness** (did the model actually learn
something — this doesn't have a single right answer and needs to be
*observed*, not just asserted). Conflating the two is a common source of
false confidence in ML code ("all my tests pass" while the model has learned
nothing useful) — keep them separate.

### 6.1 Plumbing correctness — unit tests, deterministic, fast

These should run in seconds, with no real training, and belong in a normal
`tests/` directory run via `pytest`.

| Component | What to test | Why it's worth a real test |
|---|---|---|
| `adapters/plain_text.py` | `read()` on a small fixture file yields the expected number/shape of text chunks | Silent adapter bugs (wrong chunking, dropped trailing text) are easy to introduce and easy to miss by eye |
| `tokenizer.py` | `encode(decode(ids)) == ids` and `decode(encode(text))` round-trips correctly on a handful of fixture strings, including edge cases (empty string, unknown characters, whitespace runs) | A silently broken round-trip poisons every downstream stage without an obvious symptom |
| `data/prepare.py` | packed shards have the expected dtype, shape, and no token IDs outside `[0, vocab_size)` | Out-of-range IDs crash training far from the actual bug — catch it here instead |
| `model.py` | a forward pass on a small dummy batch produces the expected output shape `(batch, seq_len, vocab_size)`; running the same input twice with the same weights gives identical output (determinism check) | Shape bugs in attention/RoPE are the single most common bug class in from-scratch transformer code, and are cheap to catch with a shape assertion before ever starting real training |
| `model.py` | a single gradient step on a tiny batch changes the weights (i.e. gradients aren't silently zero) | Catches a whole class of "training runs, loss never moves" bugs (frozen params, detached tensors, wrong loss reduction) in seconds, without waiting for a real training run to reveal it |
| `sample.py` | temperature=0 (or very low) sampling is deterministic and reproducible; output length respects `max_tokens` | Confirms the generation loop's control flow is correct independent of whether the model's outputs are any good |

Run these on every change to `model.py`, the tokenizer, or the data pipeline
— per `CLAUDE.md`, this is the "smoke test before a full run" step, not
optional cleanup.

### 6.2 Learning correctness — observed, not asserted

These don't have a pass/fail boolean; they're sanity checks you read and
judge, ideally with the same log output described in `CLAUDE.md`.

- **Overfit-one-batch test.** Before any real run, train on a single small
  batch repeated for many steps and confirm loss goes to (near) zero. If it
  doesn't, something is broken in the model or training loop — a real
  dataset will never train correctly if this fails. This is the single most
  useful sanity check in the whole project and should be the first thing
  run after any change to `model.py` or `train.py`.
- **Loss curve shape.** On a real (if small) dataset, loss should decrease
  and roughly plateau, not spike, NaN, or plateau immediately at a high
  value. Log this every run; a quick plot from the log file is enough,
  no dashboard needed.
- **Sample generations at checkpoints.** Per `CLAUDE.md`'s logging
  requirements, generate a short sample every N steps during training and
  read them. Early samples should look like noise; later samples should
  increasingly resemble the training corpus's *style* (this is the
  honest bar for a small model — see §8 on non-goals — not factual
  correctness or long-range coherence).
- **Held-out perplexity.** Split off a small validation slice the model
  never trains on, and track its loss alongside training loss. Divergence
  between the two (training loss keeps falling, validation loss rises) is
  the standard signal of overfitting, and is worth showing explicitly since
  it's a genuine, visible ML concept, not just plumbing.

### 6.3 What "passing" means for this project

Given the non-goals in §8, the acceptance bar is intentionally modest and
should be stated up front rather than discovered by disappointment:

- All §6.1 unit tests pass.
- The overfit-one-batch test succeeds.
- Loss on the real dataset decreases over training and sample generations
  visibly shift from noise toward corpus-like style by the end of a run.

That's the definition of "this project worked" — not "the model produces
impressive text." Keep that bar visible (e.g. in a project README) so it's
judged on its own honest terms.

## 7. Reasoning: why the scope is this small

This project has an explicit secondary goal beyond "train a model": it's a
**learning exercise**. A general-purpose interface is a hypothesis about
what varies across domains — and hypotheses formed by imagining future cases
are usually wrong in some detail that only shows up once you actually build
a second case. The disciplined order is:

1. Build one concrete, working adapter (plain text).
2. Get the full pipeline training and generating.
3. *Then*, if a second domain is actually wanted, build it — and let the
   adapter interface bend to fit the real second case rather than a guessed
   one.

This also keeps the project honestly scoped for what it's for: understanding
how a transformer is trained, end to end, in code you wrote and can explain —
not accumulating features.

## 8. Non-goals worth stating explicitly

- **Not** trying to produce a "good" or "useful" language model. A 15-50M
  parameter model trained on a small corpus will not be reliably coherent —
  see the model's `README` / generated samples for honest examples of this.
  The goal is understanding the mechanism, not competing with real LLMs.
- **Not** optimizing training speed or scaling to larger data than fits
  comfortably in memory on one machine.
