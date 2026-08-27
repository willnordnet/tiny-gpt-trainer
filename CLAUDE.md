# CLAUDE.md — tiny-GPT-trainer

This file tells any Claude instance (Claude Code or otherwise) how to work on
**tiny-GPT-trainer**. Read `DESIGN.md` first for the architecture and
reasoning — this file is about *how to write the code*, not *what to build*.

## What this project is for

This is an **educational** project. The person building it wants to
understand, line by line, how a transformer is tokenized, trained, and
sampled from — not to get a working model as fast as possible by any means.
That changes how code here should be written, compared to a normal
production repo. Optimize for **a human reading this code and learning
something true about transformers from it.**

Concretely, this means:

- **Prefer clarity over cleverness, always.** If there's a one-liner and a
  five-line version that's easier to follow, write the five-line version.
  Vectorized/idiomatic MLX code is fine and expected, but don't golf it.
- **Prefer explicit over implicit.** Name shapes, name dimensions, don't rely
  on the reader inferring what a tensor's shape is from context.
- **No unexplained magic numbers.** If a constant matters (vocab size,
  context length, a hyperparameter), it should be named and, where it's not
  obvious why that value, commented.

## Code comments

Every non-trivial function should have:
1. A **docstring** stating what it does and, where relevant, *why it exists*
   (not just restating the function name).
2. Inline comments at each conceptually important step — not every line, but
   every step a newcomer to transformers would need explained. Example: when
   implementing RoPE, comment *why* rotation encodes relative position, not
   just *that* the code rotates pairs of dimensions.
3. Where a design choice was made for a specific reason (e.g. RMSNorm over
   LayerNorm, tied embeddings, SwiGLU over GELU), the comment should say what
   the alternative was and why this project uses this one — tie back to
   `DESIGN.md` §3.3 rather than re-litigating it, but don't leave the choice
   unexplained in the code itself.

Bad comment: `# apply rotation`
Good comment: `# Rotate each (x_i, x_i+1) pair by an angle proportional to
# position. This is RoPE: instead of a separate learned "position N" vector,
# position is encoded as a rotation applied to the query/key vectors
# themselves, which is why attention scores end up depending on *relative*
# position between tokens rather than their absolute index.`

## Logging

Every stage of the pipeline should produce visible, human-readable log
output — this is a project meant to be *watched*, not run silently to
completion:

- **Tokenizer training**: log vocab size, a handful of example
  encode/decode round-trips.
- **Data prep**: log number of raw examples, number of tokens after
  encoding, the train/val split, and how many training windows that many
  tokens makes available at the configured context length. Note that the
  windows are a *count*, not an artifact: each split is written as one flat
  `uint16` stream and windows are sliced out of it at batch time
  (`train.py: get_batch`), so nothing pre-packed is ever stored.
- **Training loop**: log step number, loss, tokens/sec, and — at a
  configurable interval (e.g. every N steps) — a short sample generation
  from the model's current weights, so degradation/improvement is visible
  over the course of a run, not just inferred from a loss number.
- **Sampling script**: log the sampling parameters used (temperature,
  top-k/top-p) alongside the output, so it's clear *why* a given generation
  looks the way it does.

Use plain `print`/a simple logger — no need for a logging framework here;
simplicity and readability outrank configurability in this project.

## Examples

Every module that has meaningfully independent behavior should be runnable
and demonstrable on its own, not only as part of the full pipeline:

- `tokenizer.py` should have a small `if __name__ == "__main__":` block that
  trains on a tiny inline string and shows an encode/decode round trip.
- `model.py` should have a similar block that builds a tiny untrained model
  and runs one forward pass on dummy input, printing the output shape —
  proving the architecture is wired correctly before ever touching real data.
- `sample.py` should be runnable standalone against any checkpoint with a
  clear CLI (`--checkpoint`, `--prompt`, `--temperature`, etc.).

The goal: someone should be able to read and run any one file in isolation
to understand what it does, without having to run the entire pipeline first.

## Scope discipline

- **Do not** add a second adapter, multi-domain support, or domain-tag
  conditioning unless explicitly asked. `DESIGN.md` §6 explains why the
  scope is deliberately narrow — resist the pull to generalize before
  there's a second real case to generalize against.
- **Do not** add dependencies beyond what's strictly needed (MLX, and a
  minimal BPE implementation or a small well-known tokenizer training
  library if hand-rolling BPE is a bigger detour than the project warrants —
  ask before adding anything heavier).
- If a task seems to require touching `model.py` or `train.py` to support
  something adapter-specific, stop and flag it — per `DESIGN.md`, that's a
  sign the adapter interface is wrong, not a reason to special-case the
  model or training loop.

## When making changes

See `DESIGN.md` §6 for the full testing strategy (unit tests vs. observed
learning-correctness checks) — the summary for day-to-day work:

- After changing `model.py`, run its own `__main__` smoke test *and* the
  `pytest` suite in `tests/` before running a full training loop against it
  — cheap ways to catch shape bugs early.
- After any change to `model.py` or `train.py`, run the overfit-one-batch
  test (`DESIGN.md` §6.2) before starting a real run. If a tiny repeated
  batch doesn't drive loss to near zero, something is broken — don't burn
  time on a full training run until this passes.
- After changing the tokenizer, re-run its round-trip example and eyeball
  it — silent tokenizer bugs (e.g. off-by-one vocab issues) are easy to miss
  and expensive to debug once training is underway.
- Prefer small, runnable increments over large multi-file changes — this
  project is meant to be followed step by step, and commits/changes should
  be readable as a learning sequence, not just a diff.

## Tone of the code itself

Write this codebase the way a good textbook writes example code: complete,
correct, and unafraid to explain itself. It's fine — good, even — for this
code to be more heavily commented than production code would be. That's a
feature of this project, not a smell.
