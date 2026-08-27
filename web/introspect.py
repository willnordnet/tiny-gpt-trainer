"""Look inside a saved checkpoint: next-token distributions and attention.

Everything here loads a checkpoint from disk rather than reaching into the
running trainer. That is a consequence of running training as a subprocess
(see web/runner.py): the weights being optimised live in another process, and
the only thing that crosses the boundary is a .safetensors file every
`checkpoint_interval` steps.

It turns out to be the more useful arrangement anyway. Comparing step 500
against step 2000 side by side is the interesting question, and that needs
checkpoints regardless of whether the live weights are reachable.

Note that the model and the trainer will both be using the GPU when a run is
in progress. Prompting mid-run works, and slows training down while it does.
That contention is real and the UI says so rather than hiding it.
"""

import math
from pathlib import Path

import mlx.core as mx

from tinygpt.sample import reshape_logits
from tinygpt.tokenizer.tokenizer import BPETokenizer
from tinygpt.train import CHECKPOINT_SUFFIX, load_checkpoint

REPO_ROOT = Path(__file__).resolve().parent.parent

# Attention over a long prompt is a big matrix that renders as an unreadable
# smear and serialises to a lot of JSON: 64 tokens is already a 64x64 grid per
# head. Truncation keeps the *end* of the prompt, which is the part whose
# attention anyone is actually asking about.
MAX_ATTENTION_TOKENS = 64

# Loading a checkpoint is ~22 MB off disk plus an mx.eval. Caching by path
# makes slider-dragging in the probability lab feel instant instead of
# re-reading the file for every keystroke.
_model_cache: dict[str, tuple] = {}


def _load(checkpoint: str) -> tuple:
    """Load (model, metadata), memoised by path.

    Keyed on the resolved path and mtime, so overwriting a checkpoint in place
    during a run does not serve stale weights forever.
    """
    path = (REPO_ROOT / checkpoint).resolve()
    key = f"{path}:{path.stat().st_mtime_ns}"
    if key not in _model_cache:
        _model_cache.clear()  # one model at a time; these are tens of MB each
        _model_cache[key] = load_checkpoint(path)
    return _model_cache[key]


# The tokenizer is small and immutable for the life of a vocab file, but
# BPETokenizer.load prints a line every time. Uncached, dragging a slider
# would bury the server's own output under identical load messages.
_tokenizer_cache: dict[str, BPETokenizer] = {}


def load_tokenizer(vocab_path: str = "vocab.json") -> BPETokenizer:
    """Load vocab.json, memoised by path and mtime (a new run rewrites it)."""
    path = (REPO_ROOT / vocab_path).resolve()
    key = f"{path}:{path.stat().st_mtime_ns}"
    if key not in _tokenizer_cache:
        _tokenizer_cache.clear()
        _tokenizer_cache[key] = BPETokenizer.load(path)
    return _tokenizer_cache[key]


class VocabMismatch(RuntimeError):
    """The tokenizer on disk is not the one this checkpoint was trained with."""


def _checked_tokenizer(model, checkpoint: str, vocab_path: str) -> BPETokenizer:
    """Load the tokenizer and refuse to use it if it does not fit the model.

    This guards a failure mode that is silent and therefore nasty. A checkpoint
    stores weights and a vocab_size, but not the vocabulary itself; the ids it
    predicts only mean something when decoded with the tokenizer it was trained
    on. Training a new BPE vocab overwrites vocab.json in place, so every older
    checkpoint is suddenly being read with the wrong codebook -- and since the
    ids are still in range, nothing crashes. The panels just quietly relabel
    every token, which in a project about seeing what the model is doing is
    about the worst way to be wrong.

    tinygpt/data/prepare.py has verify_tokenizer_matches for the same reason on
    the token shards; this is the checkpoint-side counterpart.
    """
    tokenizer = load_tokenizer(vocab_path)
    if tokenizer.vocab_size != model.cfg.vocab_size:
        raise VocabMismatch(
            f"{vocab_path} has vocab_size={tokenizer.vocab_size}, but "
            f"{checkpoint} was trained with vocab_size={model.cfg.vocab_size}. "
            "Decoding its output with this tokenizer would silently mislabel "
            "every token. Training a new BPE vocab overwrites vocab.json, so "
            "checkpoints from before that run can no longer be read."
        )
    return tokenizer


def list_checkpoints(out_dir: str = "checkpoints") -> list[dict]:
    """Every checkpoint on disk, newest step first.

    A checkpoint carries its own preset, step and val_loss in its metadata
    (train.py: save_checkpoint), so no sidecar index file is needed and this
    stays correct even for checkpoints written by a run the server never saw.
    """
    directory = REPO_ROOT / out_dir
    if not directory.is_dir():
        return []

    found = []
    for path in sorted(directory.glob(f"*{CHECKPOINT_SUFFIX}")):
        try:
            _, metadata = mx.load(str(path), return_metadata=True)
        except Exception:  # noqa: BLE001 - a half-written file must not 500 the page
            continue
        val_loss = metadata.get("val_loss", "")
        found.append({
            "path": f"{out_dir}/{path.name}",
            "name": path.name,
            "preset": metadata.get("preset", "?"),
            "step": int(metadata.get("step", 0)),
            "val_loss": float(val_loss) if val_loss else None,
            "size_bytes": path.stat().st_size,
        })
    return sorted(found, key=lambda entry: entry["step"], reverse=True)


def next_token_distribution(
    checkpoint: str,
    prompt: str,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    top_n: int = 20,
    vocab_path: str = "vocab.json",
) -> dict:
    """The model's next-token candidates, before and after the sampling knobs.

    Returns both distributions on purpose. Showing only the post-knob one
    would hide what the knobs actually did; showing both, with the eliminated
    candidates still visible, is the entire teaching point of the panel.

    The reshaping runs here rather than in JavaScript, calling the very same
    `reshape_logits` that `generate()` uses. A second implementation of
    temperature/top-k/top-p in the browser would be a copy of the exact
    mechanics this project exists to explain, sitting somewhere it could
    silently drift from sample.py.
    """
    model, metadata = _load(checkpoint)
    tokenizer = _checked_tokenizer(model, checkpoint, vocab_path)

    prompt_ids = tokenizer.encode(prompt) or [0]
    context = prompt_ids[-model.cfg.context_len:]

    logits = model(mx.array([context]))[0, -1]

    raw_probs = mx.softmax(logits, axis=-1)
    reshaped = reshape_logits(logits, temperature, top_k, top_p)
    knob_probs = mx.softmax(reshaped, axis=-1)

    # Rank by the *raw* distribution so the bars keep a stable order as the
    # sliders move. Re-sorting on every change would make eliminated tokens
    # jump around instead of visibly dropping out where they stood.
    order = mx.argsort(-raw_probs)[:top_n].tolist()

    candidates = []
    for token_id in order:
        after = float(knob_probs[token_id])
        candidates.append({
            "id": int(token_id),
            "token": tokenizer.decode([int(token_id)]),
            "prob": float(raw_probs[token_id]),
            "prob_after": after,
            # "Eliminated" means top-k or top-p cut it, which shows up as an
            # exactly-zero probability after the -inf mask goes through softmax.
            "eliminated": after == 0.0,
        })

    return {
        "checkpoint": checkpoint,
        "step": int(metadata.get("step", 0)),
        "prompt": prompt,
        "prompt_token_count": len(prompt_ids),
        "candidates": candidates,
        "settings": {"temperature": temperature, "top_k": top_k, "top_p": top_p},
        # Entropy of the reshaped distribution, in nats, on the same scale as
        # the training loss: it is literally the loss the model would take if
        # its own sample turned out to be the right answer. Watching it fall
        # as temperature drops connects the two panels.
        "entropy": _entropy(knob_probs),
        "entropy_raw": _entropy(raw_probs),
        "uniform_entropy": math.log(model.cfg.vocab_size),
    }


def _entropy(probs: mx.array) -> float:
    """Shannon entropy in nats, guarding log(0) for masked-out tokens."""
    safe = mx.where(probs > 0, probs, mx.ones_like(probs))
    return float(-mx.sum(probs * mx.log(safe)))


def attention_grid(
    checkpoint: str,
    prompt: str,
    layer: int = 0,
    head: int = 0,
    vocab_path: str = "vocab.json",
) -> dict:
    """One layer/head's attention matrix over a prompt, with token labels.

    The grid is strictly lower-triangular: position i can only attend to
    positions at or before i. That empty upper triangle is the causal mask,
    made visible rather than described.
    """
    model, metadata = _load(checkpoint)
    tokenizer = _checked_tokenizer(model, checkpoint, vocab_path)

    prompt_ids = tokenizer.encode(prompt) or [0]
    context = prompt_ids[-min(MAX_ATTENTION_TOKENS, model.cfg.context_len):]

    collected: list[mx.array] = []
    model(mx.array([context]), attention_out=collected)

    layer = max(0, min(layer, len(collected) - 1))
    head = max(0, min(head, model.cfg.n_heads - 1))

    # (B, H, T, T) -> the one (T, T) grid being asked for.
    grid = collected[layer][0, head]

    return {
        "checkpoint": checkpoint,
        "step": int(metadata.get("step", 0)),
        "layer": layer,
        "head": head,
        "n_layers": model.cfg.n_layers,
        "n_heads": model.cfg.n_heads,
        "tokens": [tokenizer.decode([token_id]) for token_id in context],
        "truncated": len(prompt_ids) > len(context),
        "weights": grid.tolist(),
    }


if __name__ == "__main__":
    # Standalone demo against whatever checkpoints happen to be on disk.
    checkpoints = list_checkpoints()
    if not checkpoints:
        raise SystemExit("[introspect] no checkpoints/ yet - run a training first")

    newest = checkpoints[0]
    print(f"[introspect] {len(checkpoints)} checkpoint(s); using {newest['name']} "
          f"(step {newest['step']}, val loss {newest['val_loss']})")

    print()
    result = next_token_distribution(newest["path"], "ROMEO:", temperature=0.8, top_k=5)
    print(f"[introspect] next token after {result['prompt']!r}, "
          f"temperature=0.8 top_k=5")
    print(f"[introspect] entropy {result['entropy']:.3f} nats "
          f"(raw {result['entropy_raw']:.3f}, uniform {result['uniform_entropy']:.3f})")
    for candidate in result["candidates"][:8]:
        bar = "#" * int(candidate["prob"] * 60)
        cut = "  (cut)" if candidate["eliminated"] else ""
        print(f"[introspect]   {candidate['token']!r:<12} {candidate['prob']:.4f} {bar}{cut}")

    print()
    grid = attention_grid(newest["path"], "To be or not to be", layer=0, head=0)
    print(f"[introspect] attention layer {grid['layer']} head {grid['head']}, "
          f"{len(grid['tokens'])} tokens")
    for token, row in zip(grid["tokens"], grid["weights"]):
        cells = " ".join(f"{value:.2f}" if value > 0 else "   . " for value in row)
        print(f"[introspect]   {token!r:<10} {cells}")
