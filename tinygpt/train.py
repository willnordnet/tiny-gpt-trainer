"""The training loop.

Everything before this file turned text into integers. This file is where those
integers turn into weights. It is deliberately the least clever file in the
project: one loop, one optimiser, one loss, and a lot of printing, because the
point of this project is to *watch* a model train rather than to hand it off to
a framework and read a final number.

The four things this file has to get right:

  1. Sampling batches from the flat token stream, with targets shifted by one.
  2. A learning-rate schedule (linear warmup, then cosine decay).
  3. AdamW with gradient clipping, and weight decay applied only where it
     belongs.
  4. Enough logging that a broken run is obvious within the first fifty steps.

Before any real run, use the gate:

    python -m tinygpt.train --preset tiny --data data/tokens --overfit-one-batch

which is DESIGN.md section 6.2. If a tiny fixed batch cannot be memorised, the
architecture or the gradient path is broken and a real run is a waste of time.

Then the real thing:

    python -m tinygpt.train --preset tiny --data data/tokens --out checkpoints/
"""

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from tinygpt.config import PRESETS, ModelConfig, TrainConfig, describe
from tinygpt.data.prepare import load_tokens, verify_tokenizer_matches
from tinygpt.model import TinyGPT
from tinygpt.sample import generate_text
from tinygpt.tokenizer.tokenizer import BPETokenizer

# Checkpoints are written as safetensors rather than .npz because safetensors
# files carry a string metadata dict alongside the arrays. That lets a
# checkpoint record its own ModelConfig and step number, so sample.py can
# rebuild the right architecture from the file alone with nothing passed in on
# the side. A checkpoint that needs the caller to remember what shape it was is
# a checkpoint that will eventually be loaded into the wrong shape.
CHECKPOINT_SUFFIX = ".safetensors"

# Steps used by --overfit-one-batch. Measured rather than guessed: on `tiny`,
# loss reaches ~0.007 by step 500 (0.1% of the uniform-guess loss) and is still
# creeping down after that. 200 steps only reaches ~1.67, which looks like a
# failure but is really an unfinished descent, so the default has to be past
# that point or the check cries wolf.
OVERFIT_STEPS = 500


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------


def get_batch(
    tokens: np.ndarray, batch_size: int, context_len: int
) -> tuple[mx.array, mx.array]:
    """Draw a random batch of (inputs, targets) from a flat token stream.

    data/prepare.py stores one long uncut sequence per split rather than
    pre-cut windows, so a "batch" is just `batch_size` random offsets into that
    stream. Two consequences worth knowing:

      - Nearly every possible window is reachable. Pre-cutting at stride
        context_len would have thrown away all the windows that straddle a cut.
      - Windows in a batch can overlap each other. At 309,425 available offsets
        and 32 per batch, collisions are rare enough not to matter.

    Targets are inputs shifted left by exactly one position, which is the whole
    definition of next-token prediction: at every position i the model sees
    tokens[0..i] and is scored on how well it predicted tokens[i+1]. Because
    attention is causal, all context_len of those predictions are legitimate
    training signal from a single forward pass.

    Returns two (batch_size, context_len) int32 arrays.
    """
    # An offset is usable only if a full window *plus one more token* for the
    # shifted target fits after it. Hence the -1.
    highest_offset = len(tokens) - context_len - 1
    if highest_offset < 0:
        raise ValueError(
            f"stream of {len(tokens):,} tokens is too short for a "
            f"context_len={context_len} window plus its shifted target"
        )

    offsets = np.random.randint(0, highest_offset + 1, size=batch_size)

    inputs = np.stack([tokens[o : o + context_len] for o in offsets])
    targets = np.stack([tokens[o + 1 : o + context_len + 1] for o in offsets])

    # Shards are uint16 to halve their size on disk; widen to int32 here because
    # that is what an embedding lookup wants as an index.
    return mx.array(inputs.astype(np.int32)), mx.array(targets.astype(np.int32))


# ---------------------------------------------------------------------------
# Learning-rate schedule
# ---------------------------------------------------------------------------


def learning_rate_at(step: int, cfg: TrainConfig) -> float:
    """Linear warmup, then cosine decay to a floor. Written out rather than
    assembled from mlx.optimizers' schedule helpers, because the shape of this
    curve explains two real failure modes and is worth seeing as arithmetic.

    WARMUP (steps 0 .. warmup_steps): the learning rate ramps from ~0 to peak.
    Adam adapts its step size using running estimates of the gradient's first
    and second moments, and at step 0 those estimates are based on a single
    sample of a randomly-initialised model. Taking a full-size step on that
    guess can throw the weights somewhere they never recover from. Warmup buys
    the moment estimates time to become meaningful.

    COSINE DECAY (the rest): the rate falls along half a cosine from peak down
    to `min_lr_ratio` of peak. Big steps early cover ground; small steps late
    settle into a minimum instead of bouncing around it. Cosine rather than a
    step schedule mostly because it has no cliff edges and no extra knobs.

    THE FLOOR: decaying all the way to zero makes the last few hundred steps do
    nothing at all. A 10% floor keeps them mildly productive.
    """
    if step < cfg.warmup_steps:
        # +1 so step 0 gets a small nonzero rate rather than exactly zero.
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps

    decay_steps = max(1, cfg.max_steps - cfg.warmup_steps)
    progress = min(1.0, (step - cfg.warmup_steps) / decay_steps)

    # cos goes 1 -> -1 over [0, pi], so this factor goes 1 -> 0.
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

    return cfg.learning_rate * (cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * cosine)


def build_optimizer(cfg: TrainConfig) -> optim.Optimizer:
    """AdamW, with weight decay applied to matrices but not to 1-D parameters.

    WHY ADAMW AND NOT ADAM: in plain Adam, L2 regularisation is folded into the
    gradient, so Adam's per-parameter rescaling shrinks the regularisation for
    exactly the parameters that have the largest gradients. AdamW *decouples*
    it: decay is subtracted from the weights directly, independent of the
    adaptive step. That is the "W".

    WHY EXCLUDE 1-D PARAMETERS: the only 1-D parameters in this model are the
    RMSNorm scales. Decaying a normalisation scale toward zero fights the exact
    thing the layer exists to do, and it regularises nothing useful. The
    `ndim >= 2` rule is the standard shorthand for "matrices yes, scales and
    biases no", and it needs no hardcoded list of parameter names to stay
    correct as the model changes.

    MultiOptimizer routes each gradient to the first optimiser whose filter
    accepts it, falling through to the last one. Its learning_rate setter
    forwards to both, so the schedule above still drives the whole thing.
    """
    with_decay = optim.AdamW(
        learning_rate=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    without_decay = optim.AdamW(learning_rate=cfg.learning_rate, weight_decay=0.0)

    return optim.MultiOptimizer(
        [with_decay, without_decay],
        [lambda _path, value: value.ndim >= 2],
    )


# ---------------------------------------------------------------------------
# One training step
# ---------------------------------------------------------------------------


def make_step_fn(model: TinyGPT, optimizer: optim.Optimizer, grad_clip: float):
    """Build the function that performs a single optimiser step.

    Returns a closure so `nn.value_and_grad` is constructed once rather than
    per step, and so the loop below reads as `loss, norm = step(inputs,
    targets)` with no ceremony.
    """
    loss_and_grad = nn.value_and_grad(
        model, lambda m, inputs, targets: m.loss(inputs, targets)
    )

    def step(inputs: mx.array, targets: mx.array) -> tuple[mx.array, mx.array]:
        loss, grads = loss_and_grad(model, inputs, targets)

        # Rescale all gradients together if their combined L2 norm exceeds the
        # limit. One freak batch can otherwise produce a gradient large enough
        # to throw the weights somewhere useless, which shows up as a sudden
        # loss spike or a NaN partway through an otherwise healthy run. Note it
        # clips the *global* norm, so the relative direction is preserved and
        # only the step length shrinks.
        grads, grad_norm = optim.clip_grad_norm(grads, grad_clip)

        optimizer.update(model, grads)

        # MLX is lazy: everything above merely built a graph of operations, and
        # nothing has actually been computed on the GPU yet. Without this
        # mx.eval the graph would keep growing across steps until memory ran
        # out. Evaluating the parameters and the optimiser state here is what
        # makes "one step" a real, completed unit of work.
        mx.eval(model.parameters(), optimizer.state, loss)

        return loss, grad_norm

    return step


def evaluate(
    model: TinyGPT, tokens: np.ndarray, cfg: TrainConfig, context_len: int
) -> float:
    """Mean loss over several random held-out batches.

    Averaged over `eval_batches` rather than measured on one, because a single
    batch is noisy enough that its loss can move more between two adjacent
    steps than the real trend moves in a hundred.

    No gradient bookkeeping is needed to switch off: MLX only computes
    gradients where `value_and_grad` asks for them, so a plain forward pass is
    already the equivalent of torch.no_grad().
    """
    losses = []
    for _ in range(cfg.eval_batches):
        inputs, targets = get_batch(tokens, cfg.batch_size, context_len)
        losses.append(model.loss(inputs, targets))

    return float(mx.mean(mx.stack(losses)))


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


def save_checkpoint(
    path: Path, model: TinyGPT, preset_name: str, step: int, val_loss: float | None
) -> None:
    """Write weights plus enough metadata to reconstruct the model from scratch.

    The config travels *inside* the file as JSON metadata. Loading is then a
    single argument (the path) rather than a path plus a promise that you
    remembered which preset it was, which is the kind of promise that gets
    broken three weeks later.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    weights = dict(tree_flatten(model.parameters()))
    metadata = {
        "preset": preset_name,
        "step": str(step),
        "model_config": json.dumps(asdict(model.cfg)),
        "val_loss": "" if val_loss is None else f"{val_loss:.6f}",
    }

    mx.save_safetensors(str(path), weights, metadata=metadata)


def load_checkpoint(path: str | Path) -> tuple[TinyGPT, dict]:
    """Rebuild a model from a checkpoint file alone.

    Lives here rather than in sample.py for the same reason load_tokens lives in
    data/prepare.py: whatever defines a file format also defines how to read it,
    so the two can never drift apart.
    """
    weights, metadata = mx.load(str(path), return_metadata=True)

    cfg = ModelConfig(**json.loads(metadata["model_config"]))
    model = TinyGPT(cfg)

    # tree_unflatten turns the flat "blocks.0.attention.q_proj.weight" keys back
    # into the nested structure update() expects.
    model.update(tree_unflatten(list(weights.items())))
    mx.eval(model.parameters())

    return model, metadata


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


class RunLogger:
    """Print to stdout and, optionally, tee to a file.

    A training run is worth keeping: the loss curve, the samples, and the exact
    config are the evidence for whether a change helped. Scrollback is not
    evidence.
    """

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._file = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._file = path.open("w", encoding="utf-8")

    def __call__(self, message: str = "") -> None:
        print(message)
        if self._file is not None:
            self._file.write(message + "\n")
            self._file.flush()  # flushed per line so a killed run keeps its log

    def close(self) -> None:
        if self._file is not None:
            self._file.close()


# ---------------------------------------------------------------------------
# The training loops
# ---------------------------------------------------------------------------


def overfit_one_batch(
    model: TinyGPT,
    tokens: np.ndarray,
    train_cfg: TrainConfig,
    context_len: int,
    steps: int,
    log: RunLogger,
) -> bool:
    """DESIGN.md section 6.2: memorise one fixed batch, and see if loss collapses.

    This is the single most useful check in the project, and it costs seconds.
    A model with enough parameters to memorise 8,192 tokens *should* be able to
    drive the loss on those exact tokens to near zero, because it is allowed to
    simply learn the answer. Generalisation is not being tested here. What is
    being tested is that the machinery works at all:

      - gradients actually reach every parameter (a detached tensor or a
        parameter left out of the optimiser shows up as a loss that stalls),
      - the loss is aligned with the targets (an off-by-one in the shift makes
        the task impossible and the loss sits near ln(vocab) forever),
      - the learning rate and clipping are not fighting the update.

    If loss does *not* collapse here, no amount of real data will fix it, and a
    real training run is only a slower way to learn the same thing.

    Returns True if the check passed.
    """
    log("=" * 72)
    log("OVERFIT-ONE-BATCH  (DESIGN.md section 6.2)")
    log("=" * 72)
    log("")
    log("Training on ONE fixed batch, repeatedly. The model is allowed to")
    log("memorise it outright, so loss should collapse toward zero. This tests")
    log("the plumbing (gradient flow, target alignment, the optimiser), not")
    log("learning. If this fails, a real run cannot succeed.")
    log("")

    inputs, targets = get_batch(tokens, train_cfg.batch_size, context_len)
    n_tokens = inputs.size
    log(f"fixed batch: {inputs.shape} = {n_tokens:,} tokens, held constant for {steps} steps")

    # Not the schedule: a flat rate keeps the diagnostic about the model rather
    # than about the warmup curve. Higher than the real peak, because memorising
    # a single batch is meant to be fast.
    flat_lr = 1e-3
    optimizer = build_optimizer(train_cfg)
    optimizer.learning_rate = flat_lr
    log(f"flat learning rate {flat_lr:g} (no schedule, to keep this about the model)")
    log("")

    step_fn = make_step_fn(model, optimizer, train_cfg.grad_clip)

    first_loss = None
    last_loss = None
    # Loss as of 90% of the way through, so the verdict below can tell a broken
    # gradient path (flat curve) from a healthy one that simply ran out of steps.
    late_loss = None
    late_step = int(steps * 0.9)

    for step in range(steps):
        loss, grad_norm = step_fn(inputs, targets)
        value = float(loss)

        if first_loss is None:
            first_loss = value
        if step == late_step:
            late_loss = value
        last_loss = value

        if step % 10 == 0 or step == steps - 1:
            log(f"  step {step:4d}  loss {value:7.4f}  grad_norm {float(grad_norm):7.4f}")

    log("")

    # ln(vocab_size) is the loss of a model that has learned nothing and spreads
    # its probability mass uniformly. Reporting the ratio makes "collapsed" a
    # scale-free judgement rather than a raw number that means little on its own.
    uniform_loss = math.log(model.cfg.vocab_size)
    passed = last_loss < 0.1 * uniform_loss

    log(f"start loss    {first_loss:7.4f}")
    log(f"final loss    {last_loss:7.4f}")
    log(f"uniform-guess loss  ln({model.cfg.vocab_size}) = {uniform_loss:.4f}")
    log("")
    if passed:
        log(f"PASS: final loss is {last_loss / uniform_loss:.1%} of the uniform-guess loss.")
        log("The model can memorise a fixed batch, so gradients reach the")
        log("weights and the targets line up. Safe to start a real run.")
        return True

    # Two different failures wear the same number, and confusing them wastes a
    # lot of time. A curve still descending steeply at the end is not a broken
    # model, it is a step budget that ran out. A curve that has gone flat well
    # above zero is the real failure.
    still_descending = late_loss is not None and last_loss < late_loss * 0.95

    log(f"final loss is {last_loss / uniform_loss:.1%} of the uniform-guess loss, "
        f"above the 10% pass mark.")
    log("")
    if still_descending:
        log(f"INCONCLUSIVE: loss is still falling ({late_loss:.4f} at step "
            f"{late_step} -> {last_loss:.4f} at step {steps - 1}), so this is")
        log("very likely just too few steps rather than a broken model.")
        log(f"Re-run with --steps {steps * 3} before concluding anything.")
    else:
        log(f"FAIL: loss has gone flat ({late_loss:.4f} at step {late_step} -> "
            f"{last_loss:.4f} at step {steps - 1}) well above zero.")
        log("That is the real failure signature. Do not start a real run.")
        log("Check that targets are inputs shifted by one, that every parameter")
        log("is in the optimiser, and that the causal mask is not hiding the")
        log("answer from the position that has to predict it.")

    return passed


def train(
    model: TinyGPT,
    train_tokens: np.ndarray,
    val_tokens: np.ndarray,
    train_cfg: TrainConfig,
    preset_name: str,
    out_dir: Path | None,
    tokenizer: BPETokenizer | None,
    sample_prompt: str,
    log: RunLogger,
) -> None:
    """The real training loop.

    Deliberately one flat loop with the periodic work inline, rather than a
    callback system or a Trainer class. Everything that happens during training
    is visible in one screen of code, in the order it happens.
    """
    context_len = model.cfg.context_len
    tokens_per_step = train_cfg.batch_size * context_len

    log("=" * 72)
    log(f"TRAINING  preset '{preset_name}'  {model.num_parameters():,} parameters")
    log("=" * 72)
    log("")
    log(f"steps          {train_cfg.max_steps:,}")
    log(f"batch          {train_cfg.batch_size} x {context_len} = {tokens_per_step:,} tokens/step")
    log(f"total tokens   {train_cfg.max_steps * tokens_per_step:,} seen "
        f"(~{train_cfg.max_steps * tokens_per_step / len(train_tokens):.1f} epochs "
        f"over {len(train_tokens):,} training tokens)")
    log(f"learning rate  {train_cfg.learning_rate:g} peak, "
        f"{train_cfg.warmup_steps} warmup steps, cosine to "
        f"{train_cfg.min_lr_ratio:.0%}")
    log(f"weight decay   {train_cfg.weight_decay:g} on matrices, 0 on RMSNorm scales")
    log(f"grad clip      {train_cfg.grad_clip:g} (global norm)")
    log("")

    # The number to beat. A model that has learned nothing at all spreads its
    # probability uniformly over the vocabulary and scores exactly this, so it
    # is the honest zero point for reading every loss below.
    uniform_loss = math.log(model.cfg.vocab_size)
    log(f"a model that has learned nothing scores ln({model.cfg.vocab_size}) = "
        f"{uniform_loss:.4f}; everything below that is real learning")
    log("")

    optimizer = build_optimizer(train_cfg)
    step_fn = make_step_fn(model, optimizer, train_cfg.grad_clip)

    # Timing is measured over each logging interval rather than cumulatively, so
    # tokens/sec reflects current throughput rather than being dragged down by
    # the first step (which pays for Metal kernel compilation).
    interval_start = time.perf_counter()
    interval_loss_sum = 0.0
    interval_steps = 0

    for step in range(train_cfg.max_steps):
        # The schedule is applied by setting the rate before each step.
        # MultiOptimizer forwards this to both sub-optimisers.
        learning_rate = learning_rate_at(step, train_cfg)
        optimizer.learning_rate = learning_rate

        loss, grad_norm = step_fn(*get_batch(train_tokens, train_cfg.batch_size, context_len))

        interval_loss_sum += float(loss)
        interval_steps += 1

        is_last = step == train_cfg.max_steps - 1

        # Held-out loss is measured at most once per step and reused. A step
        # that is both an eval step and a checkpoint step (every 500th here,
        # and always the last one) would otherwise call evaluate() twice over
        # different random batches, and report two different validation losses
        # for the same weights, which reads as instability that is not there.
        val_loss: float | None = None

        if (step + 1) % train_cfg.log_interval == 0 or is_last:
            elapsed = time.perf_counter() - interval_start
            mean_loss = interval_loss_sum / interval_steps
            tokens_per_sec = interval_steps * tokens_per_step / elapsed

            log(
                f"step {step + 1:5d}/{train_cfg.max_steps}  "
                f"loss {mean_loss:6.4f}  "
                f"lr {learning_rate:.2e}  "
                f"grad_norm {float(grad_norm):6.3f}  "
                f"{tokens_per_sec:8,.0f} tok/s"
            )

            interval_start = time.perf_counter()
            interval_loss_sum = 0.0
            interval_steps = 0

        if (step + 1) % train_cfg.eval_interval == 0 or is_last:
            val_loss = evaluate(model, val_tokens, train_cfg, context_len)
            # Perplexity is exp(loss): loosely, "how many tokens is the model
            # effectively choosing between at each position". Easier to feel
            # than a log-loss, and it makes the gap to the 4096-way uniform
            # guess concrete.
            log(
                f"  [eval] val loss {val_loss:6.4f}  "
                f"perplexity {math.exp(val_loss):8.1f}  "
                f"(uniform guess would be {model.cfg.vocab_size})"
            )
            interval_start = time.perf_counter()  # do not bill eval time to training

        if tokenizer is not None and ((step + 1) % train_cfg.sample_interval == 0 or is_last):
            # top_k and top_p are deliberately left at their defaults, which
            # means off. A mid-run preview is a progress check, and a progress
            # check wants the distribution the model actually learned, tail and
            # all. Truncating that tail is exactly what those two knobs are for
            # at sampling time, and it would make a bad model read as better
            # than it is, which is the opposite of what this output is for.
            text = generate_text(
                model, tokenizer, sample_prompt, max_tokens=80, temperature=0.8
            )
            log("")
            log(f"  [sample @ step {step + 1}] prompt={sample_prompt!r} temperature=0.8")
            log(f"  {text!r}")
            log("")
            interval_start = time.perf_counter()

        if out_dir is not None and ((step + 1) % train_cfg.checkpoint_interval == 0 or is_last):
            path = out_dir / f"{preset_name}-step{step + 1}{CHECKPOINT_SUFFIX}"
            if val_loss is None:
                val_loss = evaluate(model, val_tokens, train_cfg, context_len)
            save_checkpoint(path, model, preset_name, step + 1, val_loss)
            log(f"  [checkpoint] {path} (val loss {val_loss:.4f})")
            interval_start = time.perf_counter()

    log("")
    log("done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="tiny", choices=sorted(PRESETS))
    parser.add_argument("--data", default="data/tokens", help="directory of token shards")
    parser.add_argument("--out", default=None, help="directory to write checkpoints into")
    parser.add_argument("--vocab", default="vocab.json", help="tokenizer, for sample previews")
    parser.add_argument(
        "--overfit-one-batch",
        action="store_true",
        help="run the DESIGN.md section 6.2 sanity check instead of training",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="override max_steps (or the overfit step count, default 500)",
    )
    parser.add_argument(
        "--prompt",
        default="ROMEO:",
        help="prompt used for the periodic sample previews",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--log-dir",
        default="logs",
        help="where to tee the run log; pass '' to log to stdout only",
    )
    args = parser.parse_args()

    # Seeding both generators: MLX drives weight init and sampling, NumPy drives
    # which offsets get chosen for batches. Seeding only one leaves the run half
    # reproducible, which is worse than not being reproducible at all because it
    # looks deterministic until it is not.
    mx.random.seed(args.seed)
    np.random.seed(args.seed)

    log_path = None
    if args.log_dir:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        suffix = "overfit" if args.overfit_one_batch else args.preset
        log_path = Path(args.log_dir) / f"run-{stamp}-{suffix}.log"
    log = RunLogger(log_path)

    try:
        preset = PRESETS[args.preset]
        model_cfg = preset.model
        train_cfg = preset.train

        train_tokens, val_tokens, meta = load_tokens(args.data)

        # The shards were built by some tokenizer; the model's vocab_size has to
        # agree with it or embedding lookups silently index the wrong rows.
        if meta["vocab_size"] != model_cfg.vocab_size:
            raise ValueError(
                f"shards were built with vocab_size={meta['vocab_size']} but "
                f"preset '{args.preset}' expects {model_cfg.vocab_size}"
            )

        tokenizer = None
        if Path(args.vocab).exists():
            tokenizer = BPETokenizer.load(args.vocab)
            verify_tokenizer_matches(meta, tokenizer)
        else:
            log(f"[warn] {args.vocab} not found; skipping sample previews")

        log("")
        log(describe(preset))
        log("")

        model = TinyGPT(model_cfg)
        mx.eval(model.parameters())  # force init to actually happen before timing

        if args.overfit_one_batch:
            passed = overfit_one_batch(
                model=model,
                tokens=train_tokens,
                train_cfg=train_cfg,
                context_len=model_cfg.context_len,
                steps=args.steps or OVERFIT_STEPS,
                log=log,
            )
            if log_path is not None:
                log("")
                log(f"log written to {log_path}")
            raise SystemExit(0 if passed else 1)

        if args.steps is not None:
            # max_steps also sets the shape of the cosine curve, so overriding it
            # has to replace the value the schedule reads, not just the loop bound.
            train_cfg = TrainConfig(**{**asdict(train_cfg), "max_steps": args.steps})

        out_dir = Path(args.out) if args.out else None
        if out_dir is None:
            log("[warn] no --out given; this run will not save checkpoints")

        train(
            model=model,
            train_tokens=train_tokens,
            val_tokens=val_tokens,
            train_cfg=train_cfg,
            preset_name=args.preset,
            out_dir=out_dir,
            tokenizer=tokenizer,
            sample_prompt=args.prompt,
            log=log,
        )

        if log_path is not None:
            log(f"log written to {log_path}")
    finally:
        log.close()


if __name__ == "__main__":
    main()
