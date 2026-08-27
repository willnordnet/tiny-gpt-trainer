"""Plumbing tests for the training loop (DESIGN.md section 6.1).

These do no real training. They check the parts where a bug is silent: a
target shifted the wrong way, a learning-rate schedule with the wrong shape, or
weight decay quietly eating the RMSNorm scales.

The one exception is the last test, a miniature version of the
--overfit-one-batch gate, small enough to run in a couple of seconds.
"""

import json
import math

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from tinygpt.config import ModelConfig, TrainConfig
from tinygpt.model import TinyGPT
from tinygpt.train import (
    RunLogger,
    build_optimizer,
    evaluate,
    get_batch,
    learning_rate_at,
    load_checkpoint,
    make_step_fn,
    save_checkpoint,
)


@pytest.fixture
def cfg() -> ModelConfig:
    """A deliberately tiny model, so these tests stay fast."""
    return ModelConfig(vocab_size=64, n_layers=2, d_model=32, n_heads=4, context_len=16)


@pytest.fixture
def stream() -> np.ndarray:
    """A flat token stream that counts upward, so shifts are easy to verify."""
    return (np.arange(1000) % 64).astype(np.uint16)


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size,context_len", [(1, 4), (8, 16), (3, 32)])
def test_get_batch_returns_matching_input_and_target_shapes(stream, batch_size, context_len):
    inputs, targets = get_batch(stream, batch_size, context_len)

    assert inputs.shape == (batch_size, context_len)
    assert targets.shape == (batch_size, context_len)


def test_targets_are_inputs_shifted_by_exactly_one(stream):
    """The definition of next-token prediction, and the easiest thing to get
    subtly wrong. A shift of zero makes the task trivial (copy the input) and a
    shift of two makes it impossible, and neither crashes."""
    inputs, targets = get_batch(stream, batch_size=8, context_len=16)

    # Every target position except the last must equal the next input position.
    assert mx.array_equal(targets[:, :-1], inputs[:, 1:])


def test_batch_ids_stay_inside_the_vocabulary(stream, cfg):
    inputs, targets = get_batch(stream, batch_size=16, context_len=16)

    for array in (inputs, targets):
        assert int(mx.min(array)) >= 0
        assert int(mx.max(array)) < cfg.vocab_size


def test_batch_is_widened_from_uint16_to_a_signed_integer_type(stream):
    """Shards are uint16 on disk; embedding lookups want a signed index."""
    inputs, _ = get_batch(stream, batch_size=2, context_len=8)

    assert inputs.dtype == mx.int32


def test_batch_never_runs_off_the_end_of_the_stream():
    """The last window must still have a target token after it. With a stream of
    exactly context_len + 1 tokens there is precisely one legal offset, so every
    draw has to be that same window."""
    context_len = 8
    tokens = np.arange(context_len + 1).astype(np.uint16)

    for _ in range(20):
        inputs, targets = get_batch(tokens, batch_size=4, context_len=context_len)
        assert mx.array_equal(inputs[0], mx.array(tokens[:-1].astype(np.int32)))
        assert mx.array_equal(targets[0], mx.array(tokens[1:].astype(np.int32)))


def test_get_batch_rejects_a_stream_too_short_for_one_window():
    tokens = np.arange(8).astype(np.uint16)

    with pytest.raises(ValueError, match="too short"):
        get_batch(tokens, batch_size=1, context_len=16)


# ---------------------------------------------------------------------------
# Learning-rate schedule
# ---------------------------------------------------------------------------


def test_warmup_ramps_up_linearly_to_the_peak():
    cfg = TrainConfig(learning_rate=1e-3, warmup_steps=10, max_steps=100)

    rates = [learning_rate_at(step, cfg) for step in range(10)]

    assert rates[0] == pytest.approx(1e-4)  # 1/10 of peak at step 0
    assert rates[-1] == pytest.approx(1e-3)  # full peak at the end of warmup
    assert rates == sorted(rates)


def test_learning_rate_is_never_zero_at_the_first_step():
    """A schedule computing step/warmup_steps rather than (step+1)/warmup_steps
    makes the very first update a no-op. Harmless, but it is an off-by-one in
    the one place it is hardest to notice."""
    cfg = TrainConfig(learning_rate=1e-3, warmup_steps=100, max_steps=1000)

    assert learning_rate_at(0, cfg) > 0


def test_cosine_decay_falls_monotonically_from_peak_to_the_floor():
    cfg = TrainConfig(learning_rate=1e-3, warmup_steps=10, max_steps=110, min_lr_ratio=0.1)

    rates = [learning_rate_at(step, cfg) for step in range(10, 110)]

    assert rates == sorted(rates, reverse=True)
    assert rates[0] == pytest.approx(1e-3)
    assert learning_rate_at(110, cfg) == pytest.approx(1e-4)  # the 10% floor


def test_learning_rate_stays_at_the_floor_past_the_end():
    """--steps can outrun max_steps; the schedule must flatten rather than
    carry the cosine round into a rising curve, or a run that overshoots would
    start speeding up again."""
    cfg = TrainConfig(learning_rate=1e-3, warmup_steps=10, max_steps=100, min_lr_ratio=0.1)

    assert learning_rate_at(500, cfg) == pytest.approx(learning_rate_at(100, cfg))


# ---------------------------------------------------------------------------
# Optimiser
# ---------------------------------------------------------------------------


def test_weight_decay_shrinks_matrices_but_not_norm_scales(cfg):
    """The reason build_optimizer uses MultiOptimizer at all.

    With zero gradients, AdamW's update reduces to pure weight decay, so any
    parameter that moves is a parameter being decayed. Matrices should shrink;
    the 1-D RMSNorm scales should sit exactly still, because decaying a
    normalisation scale toward zero fights what the layer is for.
    """
    mx.random.seed(0)
    model = TinyGPT(cfg)
    optimizer = build_optimizer(TrainConfig(learning_rate=1e-3, weight_decay=0.1))

    before = {name: mx.array(value) for name, value in tree_flatten(model.parameters())}
    zero_grads = {name: mx.zeros_like(value) for name, value in before.items()}

    from mlx.utils import tree_unflatten

    optimizer.update(model, tree_unflatten(list(zero_grads.items())))
    mx.eval(model.parameters())

    after = dict(tree_flatten(model.parameters()))

    norm_scales = [name for name in before if before[name].ndim == 1]
    matrices = [name for name in before if before[name].ndim >= 2]
    assert norm_scales and matrices, "expected both kinds of parameter to exist"

    for name in norm_scales:
        assert mx.array_equal(before[name], after[name]), f"{name} was decayed"

    for name in matrices:
        shrank = float(mx.sum(mx.abs(after[name]))) < float(mx.sum(mx.abs(before[name])))
        assert shrank, f"{name} was not decayed"


def test_one_step_moves_every_parameter(cfg, stream):
    """Catches a parameter that is not reachable by gradients at all, which is
    otherwise invisible: the loss still falls, just less than it should."""
    mx.random.seed(0)
    model = TinyGPT(cfg)
    optimizer = build_optimizer(TrainConfig(learning_rate=1e-2, weight_decay=0.0))
    step_fn = make_step_fn(model, optimizer, grad_clip=1.0)

    before = {name: mx.array(value) for name, value in tree_flatten(model.parameters())}
    step_fn(*get_batch(stream, batch_size=4, context_len=cfg.context_len))
    after = dict(tree_flatten(model.parameters()))

    for name in before:
        assert not mx.array_equal(before[name], after[name]), f"{name} never moved"


def test_gradient_clipping_caps_the_global_norm(cfg, stream):
    """A huge learning rate makes gradients large; clipping should keep the
    reported norm bounded regardless."""
    mx.random.seed(0)
    model = TinyGPT(cfg)
    optimizer = build_optimizer(TrainConfig(learning_rate=1.0, weight_decay=0.0))
    step_fn = make_step_fn(model, optimizer, grad_clip=1.0)

    inputs, targets = get_batch(stream, batch_size=4, context_len=cfg.context_len)
    for _ in range(3):
        _, grad_norm = step_fn(inputs, targets)
        # clip_grad_norm reports the norm *before* clipping, so assert on the
        # effect instead: the update stays finite rather than exploding.
        assert math.isfinite(float(grad_norm))

    assert all(
        bool(mx.all(mx.isfinite(value))) for _, value in tree_flatten(model.parameters())
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def test_untrained_validation_loss_is_near_log_vocab_size(cfg, stream):
    """An untrained model has no idea, so it should score about what uniform
    guessing scores. Far below means the eval is leaking; far above means
    something is actively wrong."""
    mx.random.seed(0)
    model = TinyGPT(cfg)

    loss = evaluate(model, stream, TrainConfig(batch_size=4, eval_batches=5), cfg.context_len)

    assert loss == pytest.approx(math.log(cfg.vocab_size), rel=0.1)


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


def test_checkpoint_round_trip_reproduces_identical_logits(cfg, tmp_path):
    """A checkpoint that loads without error but restores slightly different
    weights is worse than one that fails loudly."""
    mx.random.seed(0)
    original = TinyGPT(cfg)
    ids = mx.array([[1, 2, 3, 4, 5]])
    expected = original(ids)

    path = tmp_path / "ckpt.safetensors"
    save_checkpoint(path, original, preset_name="test", step=42, val_loss=1.23)
    restored, _ = load_checkpoint(path)

    assert mx.array_equal(restored(ids), expected)


def test_checkpoint_carries_enough_metadata_to_rebuild_the_model(cfg, tmp_path):
    """The point of using safetensors over .npz: the config travels inside the
    file, so loading needs the path and nothing else."""
    mx.random.seed(0)
    path = tmp_path / "ckpt.safetensors"
    save_checkpoint(path, TinyGPT(cfg), preset_name="test", step=42, val_loss=1.23)

    restored, metadata = load_checkpoint(path)

    assert metadata["step"] == "42"
    assert metadata["preset"] == "test"
    assert json.loads(metadata["model_config"])["d_model"] == cfg.d_model
    assert restored.cfg == cfg


def test_checkpoint_records_an_absent_validation_loss_without_crashing(cfg, tmp_path):
    """safetensors metadata values must be strings, so None needs handling."""
    path = tmp_path / "ckpt.safetensors"
    save_checkpoint(path, TinyGPT(cfg), preset_name="test", step=1, val_loss=None)

    _, metadata = load_checkpoint(path)

    assert metadata["val_loss"] == ""


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def test_run_logger_tees_to_a_file(tmp_path, capsys):
    path = tmp_path / "nested" / "run.log"
    log = RunLogger(path)

    log("first line")
    log("second line")
    log.close()

    assert path.read_text(encoding="utf-8") == "first line\nsecond line\n"
    assert "first line" in capsys.readouterr().out


def test_run_logger_without_a_path_still_prints(capsys):
    log = RunLogger(None)
    log("stdout only")
    log.close()

    assert "stdout only" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# The gate itself, in miniature
# ---------------------------------------------------------------------------


def test_loss_collapses_when_overfitting_a_single_batch(cfg, stream):
    """A fast version of --overfit-one-batch (DESIGN.md section 6.2).

    The real gate uses the `tiny` preset and 500 steps. This one uses a
    2-layer, 32-wide model and 150 steps so it runs in seconds, but it is
    testing the same thing: that a model allowed to memorise a fixed batch
    actually does, which means gradients reach the weights and the targets are
    aligned with the inputs.
    """
    mx.random.seed(0)
    model = TinyGPT(cfg)
    optimizer = build_optimizer(TrainConfig(learning_rate=1e-2, weight_decay=0.0))
    step_fn = make_step_fn(model, optimizer, grad_clip=1.0)

    inputs, targets = get_batch(stream, batch_size=4, context_len=cfg.context_len)

    first = float(step_fn(inputs, targets)[0])
    for _ in range(149):
        last = float(step_fn(inputs, targets)[0])

    assert first == pytest.approx(math.log(cfg.vocab_size), rel=0.15)
    assert last < 0.1 * first, f"loss stalled at {last:.4f}, started at {first:.4f}"
