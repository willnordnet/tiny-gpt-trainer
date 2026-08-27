"""Plumbing tests for model.py, per DESIGN.md section 6.1.

These check that the architecture is *wired* correctly - shapes line up,
gradients flow, the causal mask actually masks. They deliberately say nothing
about whether the model learns anything useful; that is section 6.2's job and
is observed during a real run, not asserted in pytest.

Everything here runs on a deliberately absurd 2-layer, 32-wide model so the
whole file finishes in well under a second.
"""

import math

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import pytest
from mlx.utils import tree_flatten

from tinygpt.config import ModelConfig, param_count
from tinygpt.model import (
    RMSNorm,
    SwiGLU,
    CausalSelfAttention,
    TinyGPT,
    apply_rope,
    causal_mask,
    rope_frequencies,
)


@pytest.fixture
def cfg() -> ModelConfig:
    """A model small enough to build and differentiate many times per second."""
    return ModelConfig(vocab_size=64, n_layers=2, d_model=32, n_heads=4, context_len=16)


@pytest.fixture
def model(cfg: ModelConfig) -> TinyGPT:
    mx.random.seed(0)
    return TinyGPT(cfg)


def random_ids(cfg: ModelConfig, batch: int, seq_len: int) -> mx.array:
    return mx.random.randint(0, cfg.vocab_size, shape=(batch, seq_len))


# --- shapes ---------------------------------------------------------------


@pytest.mark.parametrize("batch,seq_len", [(1, 1), (1, 8), (4, 16), (2, 5)])
def test_forward_returns_logits_per_position(cfg, model, batch, seq_len):
    logits = model(random_ids(cfg, batch, seq_len))
    assert logits.shape == (batch, seq_len, cfg.vocab_size)


def test_forward_works_beyond_trained_context_length(cfg, model):
    """RoPE has no fixed-size table, so longer-than-trained input must not crash.

    Quality would degrade in practice; this only asserts nothing indexes out of
    bounds, which is the concrete difference from a learned position embedding.
    """
    longer = cfg.context_len * 2
    logits = model(random_ids(cfg, 1, longer))
    assert logits.shape == (1, longer, cfg.vocab_size)


def test_parameter_count_matches_config_arithmetic(cfg, model):
    """The built model must agree with config.py's independent hand count.

    config.param_count derives its number from the config alone, so this is a
    genuine cross-check that nothing extra was allocated (an untied output
    matrix, a cached RoPE table registered as a parameter, a stray bias).
    """
    assert model.num_parameters() == param_count(cfg)["total"]


# --- determinism ----------------------------------------------------------


def test_same_input_gives_identical_output(cfg, model):
    ids = random_ids(cfg, 2, 8)
    assert mx.array_equal(model(ids), model(ids))


# --- the causal property --------------------------------------------------


def test_causal_mask_is_zero_at_and_below_diagonal():
    mask = causal_mask(4)
    assert mask.shape == (4, 4)
    for i in range(4):
        for j in range(4):
            if j <= i:
                assert float(mask[i, j]) == 0.0
            else:
                assert float(mask[i, j]) == float("-inf")


def test_editing_a_token_leaves_earlier_positions_untouched(cfg, model):
    """The single most important correctness property of a language model.

    If a later token could influence an earlier position's logits, the model
    would be reading the answer it is being trained to predict, training loss
    would collapse, and generation would be worthless.
    """
    seq_len = 10
    ids = random_ids(cfg, 1, seq_len)
    edit_at = 6

    changed_token = (int(ids[0, edit_at]) + 1) % cfg.vocab_size
    edited = mx.concatenate(
        [ids[:, :edit_at], mx.array([[changed_token]]), ids[:, edit_at + 1 :]], axis=1
    )

    before = model(ids)
    after = model(edited)

    # Everything strictly before the edit must be bit-for-bit identical.
    assert mx.array_equal(before[:, :edit_at], after[:, :edit_at])
    # And the edit must actually have done something, or the test proves nothing.
    assert not mx.array_equal(before[:, edit_at:], after[:, edit_at:])



# --- observing the attention weights --------------------------------------
#
# TinyGPT.__call__ takes an optional list and fills it with the per-layer
# attention weights, which is what the heatmap in web/ draws. These tests pin
# down both halves of that contract: that asking for the weights gives back
# something that really is a set of attention distributions, and that not
# asking for them leaves the forward pass exactly as it was.


def test_collected_attention_has_one_entry_per_layer(cfg, model):
    seq_len = 8
    ids = random_ids(cfg, 2, seq_len)

    collected: list[mx.array] = []
    model(ids, attention_out=collected)

    assert len(collected) == cfg.n_layers
    for weights in collected:
        assert weights.shape == (2, cfg.n_heads, seq_len, seq_len)


def test_collected_attention_rows_are_probability_distributions(cfg, model):
    """Each row is a softmax, so it must sum to 1 and never be negative."""
    ids = random_ids(cfg, 1, 8)

    collected: list[mx.array] = []
    model(ids, attention_out=collected)

    for weights in collected:
        assert float(mx.min(weights)) >= 0.0
        row_sums = mx.sum(weights, axis=-1)
        assert mx.allclose(row_sums, mx.ones_like(row_sums), atol=1e-5)


def test_collected_attention_is_zero_above_the_diagonal(cfg, model):
    """The causal mask, now observable directly rather than inferred.

    test_editing_a_token_leaves_earlier_positions_untouched proves causality
    from the outside, by perturbing an input and watching what moves. This
    proves the same thing from the inside: position i simply has no weight on
    any position j > i. Worth having both - if the mask were ever dropped,
    this test names the cause where the other only reports a symptom.
    """
    seq_len = 8
    ids = random_ids(cfg, 1, seq_len)

    collected: list[mx.array] = []
    model(ids, attention_out=collected)

    upper_triangle = mx.triu(mx.ones((seq_len, seq_len)), k=1)
    for weights in collected:
        assert float(mx.max(weights * upper_triangle)) == 0.0


def test_not_collecting_attention_leaves_the_logits_unchanged(cfg, model):
    """The default path must be untouched by the existence of the collector."""
    ids = random_ids(cfg, 1, 8)

    without = model(ids)
    with_collection = model(ids, attention_out=[])

    assert mx.array_equal(without, with_collection)


# --- RoPE -----------------------------------------------------------------


def test_rope_scores_depend_only_on_relative_position():
    """The defining property of RoPE, asserted rather than just commented.

    Place the same query and key content at several absolute positions with a
    constant gap between them. The attention score must come out the same every
    time: only the distance matters, not where in the sequence it happens.
    """
    head_dim, seq_len, offset = 16, 24, 3
    cos, sin = rope_frequencies(head_dim, seq_len, theta=10_000.0)

    mx.random.seed(1)
    q_vec = mx.random.normal(shape=(head_dim,))
    k_vec = mx.random.normal(shape=(head_dim,))

    def at_every_position(vec: mx.array) -> mx.array:
        stacked = mx.stack([vec] * seq_len).reshape(1, 1, seq_len, head_dim)
        return apply_rope(stacked, cos, sin)[0, 0]

    q_at = at_every_position(q_vec)
    k_at = at_every_position(k_vec)

    scores = [
        float(mx.sum(q_at[i] * k_at[i - offset])) for i in range(offset, seq_len)
    ]
    assert max(scores) - min(scores) < 1e-4


def test_rope_scores_differ_across_relative_positions():
    """The flip side: if every offset scored the same, RoPE would encode nothing."""
    head_dim, seq_len = 16, 24
    cos, sin = rope_frequencies(head_dim, seq_len, theta=10_000.0)

    mx.random.seed(1)
    vec = mx.random.normal(shape=(head_dim,))
    stacked = mx.stack([vec] * seq_len).reshape(1, 1, seq_len, head_dim)
    rotated = apply_rope(stacked, cos, sin)[0, 0]

    scores = [float(mx.sum(rotated[8] * rotated[8 - gap])) for gap in (1, 2, 4, 8)]
    assert len(set(round(s, 4) for s in scores)) == len(scores)


def test_rope_is_identity_at_position_zero():
    """Position 0 is rotated by an angle of 0, so it must pass through unchanged."""
    head_dim = 8
    cos, sin = rope_frequencies(head_dim, seq_len=4, theta=10_000.0)
    x = mx.random.normal(shape=(1, 1, 4, head_dim))
    assert mx.allclose(apply_rope(x, cos, sin)[:, :, 0], x[:, :, 0], atol=1e-6)


def test_rope_preserves_vector_length():
    """A rotation changes direction, never magnitude. If the norm moved, the
    'rotation' would be a scaling in disguise and attention scores would drift
    with absolute position."""
    head_dim = 8
    cos, sin = rope_frequencies(head_dim, seq_len=6, theta=10_000.0)
    x = mx.random.normal(shape=(2, 3, 6, head_dim))
    before = mx.sqrt(mx.sum(mx.square(x), axis=-1))
    after = mx.sqrt(mx.sum(mx.square(apply_rope(x, cos, sin)), axis=-1))
    assert mx.allclose(before, after, atol=1e-5)


# --- the hand-written attention is really attention -----------------------


def test_manual_attention_matches_the_fused_mlx_kernel(cfg):
    """model.py writes the softmax out by hand for readability. This proves the
    readable version is numerically the same thing MLX's fused kernel computes,
    so clarity was not bought with correctness.
    """
    attention = CausalSelfAttention(cfg)
    batch, seq_len = 2, 9
    x = mx.random.normal(shape=(batch, seq_len, cfg.d_model))

    cos, sin = rope_frequencies(cfg.head_dim, seq_len, cfg.rope_theta)
    mask = causal_mask(seq_len)

    ours = attention(x, cos, sin, mask)

    def to_heads(v: mx.array) -> mx.array:
        return v.reshape(batch, seq_len, cfg.n_heads, cfg.head_dim).transpose(
            0, 2, 1, 3
        )

    queries = apply_rope(to_heads(attention.q_proj(x)), cos, sin)
    keys = apply_rope(to_heads(attention.k_proj(x)), cos, sin)
    values = to_heads(attention.v_proj(x))

    fused = mx.fast.scaled_dot_product_attention(
        queries, keys, values, scale=attention.scale, mask=mask
    )
    fused = attention.o_proj(
        fused.transpose(0, 2, 1, 3).reshape(batch, seq_len, cfg.d_model)
    )

    assert mx.allclose(ours, fused, atol=1e-5)


# --- RMSNorm and SwiGLU ---------------------------------------------------


@pytest.mark.parametrize("scale", [0.5, 1.0, 50.0, 1000.0])
def test_rmsnorm_produces_unit_rms_regardless_of_input_scale(scale):
    norm = RMSNorm(dims=64, eps=1e-5)
    x = mx.random.normal(shape=(2, 3, 64)) * scale
    y = norm(x)
    rms = mx.sqrt(mx.mean(mx.square(y), axis=-1))
    assert mx.allclose(rms, mx.ones_like(rms), atol=1e-3)


def test_rmsnorm_does_not_recentre():
    """The concrete difference from LayerNorm: a nonzero input mean survives,
    scaled, rather than being subtracted away."""
    norm = RMSNorm(dims=32, eps=1e-5)
    x = mx.random.normal(shape=(1, 1, 32)) + 5.0  # deliberately far off-centre
    assert float(mx.mean(norm(x))) > 0.5


def test_swiglu_preserves_shape_and_gates(cfg):
    ffn = SwiGLU(cfg)
    x = mx.random.normal(shape=(2, 5, cfg.d_model))
    assert ffn(x).shape == (2, 5, cfg.d_model)

    # A closed gate must silence the unit no matter how large the up branch is.
    closed = nn.silu(mx.array([-30.0])) * mx.array([1000.0])
    assert abs(float(closed)) < 1e-6


# --- loss and gradients ---------------------------------------------------


def test_untrained_loss_is_near_log_vocab_size(cfg, model):
    """An untrained model spreads probability roughly uniformly over the vocab,
    which is a cross-entropy of ln(vocab_size). Far above means initialisation
    is broken; far below on random targets means something has leaked."""
    ids = random_ids(cfg, 4, 8)
    targets = random_ids(cfg, 4, 8)
    loss = float(model.loss(ids, targets))
    assert loss == pytest.approx(math.log(cfg.vocab_size), abs=0.5)


def test_loss_is_a_scalar(cfg, model):
    ids = random_ids(cfg, 2, 8)
    assert model.loss(ids, ids).shape == ()


def test_gradients_are_finite_and_not_all_zero(cfg, model):
    """A dead gradient somewhere in the stack is silent - the model just fails
    to learn that part. Checking every parameter individually catches a whole
    sub-layer that never receives signal."""
    ids = random_ids(cfg, 2, 8)
    targets = random_ids(cfg, 2, 8)

    loss, grads = nn.value_and_grad(model, lambda m: m.loss(ids, targets))(model)
    mx.eval(loss, grads)

    for name, grad in tree_flatten(grads):
        assert bool(mx.all(mx.isfinite(grad))), f"non-finite gradient in {name}"
        assert float(mx.sum(mx.abs(grad))) > 0.0, f"all-zero gradient in {name}"


def test_one_optimiser_step_changes_the_weights(cfg, model):
    ids = random_ids(cfg, 2, 8)
    targets = random_ids(cfg, 2, 8)

    before = mx.array(model.embed.weight)  # copy, not a view

    _, grads = nn.value_and_grad(model, lambda m: m.loss(ids, targets))(model)
    optim.AdamW(learning_rate=1e-2).update(model, grads)
    mx.eval(model.parameters())

    assert not mx.array_equal(before, model.embed.weight)


def test_loss_decreases_when_overfitting_a_single_batch(cfg, model):
    """A miniature of DESIGN.md section 6.2's overfit-one-batch gate.

    If repeatedly training on one fixed batch does not drive loss down, the
    forward pass, the loss, or the gradient path is broken - and no amount of
    real training will fix it. Kept to 30 steps so it stays a unit test.
    """
    ids = random_ids(cfg, 2, 8)
    targets = random_ids(cfg, 2, 8)
    optimiser = optim.AdamW(learning_rate=1e-2)

    first = float(model.loss(ids, targets))
    for _ in range(30):
        _, grads = nn.value_and_grad(model, lambda m: m.loss(ids, targets))(model)
        optimiser.update(model, grads)
        mx.eval(model.parameters())
    last = float(model.loss(ids, targets))

    assert last < first * 0.5
