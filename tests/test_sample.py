"""Plumbing tests for the sampler (DESIGN.md section 6.1).

DESIGN.md's bar for this file is "temperature=0 sampling is deterministic and
reproducible; output length respects max_tokens", which confirms the generation
loop's control flow independent of whether the model's outputs are any good.
These tests run against a randomly-initialised model for exactly that reason: a
sampler bug has to be visible without a trained checkpoint, or it will be
mistaken for the model being bad.

The distribution-reshaping tests use hand-made logits with no model at all, so
each claim in sample.py's comments is checked as arithmetic.
"""

import mlx.core as mx
import pytest

from tinygpt.config import ModelConfig
from tinygpt.model import TinyGPT
from tinygpt.sample import (
    apply_temperature,
    apply_top_k,
    apply_top_p,
    generate,
    generate_text,
    reshape_logits,
)
from tinygpt.tokenizer.tokenizer import BPETokenizer


@pytest.fixture
def cfg() -> ModelConfig:
    """A tiny model with a byte-sized vocabulary, so these tests stay fast.

    vocab_size=256 matches a zero-merge BPETokenizer exactly, which lets the
    text-level tests use a real tokenizer rather than a stub.
    """
    return ModelConfig(vocab_size=256, n_layers=2, d_model=32, n_heads=4, context_len=16)


@pytest.fixture
def model(cfg: ModelConfig) -> TinyGPT:
    mx.random.seed(0)
    built = TinyGPT(cfg)
    mx.eval(built.parameters())
    return built


@pytest.fixture
def confident() -> mx.array:
    """Logits with one clear winner and a long thin tail."""
    return mx.array([4.0, 3.2, 2.5, 0.4, 0.1, -0.6])


@pytest.fixture
def torn() -> mx.array:
    """Logits with no clear winner. Same six candidates, nearly flat."""
    return mx.array([1.2, 1.1, 1.0, 0.9, 0.8, 0.6])


def eligible(logits: mx.array) -> list[int]:
    """Indices that can still be sampled, i.e. those not masked to -inf."""
    probabilities = mx.softmax(logits, axis=-1)
    return [i for i, p in enumerate(probabilities.tolist()) if p > 0]


# ---------------------------------------------------------------------------
# Generation loop: length and control flow
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("max_tokens", [0, 1, 5, 20])
def test_generate_appends_exactly_max_tokens_to_the_prompt(model, max_tokens):
    """The DESIGN.md section 6.1 bar: output length respects max_tokens."""
    prompt_ids = [1, 2, 3]
    ids = generate(model, prompt_ids, max_tokens=max_tokens)

    assert len(ids) == len(prompt_ids) + max_tokens
    assert ids[: len(prompt_ids)] == prompt_ids, "the prompt must survive unaltered"


def test_generate_rejects_a_negative_token_budget(model):
    with pytest.raises(ValueError, match="max_tokens"):
        generate(model, [1, 2], max_tokens=-1)


def test_generate_from_an_empty_prompt_still_produces_tokens(model):
    """There is no BOS token in this project, so an empty prompt has to be
    seeded with something or the model has no logits to sample from at all."""
    ids = generate(model, [], max_tokens=4)

    assert len(ids) == 5, "one seeded token plus four generated"


def test_generated_ids_stay_inside_the_vocabulary(model, cfg):
    """A sampler that can emit an out-of-range id produces a decode crash much
    later, at a point that gives no hint the sampler was responsible."""
    ids = generate(model, [1], max_tokens=50, temperature=1.5)

    assert all(0 <= token_id < cfg.vocab_size for token_id in ids)


def test_a_prompt_longer_than_the_context_is_truncated_not_rejected(model, cfg):
    """RoPE has no learned position table, so an over-long prompt would not
    crash the model. It would just be evaluated at distances never trained on,
    which is worse than truncating, because it fails silently."""
    long_prompt = list(range(cfg.context_len * 3))

    ids = generate(model, long_prompt, max_tokens=3)

    assert len(ids) == len(long_prompt) + 3
    assert ids[: len(long_prompt)] == long_prompt


def test_the_token_callback_sees_every_generated_id_in_order(model):
    """Streaming output depends on the callback being complete and ordered."""
    seen: list[int] = []
    ids = generate(model, [7], max_tokens=6, on_token=seen.append)

    assert seen == ids[1:], "callback must report exactly the new tokens, in order"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "knobs",
    [
        {"temperature": 0.0},
        {"temperature": 1e-9},
        {"temperature": 1.0, "top_k": 1},
    ],
    ids=["temperature-zero", "temperature-near-zero", "top-k-of-one"],
)
def test_collapsing_to_a_single_candidate_is_deterministic(model, knobs):
    """The other half of the DESIGN.md section 6.1 bar.

    All three settings leave exactly one token eligible, so the sample is forced
    and the seed cannot matter. Running under deliberately different seeds is
    the point: if any of these still varies, something downstream of the mask is
    reintroducing randomness.
    """
    mx.random.seed(1)
    first = generate(model, [3, 4], max_tokens=12, **knobs)

    mx.random.seed(99)
    second = generate(model, [3, 4], max_tokens=12, **knobs)

    assert first == second


def test_the_same_seed_reproduces_the_same_sample(model):
    """Reproducibility under real sampling, which is what --seed promises."""
    mx.random.seed(7)
    first = generate(model, [3, 4], max_tokens=20, temperature=1.0)

    mx.random.seed(7)
    second = generate(model, [3, 4], max_tokens=20, temperature=1.0)

    assert first == second


def test_different_seeds_produce_different_samples(model):
    """Guards the opposite failure: a sampler stuck on argmax would pass every
    determinism test above while never actually sampling."""
    mx.random.seed(1)
    first = generate(model, [3, 4], max_tokens=30, temperature=1.0)

    mx.random.seed(2)
    second = generate(model, [3, 4], max_tokens=30, temperature=1.0)

    assert first != second


# ---------------------------------------------------------------------------
# Temperature
# ---------------------------------------------------------------------------


def test_temperature_zero_puts_all_the_mass_on_the_argmax(confident):
    reshaped = apply_temperature(confident, 0.0)

    assert eligible(reshaped) == [int(mx.argmax(confident))]


def test_temperature_never_reorders_tokens(confident):
    """The property that makes top-k insensitive to temperature and top-p
    sensitive to it. Worth pinning down rather than asserting in a comment."""
    ranking = sorted(range(confident.size), key=lambda i: -confident[i].item())

    for temperature in (0.1, 0.5, 1.0, 2.0, 10.0):
        scaled = apply_temperature(confident, temperature)
        assert sorted(range(scaled.size), key=lambda i: -scaled[i].item()) == ranking


def test_lower_temperature_sharpens_and_higher_flattens(confident):
    """Sharpness measured as the probability of the single likeliest token."""

    def peak(temperature: float) -> float:
        return float(mx.max(mx.softmax(apply_temperature(confident, temperature), axis=-1)))

    assert peak(0.5) > peak(1.0) > peak(2.0)


def test_negative_temperature_is_rejected(confident):
    """A negative temperature inverts the distribution, making the *least*
    likely token the most likely. That is never what anyone wants, and it looks
    like a badly broken model rather than a bad flag."""
    with pytest.raises(ValueError, match="temperature"):
        apply_temperature(confident, -1.0)


# ---------------------------------------------------------------------------
# Top-k
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [1, 2, 3, 5])
def test_top_k_keeps_exactly_the_k_highest_scoring_tokens(confident, k):
    kept = eligible(apply_top_k(confident, k))

    ranked = sorted(range(confident.size), key=lambda i: -confident[i].item())
    assert kept == sorted(ranked[:k])


@pytest.mark.parametrize("k", [0, -1, 6, 999])
def test_top_k_outside_the_useful_range_is_a_no_op(confident, k):
    """0 is the conventional 'off' switch; a k at or past the vocabulary size
    has nothing left to remove."""
    assert eligible(apply_top_k(confident, k)) == list(range(confident.size))


def test_top_k_keeps_tied_tokens_together():
    """Breaking a tie by sort order would silently favour the lower token id,
    which is a bias with no justification behind it."""
    tied = mx.array([5.0, 3.0, 3.0, 1.0])

    assert eligible(apply_top_k(tied, 2)) == [0, 1, 2]


def test_top_k_ignores_temperature(confident):
    """Because temperature preserves ranking and top-k reads only the ranking."""
    at_one = eligible(reshape_logits(confident, 1.0, 3, 1.0))
    at_five = eligible(reshape_logits(confident, 5.0, 3, 1.0))

    assert at_one == at_five


# ---------------------------------------------------------------------------
# Top-p
# ---------------------------------------------------------------------------


def test_top_p_keeps_the_smallest_prefix_covering_p(confident):
    """The confident distribution is 0.578 / 0.260 / 0.129 / ... so two tokens
    reach 0.838 and three reach 0.967. p=0.90 must therefore keep three."""
    assert eligible(apply_top_p(confident, 0.90)) == [0, 1, 2]
    assert eligible(apply_top_p(confident, 0.80)) == [0, 1]
    assert eligible(apply_top_p(confident, 0.50)) == [0]


@pytest.mark.parametrize("p", [0.5, 0.01, 1e-9])
def test_top_p_always_leaves_at_least_one_token(torn, p):
    """The edge case that motivates subtracting each token's own probability
    from the running total. Comparing raw cumulative sums against a small p
    would mask every token and leave nothing to sample from."""
    assert len(eligible(apply_top_p(torn, p))) >= 1


def test_top_p_adapts_to_the_shape_of_the_distribution_and_top_k_does_not(
    confident, torn
):
    """The single property that distinguishes the two knobs.

    Same p, same k, two distributions: the k-set is the same size in both, the
    p-set is not.
    """
    assert len(eligible(apply_top_k(confident, 3))) == len(eligible(apply_top_k(torn, 3)))

    confident_nucleus = len(eligible(apply_top_p(confident, 0.90)))
    torn_nucleus = len(eligible(apply_top_p(torn, 0.90)))
    assert torn_nucleus > confident_nucleus


def test_top_p_of_one_is_a_no_op(confident):
    assert eligible(apply_top_p(confident, 1.0)) == list(range(confident.size))


@pytest.mark.parametrize("p", [0.0, -0.5, 1.5])
def test_top_p_outside_zero_to_one_is_rejected(confident, p):
    with pytest.raises(ValueError, match="top_p"):
        apply_top_p(confident, p)


def test_higher_temperature_widens_the_nucleus(confident):
    """Temperature and top-p compound: flattening moves mass into the tail, so
    the same p admits more tokens."""
    at_one = len(eligible(reshape_logits(confident, 1.0, 0, 0.90)))
    at_two = len(eligible(reshape_logits(confident, 2.0, 0, 0.90)))

    assert at_two > at_one


# ---------------------------------------------------------------------------
# Composition, and the guarantee that masking actually holds
# ---------------------------------------------------------------------------


def test_masked_tokens_are_never_sampled(model, cfg):
    """The reason masked logits are set to -inf and not to a large negative
    number: exp(-inf) is exactly zero, so a rejected token cannot be drawn even
    once in many thousands of samples. A -1e9 mask would eventually fire.
    """
    mx.random.seed(0)
    logits = model(mx.array([[1, 2, 3]], dtype=mx.int32))[0, -1]

    allowed = set(eligible(apply_top_k(logits, 5)))
    reshaped = apply_top_k(logits, 5)

    drawn = {int(mx.random.categorical(reshaped)) for _ in range(2000)}
    assert drawn <= allowed


def test_the_knobs_compose_to_the_narrower_of_the_two(confident):
    """top-k and top-p are applied in sequence, so using both keeps only what
    survives both filters."""
    k_only = set(eligible(reshape_logits(confident, 1.0, 5, 1.0)))
    p_only = set(eligible(reshape_logits(confident, 1.0, 0, 0.90)))
    both = set(eligible(reshape_logits(confident, 1.0, 5, 0.90)))

    assert both == k_only & p_only


# ---------------------------------------------------------------------------
# The text-level wrapper
# ---------------------------------------------------------------------------


def test_generate_text_returns_the_prompt_followed_by_a_continuation(model):
    """A zero-merge tokenizer is pure byte-level, so its vocabulary is exactly
    the 256 ids this model was built with."""
    tokenizer = BPETokenizer([])
    mx.random.seed(0)

    text = generate_text(model, tokenizer, "ROMEO:", max_tokens=10, temperature=0.0)

    assert text.startswith("ROMEO:")
    assert len(text) > len("ROMEO:")


def test_generate_text_is_reproducible_under_a_fixed_seed(model):
    tokenizer = BPETokenizer([])

    mx.random.seed(5)
    first = generate_text(model, tokenizer, "ROMEO:", max_tokens=15, temperature=1.0)
    mx.random.seed(5)
    second = generate_text(model, tokenizer, "ROMEO:", max_tokens=15, temperature=1.0)

    assert first == second
