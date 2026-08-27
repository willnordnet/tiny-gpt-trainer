"""The transformer itself: a decoder-only GPT, written bottom-up.

Read this file top to bottom and you will have read the whole architecture:
RMSNorm, then RoPE, then causal self-attention, then SwiGLU, then the block
that stacks them, then the model that stacks the blocks. Nothing is imported
from a "transformer library" - every matrix multiply that matters is visible
here.

This is a *modern* small-LM recipe, not the 2019 GPT-2 recipe. The four places
it differs are the four terms most worth understanding, and each gets a full
explanation at its definition below:

  RMSNorm            instead of LayerNorm
  RoPE               instead of a learned position-embedding table
  causal attention   (the same as GPT-2, but written out rather than called)
  SwiGLU             instead of a plain GELU feed-forward

Run this file directly for a guided numerical tour of all four, plus a forward
pass and a parameter count:

    python -m tinygpt.model
"""

import math

import mlx.core as mx
import mlx.nn as nn

from tinygpt.config import PRESETS, ModelConfig, param_count

# Standard deviation for weight initialisation. 0.02 is the GPT-2 value and has
# stuck around because it works: small enough that the initial forward pass does
# not blow up through many layers, large enough that gradients are not vanishing
# from step one.
INIT_STD = 0.02


class RMSNorm(nn.Module):
    """Root Mean Square normalisation.

    WHAT IT DOES: rescales each token's activation vector to have roughly unit
    root-mean-square magnitude, then multiplies by a learned per-channel scale.
    In one line:  y = x / sqrt(mean(x^2) + eps) * weight.

    WHY NORMALISE AT ALL: a deep stack of residual additions makes activation
    magnitudes drift - each layer adds to the residual stream, so by layer 6 the
    numbers can be far larger than at layer 1. That drift makes gradients and
    learning rates behave inconsistently across depth. Normalising before each
    sub-layer pins the scale so every layer sees inputs in the same range.

    WHY RMSNorm AND NOT LayerNorm (the rejected alternative): LayerNorm does two
    things - it subtracts the mean (re-centering) and divides by the standard
    deviation (re-scaling), plus it has a learned bias. RMSNorm drops the mean
    subtraction and the bias, keeping only the rescaling. It turns out the
    re-centering contributes very little, so RMSNorm gets the same training
    stability with fewer operations and half the parameters. Llama, Qwen, Gemma
    and Mistral all use it. See DESIGN.md section 3.3.

    Note that the mean is taken over the *feature* axis (d_model), not over the
    batch or the sequence: each token is normalised entirely on its own, which
    is why this works identically at batch size 1 and at inference time.
    """

    def __init__(self, dims: int, eps: float) -> None:
        super().__init__()
        # One learned scale per channel, initialised to 1.0 so the layer starts
        # as a pure normalisation and learns to depart from it.
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, T, d_model) -> same shape out.
        # keepdims=True so the (B, T, 1) mean broadcasts back against (B, T, d).
        mean_square = mx.mean(mx.square(x), axis=-1, keepdims=True)
        # rsqrt is 1/sqrt, done in one op. eps guards against an all-zero vector.
        normalised = x * mx.rsqrt(mean_square + self.eps)
        return normalised * self.weight


def rope_frequencies(
    head_dim: int, seq_len: int, theta: float
) -> tuple[mx.array, mx.array]:
    """Precompute the cosine and sine tables RoPE rotates by.

    Returns two arrays of shape (seq_len, head_dim // 2): the cosine and sine of
    the rotation angle for every (position, dimension-pair) combination.

    THE FREQUENCY LADDER: dimension pair i rotates at angular frequency
    1 / theta^(2i / head_dim). At i = 0 that is 1 radian per position - fast,
    so it distinguishes "one token apart" from "two tokens apart" but wraps
    around quickly. At the top pair the frequency is ~1/theta, so over a whole
    context the angle barely moves - slow, so it encodes coarse, long-range
    position. Together the pairs form a positional "clock" with hands at many
    speeds, the same trick as the sinusoidal encodings in the original
    Transformer paper, but applied as a rotation rather than added on.

    These tables are recomputed on every forward pass rather than cached on the
    module. That is deliberate: an mx.array stored as a module attribute becomes
    a *parameter* in MLX, so it would show up in the parameter count, get weight
    decay applied, and be written into every checkpoint. It is cheap to rebuild.
    """
    # head_dim must be even because RoPE rotates adjacent dimensions in pairs.
    half_dim = head_dim // 2

    # exponent[i] = 2i / head_dim, giving inv_freq[i] = theta^(-2i/head_dim).
    exponent = mx.arange(0, half_dim, dtype=mx.float32) * 2.0 / head_dim
    inv_freq = 1.0 / (theta**exponent)  # (half_dim,)

    positions = mx.arange(seq_len, dtype=mx.float32)  # (T,)

    # Outer product: angles[t, i] = t * inv_freq[i]. Angle grows linearly with
    # position, which is exactly what makes the relative-position property below
    # hold.
    angles = positions[:, None] * inv_freq[None, :]  # (T, half_dim)

    return mx.cos(angles), mx.sin(angles)


def apply_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Rotary Position Embedding: rotate query/key vectors by their position.

    x is (B, n_heads, T, head_dim); the output has the same shape.

    WHAT IT DOES: treat the head_dim numbers of each vector as head_dim/2 points
    in a 2D plane - (x0, x1), (x2, x3), and so on - and rotate each of those
    points by an angle proportional to the token's position in the sequence.
    Position 0 is rotated by nothing, position 5 by five times the base angle.

    WHY A ROTATION IS THE RIGHT WAY TO ENCODE POSITION: attention compares a
    query at position i against a key at position j with a dot product. Rotating
    two vectors by angles a and b and then taking their dot product gives the
    same answer as rotating one of them by (a - b) and leaving the other alone -
    that is just the geometry of rotation. So after RoPE, the attention score
    between token i and token j depends on i - j, their *relative* distance, and
    not on where the pair sits in the sequence. "The word three tokens back"
    means the same thing at position 5 as at position 200. The __main__ block in
    this file demonstrates this numerically.

    THE REJECTED ALTERNATIVE: GPT-2 learns a separate embedding vector for
    "position 0", "position 1", ... up to a fixed maximum, and adds it to the
    token embedding. That has three drawbacks RoPE fixes: it burns parameters
    (context_len x d_model of them), it hard-caps the sequence length at the
    table size, and it gives the model absolute indices when what actually
    matters for language is relative offsets. See DESIGN.md section 3.3.

    Note the pairing convention: this implementation rotates *adjacent* pairs
    (x0 with x1). Llama's reference code instead pairs the first half with the
    second half (x0 with x[d/2]). Both are valid RoPE - they are the same
    operation under a permutation of the dimensions, and a model trained with
    one is simply inconsistent with the other. The only rule is to be
    consistent, which is why this is spelled out rather than left to chance.
    """
    batch, n_heads, seq_len, head_dim = x.shape

    # Split the last axis into (head_dim/2) pairs: (B, H, T, half, 2).
    pairs = x.reshape(batch, n_heads, seq_len, head_dim // 2, 2)
    x_first = pairs[..., 0]  # (B, H, T, half) - the "x" of each 2D point
    x_second = pairs[..., 1]  # (B, H, T, half) - the "y" of each 2D point

    # Broadcast the (T, half) tables across batch and heads: every head at a
    # given position gets rotated by the same angle.
    cos_b = cos[None, None, :, :]  # (1, 1, T, half)
    sin_b = sin[None, None, :, :]  # (1, 1, T, half)

    # The standard 2D rotation matrix, applied pairwise:
    #   [ cos  -sin ] [ x ]
    #   [ sin   cos ] [ y ]
    rotated_first = x_first * cos_b - x_second * sin_b
    rotated_second = x_first * sin_b + x_second * cos_b

    # Interleave the pairs back into the original layout.
    rotated = mx.stack([rotated_first, rotated_second], axis=-1)
    return rotated.reshape(batch, n_heads, seq_len, head_dim)


def causal_mask(seq_len: int) -> mx.array:
    """Build the additive mask that makes attention causal.

    Returns (T, T), zero on and below the diagonal and -inf above it.

    WHAT "CAUSAL" MEANS: this is a *language model* - its whole job is to
    predict token t+1 from tokens 0..t. If position t were allowed to attend to
    position t+1, the answer would be visible in the input and the model would
    learn nothing except to copy it. Training loss would collapse to near zero
    and generation would produce garbage, because at generation time the future
    genuinely does not exist yet. The mask is what keeps training honest.

    WHY -inf AND NOT ZEROING THE WEIGHTS: the mask is added to the raw attention
    scores *before* softmax. exp(-inf) is 0, so forbidden positions receive
    exactly zero probability and, crucially, the surviving positions still sum
    to 1. Zeroing after softmax would leave rows summing to less than 1, which
    quietly shrinks the attention output for early tokens.
    """
    # mx.triu keeps the strict upper triangle (k=1 excludes the diagonal, so a
    # token can always attend to itself) and zeros everything else.
    return mx.triu(mx.full((seq_len, seq_len), float("-inf")), k=1)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention.

    Each token produces a query ("what am I looking for?"), a key ("what do I
    offer?") and a value ("what do I contribute if chosen?"). Every token's
    query is compared against every earlier token's key; the resulting scores
    become weights over the values, and the weighted sum is what the token takes
    away. This is the only place in the whole architecture where information
    moves *between* positions - everything else operates on each token alone.

    "Multi-head" means d_model is split into n_heads independent slices, each
    running the whole procedure separately. One head can track subject-verb
    agreement while another tracks quote nesting, without having to share one
    set of attention weights between those jobs.

    A NOTE ON THE IMPLEMENTATION: MLX ships mx.fast.scaled_dot_product_attention,
    a fused kernel that does everything in __call__ below in one call and runs
    faster. This file writes the softmax out by hand instead, because seeing the
    four lines that *are* attention is the point of the project. The test suite
    checks the two against each other numerically, so this stays honest rather
    than merely well-intentioned.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim

        # 1/sqrt(head_dim). Without it, the dot product of two random head_dim
        # vectors grows with head_dim, pushing softmax into a regime where one
        # weight is ~1 and the rest are ~0 - and where the gradient is ~0 too.
        self.scale = 1.0 / math.sqrt(cfg.head_dim)

        # Four bias-free projections. Bias-free because the next thing every
        # output meets is an RMSNorm with its own learned scale, which makes a
        # bias here nearly redundant; modern LMs drop them throughout.
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def __call__(
        self,
        x: mx.array,
        cos: mx.array,
        sin: mx.array,
        mask: mx.array,
        attention_out: list[mx.array] | None = None,
    ) -> mx.array:
        batch, seq_len, d_model = x.shape

        # (B, T, d_model) -> (B, T, d_model), still one flat vector per token.
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        # Split d_model into heads and move the head axis in front of time, so
        # the last two axes are (T, head_dim) and every matmul below operates
        # per-head automatically:
        #   (B, T, d_model) -> (B, T, H, head_dim) -> (B, H, T, head_dim)
        def to_heads(v: mx.array) -> mx.array:
            return v.reshape(batch, seq_len, self.n_heads, self.head_dim).transpose(
                0, 2, 1, 3
            )

        queries = to_heads(queries)
        keys = to_heads(keys)
        values = to_heads(values)

        # Position enters here and only here. Values are deliberately NOT
        # rotated: position should influence *who attends to whom*, not the
        # content that gets passed along once the choice is made.
        queries = apply_rope(queries, cos, sin)
        keys = apply_rope(keys, cos, sin)

        # Compare every query against every key:
        #   (B, H, T, head_dim) @ (B, H, head_dim, T) -> (B, H, T, T)
        # scores[b, h, i, j] is how much token i wants to attend to token j.
        scores = (queries @ keys.transpose(0, 1, 3, 2)) * self.scale

        # Add -inf wherever j > i, so the softmax below gives those exactly zero.
        scores = scores + mask

        # Normalise each row into a probability distribution over the tokens at
        # or before i. mx.softmax subtracts the row max internally for numerical
        # stability, so no exp() overflow even with large scores.
        weights = mx.softmax(scores, axis=-1)

        # Attention weights are normally a means to an end: they exist for the
        # one matmul below and are then thrown away. A caller that wants to
        # *look* at them - the heatmap in web/ - passes a list to collect them
        # into. Left at None, which is what training and sampling do, this
        # branch does nothing at all, so there is still exactly one attention
        # implementation in this repo. The alternative was to recompute q/k,
        # RoPE and the softmax in a separate viewer module, which is the same
        # two-code-paths-that-must-stay-identical cost that generate() in
        # sample.py declines to pay for a KV cache.
        #
        # This is free: `weights` is already materialised, because `attended`
        # on the next line depends on it. Appending only keeps a reference.
        if attention_out is not None:
            attention_out.append(weights)

        # Weighted sum of values:
        #   (B, H, T, T) @ (B, H, T, head_dim) -> (B, H, T, head_dim)
        attended = weights @ values

        # Put the heads back side by side into one d_model vector per token:
        #   (B, H, T, head_dim) -> (B, T, H, head_dim) -> (B, T, d_model)
        attended = attended.transpose(0, 2, 1, 3).reshape(batch, seq_len, d_model)

        # Let the model mix across heads before returning to the residual stream.
        return self.o_proj(attended)


class SwiGLU(nn.Module):
    """The position-wise feed-forward block, with SwiGLU gating.

    Attention moves information between tokens; this moves information between
    *features* of a single token. It runs identically and independently at every
    position, which is where "position-wise" comes from.

    WHAT SwiGLU DOES: project the token up into a wider hidden space twice, in
    parallel, through two different matrices. Call one output the "gate" and the
    other the "up" projection. Pass the gate through SiLU (x * sigmoid(x), also
    called Swish), multiply the two elementwise, and project back down:

        down( silu(gate(x)) * up(x) )

    WHY GATE AT ALL: in a plain FFN - down(gelu(up(x))) - each hidden unit's
    activation is a fixed function of its own pre-activation. With a gate, one
    half of the network decides, per input, *how much* of the other half to let
    through. That multiplicative interaction is strictly more expressive than
    the additive one, and empirically it is worth the third matrix.

    THE REJECTED ALTERNATIVE: GPT-2 uses a two-matrix GELU FFN with a 4x hidden
    expansion. SwiGLU needs three matrices, so to keep the parameter count level
    the expansion is scaled by 2/3 to about 2.67x - see ModelConfig.ffn_hidden.
    At equal parameter count SwiGLU consistently reaches lower loss, which is
    why Llama, Mistral, Qwen and Gemma all adopted it. DESIGN.md section 3.3.

    "GLU" is Gated Linear Unit; the "Swi" is the SiLU/Swish activation on the
    gate. Swap in a different activation and you get ReGLU, GeGLU, and so on.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        hidden = cfg.ffn_hidden
        self.gate_proj = nn.Linear(cfg.d_model, hidden, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, cfg.d_model, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        # (B, T, d_model) -> (B, T, ffn_hidden) for both branches.
        gate = self.gate_proj(x)
        up = self.up_proj(x)

        # silu(g) = g * sigmoid(g). Unlike ReLU it is smooth and lets small
        # negative values through, so a unit that is currently "off" still has a
        # nonzero gradient and can come back.
        activated = nn.silu(gate) * up

        # Back down to the residual stream width: (B, T, d_model).
        return self.down_proj(activated)


class Block(nn.Module):
    """One transformer layer: attention, then feed-forward, both residual.

    The structure is:

        x = x + attention(norm(x))
        x = x + feed_forward(norm(x))

    WHY THE RESIDUAL (the `x +`): it gives gradients a path from the loss back
    to early layers that does not pass through any weight matrix. Without it,
    the gradient is a product of many Jacobians and shrinks or explodes with
    depth. With it, each sub-layer only has to learn a *correction* to what came
    before, which is a much easier thing to learn.

    WHY PRE-NORM AND NOT POST-NORM: the original Transformer normalised *after*
    the residual add - x = norm(x + attention(x)). That makes the residual path
    pass through a normalisation at every layer, and it needs a learning-rate
    warmup schedule to train stably at all. Pre-norm normalises only the input
    *to* the sub-layer, leaving the residual stream a clean unbroken sum from
    embedding to output. It trains far more reliably, which is why essentially
    every model since GPT-2 uses it. DESIGN.md section 3.3.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attention = CausalSelfAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.feed_forward = SwiGLU(cfg)

    def __call__(
        self,
        x: mx.array,
        cos: mx.array,
        sin: mx.array,
        mask: mx.array,
        attention_out: list[mx.array] | None = None,
    ) -> mx.array:
        x = x + self.attention(
            self.attention_norm(x), cos, sin, mask, attention_out
        )
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class TinyGPT(nn.Module):
    """The whole model: embed, N blocks, final norm, project back to vocab.

    Shapes end to end, for a batch of B sequences of T token ids:

        ids      (B, T)            integers in [0, vocab_size)
        embed    (B, T, d_model)   one learned vector per token
        blocks   (B, T, d_model)   unchanged shape, N times over
        norm     (B, T, d_model)
        logits   (B, T, vocab)     an unnormalised score per vocab entry

    Note the output is (B, T, vocab), not (B, vocab): the model predicts the
    next token at *every* position at once, so a single forward pass over a
    256-token window yields 256 training signals rather than one. Causal masking
    is what makes that sound - each of those predictions only saw its own past.

    TIED EMBEDDINGS: the output projection reuses the input embedding matrix,
    transposed, instead of learning a separate (d_model, vocab) matrix. The
    justification is that both matrices answer the same question - "which
    direction in d_model space means this token?" - so learning it twice wastes
    parameters. At `tiny` scale the saving is about 18% of the whole model.
    DESIGN.md section 3.3.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        # Plain attribute, not an mx.array, so MLX does not treat it as a
        # parameter. It is here so checkpoints can record the shape they belong
        # to and sample.py needs no outside knowledge to reload a model.
        self.cfg = cfg

        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = [Block(cfg) for _ in range(cfg.n_layers)]
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)

        self._init_parameters(cfg)

    def _init_parameters(self, cfg: ModelConfig) -> None:
        """Overwrite MLX's default initialisation with the GPT-2 scheme.

        Two things happen here:

        1. Every weight is drawn from normal(0, 0.02) instead of MLX's default
           uniform(+/- 1/sqrt(fan_in)). Nothing deep - it is the value GPT-2
           used and the one small-model recipes have kept.

        2. The two projections that write *into* the residual stream (attention
           output and the feed-forward's down projection) get their standard
           deviation divided by sqrt(2 * n_layers). Reason: the residual stream
           is a sum of 2*n_layers contributions, and the variance of a sum of
           independent terms is the sum of their variances. Left unscaled, the
           stream's magnitude would grow with depth before a single step of
           training. This scaling keeps it roughly constant regardless of
           n_layers.
        """
        residual_std = INIT_STD / math.sqrt(2 * cfg.n_layers)
        residual_projections = ("attention.o_proj", "feed_forward.down_proj")

        def init_module(name: str, module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                std = residual_std if name.endswith(residual_projections) else INIT_STD
                module.weight = mx.random.normal(shape=module.weight.shape) * std
            elif isinstance(module, nn.Embedding):
                module.weight = mx.random.normal(shape=module.weight.shape) * INIT_STD

        # Walks this module and every submodule, giving each its dotted name
        # (e.g. "blocks.3.attention.o_proj") so the check above can match.
        self.apply_to_modules(init_module)

    def __call__(
        self, ids: mx.array, attention_out: list[mx.array] | None = None
    ) -> mx.array:
        """Forward pass. ids is (B, T) of token ids; returns (B, T, vocab).

        Pass a list as `attention_out` to also collect the attention weights:
        it comes back holding one (B, n_heads, T, T) array per layer, in layer
        order, which is what the heatmap in web/ renders. Omitting it - the
        normal case, and what train.py and sample.py do - costs nothing.
        """
        _, seq_len = ids.shape

        # Position tables and mask are built for exactly this sequence length.
        # Because RoPE has no fixed-size learned table, T may legitimately exceed
        # cfg.context_len - the model was not trained there and quality will
        # suffer, but nothing indexes out of bounds.
        cos, sin = rope_frequencies(
            self.cfg.head_dim, seq_len, self.cfg.rope_theta
        )
        mask = causal_mask(seq_len)

        # (B, T) integer ids -> (B, T, d_model) vectors, by table lookup.
        h = self.embed(ids)

        for block in self.blocks:
            h = block(h, cos, sin, mask, attention_out)

        h = self.final_norm(h)

        # Tied output projection: (B, T, d_model) @ (d_model, vocab).
        # nn.Embedding also offers .as_linear(h) which does exactly this; the
        # matmul is written out so the weight sharing is impossible to miss.
        return h @ self.embed.weight.T

    def loss(self, inputs: mx.array, targets: mx.array) -> mx.array:
        """Mean next-token cross-entropy.

        inputs and targets are both (B, T); targets is inputs shifted left by
        one, so the prediction made at position i is scored against the token
        that actually followed it.

        Cross-entropy here is the negative log probability the model assigned to
        the correct token, averaged over every position in the batch. A useful
        reference point: a model that has learned nothing spreads its
        probability uniformly over the vocabulary, giving a loss of
        ln(vocab_size) - about 8.32 at vocab 4096. Any real run should start
        near there and go down.
        """
        logits = self(inputs)
        vocab = logits.shape[-1]

        # Flatten (B, T, vocab) -> (B*T, vocab) and (B, T) -> (B*T,), since
        # cross_entropy scores a flat list of independent predictions.
        return nn.losses.cross_entropy(
            logits.reshape(-1, vocab),
            targets.reshape(-1),
            reduction="mean",
        )

    def num_parameters(self) -> int:
        """Total learnable scalars, counting the tied embedding only once."""
        # Imported here rather than at module top because it is only needed for
        # reporting, not for the forward pass.
        from mlx.utils import tree_flatten

        return sum(array.size for _, array in tree_flatten(self.parameters()))


# ---------------------------------------------------------------------------
# Standalone demonstration.
#
# Everything below exists to make the four concepts above *visible* rather than
# merely asserted. Each section states the claim the comments made, then shows
# the numbers that back it up.
# ---------------------------------------------------------------------------


def _demo_rmsnorm() -> None:
    print("1. RMSNorm - normalise each token vector by its own RMS magnitude")
    print("   claim: whatever scale a vector arrives at, it leaves at RMS ~1.0")

    norm = RMSNorm(dims=8, eps=1e-5)

    for scale in (0.01, 1.0, 100.0):
        # One batch, one token, 8 features, at wildly different magnitudes.
        x = mx.random.normal(shape=(1, 1, 8)) * scale
        y = norm(x)
        rms_in = float(mx.sqrt(mx.mean(mx.square(x))))
        rms_out = float(mx.sqrt(mx.mean(mx.square(y))))
        print(f"   input RMS {rms_in:>10.4f}  ->  output RMS {rms_out:>7.4f}")

    print("   the first row falls short of 1.0 on purpose: at an input RMS of")
    print("   ~0.007 the squared mean is ~5e-5, the same order as eps=1e-5, so the")
    print("   guard term is no longer negligible. That is eps doing its job -")
    print("   damping the normalisation of a near-zero vector instead of")
    print("   amplifying its noise by a factor of a thousand.")
    print("   note: no mean subtraction and no bias - that is the whole")
    print("   difference from LayerNorm, and it costs nothing in quality.")
    print()


def _demo_rope() -> None:
    print("2. RoPE - position as a rotation, so attention sees *relative* distance")
    print("   claim: dot(rope(q, i), rope(k, j)) depends only on i - j")

    head_dim, seq_len = 8, 16
    cos, sin = rope_frequencies(head_dim, seq_len, theta=10_000.0)

    # One fixed query vector and one fixed key vector, placed at every position.
    # Any difference in their dot product across positions is caused purely by
    # RoPE, since the underlying content is identical everywhere.
    q_vec = mx.random.normal(shape=(head_dim,))
    k_vec = mx.random.normal(shape=(head_dim,))
    q_at = apply_rope(mx.stack([q_vec] * seq_len).reshape(1, 1, seq_len, head_dim), cos, sin)[0, 0]
    k_at = apply_rope(mx.stack([k_vec] * seq_len).reshape(1, 1, seq_len, head_dim), cos, sin)[0, 0]

    print("   same content, different absolute positions, constant offset of 2:")
    for i, j in ((2, 0), (7, 5), (11, 9), (15, 13)):
        score = float(mx.sum(q_at[i] * k_at[j]))
        print(f"     q at position {i:>2}  .  k at position {j:>2}   ->  {score:+.6f}")

    print("   identical scores: absolute index is invisible, only the gap matters.")
    print("   now vary the offset, holding the query position fixed:")
    for j in (11, 9, 7, 3):
        score = float(mx.sum(q_at[12] * k_at[j]))
        print(f"     q at position 12  .  k at position {j:>2}   ->  {score:+.6f}  (offset {12 - j})")

    print("   different offsets give different scores - which is the signal the")
    print("   model uses to learn things like 'agree with the noun 3 tokens back'.")
    print()


def _demo_swiglu() -> None:
    print("3. SwiGLU - one branch decides how much of the other branch gets through")
    print("   claim: silu(gate) acts as a soft, per-unit volume knob on up(x)")

    gate_values = mx.array([-4.0, -1.0, 0.0, 1.0, 4.0])
    up_values = mx.array([2.0, 2.0, 2.0, 2.0, 2.0])
    activated = nn.silu(gate_values)
    gated = activated * up_values

    print("     gate    silu(gate)    up   silu(gate)*up")
    for idx in range(gate_values.size):
        print(
            f"   {float(gate_values[idx]):>6.1f}  {float(activated[idx]):>10.4f}  "
            f"{float(up_values[idx]):>4.1f}  {float(gated[idx]):>13.4f}"
        )

    print("   a strongly negative gate closes the unit to ~0; a positive gate opens")
    print("   it roughly linearly. A plain GELU FFN has no such per-input control.")
    print()


def _demo_causality(model: TinyGPT) -> None:
    print("4. Causal attention - no position can see the future")
    print("   claim: editing token t leaves the logits at every position < t identical")

    seq_len = 12
    ids = mx.random.randint(0, model.cfg.vocab_size, shape=(1, seq_len))
    logits_before = model(ids)

    # Change the last token only.
    edited = mx.concatenate(
        [ids[:, : seq_len - 1], mx.array([[(int(ids[0, -1]) + 1) % model.cfg.vocab_size]])],
        axis=1,
    )
    logits_after = model(edited)

    difference = mx.abs(logits_before - logits_after).max(axis=-1)[0]  # (T,)
    print("     position   max |logit change|")
    for position in range(seq_len):
        marker = "  <- the edited token" if position == seq_len - 1 else ""
        print(f"     {position:>8}   {float(difference[position]):>17.6f}{marker}")

    print("   zeros everywhere before the edit: the mask worked. Without it the")
    print("   model could read the answer it is being asked to predict.")
    print()


def _demo_forward_and_loss(model: TinyGPT, cfg: ModelConfig) -> None:
    print("5. Forward pass and parameter count")

    batch, seq_len = 2, 16
    ids = mx.random.randint(0, cfg.vocab_size, shape=(batch, seq_len))
    logits = model(ids)
    mx.eval(logits)  # MLX is lazy; force the computation before reporting.

    print(f"   input  ids    {tuple(ids.shape)}")
    print(f"   output logits {tuple(logits.shape)}  (a score per vocab entry, per position)")

    built = model.num_parameters()
    expected = param_count(cfg)["total"]
    agreement = "matches" if built == expected else "MISMATCH"
    print(f"   built model   {built:,} parameters")
    print(f"   config.py     {expected:,} parameters  ({agreement})")
    print()

    print("6. Initial loss - the sanity check before any training happens")
    targets = mx.random.randint(0, cfg.vocab_size, shape=(batch, seq_len))
    loss = float(model.loss(ids, targets))
    uniform = math.log(cfg.vocab_size)
    print(f"   loss on random data   {loss:.4f}")
    print(f"   ln(vocab_size)        {uniform:.4f}   (loss of a model that knows nothing)")
    print(f"   ratio                 {loss / uniform:.3f}")
    print("   an untrained model should land near ln(vocab_size). Much higher means")
    print("   initialisation is broken; much lower means the targets leaked in.")
    print()


if __name__ == "__main__":
    mx.random.seed(0)  # so this tour prints the same numbers every run

    cfg = PRESETS["tiny"].model

    print("=" * 72)
    print("TinyGPT architecture tour")
    print(
        f"config: {cfg.n_layers} layers, d_model {cfg.d_model}, "
        f"{cfg.n_heads} heads x {cfg.head_dim}, ffn_hidden {cfg.ffn_hidden}, "
        f"vocab {cfg.vocab_size}"
    )
    print("=" * 72)
    print()

    _demo_rmsnorm()
    _demo_rope()
    _demo_swiglu()

    model = TinyGPT(cfg)

    _demo_causality(model)
    _demo_forward_and_loss(model, cfg)

    print("=" * 72)
    print("All four ideas above are what separate this from a 2019 GPT-2:")
    print("  RMSNorm  replaces LayerNorm  - same stability, fewer ops")
    print("  RoPE     replaces a learned position table - relative, unbounded")
    print("  SwiGLU   replaces a GELU MLP - multiplicative gating")
    print("  tied embeddings - the output projection reuses the input table")
    print("Causal masking is the one piece GPT-2 already had, and still the one")
    print("that makes any of it a language model rather than an autoencoder.")
    print("=" * 72)
