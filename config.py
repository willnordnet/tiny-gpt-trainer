"""Model and training size presets.

Why a config module at all, rather than constants scattered across model.py and
train.py: the same handful of numbers (how wide, how deep, how long a context)
determine the shape of nearly every tensor in the project. Naming them in one
place means `model.py` can talk about `cfg.d_model` instead of a bare 256, and
means changing model size is one flag rather than an edit in four files.

Run this file directly to see the parameter-count breakdown per preset:

    python config.py
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Everything that determines the shape of the network.

    Frozen (immutable) on purpose: a config that gets mutated halfway through a
    run is how you end up with a checkpoint whose weights do not match the
    config saved alongside them.
    """

    # Size of the BPE vocabulary. 4096 = 256 raw byte tokens + 3840 learned
    # merges. Chosen so the embedding table stays a modest slice of a tiny
    # model: at d_model=256 it is ~1.0M of ~5.9M params (~18%). A GPT-4-sized
    # 100k vocab would be ~25M params of embedding alone, dwarfing everything
    # else and leaving almost no capacity in the actual transformer.
    vocab_size: int = 4096

    n_layers: int = 6
    d_model: int = 256

    # Attention splits d_model across heads, so d_model must divide evenly by
    # n_heads (checked below). More heads at fixed d_model means more, narrower
    # subspaces to attend in; head_dim below is what each one gets.
    n_heads: int = 4

    # Maximum sequence length the model is trained on. Note that RoPE has no
    # learned position table, so a RoPE model can be *run* on longer sequences
    # than it was trained on (quality degrades, but it does not crash on an
    # out-of-range index the way a learned position embedding would). This is
    # one of the practical reasons RoPE won out; see DESIGN.md section 3.3.
    context_len: int = 256

    # RoPE base frequency. 10000 is the value from the original RoPE paper and
    # what Llama/Qwen/Gemma use. It sets how fast the rotation angle decays
    # across dimension pairs: low dimensions rotate fast (encoding fine,
    # local position differences), high dimensions rotate slowly (encoding
    # coarse, long-range position differences). Raising it stretches the
    # effective context; not needed at these lengths.
    rope_theta: float = 10_000.0

    # Added inside the RMSNorm denominator to avoid dividing by zero when an
    # activation vector is all zeros.
    norm_eps: float = 1e-5

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model={self.d_model} must be divisible by "
                f"n_heads={self.n_heads} so each head gets an equal slice"
            )
        # data/prepare.py stores token ids as uint16 to halve shard size. That
        # is only safe while the vocab fits. Assert it here, at the point the
        # vocab size is declared, rather than discovering it as silently
        # wrapped-around token ids much later.
        if self.vocab_size > 65_535:
            raise ValueError(
                f"vocab_size={self.vocab_size} exceeds uint16 range; "
                "data/prepare.py stores token ids as uint16"
            )

    @property
    def head_dim(self) -> int:
        """Width of each attention head's query/key/value vectors."""
        return self.d_model // self.n_heads

    @property
    def ffn_hidden(self) -> int:
        """Inner width of the SwiGLU feed-forward block.

        A classic transformer FFN is two matrices with a 4x hidden expansion.
        SwiGLU uses *three* matrices (gate, up, down) instead of two, so to
        keep the parameter count comparable the expansion is scaled by 2/3:
        4 * 2/3 = 8/3, roughly 2.67x rather than 4x.

        The result is then rounded *up* to a multiple of 64, because matmuls on
        GPU hardware are meaningfully faster at dimensions that align to the
        underlying tile size. This is exactly the convention Llama uses.
        """
        target = int(2 / 3 * 4 * self.d_model)
        multiple_of = 64
        return multiple_of * math.ceil(target / multiple_of)


@dataclass(frozen=True)
class TrainConfig:
    """Everything about *how* the model is trained, as opposed to its shape."""

    batch_size: int = 32
    max_steps: int = 2000

    # Peak learning rate, reached at the end of warmup and then decayed.
    learning_rate: float = 3e-4

    # Linear warmup from ~0 to peak over these steps. Without warmup, the very
    # first updates are applied to randomly-initialised weights while Adam's
    # running moment estimates are still meaningless, which can knock the model
    # into a bad region it never recovers from.
    warmup_steps: int = 100

    # Cosine decay floor, as a fraction of peak. Decaying all the way to zero
    # wastes the final steps; a small floor keeps them mildly useful.
    min_lr_ratio: float = 0.1

    # AdamW's decoupled weight decay. Applied to matrices but conventionally
    # *not* to norm weights or biases, since shrinking a normalisation scale
    # toward zero fights what the layer is for.
    weight_decay: float = 0.1

    # Global gradient-norm clip. A single freak batch can otherwise produce a
    # huge gradient that blows the weights apart, showing up as a loss spike or
    # a NaN partway through an otherwise healthy run.
    grad_clip: float = 1.0

    # How often (in steps) to do each periodic thing. These exist so training
    # is watchable: see CLAUDE.md on logging.
    log_interval: int = 10
    eval_interval: int = 100
    sample_interval: int = 250
    checkpoint_interval: int = 500

    # Batches averaged when measuring held-out loss. One batch is too noisy to
    # tell a real trend from sampling variance.
    eval_batches: int = 20


@dataclass(frozen=True)
class Preset:
    """A named (model shape, training recipe) pair, selected with --preset."""

    name: str
    model: ModelConfig
    train: TrainConfig
    note: str


PRESETS: dict[str, Preset] = {
    # Deliberately small enough that a full run finishes in minutes. The point
    # of this preset is to prove the pipeline works end to end, so that a bug
    # is found in a cheap run rather than an expensive one.
    "tiny": Preset(
        name="tiny",
        model=ModelConfig(
            n_layers=6,
            d_model=256,
            n_heads=4,
            context_len=256,
        ),
        train=TrainConfig(
            batch_size=32,
            max_steps=2000,
            learning_rate=3e-4,
        ),
        note="fast iteration: prove the pipeline works end to end",
    ),
    # The "real" run, once tiny is verified. Roughly 4.7x the parameters and a
    # 2x longer context, so it sees more text per step and takes proportionally
    # longer. Smaller learning rate because wider models tolerate less.
    "small": Preset(
        name="small",
        model=ModelConfig(
            n_layers=8,
            d_model=512,
            n_heads=8,
            context_len=512,
        ),
        train=TrainConfig(
            batch_size=16,
            max_steps=5000,
            learning_rate=2e-4,
            warmup_steps=200,
        ),
        note="the real run, once tiny is verified",
    ),
}


def param_count(cfg: ModelConfig) -> dict[str, int]:
    """Count parameters analytically, broken down by where they live.

    This is computed from the config rather than by building a model and
    counting arrays, so it can be checked before model.py exists and stays a
    genuinely independent number to compare the built model against. model.py's
    own __main__ prints its real count; the two agreeing is a small but real
    confirmation that the architecture was wired as designed.

    All projections are bias-free, which is standard in modern LMs: the
    following normalisation layer has its own learned scale, so a bias just
    before it is close to redundant.
    """
    d = cfg.d_model
    hidden = cfg.ffn_hidden

    # Token embedding table: one learned d_model vector per vocabulary entry.
    # Counted ONCE even though it is used twice, because input embedding and
    # output projection are tied (they share the same matrix, transposed for
    # the output side). At this scale that saving is large: untied, `tiny`
    # would carry another ~1.0M params, ~18% more, for no reliable gain.
    embedding = cfg.vocab_size * d

    # Attention: four square projections (query, key, value, output).
    attn_per_layer = 4 * d * d

    # SwiGLU: gate and up both project d_model -> hidden, down projects back
    # hidden -> d_model. Three matrices, hence the 2/3 scaling in ffn_hidden.
    ffn_per_layer = 3 * d * hidden

    # RMSNorm holds one scale per channel and no bias (that is the whole
    # simplification vs LayerNorm). Two per block: one before attention, one
    # before the feed-forward, since this is a pre-norm architecture.
    norm_per_layer = 2 * d

    per_layer = attn_per_layer + ffn_per_layer + norm_per_layer

    return {
        "embedding": embedding,
        "attention": attn_per_layer * cfg.n_layers,
        "feed_forward": ffn_per_layer * cfg.n_layers,
        "norms": norm_per_layer * cfg.n_layers + d,  # + final norm before output
        "per_layer": per_layer,
        "total": embedding + per_layer * cfg.n_layers + d,
    }


def describe(preset: Preset) -> str:
    """Render a preset and its parameter breakdown as human-readable text."""
    cfg = preset.model
    counts = param_count(cfg)
    total = counts["total"]

    lines = [
        f"preset '{preset.name}' - {preset.note}",
        f"  layers      {cfg.n_layers}",
        f"  d_model     {cfg.d_model}  ({cfg.n_heads} heads x {cfg.head_dim} head_dim)",
        f"  ffn_hidden  {cfg.ffn_hidden}  ({cfg.ffn_hidden / cfg.d_model:.2f}x d_model, SwiGLU)",
        f"  context     {cfg.context_len} tokens",
        f"  vocab       {cfg.vocab_size}",
        "",
        f"  {'embedding (tied)':<20} {counts['embedding']:>12,}  {counts['embedding'] / total:>5.1%}",
        f"  {'attention':<20} {counts['attention']:>12,}  {counts['attention'] / total:>5.1%}",
        f"  {'feed_forward':<20} {counts['feed_forward']:>12,}  {counts['feed_forward'] / total:>5.1%}",
        f"  {'norms':<20} {counts['norms']:>12,}  {counts['norms'] / total:>5.1%}",
        f"  {'TOTAL':<20} {total:>12,}  ({total / 1e6:.2f}M)",
        "",
        f"  per layer   {counts['per_layer']:,} params",
        f"  batch       {preset.train.batch_size} x {cfg.context_len} = "
        f"{preset.train.batch_size * cfg.context_len:,} tokens/step",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    for preset in PRESETS.values():
        print(describe(preset))
        print()

    print("Note: DESIGN.md section 3.3 estimates these at ~15M and ~50M params.")
    print("The counts above are the arithmetic for the dimensions it specifies,")
    print("with tied embeddings and a 4096 vocab. Treat this file as the")
    print("authority and DESIGN.md's figures as a rough early guess.")
