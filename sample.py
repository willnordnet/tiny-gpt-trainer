"""Autoregressive generation: temperature, top-k, and top-p.

Training is where a model learns a probability distribution over next tokens.
Sampling is where you decide what to *do* with that distribution, and it is a
genuinely separate question. The same checkpoint can look fluent or deranged
depending only on the three knobs in this file, which is why they are written
out here as arithmetic rather than delegated to a library call.

The core loop is four lines and never changes:

    1. Feed the tokens so far through the model.
    2. Take the logits at the *last* position (the prediction for what comes
       next). Everything earlier is a prediction about a token already known.
    3. Reshape that distribution with temperature / top-k / top-p.
    4. Sample one token from it, append, repeat.

Run it against a checkpoint:

    python sample.py --checkpoint checkpoints/tiny-step2000.safetensors \
        --prompt "ROMEO:" --max-tokens 200 --temperature 0.8 --top-k 40

Or run this file with no arguments to see what each knob does to a small
hand-made distribution, with no model and no checkpoint involved:

    python sample.py
"""

import argparse
import time
from pathlib import Path
from typing import Callable

import mlx.core as mx

from model import TinyGPT
from tokenizer.tokenizer import BPETokenizer

# Masked-out tokens are set to negative infinity rather than to a large negative
# number. exp(-inf) is exactly 0, so a rejected token receives exactly zero
# probability. A "large negative number" like -1e9 leaves a vanishingly small
# but nonzero probability, which over thousands of sampled tokens does
# eventually fire, and the resulting one-in-a-thousand nonsense token is close
# to impossible to debug after the fact.
NEGATIVE_INFINITY = -mx.inf

# Below this, temperature is treated as exactly 0 (greedy/argmax) instead of
# dividing by a near-zero float. At 1e-6 the division has already overflowed the
# useful range of float32 anyway, so this changes no real output; it only avoids
# producing inf/NaN logits and sampling from garbage.
GREEDY_TEMPERATURE_EPS = 1e-6


# ---------------------------------------------------------------------------
# The three knobs
#
# All three take a 1-D array of logits for one position and return a 1-D array
# of the same shape. Nothing here normalises to probabilities: softmax is
# invariant to what the sampler does downstream, and mx.random.categorical
# takes unnormalised logits directly, so converting to probabilities and back
# would only lose precision.
# ---------------------------------------------------------------------------


def apply_temperature(logits: mx.array, temperature: float) -> mx.array:
    """Divide the logits by `temperature`, rescaling how peaked the distribution is.

    Softmax turns logits into probabilities via exp(logit_i) / sum(exp(logit_j)).
    Dividing every logit by T before that exponential stretches or compresses
    the *gaps* between them, and since exp turns gaps into ratios, small changes
    here have large effects:

      T < 1  divides by a fraction, so gaps grow. The already-likely tokens take
             an even bigger share. Output is more confident and more repetitive.
      T = 1  the distribution the model actually learned, untouched.
      T > 1  gaps shrink toward zero, so the distribution flattens toward
             uniform. Output is more surprising, and past roughly 1.2 on a small
             model, more incoherent.
      T -> 0 gaps grow without bound until all the probability is on the single
             highest logit. This is argmax, handled separately below.

    Worth knowing: temperature does not change the *ranking* of tokens, only the
    spacing between them. That is why top-k (which only cares about rank) is
    unaffected by temperature, while top-p (which cares about accumulated
    probability mass) very much is.
    """
    if temperature < 0:
        raise ValueError(f"temperature={temperature} must be >= 0")

    if temperature < GREEDY_TEMPERATURE_EPS:
        # Greedy decoding, expressed as a distribution so the rest of the
        # pipeline does not need a special case: all the mass on the argmax.
        best = mx.argmax(logits)
        return mx.where(
            mx.arange(logits.size) == best, logits, NEGATIVE_INFINITY
        )

    return logits / temperature


def apply_top_k(logits: mx.array, k: int) -> mx.array:
    """Keep only the `k` highest-scoring tokens; give everything else zero probability.

    The problem this solves: a 4096-token vocabulary has a long tail of
    thousands of tokens that are each individually near-impossible but which,
    summed, can hold a few percent of the probability mass. Sample often enough
    and one of them fires, derailing the sentence. Top-k truncates that tail
    outright.

    The trade-off is that k is a fixed count applied to a distribution whose
    sharpness varies wildly by position. After "ROMEO" the next token is almost
    certainly ":" and k=40 is 39 tokens of pure noise held in reserve. At the
    start of a line, forty candidates may be genuinely too few. Top-p exists
    because of exactly this mismatch.

    Implementation note: rather than scattering a mask back into vocabulary
    order, this finds the k-th largest logit and masks everything below it. Same
    result, and it keeps top-k and top-p structurally identical (both are "find
    a cutoff value, mask below it"), which makes the difference between them
    easier to see.
    """
    if k <= 0 or k >= logits.size:
        # k=0 is the conventional "off" switch, and a k at or beyond the vocab
        # size cannot remove anything.
        return logits

    # Descending sort, so index k-1 is the k-th largest value.
    descending = mx.sort(logits)[::-1]
    kth_largest = descending[k - 1]

    # Strictly-less-than, so ties at the boundary are all kept. That can leave
    # slightly more than k candidates, which is the right call: breaking a tie
    # by sort order would silently favour whichever token happened to have the
    # lower id.
    return mx.where(logits < kth_largest, NEGATIVE_INFINITY, logits)


def apply_top_p(logits: mx.array, p: float) -> mx.array:
    """Keep the smallest set of top tokens whose probabilities sum to at least `p`.

    Also called nucleus sampling. Where top-k asks "how many candidates?",
    top-p asks "how much probability mass?" and lets the count follow from the
    distribution's own shape:

      - Where the model is confident, one or two tokens already cover p=0.9, so
        the candidate set collapses to those and nothing else can be chosen.
      - Where the model is genuinely uncertain, it may take fifty tokens to
        reach 0.9, and all fifty stay eligible.

    That adaptivity is the whole point, and it is why top-p is the more common
    default in practice. p=0.9 to 0.95 is the usual range; p=1.0 disables it.

    Note this runs *after* temperature, and unlike top-k it is sensitive to that
    ordering: temperature changes the probability mass, so a higher temperature
    flattens the distribution and widens the nucleus, letting more tokens
    through. The two knobs compound rather than acting independently.
    """
    if not 0.0 < p <= 1.0:
        raise ValueError(f"top_p={p} must be in (0, 1]")
    if p == 1.0:
        return logits

    descending = mx.sort(logits)[::-1]
    probabilities = mx.softmax(descending, axis=-1)

    # Cumulative mass *strictly before* each token. Subtracting the token's own
    # probability from the running total is what makes the token that crosses
    # the p boundary get included rather than excluded, and it guarantees the
    # single most likely token always survives: its "mass before" is 0, which is
    # below any positive p. Without that guarantee, a small p against a flat
    # distribution would mask every token and leave nothing to sample.
    cumulative_before = mx.cumsum(probabilities, axis=-1) - probabilities
    keep = cumulative_before < p

    # The kept tokens are always a prefix of the descending order, so the cutoff
    # is just the logit of the last one kept.
    n_kept = int(mx.sum(keep))
    cutoff = descending[n_kept - 1]

    return mx.where(logits < cutoff, NEGATIVE_INFINITY, logits)


def reshape_logits(
    logits: mx.array, temperature: float, top_k: int, top_p: float
) -> mx.array:
    """Apply all three knobs, in the order they conventionally compose.

    Temperature first, because it defines the probabilities the other two are
    then measured against. Then top-k, then top-p: k trims the distribution to a
    manageable candidate count, and p trims whatever of that remains that is not
    carrying real mass. Using both is common; using neither means sampling from
    the raw distribution, tail and all.
    """
    logits = apply_temperature(logits, temperature)
    logits = apply_top_k(logits, top_k)
    logits = apply_top_p(logits, top_p)
    return logits


# ---------------------------------------------------------------------------
# The generation loop
# ---------------------------------------------------------------------------


def generate(
    model: TinyGPT,
    prompt_ids: list[int],
    max_tokens: int,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    on_token: Callable[[int], None] | None = None,
) -> list[int]:
    """Generate `max_tokens` new tokens after `prompt_ids`.

    Works purely in token ids, with no knowledge of the tokenizer. That is not
    tidiness for its own sake: it means this function can be tested against a
    randomly-initialised model with no vocabulary at all, which is exactly what
    the tests in tests/test_sample.py do.

    Args:
        prompt_ids: Starting context. May be empty, in which case generation
            starts from token 0 (there is no BOS token in this project, so
            something has to be fed in to get logits at all).
        on_token: Called with each newly sampled id as it is produced, for
            streaming output. Generation itself does not depend on it.

    Returns:
        prompt_ids followed by exactly `max_tokens` newly generated ids.

    There is no KV cache. Every iteration re-runs the full forward pass over
    the entire prefix, recomputing keys and values for tokens that have not
    changed since the last step, which makes generating n tokens O(n^2) work
    instead of O(n). A cache is a pure performance optimisation: it cannot
    change a single token of the output, only how fast it arrives. Adding one
    would mean a second attention code path in the repo that has to stay
    behaviourally identical to the first, which is a real maintenance cost to
    pay for a speedup on samples that already take under a second.
    """
    if max_tokens < 0:
        raise ValueError(f"max_tokens={max_tokens} must be >= 0")

    ids = list(prompt_ids) or [0]

    for _ in range(max_tokens):
        # Feed at most the trained context length. RoPE has no learned position
        # table, so a longer window would not crash the way a learned position
        # embedding would; it would just place tokens at distances the model has
        # never been trained on. Truncating from the left keeps the most recent
        # context, which is the part that matters most for the next token.
        window = ids[-model.cfg.context_len :]

        # (1, T) in, (1, T, vocab) out. The batch dimension of 1 exists only
        # because the model always expects one.
        logits = model(mx.array([window], dtype=mx.int32))

        # Position -1 is the prediction for the token that follows everything
        # seen so far. The other T-1 predictions are about tokens already in the
        # window, and were only useful during training.
        next_logits = reshape_logits(logits[0, -1], temperature, top_k, top_p)

        # categorical() samples one index in proportion to softmax(logits),
        # which is precisely "draw a token from the model's distribution".
        next_id = int(mx.random.categorical(next_logits))

        ids.append(next_id)
        if on_token is not None:
            on_token(next_id)

    return ids


def generate_text(
    model: TinyGPT,
    tokenizer: BPETokenizer,
    prompt: str,
    max_tokens: int,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    stream: bool = False,
) -> str:
    """Encode a prompt, generate, decode. The text-level wrapper around generate().

    When `stream` is set, text is printed as it is produced rather than at the
    end. Streaming re-decodes the whole id list each step and prints whatever is
    new, because a BPE token is a byte sequence and can end partway through a
    multi-byte character: decoding tokens one at a time in isolation would print
    a replacement character where a perfectly valid character was being built up
    across two tokens.
    """
    prompt_ids = tokenizer.encode(prompt)

    # The streaming callback accumulates its own copy of the id list, rather
    # than reading generate()'s. generate() owns its list and hands out only the
    # newly sampled id, which keeps it independent of any tokenizer; the cost is
    # that a streaming caller has to keep count itself.
    streamed_ids = list(prompt_ids) or [0]
    printed = 0

    def emit(new_id: int) -> None:
        nonlocal printed
        streamed_ids.append(new_id)
        text_so_far = tokenizer.decode(streamed_ids)
        print(text_so_far[printed:], end="", flush=True)
        printed = len(text_so_far)

    if stream:
        # Print the prompt first so the continuation reads as one passage, and
        # prime `printed` with its length so the first emitted token does not
        # reprint it.
        prompt_text = tokenizer.decode(streamed_ids)
        print(prompt_text, end="", flush=True)
        printed = len(prompt_text)

    all_ids = generate(
        model=model,
        prompt_ids=prompt_ids,
        max_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        on_token=emit if stream else None,
    )

    if stream:
        print()

    return tokenizer.decode(all_ids)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def describe_settings(
    checkpoint: Path,
    metadata: dict,
    model: TinyGPT,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
) -> str:
    """Render the settings block printed above every generation.

    CLAUDE.md asks for this explicitly: an output with no record of the
    parameters that produced it is an anecdote. With the block above it, a
    sample is reproducible and its failure modes are attributable.
    """
    step = metadata.get("step", "?")
    val_loss = metadata.get("val_loss") or "not recorded"

    if temperature < GREEDY_TEMPERATURE_EPS:
        temp_note = "greedy (argmax, deterministic)"
    elif temperature < 1.0:
        temp_note = "sharpened, more confident and more repetitive than trained"
    elif temperature == 1.0:
        temp_note = "the distribution the model actually learned"
    else:
        temp_note = "flattened, more surprising and less coherent than trained"

    k_note = "off, full tail eligible" if top_k <= 0 else f"only the top {top_k} tokens eligible"
    p_note = (
        "off, full tail eligible"
        if top_p >= 1.0
        else f"smallest token set covering {top_p:.0%} of the mass"
    )

    return "\n".join(
        [
            "sampling parameters",
            f"  checkpoint    {checkpoint}",
            f"  preset        {metadata.get('preset', '?')} "
            f"({model.num_parameters() / 1e6:.2f}M params, "
            f"context {model.cfg.context_len})",
            f"  trained to    step {step}, val loss {val_loss}",
            f"  prompt        {prompt!r}",
            f"  max_tokens    {max_tokens}",
            f"  temperature   {temperature}  ({temp_note})",
            f"  top_k         {top_k}  ({k_note})",
            f"  top_p         {top_p}  ({p_note})",
        ]
    )


def demo_knobs() -> None:
    """Show what each knob does to a distribution, with no model involved.

    Two hand-made distributions over the same six candidate tokens: one where
    the model is confident, one where it is torn. Running every setting against
    both is what makes the difference between top-k and top-p visible, since on
    any single distribution the two can easily coincide.
    """
    words = ["the", "a", "and", "quixotic", "zephyr", "xylem"]

    # "Confident": one clear winner, the rest an afterthought. This is what the
    # distribution looks like after "ROMEO" when the next token is almost
    # certainly ":".
    confident = mx.array([4.0, 3.2, 2.5, 0.4, 0.1, -0.6])

    # "Torn": no strong winner. This is what it looks like at the start of a
    # line, where many continuations are about equally reasonable.
    torn = mx.array([1.2, 1.1, 1.0, 0.9, 0.8, 0.6])

    def show(label: str, logits: mx.array, **knobs) -> int:
        reshaped = reshape_logits(
            logits,
            knobs.get("temperature", 1.0),
            knobs.get("top_k", 0),
            knobs.get("top_p", 1.0),
        )
        probabilities = mx.softmax(reshaped, axis=-1)
        cells = [
            f"{p:.3f}" if p > 0 else "  -  " for p in probabilities.tolist()
        ]
        alive = int(mx.sum(probabilities > 0))
        print(f"  {label:<30} {'  '.join(cells)}   [{alive} eligible]")
        return alive

    header = "  " + " " * 30 + "  ".join(f"{w:>5.5}" for w in words)

    print("what the sampling knobs do to a distribution")
    print()
    print("A CONFIDENT distribution (one clear winner)")
    print(header)
    show("temperature=1.0 (as-is)", confident)
    show("temperature=0.5 (sharper)", confident, temperature=0.5)
    show("temperature=2.0 (flatter)", confident, temperature=2.0)
    show("temperature=0.0 (greedy)", confident, temperature=0.0)
    print()
    confident_k = show("top_k=3", confident, top_k=3)
    show("top_k=1 (identical to greedy)", confident, top_k=1)
    confident_p = show("top_p=0.90", confident, top_p=0.90)
    show("top_p=0.50", confident, top_p=0.50)
    show("top_p=0.01 (still keeps one)", confident, top_p=0.01)

    print()
    print("A TORN distribution (no clear winner), same settings")
    print(header)
    torn_k = show("top_k=3", torn, top_k=3)
    torn_p = show("top_p=0.90", torn, top_p=0.90)
    show("top_p=0.50", torn, top_p=0.50)

    print()
    print("  This is the difference between the two knobs, in one comparison:")
    print(f"    top_k=3    kept {confident_k} tokens when confident, "
          f"{torn_k} when torn.  Always {torn_k}.")
    print(f"    top_p=0.90 kept {confident_p} tokens when confident, "
          f"{torn_p} when torn.  It follows the shape.")
    print("  top-k commits to a candidate count in advance. top-p commits to an")
    print("  amount of probability mass and lets the count fall out of the")
    print("  distribution, which is why it is the more common default.")

    print()
    print("Temperature and top-p compound (confident distribution)")
    print(header)
    at_one = show("temperature=1.0 + top_p=0.90", confident, top_p=0.90)
    at_two = show("temperature=2.0 + top_p=0.90", confident, temperature=2.0, top_p=0.90)
    print()
    print(f"  The same p=0.90 admits {at_one} tokens at temperature 1.0 and "
          f"{at_two} at 2.0,")
    print("  because flattening the distribution pushed mass out into the tail.")
    print("  top-k would be unmoved: temperature changes the spacing between")
    print("  logits but never their order, and top-k only reads the order.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="a .safetensors checkpoint; omit to run the no-model knob demo",
    )
    parser.add_argument("--vocab", default="vocab.json", help="tokenizer used for training")
    parser.add_argument("--prompt", default="ROMEO:", help="text to continue")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="0 is greedy; <1 sharpens; >1 flattens",
    )
    parser.add_argument(
        "--top-k", type=int, default=40, help="keep only the k likeliest tokens; 0 disables"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="keep the smallest token set covering p of the mass; 1.0 disables",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--num-samples", type=int, default=1, help="generate this many continuations"
    )
    args = parser.parse_args()

    if args.checkpoint is None:
        demo_knobs()
        return

    # Seeding makes a generation reproducible, which matters because "the model
    # produced this" is only a meaningful claim if the run can be repeated.
    mx.random.seed(args.seed)

    # Imported here rather than at the top of the file, and the reason is worth
    # stating: the library half of this module (generate, and the three knobs)
    # has no dependency on training at all, which is what lets it be tested
    # against a model that was never trained. Only the CLI needs to read a
    # checkpoint off disk. Keeping the import local says that out loud, and it
    # also breaks what would otherwise be an import cycle, since train.py calls
    # generate_text() for its mid-run previews.
    from train import load_checkpoint

    checkpoint = Path(args.checkpoint)
    model, metadata = load_checkpoint(checkpoint)
    tokenizer = BPETokenizer.load(args.vocab)

    # A checkpoint knows the vocab size it was trained with; the tokenizer knows
    # the vocab size it produces. If they disagree, every token id means
    # something other than what it meant during training, and the output is
    # gibberish that looks exactly like an undertrained model. Catch it here.
    if tokenizer.vocab_size != model.cfg.vocab_size:
        raise ValueError(
            f"tokenizer has vocab_size={tokenizer.vocab_size} but the "
            f"checkpoint was trained with vocab_size={model.cfg.vocab_size}; "
            "these are not the same vocabulary"
        )

    print()
    print(
        describe_settings(
            checkpoint=checkpoint,
            metadata=metadata,
            model=model,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
    )
    print(f"  seed          {args.seed}")

    for index in range(args.num_samples):
        print()
        print(f"--- sample {index + 1} of {args.num_samples} " + "-" * 40)
        started = time.perf_counter()
        generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            stream=True,
        )
        elapsed = time.perf_counter() - started
        print(
            f"--- {args.max_tokens} tokens in {elapsed:.2f}s "
            f"({args.max_tokens / elapsed:.1f} tok/s, no KV cache)"
        )


if __name__ == "__main__":
    main()
