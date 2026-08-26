"""Train a byte-level BPE vocabulary from whatever an adapter yields.

Why train a tokenizer instead of importing a pretrained one (tiktoken, GPT-2's
vocab): doing it yourself is a large part of what this project is for, and a
tiny model genuinely wants a tiny vocabulary. At 4096 tokens the embedding
table is ~18% of the `tiny` preset; GPT-2's 50257-token vocabulary at the same
d_model would be ~73%, leaving the actual transformer as a minority of its own
parameter count. See DESIGN.md section 3.2.

The algorithm, in full:

  1. Pre-tokenize the corpus into words, and count how often each occurs.
     Everything after this works on ~15k distinct words with multiplicities
     rather than ~1.1M individual bytes, which is both faster and how BPE is
     actually specified.
  2. Represent each word as a list of raw byte ids.
  3. Count every adjacent pair of ids across all words, weighted by word
     frequency.
  4. Fuse the most frequent pair into a new token id.
  5. Repeat from 3 until the vocabulary is full.

Usage:

    python -m tokenizer.train_tokenizer \
        --input data/raw/tinyshakespeare.txt \
        --vocab-size 4096 \
        --out vocab.json
"""

import argparse
import collections
import time
from pathlib import Path

from adapters.plain_text import PlainTextAdapter
from tokenizer.tokenizer import (
    BYTE_TOKENS,
    PRETOKENIZE,
    BPETokenizer,
    merge_pair,
)


def train_bpe(
    texts: list[str],
    vocab_size: int,
    log_every: int = 250,
) -> list[tuple[int, int]]:
    """Learn a merge list from `texts`.

    Args:
        texts: Raw text examples, e.g. whatever an adapter yielded.
        vocab_size: Target vocabulary size, including the 256 byte tokens.
        log_every: Print a progress line every this many merges.

    Returns:
        The learned merges, in the order learned. That order is the tokenizer's
        merge priority, so it must be preserved.
    """
    if vocab_size <= BYTE_TOKENS:
        raise ValueError(
            f"vocab_size={vocab_size} must exceed {BYTE_TOKENS}, since the "
            "raw byte tokens alone occupy that many ids"
        )

    target_merges = vocab_size - BYTE_TOKENS

    # Step 1: pre-tokenize into words and count them. Collapsing to unique
    # words is the single most important thing that makes this tractable in
    # pure Python: the corpus below has ~298k word occurrences but only ~15k
    # distinct words, so every subsequent pass does ~20x less work.
    word_freqs: collections.Counter[str] = collections.Counter()
    for text in texts:
        word_freqs.update(PRETOKENIZE.findall(text))

    total_words = sum(word_freqs.values())
    total_bytes = sum(len(w.encode("utf-8")) * n for w, n in word_freqs.items())
    print(
        f"[bpe] corpus: {total_bytes:,} bytes, {total_words:,} word "
        f"occurrences, {len(word_freqs):,} distinct words"
    )
    print(f"[bpe] learning {target_merges:,} merges -> vocab_size={vocab_size}")

    # Step 2: each distinct word becomes a list of byte ids that gets fused
    # in place as merges are learned.
    splits: dict[str, list[int]] = {
        word: list(word.encode("utf-8")) for word in word_freqs
    }

    # id -> bytes, needed only so progress logging can show what a merge
    # actually spells. The tokenizer rebuilds this itself from the merge list.
    token_bytes: dict[int, bytes] = {i: bytes([i]) for i in range(BYTE_TOKENS)}

    merges: list[tuple[int, int]] = []
    started = time.perf_counter()

    while len(merges) < target_merges:
        # Step 3: count adjacent pairs across all words, weighted by how often
        # each word occurs in the corpus.
        pair_counts: collections.Counter[tuple[int, int]] = collections.Counter()
        for word, freq in word_freqs.items():
            ids = splits[word]
            for pair in zip(ids, ids[1:]):
                pair_counts[pair] += freq

        if not pair_counts:
            # Every word has been fused down to a single token. The corpus
            # simply does not contain enough distinct structure to fill the
            # requested vocabulary, which is normal on tiny inputs.
            print(
                f"[bpe] no pairs left after {len(merges):,} merges; "
                f"stopping early at vocab_size={BYTE_TOKENS + len(merges)}"
            )
            break

        # Step 4: pick the most frequent pair. Ties are broken by the pair
        # itself so that two runs on the same corpus always produce the same
        # vocabulary; leaving it to dict ordering would work today but is not
        # something to rely on.
        best_pair, best_count = min(
            pair_counts.items(), key=lambda item: (-item[1], item[0])
        )

        new_id = BYTE_TOKENS + len(merges)
        token_bytes[new_id] = token_bytes[best_pair[0]] + token_bytes[best_pair[1]]
        merges.append(best_pair)

        # Apply the merge everywhere. Only words that actually contain the
        # pair change, but checking costs about as much as doing it, so this
        # just rewrites all of them.
        for word in splits:
            splits[word] = merge_pair(splits[word], best_pair, new_id)

        if len(merges) % log_every == 0 or len(merges) == 1:
            spelled = token_bytes[new_id].decode("utf-8", errors="replace")
            elapsed = time.perf_counter() - started
            rate = len(merges) / elapsed
            eta = (target_merges - len(merges)) / rate
            print(
                f"[bpe] merge {len(merges):>5}/{target_merges}  "
                f"count={best_count:>7,}  -> {spelled!r}  "
                f"({rate:.0f} merges/s, eta {eta:.0f}s)"
            )

    elapsed = time.perf_counter() - started
    print(f"[bpe] learned {len(merges):,} merges in {elapsed:.1f}s")
    return merges


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        help="a .txt file or a directory of them",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=4096,
        help="target vocabulary size including the 256 byte tokens",
    )
    parser.add_argument(
        "--out",
        default="vocab.json",
        help="where to write the trained vocabulary",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=250,
        help="print a progress line every this many merges",
    )
    args = parser.parse_args()

    texts = list(PlainTextAdapter().read(args.input))

    merges = train_bpe(texts, args.vocab_size, log_every=args.log_every)
    tokenizer = BPETokenizer(merges)
    tokenizer.save(args.out)

    # A round trip on real corpus text, printed rather than asserted. Per
    # CLAUDE.md, a silent tokenizer bug poisons every stage downstream with no
    # obvious symptom, so this is the cheapest possible place to catch one, and
    # it costs nothing to look at.
    print()
    sample = texts[0][:120]
    ids = tokenizer.encode(sample)
    restored = tokenizer.decode(ids)
    print(f"[bpe] round trip: {'ok' if restored == sample else 'FAILED'}")
    print(f"[bpe]   text  {sample!r}")
    print(f"[bpe]   ids   {ids[:24]}{' ...' if len(ids) > 24 else ''}")
    print(f"[bpe]   back  {restored!r}")

    # Compression ratio is the headline number for a tokenizer: how many raw
    # bytes of corpus each token carries. Higher means each training window
    # covers more text, so it directly affects how much the model sees.
    corpus_bytes = sum(len(t.encode("utf-8")) for t in texts)
    corpus_tokens = sum(len(tokenizer.encode(t)) for t in texts)
    print()
    print(
        f"[bpe] compression: {corpus_bytes:,} bytes -> {corpus_tokens:,} "
        f"tokens ({corpus_bytes / corpus_tokens:.2f} bytes/token)"
    )


if __name__ == "__main__":
    main()
