"""Turn a text source into packed token shards ready for training.

This is the last stage that knows anything about text. Everything after it
(model.py, train.py) sees only integer arrays.

Usage:

    python -m data.prepare \
        --input data/raw/tinyshakespeare.txt \
        --vocab vocab.json \
        --out-dir data/tokens
"""

import argparse
import json
from pathlib import Path

import numpy as np

from adapters.plain_text import PlainTextAdapter
from tokenizer.tokenizer import BPETokenizer

# Chunks are rejoined with a blank line before encoding. This is what stands in
# for a special end-of-document token in this project: the boundary between two
# paragraphs is present in the token stream as ordinary text, so the model
# learns "a blank line ends a passage" from data rather than being handed a
# reserved token id for it. One less concept, and one less id that means
# something only by convention.
CHUNK_SEPARATOR = "\n\n"

# Token ids are stored as uint16, which halves shard size versus int32 and is
# safe for any vocabulary up to 65,535. Asserted below rather than assumed,
# because the failure mode is silent wraparound into wrong-but-valid ids.
TOKEN_DTYPE = np.uint16

META_FILENAME = "meta.json"


def prepare(
    input_path: str,
    tokenizer: BPETokenizer,
    out_dir: str,
    val_fraction: float = 0.1,
    context_len: int = 256,
) -> dict:
    """Encode a text source into train/val token shards on disk.

    Args:
        input_path: A .txt file or directory, passed to the adapter.
        tokenizer: A loaded tokenizer. Its fingerprint is recorded in the
            shard metadata so training and sampling can detect being pointed
            at shards built by a different vocabulary.
        out_dir: Directory to write train.npy, val.npy and meta.json into.
        val_fraction: Fraction of tokens held out for validation.
        context_len: Only used to report how many training windows exist. It
            does not affect what is written, since windows are sliced at batch
            time rather than baked into the file.

    Returns:
        The metadata dict that was written to meta.json.
    """
    if tokenizer.vocab_size > np.iinfo(TOKEN_DTYPE).max:
        raise ValueError(
            f"vocab_size={tokenizer.vocab_size} does not fit in "
            f"{TOKEN_DTYPE.__name__}; token ids would silently wrap around"
        )
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction={val_fraction} must be in (0, 1)")

    chunks = list(PlainTextAdapter().read(input_path))
    if not chunks:
        raise ValueError(f"no text found under {input_path}")

    # Joining everything into one string before encoding is simple and
    # obviously correct at this corpus size. Encoding chunk by chunk would give
    # an identical result here (chunks are stripped, so they never end mid-word,
    # and the separator is pure whitespace, so pre-tokenization boundaries line
    # up either way) and would stream better on a corpus too large to hold in
    # memory. Not needed at ~1MB.
    text = CHUNK_SEPARATOR.join(chunks)
    corpus_bytes = len(text.encode("utf-8"))

    print(f"[prepare] encoding {corpus_bytes:,} bytes with vocab_size={tokenizer.vocab_size}")
    ids = tokenizer.encode(text)
    tokens = np.array(ids, dtype=TOKEN_DTYPE)

    print(
        f"[prepare] {corpus_bytes:,} bytes -> {len(tokens):,} tokens "
        f"({corpus_bytes / len(tokens):.2f} bytes/token)"
    )

    # The split is contiguous (the tail becomes validation), not shuffled.
    # Shuffling would leak: training windows overlap by design, so a randomly
    # chosen "held out" token almost certainly appears inside some training
    # window, and validation loss would then measure memorisation rather than
    # generalisation.
    n_val = int(len(tokens) * val_fraction)
    if n_val < context_len + 1:
        raise ValueError(
            f"val split would be {n_val} tokens, too small to form even one "
            f"window of context_len={context_len}; use a larger corpus or a "
            "larger --val-fraction"
        )
    n_train = len(tokens) - n_val
    train, val = tokens[:n_train], tokens[n_train:]

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    meta = {
        # Provenance. train.py and sample.py check these so that shards built
        # by an older vocabulary are detected instead of quietly producing
        # nonsense, which is otherwise indistinguishable from a small model
        # working as expected.
        "tokenizer_fingerprint": tokenizer.fingerprint,
        "vocab_size": tokenizer.vocab_size,
        "source": str(input_path),
        "dtype": TOKEN_DTYPE.__name__,
        "total_tokens": len(tokens),
        "train_tokens": len(train),
        "val_tokens": len(val),
        "val_fraction": val_fraction,
        "chunks": len(chunks),
        "corpus_bytes": corpus_bytes,
    }

    np.save(out / "train.npy", train)
    np.save(out / "val.npy", val)
    (out / META_FILENAME).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Number of distinct starting offsets that yield a full input window plus
    # its shifted target: offset i needs tokens[i : i + context_len + 1], so
    # the last usable offset is len - context_len - 1.
    train_windows = max(0, len(train) - context_len)
    val_windows = max(0, len(val) - context_len)

    print(
        f"[prepare] train {len(train):,} tokens, val {len(val):,} tokens "
        f"({val_fraction:.0%} held out, contiguous tail)"
    )
    print(
        f"[prepare] {train_windows:,} training windows and {val_windows:,} "
        f"validation windows at context_len={context_len}"
    )
    print(
        f"[prepare] wrote {out}/train.npy, {out}/val.npy, {out}/{META_FILENAME} "
        f"(tokenizer {tokenizer.fingerprint})"
    )
    return meta


def load_tokens(tokens_dir: str | Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load prepared shards back as (train, val, meta).

    Lives here rather than in train.py so that the shard layout is defined in
    exactly one place: whatever writes these files also decides how they are
    read.
    """
    path = Path(tokens_dir)
    meta = json.loads((path / META_FILENAME).read_text(encoding="utf-8"))
    train = np.load(path / "train.npy")
    val = np.load(path / "val.npy")
    print(
        f"[prepare] loaded {len(train):,} train / {len(val):,} val tokens "
        f"from {path} (tokenizer {meta['tokenizer_fingerprint']})"
    )
    return train, val, meta


def verify_tokenizer_matches(meta: dict, tokenizer: BPETokenizer) -> None:
    """Raise if `tokenizer` is not the one these shards were built with.

    This is the check that makes the fingerprint worth recording. Training or
    sampling with a mismatched vocabulary does not crash: token id 1841 simply
    means a different string than it did, so the model reads and writes
    gibberish while every array shape stays valid. Since gibberish is also what
    a 5.9M-parameter model produces when it is working correctly, there would
    be no symptom to notice. Failing loudly here is the only defence.
    """
    if meta["tokenizer_fingerprint"] != tokenizer.fingerprint:
        raise ValueError(
            "tokenizer mismatch: these shards were built with tokenizer "
            f"{meta['tokenizer_fingerprint']}, but the loaded tokenizer is "
            f"{tokenizer.fingerprint}. Re-run data.prepare with this "
            "vocabulary, or load the vocabulary the shards were built with."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="a .txt file or a directory")
    parser.add_argument("--vocab", default="vocab.json", help="trained vocabulary")
    parser.add_argument("--out-dir", default="data/tokens", help="where to write shards")
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="fraction of tokens held out for validation (contiguous tail)",
    )
    parser.add_argument(
        "--context-len",
        type=int,
        default=256,
        help="only used to report the number of available training windows",
    )
    args = parser.parse_args()

    tokenizer = BPETokenizer.load(args.vocab)
    prepare(
        input_path=args.input,
        tokenizer=tokenizer,
        out_dir=args.out_dir,
        val_fraction=args.val_fraction,
        context_len=args.context_len,
    )

    # Decode a window back to text and print it. Per CLAUDE.md, this stage is
    # where an off-by-one or a dtype mistake becomes invisible, so it is worth
    # actually looking at what the model will be fed.
    train, _, meta = load_tokens(args.out_dir)
    verify_tokenizer_matches(meta, tokenizer)
    preview = tokenizer.decode(train[:48].tolist())
    print()
    print("[prepare] first 48 training tokens decode to:")
    print(f"[prepare]   {preview!r}")


if __name__ == "__main__":
    main()
