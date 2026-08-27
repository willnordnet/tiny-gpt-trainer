"""Plumbing tests for the data-prep stage (DESIGN.md section 6.1).

This stage is where a mistake becomes hardest to trace. An out-of-range token
id or a wrong dtype does not fail here, it fails inside the model as an opaque
indexing error, a long way from the code that caused it. And a leaky train/val
split does not fail at all: it just makes validation loss meaningless while
looking perfectly healthy.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from tinygpt.data.prepare import (
    CHUNK_SEPARATOR,
    TOKEN_DTYPE,
    load_tokens,
    prepare,
    verify_tokenizer_matches,
)
from tinygpt.tokenizer.tokenizer import BPETokenizer
from tinygpt.tokenizer.train_tokenizer import train_bpe

# Long enough to survive a 10% split with room for several windows, repetitive
# enough that BPE learns real merges.
CORPUS = "\n\n".join(
    [
        "the cat sat on the mat and the cat sat still",
        "the rat sat on the mat and the rat sat still",
        "the bat sat on the hat and the bat sat still",
        "she sells sea shells by the sea shore today",
        "he tells tall tales by the tall trees today",
    ]
    * 6
)

CONTEXT_LEN = 4


@pytest.fixture(scope="module")
def tokenizer() -> BPETokenizer:
    return BPETokenizer(train_bpe([CORPUS], vocab_size=400, log_every=10_000))


@pytest.fixture
def prepared(tmp_path: Path, tokenizer: BPETokenizer) -> tuple[Path, dict]:
    """Run the real prepare() against a temp corpus and return (dir, meta)."""
    source = tmp_path / "corpus.txt"
    source.write_text(CORPUS, encoding="utf-8")
    out_dir = tmp_path / "tokens"

    meta = prepare(
        input_path=str(source),
        tokenizer=tokenizer,
        out_dir=str(out_dir),
        val_fraction=0.1,
        context_len=CONTEXT_LEN,
    )
    return out_dir, meta


def test_shards_have_the_expected_dtype(prepared):
    """uint16 halves shard size, and only holds while the vocab fits in it."""
    out_dir, _ = prepared
    train, val, _ = load_tokens(out_dir)

    assert train.dtype == TOKEN_DTYPE
    assert val.dtype == TOKEN_DTYPE
    assert train.ndim == 1 and val.ndim == 1


def test_no_token_id_is_out_of_range(prepared, tokenizer: BPETokenizer):
    """Every id must be a valid index into the embedding table.

    Caught here it is one line; caught in the model it is an opaque MLX
    indexing failure with no hint about which stage produced the bad id.
    """
    out_dir, _ = prepared
    train, val, _ = load_tokens(out_dir)

    for name, shard in [("train", train), ("val", val)]:
        assert shard.min() >= 0, name
        assert shard.max() < tokenizer.vocab_size, name


def test_splits_account_for_every_token(prepared):
    """No token may be dropped or duplicated by the split."""
    out_dir, meta = prepared
    train, val, _ = load_tokens(out_dir)

    assert len(train) + len(val) == meta["total_tokens"]
    assert len(train) == meta["train_tokens"]
    assert len(val) == meta["val_tokens"]
    assert len(val) > 0


def test_split_is_a_contiguous_prefix_and_tail(prepared, tokenizer: BPETokenizer):
    """train must be the head of the corpus and val the tail, in order.

    A shuffled split would leak: training windows overlap, so a randomly
    held-out token is almost certainly inside some training window, and
    validation loss would then measure memorisation rather than generalisation.
    """
    out_dir, _ = prepared
    train, val, _ = load_tokens(out_dir)

    full = np.array(tokenizer.encode(CORPUS), dtype=TOKEN_DTYPE)

    assert np.array_equal(train, full[: len(train)])
    assert np.array_equal(val, full[len(train) :])


def test_tokens_decode_back_to_the_joined_corpus(prepared, tokenizer: BPETokenizer):
    """The shards must represent exactly the text that went in.

    Chunks are rejoined with a blank line rather than separated by a special
    token, so the round trip is over the joined form, and the separator has to
    survive it.
    """
    out_dir, _ = prepared
    train, val, _ = load_tokens(out_dir)

    restored = tokenizer.decode(np.concatenate([train, val]).tolist())

    assert restored == CHUNK_SEPARATOR.join(CORPUS.split("\n\n"))
    assert CHUNK_SEPARATOR in restored


def test_metadata_records_the_tokenizer_that_built_the_shards(
    prepared, tokenizer: BPETokenizer
):
    out_dir, meta = prepared
    on_disk = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))

    assert on_disk == meta
    assert meta["tokenizer_fingerprint"] == tokenizer.fingerprint
    assert meta["vocab_size"] == tokenizer.vocab_size
    assert meta["dtype"] == TOKEN_DTYPE.__name__


def test_verify_tokenizer_accepts_the_matching_tokenizer(prepared, tokenizer):
    _, meta = prepared

    verify_tokenizer_matches(meta, tokenizer)  # must not raise


def test_verify_tokenizer_rejects_a_different_tokenizer(prepared, tokenizer):
    """The check that makes the fingerprint worth recording.

    A mismatched vocabulary does not crash anything: ids stay in range and
    every shape stays valid, the model simply reads and writes gibberish. Since
    gibberish is also what a small model produces when working correctly, there
    is no symptom to notice, so this must fail loudly.
    """
    _, meta = prepared
    different = BPETokenizer(tokenizer.merges[:-10])

    with pytest.raises(ValueError, match="tokenizer mismatch"):
        verify_tokenizer_matches(meta, different)


@pytest.mark.parametrize("val_fraction", [0.0, 1.0, -0.1, 1.5])
def test_invalid_val_fraction_is_rejected(tmp_path, tokenizer, val_fraction):
    source = tmp_path / "corpus.txt"
    source.write_text(CORPUS, encoding="utf-8")

    with pytest.raises(ValueError, match="val_fraction"):
        prepare(str(source), tokenizer, str(tmp_path / "out"), val_fraction, CONTEXT_LEN)


def test_a_val_split_too_small_for_one_window_is_rejected(tmp_path, tokenizer):
    """Silently producing an unusable val split would break eval, not prepare."""
    source = tmp_path / "corpus.txt"
    source.write_text(CORPUS, encoding="utf-8")

    with pytest.raises(ValueError, match="too small to form even one window"):
        prepare(str(source), tokenizer, str(tmp_path / "out"), 0.1, context_len=100_000)


def test_an_empty_source_is_rejected(tmp_path, tokenizer):
    """Pointing prepare at the wrong path must fail, not write empty shards."""
    source = tmp_path / "empty.txt"
    source.write_text("   \n\n  \n", encoding="utf-8")

    with pytest.raises(ValueError, match="no text found"):
        prepare(str(source), tokenizer, str(tmp_path / "out"), 0.1, CONTEXT_LEN)


def test_a_vocab_too_large_for_the_shard_dtype_is_rejected(tmp_path):
    """uint16 storage is only valid while ids fit; wraparound would be silent.

    Uses a stand-in rather than a real 65k-token tokenizer because the check
    fires on vocab_size alone, before any encoding happens.
    """

    class OversizedTokenizer:
        vocab_size = 70_000
        fingerprint = "oversized"

        def encode(self, text: str) -> list[int]:  # pragma: no cover
            raise AssertionError("should be rejected before encoding")

    source = tmp_path / "corpus.txt"
    source.write_text(CORPUS, encoding="utf-8")

    with pytest.raises(ValueError, match="does not fit in"):
        prepare(str(source), OversizedTokenizer(), str(tmp_path / "out"), 0.1, CONTEXT_LEN)
