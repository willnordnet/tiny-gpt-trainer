"""Plumbing tests for the byte-level BPE tokenizer (DESIGN.md section 6.1).

A silently broken round trip is the worst bug class in this project: it does
not raise, it poisons every downstream stage, and the symptom (a model emitting
nonsense) is indistinguishable from a small model behaving exactly as expected.
So the round trip gets tested hard, including on input the tokenizer was never
trained on.
"""

import json
from pathlib import Path

import pytest

from tinygpt.tokenizer.tokenizer import BYTE_TOKENS, BPETokenizer, merge_pair
from tinygpt.tokenizer.train_tokenizer import train_bpe

# Enough repetition that BPE has real pairs to learn, small enough to train
# instantly. Deliberately English/ASCII so that the unicode round-trip cases
# below are genuinely testing unseen bytes.
TRAINING_TEXT = (
    "the cat sat on the mat. the cat sat on the hat. "
    "the rat sat on the mat. the bat sat on the cat. "
    "she sells sea shells by the sea shore."
)


@pytest.fixture(scope="module")
def tokenizer() -> BPETokenizer:
    """A small tokenizer trained once and shared across tests."""
    return BPETokenizer(train_bpe([TRAINING_TEXT], vocab_size=320, log_every=10_000))


@pytest.mark.parametrize(
    "description, text",
    [
        ("text straight from the training corpus", "the cat sat on the mat."),
        ("a single trained word", "the"),
        ("empty string", ""),
        ("one space", " "),
        ("a run of spaces", "a     b"),
        ("newlines and tabs", "line one\n\nline\ttwo\n"),
        ("only whitespace", "  \n\t  \n"),
        # Not in the training text, so these exercise the fallback to raw byte
        # tokens. This is the case a vocab with an UNK token would mangle.
        ("unseen ascii punctuation", "!@#$%^&*()[]{}<>|\\/~`"),
        ("unseen accented letters", "wörds mit ümlauts"),
        ("unseen CJK", "日本語のテキスト"),
        ("unseen emoji (4-byte utf-8)", "a 🎲 b 🧊"),
        ("mixed scripts and digits", "Ünïcödé 123 日本 ✓ done"),
        ("digits, which the regex splits separately", "in 1597 and 2026"),
        ("contractions, a regex special case", "don't he'll we've I'm it's"),
        ("leading and trailing whitespace preserved", "   padded   "),
    ],
)
def test_decode_encode_round_trip(tokenizer: BPETokenizer, description: str, text: str):
    """decode(encode(text)) must return exactly the input, byte for byte.

    Byte-level BPE has no UNK token, so this must hold for *any* input, not
    just input resembling the training corpus.
    """
    assert tokenizer.decode(tokenizer.encode(text)) == text, description


@pytest.mark.parametrize(
    "text",
    ["the cat sat on the mat.", "unseen 日本語 🎲", "", "   \n\t"],
)
def test_all_ids_are_in_range(tokenizer: BPETokenizer, text: str):
    """Every emitted id must be a valid index into the embedding table.

    An out-of-range id does not fail here, it fails much later as an opaque
    indexing error inside the model, far from the code that produced it.
    """
    for token_id in tokenizer.encode(text):
        assert 0 <= token_id < tokenizer.vocab_size


def test_vocab_size_is_byte_tokens_plus_merges(tokenizer: BPETokenizer):
    assert tokenizer.vocab_size == BYTE_TOKENS + len(tokenizer.merges)
    assert len(tokenizer.vocab) == tokenizer.vocab_size


def test_training_is_deterministic():
    """The same corpus must always produce the same vocabulary.

    Tie-breaking between equally frequent pairs is explicit in train_bpe rather
    than left to dict iteration order, precisely so this holds. Without it, two
    runs could produce different vocabularies and a checkpoint would silently
    stop matching its own tokenizer.
    """
    first = train_bpe([TRAINING_TEXT], vocab_size=320, log_every=10_000)
    second = train_bpe([TRAINING_TEXT], vocab_size=320, log_every=10_000)

    assert first == second


def test_encoding_survives_save_and_load(tokenizer: BPETokenizer, tmp_path: Path):
    """A loaded tokenizer must encode identically to the one that was saved."""
    path = tmp_path / "vocab.json"
    tokenizer.save(path)

    reloaded = BPETokenizer.load(path)

    text = "the cat sat on unseen 日本 ground"
    assert reloaded.encode(text) == tokenizer.encode(text)
    assert reloaded.vocab_size == tokenizer.vocab_size
    assert reloaded.fingerprint == tokenizer.fingerprint


def test_fingerprint_changes_when_merges_change(tokenizer: BPETokenizer):
    """Different vocabularies must not share a fingerprint.

    The fingerprint's whole job is to make a vocab/shard/checkpoint mismatch
    detectable, which requires it to actually differ.
    """
    smaller = BPETokenizer(tokenizer.merges[:-5])

    assert smaller.fingerprint != tokenizer.fingerprint


def test_load_rejects_a_vocab_whose_size_does_not_match_its_merges(tmp_path: Path):
    """A hand-edited or truncated vocab.json must fail loudly at load time."""
    path = tmp_path / "corrupt.json"
    path.write_text(
        json.dumps(
            {
                "pretokenizer": "gpt2",
                "vocab_size": 9999,  # inconsistent with the merge list below
                "merges": [[104, 101]],
                "fingerprint": "irrelevant",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="reconstructs to"):
        BPETokenizer.load(path)


def test_load_rejects_a_vocab_from_a_different_pretokenizer(tmp_path: Path):
    """Changing the pre-tokenization regex must invalidate old vocabularies.

    The same text encodes differently under a different pre-tokenizer, so
    silently accepting the old file would produce a wrong token stream with no
    error anywhere.
    """
    path = tmp_path / "old.json"
    path.write_text(
        json.dumps(
            {
                "pretokenizer": "some-older-scheme",
                "vocab_size": 257,
                "merges": [[104, 101]],
                "fingerprint": "irrelevant",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="pretokenizer"):
        BPETokenizer.load(path)


def test_vocab_size_below_the_byte_tokens_is_rejected():
    """Asking for fewer tokens than there are byte values is meaningless."""
    with pytest.raises(ValueError, match="must exceed"):
        train_bpe([TRAINING_TEXT], vocab_size=BYTE_TOKENS)


def test_training_stops_early_when_the_corpus_runs_out_of_pairs():
    """A corpus too small to fill the vocabulary must stop, not loop forever."""
    merges = train_bpe(["ab ab ab"], vocab_size=4096, log_every=10_000)

    tokenizer = BPETokenizer(merges)
    assert tokenizer.vocab_size < 4096
    assert tokenizer.decode(tokenizer.encode("ab ab ab")) == "ab ab ab"


@pytest.mark.parametrize(
    "description, ids, pair, expected",
    [
        ("single occurrence", [1, 2, 3], (1, 2), [99, 3]),
        ("several occurrences", [1, 2, 1, 2], (1, 2), [99, 99]),
        ("no occurrence leaves input alone", [1, 3, 2], (1, 2), [1, 3, 2]),
        # "aaa" with pair (a, a) must give [aa, a], not [aa, aa]: consuming two
        # positions per match is what makes occurrences non-overlapping.
        ("overlapping matches consume left to right", [1, 1, 1], (1, 1), [99, 1]),
        ("pair at the very end", [3, 1, 2], (1, 2), [3, 99]),
        ("too short to contain a pair", [1], (1, 2), [1]),
        ("empty input", [], (1, 2), []),
    ],
)
def test_merge_pair(description, ids, pair, expected):
    """merge_pair is shared by training and encoding, so it must agree with
    itself: a difference here would mean text encodes differently than the
    vocabulary was built for."""
    assert merge_pair(ids, pair, 99) == expected, description
