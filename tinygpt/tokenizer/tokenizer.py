"""Byte-level BPE tokenizer: load, encode, decode.

This file is the *runtime* side of the tokenizer. Training lives in
train_tokenizer.py, because training happens once and encoding happens
constantly, and separating them keeps this file short enough to read in one
sitting.

Why byte-level, starting from the 256 single-byte tokens: it means every
possible input is encodable. There is no UNK token, no "character not in
vocabulary" branch, and no way for a stray emoji or a Windows line ending in
the corpus to produce a hole in the token stream. The cost is that rare
characters take several tokens each, which is a good trade at this scale.

Run directly to train a tiny vocab on an inline string and see round trips:

    python -m tinygpt.tokenizer.tokenizer
"""

import hashlib
import json
import re
from pathlib import Path

# Pre-tokenization regex, from GPT-2. Text is split into "words" *before* BPE
# runs, and merges are only ever learned within a word, never across the
# boundary between two.
#
# Why that restriction matters: without it, BPE would happily learn " the cat"
# as a single token because that byte sequence is frequent. Vocabulary would
# then be spent on multi-word phrases that only help in the exact contexts they
# were seen in, instead of on sub-word pieces that compose. Pre-tokenizing is
# what keeps the learned vocabulary made of reusable parts.
#
# The alternatives in the pattern, in order: common English contractions
# ('s, 'd, 'm, 't, 'll, 've, 're); a run of letters with an optional leading
# space; a run of digits with an optional leading space; a run of punctuation
# with an optional leading space; trailing whitespace; any other whitespace.
# The leading " ?" is why tokens in a BPE vocab so often start with a space:
# " the" and "the" are genuinely different tokens, which is how the model
# learns word boundaries without a separate marker.
PRETOKENIZE = re.compile(
    r"'(?:[sdmt]|ll|ve|re)| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+"
)

# Identifies which pre-tokenizer produced a saved vocab. Stored in vocab.json
# so that changing the regex above invalidates old vocabularies visibly rather
# than silently producing different encodings for the same text.
PRETOKENIZER_NAME = "gpt2"

# The 256 raw byte values always occupy ids 0..255. Every learned merge is
# appended after them, so a merge's token id is simply 256 + its position in
# the merge list. That is why vocab.json only needs to store the merge list.
BYTE_TOKENS = 256


class BPETokenizer:
    """Encodes text to token ids and back, using a learned merge list.

    Construct from a merge list (as produced by train_tokenizer.train_bpe), or
    load a previously saved one with `BPETokenizer.load(path)`.
    """

    def __init__(self, merges: list[tuple[int, int]]) -> None:
        """
        Args:
            merges: Learned merges in the order they were learned. Order is
                load-bearing, not incidental: it defines merge *priority* at
                encode time. A pair learned earlier was more frequent, so it
                must be applied before a later one, otherwise encoding would
                not reproduce the segmentation the vocabulary was built for.
        """
        self.merges = [tuple(pair) for pair in merges]

        # pair -> its rank (= priority). Lower rank wins at encode time.
        self._rank = {pair: i for i, pair in enumerate(self.merges)}

        # id -> the actual bytes that id stands for. Built by replaying the
        # merges in order, which is why the saved file needs nothing else.
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(BYTE_TOKENS)}
        for i, (a, b) in enumerate(self.merges):
            self.vocab[BYTE_TOKENS + i] = self.vocab[a] + self.vocab[b]

        # Encoding the same word twice does the same work twice, and real text
        # repeats words heavily (TinyShakespeare: ~298k word occurrences but
        # only ~15k distinct words). Caching per word makes encoding roughly
        # 20x cheaper on that corpus for the cost of one dict.
        self._cache: dict[str, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        """Total number of distinct token ids, i.e. 256 + number of merges."""
        return BYTE_TOKENS + len(self.merges)

    @property
    def fingerprint(self) -> str:
        """Short stable hash of this tokenizer's identity.

        Exists to guard a failure mode that is otherwise invisible: retrain the
        vocabulary, then reuse token shards or a checkpoint produced by the
        *old* one, and nothing crashes. The model just emits nonsense, which is
        indistinguishable from a small model behaving normally. Recording this
        alongside shards and checkpoints turns that into a detectable mismatch.
        """
        payload = json.dumps(
            {"pretokenizer": PRETOKENIZER_NAME, "merges": self.merges},
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    def encode(self, text: str) -> list[int]:
        """Turn text into token ids.

        Pre-tokenizes into words, then greedily applies learned merges within
        each word until no further merge is possible.
        """
        ids: list[int] = []
        for word in PRETOKENIZE.findall(text):
            cached = self._cache.get(word)
            if cached is None:
                cached = self._encode_word(word)
                self._cache[word] = cached
            ids.extend(cached)
        return ids

    def _encode_word(self, word: str) -> list[int]:
        """Apply merges to a single pre-tokenized word, highest priority first.

        Starts from raw bytes, then repeatedly finds the adjacent pair with the
        lowest merge rank and fuses it. Note it is *not* left-to-right: the
        best-ranked pair anywhere in the word goes first, because merge order,
        not position, is what the vocabulary was built around.
        """
        ids = list(word.encode("utf-8"))

        while len(ids) >= 2:
            # Find the adjacent pair with the best (lowest) rank. Pairs that
            # were never learned are given infinite rank so they lose.
            best_pair = min(
                zip(ids, ids[1:]),
                key=lambda pair: self._rank.get(pair, float("inf")),
            )
            if best_pair not in self._rank:
                break  # nothing left that this vocabulary knows how to merge

            ids = merge_pair(ids, best_pair, self._rank[best_pair] + BYTE_TOKENS)

        return ids

    def decode(self, ids: list[int]) -> str:
        """Turn token ids back into text.

        errors="replace" rather than raising: a token sequence can legitimately
        end partway through a multi-byte UTF-8 character, which happens
        constantly during generation when the model is cut off at max_tokens.
        Raising there would mean sampling crashes on perfectly normal output.
        """
        raw = b"".join(self.vocab[i] for i in ids)
        return raw.decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        """Write the merge list and metadata to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pretokenizer": PRETOKENIZER_NAME,
            "vocab_size": self.vocab_size,
            "fingerprint": self.fingerprint,
            # Stored as [a, b] pairs in learned order. The token id each merge
            # produces is implicit (256 + index), so it is not stored: one less
            # thing that can be inconsistent with itself.
            "merges": [list(pair) for pair in self.merges],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(
            f"[tokenizer] saved {path} "
            f"(vocab_size={self.vocab_size}, fingerprint={self.fingerprint})"
        )

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        """Read a tokenizer back from a JSON file written by `save`."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))

        saved_pretokenizer = payload.get("pretokenizer")
        if saved_pretokenizer != PRETOKENIZER_NAME:
            raise ValueError(
                f"{path} was built with pretokenizer "
                f"{saved_pretokenizer!r}, but this code uses "
                f"{PRETOKENIZER_NAME!r}; the same text would encode "
                "differently, so this vocabulary must be retrained"
            )

        tokenizer = cls([tuple(pair) for pair in payload["merges"]])

        # Cross-check what was written against what we reconstructed. Cheap,
        # and catches a hand-edited or truncated vocab.json immediately rather
        # than as a mysterious out-of-range token id during training.
        if tokenizer.vocab_size != payload["vocab_size"]:
            raise ValueError(
                f"{path} claims vocab_size={payload['vocab_size']} but its "
                f"merge list reconstructs to {tokenizer.vocab_size}"
            )

        print(
            f"[tokenizer] loaded {path} "
            f"(vocab_size={tokenizer.vocab_size}, "
            f"fingerprint={tokenizer.fingerprint})"
        )
        return tokenizer


def merge_pair(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every non-overlapping occurrence of `pair` in `ids` with `new_id`.

    Shared by training and encoding so both are guaranteed to fuse pairs the
    same way. Scanning left to right and skipping two positions on a match
    means "aaa" with pair (a, a) becomes [aa, a], not [aa, aa]: occurrences
    cannot overlap.
    """
    out: list[int] = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


if __name__ == "__main__":
    # Imported here rather than at module scope to avoid a circular import:
    # train_tokenizer imports BPETokenizer from this file.
    from tinygpt.tokenizer.train_tokenizer import train_bpe

    SAMPLE = (
        "the cat sat on the mat. the cat sat on the hat. "
        "the rat sat on the mat. the bat sat on the cat."
    )

    print("training a 300-token vocab on one sentence pair:")
    merges = train_bpe([SAMPLE], vocab_size=300, log_every=10)
    tokenizer = BPETokenizer(merges)

    print()
    for text in [SAMPLE, "the cat", "unseen wörds ✓", ""]:
        ids = tokenizer.encode(text)
        restored = tokenizer.decode(ids)
        status = "ok " if restored == text else "FAIL"
        print(f"  [{status}] {len(ids):>3} ids  {text[:40]!r}")
        pieces = [tokenizer.vocab[i].decode("utf-8", errors="replace") for i in ids]
        print(f"          -> {pieces}")
