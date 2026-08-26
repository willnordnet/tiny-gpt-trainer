"""Plumbing tests for the plain-text adapter (DESIGN.md section 6.1).

Why the adapter is worth real tests even though it is barely 60 lines: silent
chunking bugs (dropping the last paragraph, swallowing a chunk, reading files
in a different order each run) produce a subtly wrong corpus rather than an
error. Nothing downstream would fail, the model would just train on slightly
the wrong data, and you would have no symptom to chase.
"""

from pathlib import Path

import pytest

from adapters.base import Adapter
from adapters.plain_text import PlainTextAdapter


def write(tmp_path: Path, name: str, text: str) -> Path:
    """Write `text` to a file under tmp_path and return its path."""
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "description, text, expected",
    [
        (
            "two paragraphs split on a blank line",
            "alpha\n\nbeta",
            ["alpha", "beta"],
        ),
        (
            # A file that does not end in a newline must still yield its last
            # paragraph. This is the classic off-by-one in hand-rolled readers.
            "no trailing newline keeps the final chunk",
            "alpha\n\nbeta without trailing newline",
            ["alpha", "beta without trailing newline"],
        ),
        (
            "trailing newlines produce no empty chunk",
            "alpha\n\nbeta\n\n\n",
            ["alpha", "beta"],
        ),
        (
            # Real text files contain lines of spaces and tabs that look blank
            # but are not, so they must count as separators, and the resulting
            # empty chunk must be dropped rather than yielded as "".
            "whitespace-only line is a separator, not a chunk",
            "alpha\n \t \nbeta",
            ["alpha", "beta"],
        ),
        (
            "several blank lines collapse to one boundary",
            "alpha\n\n\n\nbeta",
            ["alpha", "beta"],
        ),
        (
            "a file with no blank line is a single chunk",
            "just\none\nparagraph",
            ["just\none\nparagraph"],
        ),
        (
            "internal newlines within a paragraph are preserved",
            "ROMEO:\nBut soft\n\nJULIET:\nAy me",
            ["ROMEO:\nBut soft", "JULIET:\nAy me"],
        ),
        (
            "an empty file yields nothing at all",
            "",
            [],
        ),
        (
            "a whitespace-only file yields nothing at all",
            "\n\n   \n\t\n\n",
            [],
        ),
    ],
)
def test_chunking(tmp_path: Path, description: str, text: str, expected: list[str]):
    path = write(tmp_path, "corpus.txt", text)

    chunks = list(PlainTextAdapter().read(str(path)))

    assert chunks == expected, description


def test_directory_reads_all_txt_files_in_sorted_order(tmp_path: Path):
    """A directory source must cover every .txt file, in a stable order.

    Sorted order matters beyond tidiness: BPE merges depend on the order text
    is seen, so filesystem-order iteration would make two runs over the same
    directory produce different vocabularies.
    """
    write(tmp_path, "c_third.txt", "gamma")
    write(tmp_path, "a_first.txt", "alpha")
    write(tmp_path, "b_second.txt", "beta")
    write(tmp_path, "ignored.md", "should not be read")

    chunks = list(PlainTextAdapter().read(str(tmp_path)))

    assert chunks == ["alpha", "beta", "gamma"]


def test_min_chars_filters_short_chunks(tmp_path: Path):
    """min_chars drops chunks below a length, measured after stripping."""
    path = write(tmp_path, "corpus.txt", "x\n\na longer paragraph\n\nyz")

    kept = list(PlainTextAdapter(min_chars=5).read(str(path)))

    assert kept == ["a longer paragraph"]


def test_missing_source_raises(tmp_path: Path):
    """A typo'd path should fail loudly and immediately, not yield nothing.

    Silently yielding zero chunks would look identical to an empty corpus and
    would surface much later as a confusing tokenizer failure.
    """
    stream = PlainTextAdapter().read(str(tmp_path / "does_not_exist.txt"))

    with pytest.raises(FileNotFoundError):
        next(stream)


def test_satisfies_the_adapter_protocol():
    """The concrete adapter must match the interface everything else expects."""
    assert isinstance(PlainTextAdapter(), Adapter)
