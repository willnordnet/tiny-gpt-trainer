"""Plain-text adapter: reads .txt file(s) and yields paragraph-ish chunks.

This is the only adapter in the project, deliberately (DESIGN.md section 7): a
general-purpose interface is easiest to get right once it has been proven
against one real, working case rather than several imagined ones.

Run directly to see the chunking behaviour on a small inline sample:

    python -m tinygpt.adapters.plain_text
"""

import re
from collections.abc import Iterator
from pathlib import Path

# Split on a run of one or more blank lines. The \s* in the middle means a
# "blank" line containing only spaces or tabs still counts as a separator,
# which matters because real text files are rarely perfectly clean.
#
# Why paragraphs rather than fixed-length character windows: a paragraph is a
# unit a human can read in a log and judge, and it does not slice through the
# middle of a word. The alternative, fixed windows, is easier to test but puts
# arbitrary boundaries into the token stream.
PARAGRAPH_SEPARATOR = re.compile(r"\n\s*\n+")


class PlainTextAdapter:
    """Reads UTF-8 .txt files and yields their paragraphs as strings."""

    def __init__(self, min_chars: int = 1) -> None:
        """
        Args:
            min_chars: Drop chunks shorter than this after stripping. The
                default of 1 only drops genuinely empty chunks. Raise it to
                filter out noise like stray single-character lines, but note
                that dropping content silently changes the corpus, so the
                chunk count is logged either way.
        """
        self.min_chars = min_chars

    def read(self, source_path: str) -> Iterator[str]:
        """Yield paragraph-sized text chunks from a .txt file or a directory.

        Accepts either a single file or a directory (non-recursive) of .txt
        files. Files are processed in sorted order, not filesystem order, so
        that two runs over the same directory produce the same token stream:
        the tokenizer's learned merges depend on the order it sees text, so a
        nondeterministic file order would make runs unreproducible for no
        reason.

        Note that the log lines below only appear as the caller consumes the
        iterator, since this is a generator. That is intentional (nothing is
        read into memory until asked for) but it does mean the summary line
        prints at the end, once the stream is exhausted.
        """
        paths = self._resolve_files(Path(source_path))
        print(f"[adapter] {len(paths)} file(s) under {source_path}")

        total_chunks = 0
        total_chars = 0

        for path in paths:
            text = path.read_text(encoding="utf-8")
            chunks = self._split_into_chunks(text)
            total_chunks += len(chunks)
            total_chars += sum(len(c) for c in chunks)
            print(
                f"[adapter]   {path.name}: {len(text):,} chars "
                f"-> {len(chunks):,} chunks"
            )
            yield from chunks

        # Guard against dividing by zero on an empty corpus, which is a
        # legitimate thing to point the adapter at by accident.
        mean_len = total_chars / total_chunks if total_chunks else 0.0
        print(
            f"[adapter] total {total_chunks:,} chunks, "
            f"mean {mean_len:.1f} chars/chunk"
        )

    def _resolve_files(self, path: Path) -> list[Path]:
        """Turn a file-or-directory path into a sorted list of .txt files."""
        if path.is_dir():
            return sorted(path.glob("*.txt"))
        if path.is_file():
            return [path]
        raise FileNotFoundError(f"no such file or directory: {path}")

    def _split_into_chunks(self, text: str) -> list[str]:
        """Split one file's text on blank lines, dropping empty results.

        Each chunk is stripped of surrounding whitespace. The blank line that
        separated them is *not* preserved here; data/prepare.py rejoins chunks
        with "\\n\\n" so the boundary comes back as literal text. That is what
        stands in for a special end-of-document token in this project: the
        model learns the blank line as a boundary signal from data, rather
        than being handed a reserved token id for it.
        """
        chunks = []
        for raw in PARAGRAPH_SEPARATOR.split(text):
            chunk = raw.strip()
            # A file ending without a trailing newline still produces its final
            # chunk here, because split() returns the trailing segment; the
            # only things dropped are chunks that are empty (or too short)
            # after stripping.
            if len(chunk) >= self.min_chars:
                chunks.append(chunk)
        return chunks


if __name__ == "__main__":
    import tempfile

    SAMPLE = """First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

   \t
First Citizen:
You are all resolved rather to die than to famish?"""

    # Written to a real temp file rather than passed as a string, so this
    # demonstrates the actual code path the pipeline uses.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sample.txt"
        path.write_text(SAMPLE, encoding="utf-8")

        chunks = list(PlainTextAdapter().read(str(path)))

    print()
    print(f"got {len(chunks)} chunks (the whitespace-only one was dropped):")
    for i, chunk in enumerate(chunks):
        preview = chunk.replace("\n", " / ")
        print(f"  [{i}] {preview!r}")
