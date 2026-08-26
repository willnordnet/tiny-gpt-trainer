"""Download the TinyShakespeare corpus into data/raw/.

Why this exists as a script rather than a README instruction: the corpus is
gitignored (it is downloaded data, not source), so anyone cloning this repo
needs a one-command way to get it back. Keeping it as code also means the
provenance of the training data is recorded in the repo instead of living in
someone's shell history.

TinyShakespeare is the nanoGPT-standard toy corpus: ~1.1MB of concatenated
Shakespeare, small enough that the `tiny` preset trains in minutes, and
stylistically distinctive enough that you can tell by eye whether generated
samples have started to resemble it.

Uses urllib from the standard library rather than `requests` so this adds no
dependency (see requirements.txt on why the dependency list stays short).
"""

import urllib.request
from pathlib import Path

# Karpathy's char-rnn repo is the canonical host for this file; it is the same
# bytes every other tiny-transformer tutorial trains on, which makes results
# here loosely comparable to those.
TINYSHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)

DEFAULT_DEST = Path("data/raw/tinyshakespeare.txt")


def download(url: str = TINYSHAKESPEARE_URL, dest: Path = DEFAULT_DEST) -> Path:
    """Fetch `url` to `dest`, skipping the download if the file already exists.

    Returns the path written to. Idempotent on purpose: re-running the whole
    pipeline from the top should not re-download a megabyte every time.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        print(f"[download] already present: {dest} ({dest.stat().st_size:,} bytes)")
        return dest

    print(f"[download] fetching {url}")
    with urllib.request.urlopen(url) as response:
        data = response.read()
    dest.write_bytes(data)

    # Log the shape of what we got, not just "done". A corpus that arrived
    # truncated or as an HTML error page is a confusing failure three stages
    # downstream, and a byte/line count makes it obvious right here.
    line_count = data.count(b"\n") + 1
    print(f"[download] wrote {dest} ({len(data):,} bytes, {line_count:,} lines)")

    preview = data[:120].decode("utf-8", errors="replace").replace("\n", " / ")
    print(f"[download] starts with: {preview!r}")

    return dest


if __name__ == "__main__":
    download()
