"""The adapter interface: the one abstraction in this project that matters.

Everything downstream of an adapter (tokenizer, packed shards, model, training
loop, sampler) works on text and knows nothing about where that text came from.
That is the entire point: an adapter is the *only* place domain-specific logic
is allowed to live. Adding a new domain later should mean writing one new file
here and re-running the tokenizer and data-prep steps, with no change to
model.py, train.py, or sample.py (DESIGN.md section 5).

If adding a domain ever does require touching the model or the training loop,
that is a signal this interface was designed wrong, and the right response is
to fix the interface rather than special-case the new domain.
"""

from collections.abc import Iterator
from typing import Protocol, runtime_checkable


@runtime_checkable
class Adapter(Protocol):
    """Anything that can turn a data source into a stream of text examples.

    This is a Protocol (structural typing) rather than an abstract base class.
    The practical difference: an adapter does not have to inherit from anything
    to count as an adapter, it just has to have a `read` method of the right
    shape. That keeps each adapter file standalone and readable on its own,
    which is worth more here than the compile-time enforcement an ABC would
    give, since there is exactly one implementation to keep honest.

    `runtime_checkable` means `isinstance(obj, Adapter)` works, so tests can
    assert an implementation satisfies the interface. Note it only checks that
    the method *exists*, not that its signature matches.
    """

    def read(self, source_path: str) -> Iterator[str]:
        """Yield raw text examples from a data source.

        Returns an iterator rather than a list so a corpus larger than memory
        could be streamed through the pipeline without change. Each yielded
        string is one "example": what that means is the adapter's business
        (for plain text it is roughly a paragraph), but it should be a unit
        that makes sense to read on its own, because these strings are what
        gets logged when you inspect the pipeline.
        """
        ...
