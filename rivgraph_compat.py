"""Compatibility guards for the locally patched RivGraph installation.

RivGraph's river centerline routine assumes every link-dictionary entry other
than ``n_networks`` is a per-link sequence. Deterministic network-ID metadata
adds scalar/private entries, so older centerline code attempts to index a
boolean. The context manager below sanitizes only the temporary holes-filled
graph created internally by ``mask_to_centerline``.
"""

from __future__ import annotations

from collections.abc import Iterator, MutableMapping
from contextlib import contextmanager
from typing import Any


def sanitize_centerline_link_fields(links: MutableMapping[str, Any]) -> None:
    """Retain only fields RivGraph may safely index by link position."""

    n_links = len(links.get("id", []))
    for key in list(links):
        if key in {"id", "n_networks"}:
            continue
        value = links[key]
        if key.startswith("_") or isinstance(value, (str, bytes)):
            del links[key]
            continue
        try:
            value_len = len(value)
        except TypeError:
            del links[key]
            continue
        if value_len != n_links:
            del links[key]


@contextmanager
def rivgraph_centerline_compat() -> Iterator[None]:
    """Make one RivGraph centerline call tolerant of scalar link metadata.

    The original RivGraph function is restored even if centerline generation
    raises. Exported network links and nodes are never passed to the sanitizer.
    """

    try:
        import rivgraph.ln_utils as lnu
    except ModuleNotFoundError:
        yield
        return

    original = lnu.link_widths_and_lengths

    def patched_link_widths_and_lengths(*args: Any, **kwargs: Any):
        links = original(*args, **kwargs)
        sanitize_centerline_link_fields(links)
        return links

    lnu.link_widths_and_lengths = patched_link_widths_and_lengths
    try:
        yield
    finally:
        lnu.link_widths_and_lengths = original
