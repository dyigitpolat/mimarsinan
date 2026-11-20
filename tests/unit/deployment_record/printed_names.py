"""[U-ENERGY class 2] A term's printed text read in its two parts: what it prices, and what it names unpriced.

No declared constant prices at zero unnamed: a declared row over a census the view does not produce is named beside
the estimate with the plane's reason ("not produced on this plane: ..."). A test that pins a term's text therefore
reads the priced part and the unpriced names apart, so a newly named row is told from a change of what is priced."""

from __future__ import annotations

from typing import Iterable, List, Tuple

UNPRICED = "; unpriced: "
NOT_PRODUCED = "not produced on this plane"


def split_names(text: str) -> List[str]:
    """``text`` split at its top-level ", " (a name's own parenthesis, with its commas, kept whole)."""
    names, depth, start = [], 0, 0
    for i, ch in enumerate(text):
        depth += ch == "("
        depth -= ch == ")"
        if depth == 0 and text.startswith(", ", i):
            names.append(text[start:i])
            start = i + 2
    tail = text[start:]
    return names + ([tail] if tail else [])


def parts(source: str) -> Tuple[str, List[str]]:
    """(the priced part, the unpriced names) of a term's source."""
    priced, _sep, unpriced = source.partition(UNPRICED)
    return priced, split_names(unpriced)


def unpriced_names(source: str) -> List[str]:
    return parts(source)[1]


def named(source: str, row: str) -> List[str]:
    """The unpriced names of ``row`` in a term's source (its bare name or its name with a reason)."""
    return [n for n in unpriced_names(source) if n == row or n.startswith(row + "(")]


def without(source: str, rows: Iterable[str]) -> str:
    """The source with every unpriced name of ``rows`` taken out (the unpriced list dropped where it empties)."""
    rows = tuple(rows)
    priced, names = parts(source)
    kept = [n for n in names if not any(n == r or n.startswith(r + "(") for r in rows)]
    return priced + (UNPRICED + ", ".join(kept) if kept else "")
