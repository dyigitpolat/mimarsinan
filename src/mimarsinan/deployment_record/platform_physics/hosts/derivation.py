"""[U-HOST2] The one declared form of a row derived from a host description's rows.

A row that names its host (``HOST_IDENTITY_FIELD``, ``'<description> <mode>'``) is admitted only as that mode's own
row or as a derivation of the mode's rows declared in this one form (``hosts.admission``, the one admission rule)::

    "derived_from": {"of": {"<row the named mode declares>": <factor>, ...}, "combine": "sum" | "slower"}

The derived band is, corner by corner (low, nominal, high), the stated combination of each named row's band (in SI)
times its factor: ``sum`` adds the terms (the default: a store-and-forward composition), ``slower`` takes the largest
(the pipelined bound of a series link; a time only). Each factor is a finite number above zero, every named row has
the derived row's dimension, and the row's class is ``derived``. A host row's derivation names the row itself among
its terms (it is a derivation of that row); a chip row fed over the host's link (``host_attach_block``) names the host
rows it is built from. One row at one factor, ``{"of": {"<row>": <factor>}}``, is a scaled row: a robustness grid's
point, or a sensitivity of another host stated against the identified host's rows. The loader recomputes the band
from the description and compares it (``REL_TOL``, one rounding of a product); the declaration is sealed with the row
(``PhysicsConstantValue.derived_from``), so a sealed physics is judged by the same rule. [U-HOST3] The row's words are
tied to its declaration: its printed basis states the declaration first (``hosts.rows.declaration_statement``), and a
derivation text that contradicts it (``text_refusal``: a vocabulary row that is no term, a stated factor no term
declares) is refused by name.
"""

from __future__ import annotations

import json
import math
import re
from decimal import Decimal
from typing import Any, Mapping, Optional, Sequence, Tuple

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.deployment_record.platform_physics.constants import PHYSICS_CONSTANTS, spec_for
from mimarsinan.deployment_record.units import TIME

#: The field of a row declaring its derivation from its host description's rows.
DERIVATION_FIELD = "derived_from"
SUM, SLOWER = "sum", "slower"
COMBINES = (SUM, SLOWER)
#: The comparison's one tolerance: a recomputed corner equals the declared one up to one rounding of a product.
REL_TOL = 1e-12
CORNERS = ("low", "nominal", "high")
Terms = Tuple[Tuple[str, float], ...]


class HostDerivationError(ToolchainRefusal, ValueError):
    """A ``derived_from`` declaration that is not of the one form, refused by name."""


def _factor(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) and value > 0.0 else None


def judged(key: str, declaration: Any) -> Tuple[Terms, str]:
    """(the terms, the combine) of ``key``'s ``derived_from`` declaration, or a refusal by name."""
    where = f"{key}: {DERIVATION_FIELD}"
    if not isinstance(declaration, Mapping) or set(declaration) - {"of", "combine"} or "of" not in declaration:
        raise HostDerivationError(f"{where} is {{'of': {{<row>: <factor>, ...}}, 'combine': {list(COMBINES)}}}, "
                                  f"got {declaration!r}")
    of, combine = declaration["of"], declaration.get("combine", SUM)
    if not isinstance(of, Mapping) or not of:
        raise HostDerivationError(f"{where}: 'of' names at least one row with its factor, got {of!r}")
    if combine not in COMBINES:
        raise HostDerivationError(f"{where}: combine {combine!r} is not one of {list(COMBINES)}")
    if combine == SLOWER and spec_for(key).dimension != TIME:
        raise HostDerivationError(f"{where}: combine {SLOWER!r} composes times, and {key} is a "
                                  f"{spec_for(key).dimension} row")
    terms = []
    for row, value in sorted(of.items()):
        if str(row) not in PHYSICS_CONSTANTS:
            raise HostDerivationError(f"{where}: {row!r} is not a row of the vocabulary")
        if (factor := _factor(value)) is None:
            raise HostDerivationError(f"{where}: the factor of {row!r} is a finite number above zero, got {value!r}")
        if spec_for(str(row)).dimension != spec_for(key).dimension:
            raise HostDerivationError(f"{where}: {row!r} is a {spec_for(str(row)).dimension} row and {key} a "
                                      f"{spec_for(key).dimension} row: a derivation keeps its dimension")
        terms.append((str(row), factor))
    return tuple(terms), str(combine)


def canonical(key: str, declaration: Any) -> str:
    """The judged declaration as the text a value carries and seals (``PhysicsConstantValue.derived_from``)."""
    terms, combine = judged(key, declaration)
    return json.dumps({"combine": combine, "of": dict(terms)}, sort_keys=True)


def parsed(key: str, text: str) -> Tuple[Terms, str]:
    """The terms and combine of a value's sealed ``derived_from`` text (judged again)."""
    try:
        declaration = json.loads(text)
    except ValueError:
        raise HostDerivationError(f"{key}: {DERIVATION_FIELD} {text!r} is not a declaration") from None
    return judged(key, declaration)


def combined(terms: Terms, combine: str, bands: Mapping[str, Sequence[float]]) -> Tuple[float, float, float]:
    """The derived SI band: per corner, each term's SI band times its factor, added or the largest taken."""
    pick = max if combine == SLOWER else sum
    low, nominal, high = (pick([float(bands[row][i]) * factor for row, factor in terms]) for i in range(3))
    return low, nominal, high


def same_band(declared: Sequence[float], expected: Sequence[float]) -> bool:
    """Whether two SI bands are equal corner by corner up to ``REL_TOL``."""
    return all(math.isclose(a, b, rel_tol=REL_TOL, abs_tol=0.0) for a, b in zip(declared, expected))


#: [U-HOST3] A number as a derivation text writes one.
_NUMBER = r"(\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
#: [U-HOST3] A factor a derivation text states: a number beside a multiplication sign ('x' standing alone, or × or *).
_STATED_FACTORS = re.compile(rf"(?:^|(?<=\s))(?:x\s+|[×*]\s*){_NUMBER}(?!\w)|(?<![\w.]){_NUMBER}(?:\s+x(?=\s|$)|\s*[×*])")
#: [U-HOST3] A token that may name a vocabulary row.
_TOKEN = re.compile(r"(?<![A-Za-z0-9_])[a-z][a-z0-9_]*(?![A-Za-z0-9_])")


def _written(text: str, factor: float) -> bool:
    """Whether the number ``text`` is ``factor`` as written to the text's own significant digits (a text stating more
    digits than the factor carries compares within ``REL_TOL``, as the band does)."""
    digits = len(Decimal(text).as_tuple().digits)  # the text's significant digits (no string normalization)
    return math.isclose(float(text), float(f"{factor:.{digits}g}"), rel_tol=REL_TOL)


def text_refusal(key: str, text: str, terms: Terms) -> Optional[str]:
    """[U-HOST3] Why the derivation text of ``key`` contradicts its declared ``terms``, or None: every vocabulary row
    the text names is the row itself or a term, and every factor it states (a number beside x, × or *) is a term's
    factor as written. A text that states neither contradicts nothing (the printed basis states the declaration)."""
    rows = {row for row, _factor in terms}
    foreign = sorted({t for t in _TOKEN.findall(text) if t in PHYSICS_CONSTANTS} - rows - {key})
    stated = [a or b for a, b in _STATED_FACTORS.findall(text)]
    unstated = [s for s in stated if not any(_written(s, factor) for _row, factor in terms)]
    if not foreign and not unstated:
        return None
    return (f"{key}: its derivation text {text!r} contradicts its {DERIVATION_FIELD} "
            f"{dict(terms)}: " + "; ".join(([f"it names {foreign}, no term of it"] if foreign else [])
                                           + ([f"it states the factors {unstated}, no term's"] if unstated else [])))
