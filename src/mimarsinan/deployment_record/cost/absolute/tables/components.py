"""[U-ENERGY class 2] A component table priced with every factor in one of four states, none priced at zero unnamed.

Produced: priced, constant x census. Withheld: named unpriced with the view's reason. Zero by the platform's
declaration: nothing priced, the reason printed in the basis. Not produced on this plane: named beside the estimate
with the plane's reason, never a refusal. A description's declared dataflow derives the conversion counts once
(``with_derived``): a dataflow without converters makes them zero by its declaration, with that reason printed."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from mimarsinan.deployment_record.cost.absolute.formulas import (
    DerivedFactor, Factor, factor_withheld, quantity_product, scale_band)
from mimarsinan.deployment_record.platform_physics.conversion import DIGITAL
from mimarsinan.deployment_record.quantities.spec import Quantities
from mimarsinan.deployment_record.schema.provenance import Band

ComponentTable = Sequence[Tuple[str, Tuple[Factor, ...], str]]


def _keys(factor: Factor) -> Tuple[str, ...]:
    return (factor.over + factor.per) if isinstance(factor, DerivedFactor) else (factor,)


def factor_known_zero(quantities: Quantities, factor: Factor) -> Optional[str]:
    """Why ``factor``'s work is zero by declaration (any quantity it is read off), or None."""
    return next((r for r in (quantities.known_zero_reason(k) for k in _keys(factor)) if r), None)


def factor_unproduced(quantities: Quantities, factor: Factor) -> Optional[str]:
    """Why this plane never counts ``factor`` (or a quantity it is read off), or None."""
    return next((r for r in (quantities.unproduced_reason(k) for k in _keys(factor)) if r), None)


def absent_reason(quantities: Quantities, key: str) -> str:
    """Why ``key`` is absent from the view, in the words of its state: withheld (the view's reason), zero by
    declaration (the platform's), or not produced on this plane (the plane's)."""
    withheld = quantities.withheld_reason(key)
    if withheld is not None:
        return withheld
    known = quantities.known_zero_reason(key)
    if known:
        return f"zero by declaration: {known}"
    return f"not produced on this plane: {quantities.unproduced_reason(key) or 'the view states no ' + key}"


def component_bands(ctx: Any, table: ComponentTable, scale: float) -> Tuple[List[Band], List[str], List[str]]:
    """(priced bands, evidence notes, unpriced-work notes) over a component table on a pricing context: a refused
    constant over existing work blocks its headline (``ctx.blocked``); a declared constant over an absent factor is
    named with its state's reason, or printed as a zero by declaration."""
    q = ctx.quantities
    bands: List[Band] = []
    evidence: List[str] = []
    unpriced: List[str] = []
    for constant, factors, label in table:
        product = quantity_product(q, factors)
        # A factor the view withholds: its work may exist, so the component is named, never dropped.
        withheld = next((reason for reason in (factor_withheld(q, f) for f in factors) if reason is not None), None)
        if constant in ctx.refused_constants:
            if product or withheld is not None:  # work exists that the declared constant cannot price
                ctx.blocked[constant] = ctx.refused_constants[constant]
            continue
        if ctx.priced(constant):
            if product is not None:
                value = ctx.physics.band(constant)
                bands.append(scale_band(value, product * scale, f"{constant} x {' x '.join(map(str, factors))}"))
                evidence.append(f"{label}: {constant} [{value.basis}]")
            elif withheld is not None:
                unpriced.append(f"{constant}({label}; {withheld})")
            elif (zero := next((r for r in (factor_known_zero(q, f) for f in factors) if r), None)) is not None:
                evidence.append(f"zero by declaration: {constant} ({zero})")
            else:
                why = next((r for r in (factor_unproduced(q, f) for f in factors) if r), None)
                unpriced.append(f"{constant}({label}; not produced on this plane: "
                                f"{why or 'the view states no ' + ' x '.join(map(str, factors))})")
        elif not ctx.physics.has(constant) and (product or withheld is not None):
            # Work exists (a positive census) but the target never priced it.
            unpriced.append(f"{constant}({label})" if withheld is None else f"{constant}({label}; {withheld})")
    return bands, evidence, unpriced


def with_derived(quantities: Quantities, model: Any) -> Quantities:
    """The view with what the description's declared dataflow derives (the conversion counts, a bit-sliced
    crossbar's row drives) joined beside its own quantities, never replacing one; a dataflow without converters makes
    its counts zero by declaration, with the reason printed wherever a declared row reads them."""
    derived = model.derive(quantities)
    if not derived:
        return quantities
    declared: Dict[str, str] = {k: f"the declared dataflow ({model.name}) has no converter" for k in derived
                                if model.name == DIGITAL and not quantities.has(k)}
    merged = {key: quantities.get(key) for key in quantities.keys()}
    for key, value in derived.items():
        if key not in declared:
            merged.setdefault(key, value)
    known = {**quantities.known_zero, **declared}
    withheld = {k: r for k, r in quantities.withheld.items() if k not in merged and k not in known}
    return Quantities(merged, withheld=withheld,
                      unproduced=quantities.unproduced, known_zero=known)
