"""[U-AREA] The held-out decomposition at each published array figure's scope (review MA-2, T-26's unit side).

A point's published array figure names its entries; their array scope class (``array_scope.class`` of the evidence,
the second corrected set's data) decides the comparison: at ``bitcells`` the estimate's array is cells x cell area
(the group times eta) and its macro overhead, array x (1 - eta), moves to the rest; at ``macro_or_block`` the array
stays cells x cell area / eta. The total does not change. No chip is named: the class is read from the entries.
"""

from __future__ import annotations

from typing import Any, Dict, List, NamedTuple, Optional, Sequence

from mimarsinan.deployment_record.platform_physics.area_rule import units as u
from mimarsinan.deployment_record.platform_physics.area_rule.core.door import door
from mimarsinan.deployment_record.platform_physics.area_rule.core.judge import NONNEGATIVE, judged

BITCELLS, BLOCK = "bitcells", "macro_or_block"
#: The cell family of the SRAM array fact; each type table states its own (``cell_family``, the fix round's mi-4).
SRAM_FAMILY = "sram"


class Stack(NamedTuple):
    chip: str
    scope: str
    type: str
    scope_class: str
    published_um2: float
    array_um2: float
    array_ratio: float
    moved_to_rest_um2: float
    total_um2: Optional[float]
    total_macro_scope_um2: Optional[float]
    family: Optional[str] = None


@door
def scope_class(registry: Any, ids: Sequence[str]) -> str:
    """The array scope class the figure's entries declare: bitcells where one does, else the array block's."""
    classes = {registry.evidence.entry(i).get("array_scope", {}).get("class") for i in ids}
    return BITCELLS if BITCELLS in classes else BLOCK


@door(rows="figure rows")
def array_decomposition(registry: Any, rows: Sequence[Any], scope_rule: str = "published") -> List[Stack]:
    """The array group of each point with a published array figure, at the figure's scope (``scope_rule``
    'published'), or at the array block's scope for every figure ('macro', the mutant of T-26)."""
    by_key = {(r.chip, r.scope): r for r in rows}
    out = []
    for point in registry.published_points():
        figure = point.get("array_figure")
        if not figure:
            continue
        row = by_key[(point["name"], figure["scope"])]
        k = row.count
        published = registry.evaluate(figure["um2"])
        array = k * row.array_um2
        cls = scope_class(registry, figure["ids"]) if scope_rule == "published" else BLOCK
        moved = u.NONE
        if cls == BITCELLS:
            moved = array * (u.UNITY - row.eta)
            array = array * row.eta
        total = total_macro = None
        if row.terms is not None:
            outside = (k * row.terms.core_um2 * (u.UNITY / row.gamma - u.UNITY)) if row.gamma else u.NONE
            total_macro = k * row.terms.core_um2 + outside
            rest = total_macro - k * row.array_um2 - k * row.terms.converters_um2
            total = array + k * row.terms.converters_um2 + rest + moved
        stack = Stack(point["name"], figure["scope"], point["type"], cls, published, array, array / published, moved,
                      total, total_macro, registry.type_table(point["type"]).get("cell_family"))
        for name in ("published_um2", "array_um2", "array_ratio", "moved_to_rest_um2", "total_um2",
                     "total_macro_scope_um2"):  # judged where made (the fourth delta round, MA-C6)
            if getattr(stack, name) is not None:
                judged("the stack of {} at {}: its {}", getattr(stack, name), NONNEGATIVE, stack.chip, stack.scope,
                       name)
        out.append(stack)
    return out


@door
def sram_array_fact(stacks: Sequence[Stack]) -> Dict[str, Any]:
    """The largest |array ratio - 1| over the SRAM points (the types whose table states the SRAM cell family) and
    its chip."""
    sram = [s for s in stacks if s.family == SRAM_FAMILY]
    worst = max(sram, key=lambda s: abs(s.array_ratio - u.UNITY))
    return {"chip": worst.chip, "max_deviation": abs(worst.array_ratio - u.UNITY), "chips": [s.chip for s in sram],
            "ratios": {s.chip: s.array_ratio for s in sram}}
