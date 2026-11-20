"""[U-AREA] A cell's area by its kind (design revision 3, section 2.1), moved from rule.py in the third delta round for
the module budget (rule.py re-exports every public name):

    a_cell = kappa(kind) x sigma(n)  (SRAM; compute cells k_kind x k_L x sigma)  |  f x n^2 (1T1R)  |  a_PCM

The 1T1R cell's square of its node is the judge's ``power``, infinity beyond the floats, which the cell's judge refuses
where cell_area makes it (the fourth delta round, MA-C6; MA-K2 judged it where the core's cell is resolved). Every
public function is a door (core/door.py).
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from mimarsinan.deployment_record.platform_physics.area_rule import units as u
from mimarsinan.deployment_record.platform_physics.area_rule.core.door import door
from mimarsinan.deployment_record.platform_physics.area_rule.core.judge import CELL, judged, power


def _kind_value(registry: Any, path: str, excl: Sequence[str], declared: bool) -> float:
    """A cell kind's value: a kind the description declares reads its own printed units as inputs (a view's
    ``declared_value``); a default kind and every shared value read under the exclusions."""
    return registry.declared_value(path) if declared else registry.value(path, excl)


@door
def logic_rule_factor(registry: Any, excl: Sequence[str] = ()) -> float:
    """k_L: the logic-rule 6T over the node's reference cell (its kind named in cell_kinds.json, mi-4)."""
    return registry.value(f"cell_kinds:kinds.{registry.files['cell_kinds']['logic_rule_reference']}.factor_of_sigma",
                          excl)


@door(declared="flag")
def cell_factor(registry: Any, kind: str, excl: Sequence[str] = (), declared: bool = False) -> float:
    """A compute cell over a logic-rule 6T (factor_of_logic_6t) or the series' own cell (factor_of_sigma)."""
    spec = registry.cell_kind(kind)
    name = next(iter(k for k in ("factor_of_logic_6t", "factor_of_sigma") if k in spec))
    return _kind_value(registry, f"cell_kinds:kinds.{kind}.{name}", excl, declared)


@door(declared="flag")
def cell_area(registry: Any, kind: str, node_nm: float, excl: Sequence[str] = (),
              declared_um2: Optional[float] = None, declared: bool = False) -> float:
    """One cell (um^2) of a kind at a node: an SRAM cell over sigma(n), a 1T1R cell in F^2, a node-flat PCM cell
    (``declared``: the description's own kind, read with its own printed units as inputs); judged where it is made, a
    cell's area above 0 (the fourth delta round, MA-C6: an F2 cell beyond the floats or below them, a factor over a
    declared cell of the floats' top, is a named refusal naming the kind and the node)."""
    spec = registry.cell_kind(kind)
    if "F2" in spec:
        area = _kind_value(registry, f"cell_kinds:kinds.{kind}.F2", excl, declared) * power(
            node_nm / u.NM_PER_UM, u.BINARY)
    elif "um2" in spec:
        area = _kind_value(registry, f"cell_kinds:kinds.{kind}.um2", excl, declared)
    else:
        sigma = registry.sigma(node_nm, excl, declared_um2)
        factor = cell_factor(registry, kind, excl, declared)
        area = factor * logic_rule_factor(registry, excl) * sigma if "factor_of_logic_6t" in spec else factor * sigma
    return judged("the {} cell at {!r} nm", area, CELL, kind, node_nm)


@door
def reads_sigma(registry: Any, kind: str) -> bool:
    """Whether ``cell_area`` reads sigma(n) for a kind, and so a declared cell in its place (mi-D1): every kind but a
    1T1R cell in F^2 and a node-flat cell."""
    spec = registry.cell_kind(kind)
    return "F2" not in spec and "um2" not in spec



__all__ = ["cell_area", "cell_factor", "logic_rule_factor", "reads_sigma"]
