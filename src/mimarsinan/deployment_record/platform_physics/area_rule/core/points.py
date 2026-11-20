"""[U-AREA] The door's judges of the published points and their rows and stacks (moved from core/arguments.py in the
fourth delta round for the module budget): a point, a scope, the registry's points, a list of parity rows (as the
headline reads them, as the strict spread and the strict form read them, as the array figures read them), a list of
stacks. Each states what the door's body reads (the fifth check's MA-C3: the strict rows hold at most one headline row
per chip, at least one priced, and a type-default reading on every row refused by converter kind).
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from mimarsinan.deployment_record.platform_physics.area_rule import units as u
from mimarsinan.deployment_record.platform_physics.area_rule.core import fields
from mimarsinan.deployment_record.platform_physics.area_rule.core.door import remembered
from mimarsinan.deployment_record.platform_physics.area_rule.evidence import AreaRuleRefusal
from mimarsinan.deployment_record.platform_physics.area_rule.published import decomposition, parity
from mimarsinan.deployment_record.platform_physics.area_rule.registry import Registry, View

Judge = Callable[[Any, str, Any, Mapping[str, Any], bool], None]


def registry_judge(registry: Any, name: str, value: Any, arguments: Mapping[str, Any], memo: bool) -> None:
    if not isinstance(value, (Registry, View)):
        raise AreaRuleRefusal(f"{name} {value!r:.80} is not the area rule's registry")


def point_judge(registry: Any, name: str, value: Any, arguments: Mapping[str, Any], memo: bool) -> None:
    """A published point: its fields by the table (its expressions in sample) and the rules that tie them."""
    if not isinstance(value, Mapping):
        raise AreaRuleRefusal(f"{name} {value!r:.80} is not a published point")
    place = f"the published point {value.get('name')!r}"
    fields.judged_holder(getattr(registry, "base", registry), place, value, "point")
    point_org, point_conv, figure = value["organization"], value.get("converter") or {}, value.get("array_figure")
    forms = [f for f in parity.COUNT_FORMS if f in point_org]
    scopes = value["scopes"]
    why = [f"its organization states {forms}, one of {list(parity.COUNT_FORMS)}" if len(forms) != int(u.UNITY)
           else None,
           "arrays without array_cells" if "arrays" in point_org and "array_cells" not in point_org else None,
           "a converter without its kind" if point_conv and "kind" not in point_conv else None,
           "a converter without its count" if point_conv and "count" not in point_conv and "per_array" not in
           point_conv else None,
           "a per-array converter without arrays" if "per_array" in point_conv and "arrays" not in point_org else None,
           "no primary scope" if not any(sc.get("primary") is True for sc in scopes) else None,
           "a sub-array scope without array_cells and a per-array converter" if any(sc.get("sub_arrays") and not (
               "array_cells" in point_org and "per_array" in point_conv) for sc in scopes) else None,
           "an array figure at no scope of the point" if figure and figure.get("scope") not in [
               sc.get("scope") for sc in scopes] else None]
    if stated := [w for w in why if w]:
        raise AreaRuleRefusal(f"{place}: {'; '.join(stated)}")


def points_judge(registry: Any, name: str, value: Any, arguments: Mapping[str, Any], memo: bool) -> None:
    registry_judge(registry, name, value, arguments, memo)
    found = value.published_points()
    if not isinstance(found, (tuple, list)) or not found:
        raise AreaRuleRefusal(f"{name}'s published points {found!r:.80} are not a list of points")
    remembered(value, "points", (found,), lambda: [point_judge(value, name, p, arguments, memo) for p in found])


def scope_judge(registry: Any, name: str, value: Any, arguments: Mapping[str, Any], memo: bool) -> None:
    if not isinstance(value, Mapping):
        raise AreaRuleRefusal(f"{name} {value!r:.80} is not a published scope")
    fields.judged_holder(getattr(registry, "base", registry), name, value, "scope")
    stated = arguments.get("point")
    point: Mapping[str, Any] = stated if isinstance(stated, Mapping) else {}
    if value.get("sub_arrays") and not ("array_cells" in (point.get("organization") or {})
                                        and "per_array" in (point.get("converter") or {})):
        raise AreaRuleRefusal(f"{name} {value!r:.80}: a sub-array scope of a point without array_cells and a per-array "
                              "converter")


def rows(spread: bool = False, priced: bool = False, figures: bool = False) -> Judge:
    """A list of parity rows (each judged where it was made: parity.Row), and what the door's body reads of them:
    ``spread`` one headline row per chip (strict_spread keys them by chip); ``priced`` at least one priced headline row
    and a type-default reading on each refused by converter kind (strict_form reads both); ``figures`` a row with its
    array at each published array figure's scope (array_decomposition)."""
    def judge(registry: Any, name: str, value: Any, arguments: Mapping[str, Any], memo: bool) -> None:
        if not isinstance(value, (list, tuple)) or not all(isinstance(r, parity.Row) for r in value):
            raise AreaRuleRefusal(f"{name} {value!r:.80} is not a list of the published points' rows")
        head = parity.headline(value) if spread or priced else ()
        if len({r.chip for r in head}) != len(head):
            raise AreaRuleRefusal(f"{name}: more than one headline row per chip (the strict spread holds one headline "
                                  "row per chip)")
        if priced and not any(r.refused is None for r in head):
            raise AreaRuleRefusal(f"{name}: no headline row is priced (a strict form needs one)")
        if priced and (lacking := [r.chip for r in head if r.refused_by == "converter" and not any(
                reading.type_default for reading in r.readings)]):
            raise AreaRuleRefusal(f"{name}: the headline rows refused by converter kind of {lacking} state no "
                                  "type-default reading (the strict form reads it)")
        by_key = {(r.chip, r.scope): r for r in value}
        for point in registry.published_points() if figures else ():
            figure = point.get("array_figure")
            row = by_key.get((point["name"], (figure or {}).get("scope")))
            if figure and (row is None or row.array_um2 is None or (row.eta is None and decomposition.scope_class(
                    registry, figure["ids"]) == decomposition.BITCELLS)):
                raise AreaRuleRefusal(f"{name}: no row of {point['name']} with its array at {figure['scope']}")
    return judge


def stacks(registry: Any, name: str, value: Any, arguments: Mapping[str, Any], memo: bool) -> None:
    if not isinstance(value, (list, tuple)) or not all(isinstance(stack, decomposition.Stack) for stack in value):
        raise AreaRuleRefusal(f"{name} {value!r:.80} is not a list of stacks")
    for i, stack in enumerate(value):
        fields.judged_holder(registry, f"{name}[{i}]", stack, "stack")
    if not any(stack.family == decomposition.SRAM_FAMILY for stack in value):
        raise AreaRuleRefusal(f"{name} {value!r:.80} holds no stack of the SRAM family")


__all__ = ["point_judge", "points_judge", "registry_judge", "rows", "scope_judge", "stacks"]
