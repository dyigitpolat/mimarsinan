"""[U-AREA] The registry viewed without a held-out chip's entries (the build's verify, MA-3): the published points'
held-out estimate prices every core through the type rule's own equation on this view, never by equations of its own.

Every value, evaluation, node cell and law of the view applies the chip's exclusions. A value the chip's own
declaration names (its cell kind's factor, its converter's own printed unit) reads the chip's own printed units as
inputs (design 3.3: "its organization and own printed units kept as inputs"); every other value is re-derived without
the chip. Every other attribute is the registry's.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Tuple

from mimarsinan.deployment_record.platform_physics.area_rule.evidence import KEPT, HeldOut, excluded


class View:
    def __init__(self, base: Any, excl: Tuple[str, ...], own: Tuple[str, ...] = ()):
        self.base, self.excl = base, tuple(excl)
        self.kept = self.excl + tuple(KEPT + i for i in own)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    def value(self, path: str, excl: Sequence[str] = ()) -> float:
        return self.base.value(path, self.excl + tuple(excl))

    def evaluate(self, expression: str, excl: Sequence[str] = ()) -> float:
        return self.base.evaluate(expression, self.excl + tuple(excl))

    def sigma(self, node_nm: float, excl: Sequence[str] = (), declared_um2: Optional[float] = None) -> float:
        return self.base.sigma(node_nm, self.excl + tuple(excl), declared_um2)

    def law(self, obj: Mapping[str, Any], excl: Sequence[str] = (), name: str = "the law") -> Tuple[
            float, float, Tuple[float, float]]:
        return self.base.law(obj, self.excl + tuple(excl), name)

    def object_value(self, obj: Mapping[str, Any], excl: Sequence[str], name: str) -> float:
        return self.base.object_value(obj, self.excl + tuple(excl), name)

    def declared_value(self, path: str) -> float:
        return self.base.value(path, self.kept)

    def declared_entry(self, entry_id: str) -> float:
        if excluded(entry_id, self.kept):
            raise HeldOut(entry_id)
        return self.base.evidence.value(entry_id)

    def held_out(self, excl: Sequence[str], own: Sequence[str] = ()) -> "View":
        return View(self.base, self.excl + tuple(excl), tuple(own))
