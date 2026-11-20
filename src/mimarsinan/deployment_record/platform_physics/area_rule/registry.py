"""[U-AREA] The area rule's registry: its data files, a value by path (``types/t1:eta``) under a held-out chip's
exclusions, the node series sigma(n) (rules A and B) and its evidence check (T-9). A published value reads its entry; a
derived value evaluates its expression (or its fallback where it reads a held-out entry); a fitted value aggregates its
points, dropping a held-out point and, by name, one that meets a named refusal (rule B), else its fallback or a named
refusal. The types are the type tables' ``type`` fields (mi-4); values, evaluations and node cells are memoized and
expressions parsed at load (mi-1); ``held_out`` views it without a chip's entries (MA-3). sigma and evaluate are doors.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from mimarsinan.deployment_record.platform_physics.area_rule import units as u
from mimarsinan.deployment_record.platform_physics.area_rule.core.door import door
from mimarsinan.deployment_record.platform_physics.area_rule.evidence import (
    AreaRuleRefusal, Evidence, HeldOut, evaluate, excluded, geometric_mean, log_linear_least_squares, parse_all,
    read_ids)

DATA = Path(__file__).resolve().parent / "data"
#: The data files beside the type tables, which are every ``types/*.json`` (found at load).
BASE_FILES = ("node_series", "cell_kinds", "converter_kinds", "bands")
TYPE_DIR = "types"


def type_files(data: Path = DATA) -> Tuple[str, ...]:
    return tuple(f"{TYPE_DIR}/{p.stem}" for p in sorted((Path(data) / TYPE_DIR).glob("*.json")))


def node_reference(values: Sequence[float]) -> float:
    """Rule A: the reference push-rule 6T cell at a node is the geometric mean of the node's published cells."""
    return geometric_mean(values)


def node_in_range(node: float, low: float, high: float) -> bool:
    """Rule B: the series prices only inside its published range."""
    return low <= node <= high


def _get(doc: Any, dotted: str) -> Any:
    for key in dotted.split("."):
        doc = doc[int(key)] if isinstance(doc, list) else doc[key]
    return doc


def _memo(memo: Dict[Any, Tuple[bool, Any]], key: Any, compute: Any) -> Any:
    """A memoized value; a memoized refusal is raised as a fresh copy each time (mi-R1: no traceback grows)."""
    if key not in memo:
        try:
            memo[key] = (True, compute())
        except (HeldOut, AreaRuleRefusal) as caught:
            memo[key] = (False, caught.with_traceback(None))
    ok, value = memo[key]
    if not ok:
        raise copy.copy(value)
    return value


@dataclass
class Registry:
    """The registry data files and the evidence, with values, evaluations and node cells memoized per exclusions."""

    files: Mapping[str, Any]
    evidence: Evidence
    points: Tuple[Mapping[str, Any], ...]
    _memo: Dict[Tuple[str, str, Tuple[str, ...]], Tuple[bool, Any]] = field(default_factory=dict)

    @classmethod
    def load(cls, data: Path = DATA) -> "Registry":
        data = Path(data)
        files = {name: json.loads((data / f"{name}.json").read_text()) for name in BASE_FILES + type_files(data)}
        points = json.loads((data / "evidence" / "published_points.json").read_text())["points"]
        loaded = cls(files=files, evidence=Evidence.load(data / "evidence" / "published_figures.json"),
                     points=tuple(points))
        parse_all(obj for _path, obj in loaded.walk())
        return loaded

    # ------------------------------------------------------------------ the type tables (mi-4: read from data)
    @cached_property
    def types(self) -> Dict[str, str]:
        """Each technology type and its table's file, by the table's own ``type`` field."""
        return {doc["type"]: name for name, doc in self.files.items() if name.startswith(f"{TYPE_DIR}/")}

    def type_file(self, type_name: str) -> str:
        if not isinstance(type_name, str) or type_name not in self.types:  # MA-D1: a name is a text
            raise AreaRuleRefusal(f"area_rule type {type_name!r} is not one of {list(self.types)}")
        return self.types[type_name]

    def type_table(self, type_name: str) -> Mapping[str, Any]:
        return self.files[self.type_file(type_name)]

    # ------------------------------------------------------------------ values
    def object(self, path: str) -> Mapping[str, Any]:
        name, dotted = path.split(":")
        return _get(self.files[name], dotted)

    @door(keep_held_out=True)
    def evaluate(self, expression: str, excl: Sequence[str] = ()) -> float:
        excl = tuple(excl)
        return _memo(self._memo, ("evaluate", expression, excl),
                     lambda: evaluate(expression, self.evidence, excl, self.value, self.sigma))

    def value(self, path: str, excl: Sequence[str] = ()) -> float:
        excl = tuple(excl)
        return _memo(self._memo, ("value", path, excl), lambda: self.object_value(self.object(path), excl, path))

    def declared_value(self, path: str) -> float:
        """A value a description's declaration names (its cell kind's factor); a view keeps the chip's own units."""
        return self.value(path)

    def declared_entry(self, entry_id: str) -> float:
        """An evidence entry a description's declaration names (its converter's own printed unit)."""
        return self.evidence.value(entry_id)

    def held_out(self, excl: Sequence[str], own: Sequence[str] = ()) -> "View":
        """The registry without the entries ``excl`` names, ``own`` ids kept as the declaration's inputs (MA-3)."""
        return View(self, tuple(excl), tuple(own))

    def object_value(self, obj: Mapping[str, Any], excl: Sequence[str], name: str) -> float:
        kind, excl = obj["evidence_kind"], tuple(excl)
        if kind == "published":
            entry = obj["entry_value"]
            if excluded(entry["id"], excl):
                raise HeldOut(entry["id"])
            return self.evidence.value(entry["id"], entry["index"], entry["key"])
        if kind == "derived":
            try:
                return self.evaluate(obj["recompute"]["expression"], excl)
            except HeldOut:
                if obj.get("fallback"):
                    return self.evaluate(obj["fallback"], excl)
                raise
        aggregate = obj["recompute"]["aggregate"]
        if aggregate == "log_linear_least_squares":
            return self.law(obj, excl, name)[u.FIRST]
        return self.fitted(obj, excl, name)[u.FIRST]

    def fitted(self, obj: Mapping[str, Any], excl: Tuple[str, ...], name: str) -> Tuple[float, Tuple[str, ...],
                                                                                        Tuple[str, ...]]:
        """(the value, the labels kept, the labels dropped with their reasons) of a fitted value."""
        recompute = obj["recompute"]
        values, kept, dropped = [], [], []
        for point in recompute["points"]:
            try:
                values.append(self.evaluate(point["expression"], excl))
                kept.append(point["label"])
            except HeldOut:
                dropped.append(f"{point['label']}: held out")
            except AreaRuleRefusal as refusal:
                dropped.append(f"{point['label']}: {refusal}")
        if not values:
            if recompute.get("fallback"):
                return self.evaluate(recompute["fallback"], excl), (), tuple(dropped)
            raise AreaRuleRefusal(f"{name}: no point remains ({'; '.join(dropped)})")
        if recompute["aggregate"] == "max":
            return max(values), tuple(kept), tuple(dropped)
        return geometric_mean(values), tuple(kept), tuple(dropped)

    def law(self, obj: Mapping[str, Any], excl: Sequence[str] = (), name: str = "the law") -> Tuple[
            float, float, Tuple[float, float]]:
        """A converter law (a_ref, alpha per bit, its published bits range) from its points still available; its
        refusal names the law (``name``: the converter kind that reads it, mi-4)."""
        recompute = obj["recompute"]
        points = []
        for point, bits in zip(recompute["points"], recompute["bits"]):
            try:
                points.append((self.evaluate(bits, excl), self.evaluate(point["expression"], excl)))
            except HeldOut:
                continue
        if len(points) < u.LAW_POINTS:
            raise AreaRuleRefusal(f"{name}: fewer than two published bit points remain")
        a_ref, alpha = log_linear_least_squares(points, recompute["reference_bits"])
        return a_ref, alpha, (min(b for b, _a in points), max(b for b, _a in points))

    # ------------------------------------------------------------------ the node series (rules A and B)
    def range_nm(self) -> Tuple[float, float]:
        nodes = [node["node_nm"] for node in self.files["node_series"]["nodes"]]
        return min(nodes), max(nodes)

    def node_reference_ids(self, node_nm: float) -> Tuple[str, ...]:
        node = next(n for n in self.files["node_series"]["nodes"] if n["node_nm"] == node_nm)
        return tuple(i for cell in node["cells"] for i in self._ids(cell))

    @door(keep_held_out=True)
    def sigma(self, node_nm: float, excl: Sequence[str] = (), declared_um2: Optional[float] = None) -> float:
        """The reference push-rule 6T cell at ``node_nm`` (um^2): a declared cell first; inside the published range
        the node's reference or the log-log interpolation of the adjacent nodes; outside it a named refusal."""
        if declared_um2 is not None:
            return float(declared_um2)
        return _memo(self._memo, ("sigma", node_nm, tuple(excl)), lambda: self._sigma(node_nm, tuple(excl)))

    def _sigma(self, node_nm: float, excl: Tuple[str, ...]) -> float:
        low, high = self.range_nm()
        if not node_in_range(node_nm, low, high):
            raise AreaRuleRefusal(f"sigma({node_nm:g} nm): the node lies outside the published 6T series ({low:g} to "
                                  f"{high:g} nm) and the description declares no bitcell of its own")
        points = []
        for node in self.files["node_series"]["nodes"]:
            values = []
            for cell in node["cells"]:
                try:
                    values.append(self.object_value(cell, excl, "node cell"))
                except HeldOut:
                    continue
            if values:
                points.append((node["node_nm"], node_reference(values)))
        exact = [a for n, a in points if n == node_nm]
        if exact:
            return exact[u.FIRST]
        below = [p for p in points if p[u.FIRST] < node_nm]
        above = [p for p in points if p[u.FIRST] > node_nm]
        if below and above:  # inside the range: the adjacent nodes
            (n0, a0), (n1, a1) = max(below), min(above)
        elif above:  # below the series (only where rule B is lifted): the first segment's slope
            (n0, a0), (n1, a1), *_rest = points
        else:  # above the series (only where rule B is lifted): the last segment's slope
            *_rest, (n0, a0), (n1, a1) = points
        slope = math.log(a1 / a0) / math.log(n1 / n0)
        return a0 * (node_nm / n0) ** slope

    # ------------------------------------------------------------------ tables
    def band_object(self, name: str) -> Mapping[str, Any]:
        return self.files["bands"]["bands"][name]

    def band_of_type(self, type_name: str) -> Tuple[float, float]:
        low, high = self.band_object(self.files["bands"]["type_band"][type_name])["value"]
        return low, high

    def band_points(self) -> Tuple[Mapping[str, Any], ...]:
        return tuple(self.files["bands"]["points"])

    def band_held_out_factors(self, name: str) -> Tuple[float, float]:
        types = self.band_object(name)["recompute"].get("types")
        values = [p["held_out"] for p in self.band_points() if p["held_out"] is not None
                  and (types is None or p["type"] in types)]
        return u.UNITY / max(values), u.UNITY / min(values)

    def converter_kind(self, kind: str) -> Mapping[str, Any]:
        kinds = self.files["converter_kinds"]["kinds"]
        if not isinstance(kind, str) or kind not in kinds:
            raise AreaRuleRefusal(f"converter kind {kind!r} is not one of {sorted(kinds)}")
        return kinds[kind]

    def cell_kind(self, kind: str) -> Mapping[str, Any]:
        kinds = self.files["cell_kinds"]["kinds"]
        if not isinstance(kind, str) or kind not in kinds:
            raise AreaRuleRefusal(f"cell kind {kind!r} is not one of {sorted(kinds)}")
        return kinds[kind]

    def published_points(self) -> Tuple[Mapping[str, Any], ...]:
        return self.points

    # ------------------------------------------------------------------ the evidence check (T-9)
    def walk(self) -> List[Tuple[str, Mapping[str, Any]]]:
        """(path, value object) of every value object of the data files."""
        found: List[Tuple[str, Mapping[str, Any]]] = []

        def visit(obj: Any, name: str, dotted: str) -> None:
            if isinstance(obj, Mapping):
                if "evidence_kind" in obj:
                    found.append((f"{name}:{dotted}", obj))
                for key, value in obj.items():
                    if key not in ("recompute", "citation", "entry_value", "held_out"):
                        visit(value, name, f"{dotted}.{key}" if dotted else str(key))
            elif isinstance(obj, list):
                for i, value in enumerate(obj):
                    visit(value, name, f"{dotted}.{i}")
        for name in self.files:
            visit(self.files[name], name, "")
        return found

    def _ids(self, obj: Mapping[str, Any]) -> Tuple[str, ...]:
        if obj["evidence_kind"] == "published":
            return (obj["entry_value"]["id"],)
        recompute = obj.get("recompute") or {}
        expressions = ([recompute["expression"]] if "expression" in recompute else
                       [p["expression"] for p in recompute.get("points") or () if isinstance(p, Mapping)])
        return tuple(i for e in expressions + list(recompute.get("bits") or ()) for i in read_ids(e))

    def unknown_citations(self) -> Tuple[str, ...]:
        return unknown_citations(self)


@lru_cache(maxsize=None)
def load_registry() -> Registry:
    """The shipped registry (cached)."""
    return Registry.load(DATA)


from mimarsinan.deployment_record.platform_physics.area_rule.check import (  # noqa: E402
    RegistryCheck, check_registry, unknown_citations)
from mimarsinan.deployment_record.platform_physics.area_rule.view import View  # noqa: E402

__all__ = ["DATA", "Registry", "RegistryCheck", "View", "check_registry", "load_registry", "node_in_range",
           "node_reference", "type_files"]
