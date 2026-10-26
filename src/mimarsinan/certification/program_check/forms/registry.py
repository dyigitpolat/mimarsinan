"""[W15] The registry of program forms the static check knows: per program element, the classes (or stage kinds) it
may hold and every attribute (record key) by name, each with the conditions whose predicates read it or a marker that
no reference executor reads it (``forms.base``).

The check reads a program only through these forms. An element of an unregistered class, kind or attribute fails the
check by name (``failed: unknown program form ...``), never a silent pass. A unit that adds a form registers it with
``register_program_form`` and either a predicate (it reads the form and returns violations under its condition) or an
``unread_reason`` (no executor reads it, a claim ``test_forms_registry``'s ratchet tests); a form registered with
neither fails by name too (``failed: unread program form ...``)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Mapping, Optional, Tuple

import numpy as np

from mimarsinan.certification.program_check.forms import base as B
from mimarsinan.certification.program_check.report import Violation

ELEMENTS = tuple(B.ATTRIBUTES)
#: A form's predicate: ``(program, delta view, identity view, declaration) -> violations`` under its condition.
FormPredicate = Callable[[Any, Any, Any, Any], List[Violation]]


@dataclass(frozen=True)
class ProgramForm:
    """A form a unit adds: the element it extends, the attributes and classes (stage kinds) it declares, and how the
    check reads it: ``predicate`` (reported under ``condition``), or ``unread_reason`` (no executor reads it)."""

    name: str
    element: str
    attributes: FrozenSet[str] = frozenset()
    classes: FrozenSet[str] = frozenset()
    condition: str = ""
    predicate: Optional[FormPredicate] = None
    unread_reason: str = ""

    def __post_init__(self) -> None:
        if self.element not in ELEMENTS:
            raise ValueError(f"ProgramForm {self.name!r}: element {self.element!r} is none of {ELEMENTS}")
        if self.predicate is not None and not self.condition:
            raise ValueError(f"ProgramForm {self.name!r}: a predicate reports under a condition; name it")
        if self.predicate is not None and self.unread_reason:
            raise ValueError(f"ProgramForm {self.name!r}: read by a predicate or by no executor, not both")

    @property
    def read(self) -> bool:
        return self.predicate is not None or bool(self.unread_reason)

    def readers(self) -> B.Readers:
        if self.predicate is not None:
            return (self.condition,)
        return (B.UNREAD,) if self.unread_reason else ()


_EXTRA: List[ProgramForm] = []


def register_program_form(form: ProgramForm) -> None:
    """Register a form a unit adds (the integration step)."""
    if any(f.name == form.name for f in _EXTRA):
        raise ValueError(f"program form {form.name!r} is registered already")
    _EXTRA.append(form)


def unregister_program_form(name: str) -> None:
    _EXTRA[:] = [f for f in _EXTRA if f.name != name]


def extra_forms() -> List[ProgramForm]:
    return list(_EXTRA)


def registered_attributes() -> Dict[str, Dict[str, B.Readers]]:
    """Per element, every known attribute and its readers (a registered form without a predicate reads ``()``)."""
    out = {element: dict(attrs) for element, attrs in B.ATTRIBUTES.items()}
    for form in _EXTRA:
        for attribute in form.attributes:
            out[form.element][attribute] = tuple(out[form.element].get(attribute, ())) + form.readers()
    return out


def _known() -> Dict[str, Tuple[Mapping[str, B.Readers], FrozenSet[str]]]:
    attrs = registered_attributes()
    classes = {e: set(c) for e, c in B.CLASSES.items()}
    for form in _EXTRA:
        classes[form.element] |= set(form.classes)
    return {e: (attrs[e], frozenset(classes[e])) for e in ELEMENTS}


class _Scan:
    def __init__(self) -> None:
        self.known = _known()
        self.unread = {(f.element, a) for f in _EXTRA if not f.read for a in f.attributes}
        self.found: List[Violation] = []
        self.seen: set = set()

    def element(self, element: str, obj: Any, where: str, *, keys: Iterable[str] = (), kind: str = "") -> None:
        attrs, classes = self.known[element]
        label = kind or type(obj).__name__
        if classes and label not in classes:
            self._add("forms", element, f"{element}:{label}", where, f"{element} of class or kind {label!r}")
        for key in keys:
            if (element, key) in self.unread:
                self._add("forms_unread", element, f"{element}.{key}", where,
                          f"{element} attribute {key!r}: registered, but no predicate reads it and no executor is "
                          "declared not to")
            elif key not in attrs or not attrs[key]:
                self._add("forms", element, f"{element}.{key}", where, f"{element} attribute {key!r}")

    def classes_of(self, element: str, items, where: str) -> None:
        """Every distinct class among ``items`` (every axon source, slice or source, not the first only)."""
        for cls in set(map(type, items)):
            self.element(element, None, where, kind=cls.__name__)

    def _add(self, condition: str, element: str, rule: str, where: str, what: str) -> None:
        if rule in self.seen:
            return
        self.seen.add(rule)
        tail = "" if condition == "forms_unread" else ": no registered program form declares it"
        self.found.append(Violation(condition, rule, where, what + tail))


def _flat(values) -> list:
    return list(np.asarray(values, dtype=object).flatten()) if values is not None else []


def _scan_unit(scan: _Scan, unit: Any, where: str) -> None:
    scan.element("stage", unit, where, keys=vars(unit), kind=str(getattr(unit, "kind", "?")))
    scan.classes_of("io_slice", list(unit.input_map) + list(unit.output_map), where)
    for s in list(unit.input_map) + list(unit.output_map):
        scan.element("io_slice", s, where, keys=vars(s))
    if getattr(unit, "kind", None) == "compute":
        op = unit.compute_op
        if op is not None:
            scan.element("ir_node", op, where, keys=vars(op))
            scan.classes_of("ir_source", _flat(op.input_sources), where)
        return
    hcm = getattr(unit, "hard_core_mapping", None)
    if hcm is None:
        return
    scan.element("core_mapping", hcm, where, keys=vars(hcm))
    scan.classes_of("axon_source", _flat(hcm.output_sources), f"{where} outputs")
    for h, core in enumerate(hcm.cores):
        scan.element("hard_core", core, f"{where} core {h}", keys=vars(core))
        scan.classes_of("matrix_placement", list(getattr(core, "matrix_placements", None) or ()), f"{where} core {h}")
        scan.classes_of("axon_source", core.axon_sources, f"{where} core {h}")
    for h, placements in enumerate(getattr(hcm, "soft_core_placements_per_hard_core", None) or []):
        for p in placements:
            scan.element("placement", p, f"{where} core {h}", keys=list(dict(p)))


def scan_forms(program: Any, ir_graph: Any = None) -> List[Violation]:
    """Every program element no registered form declares (condition ``forms``) and every registered attribute that
    no predicate reads (``forms_unread``), each named once (its first occurrence)."""
    scan = _Scan()
    scan.element("program", program, "program", keys=vars(program))
    scan.classes_of("ir_source", _flat(getattr(program, "output_sources", None)), "program outputs")
    for s, stage in enumerate(getattr(program, "stages", ()) or ()):
        units = [("", stage)] + [(f" level {u}", lv) for u, lv in
                                 enumerate(getattr(stage, "retimed_level_stages", None) or ())]
        for suffix, unit in units:
            _scan_unit(scan, unit, f"stage {s}{suffix} ({getattr(unit, 'name', '?')})")
    if ir_graph is not None:
        scan.element("ir_graph", ir_graph, "IR graph", keys=vars(ir_graph))
        for node in getattr(ir_graph, "nodes", None) or ():
            where = f"IR node {getattr(node, 'id', '?')}"
            scan.element("ir_node", node, where, keys=vars(node))
    return scan.found


def unknown_forms(program: Any, ir_graph: Any = None) -> List[Violation]:
    return [v for v in scan_forms(program, ir_graph) if v.condition == "forms"]


def run_form_predicates(program: Any, delta: Any, iota: Any, declaration: Any, results: Mapping[str, Any]) -> None:
    """Every registered form's predicate, its violations under its condition."""
    for form in _EXTRA:
        if form.predicate is None:
            continue
        result = results[form.condition]
        result.checked += 1
        for violation in form.predicate(program, delta, iota, declaration):
            result.add(violation)
