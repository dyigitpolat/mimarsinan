"""The requirement table: one op-kind-to-capability-class map, its drift against the scout oracle, its refusals."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.capability.requirements import (
    CAPABILITY_CLASSES,
    CROSSBAR,
    HOST_ONLY,
    LANE_CLASSES,
    PLACEMENTS,
    REQUIREMENTS,
    WORK_RULES,
    UnknownOpKindError,
    requirement_of,
)

REPO = Path(__file__).resolve().parents[4]
SRC = REPO / "src" / "mimarsinan"
FIXTURES = REPO / "tests" / "fixtures" / "requirement_census"
TABLE_MODULE = "mapping/capability/requirements.py"

#: Exempt by name until CL4, when G1's conversion predicate reads this table (plan §11.3 acceptance 2, §13 rule 6).
SECOND_MAP_EXEMPTIONS = frozenset({"core_lowering_requirement"})


def _oracle_map() -> dict:
    return json.loads((FIXTURES / "oracle_kind_map.json").read_text())


def test_the_oracle_fixture_is_the_pinned_copy():
    assert _oracle_map()["source_sha256_16"] == "296ff421dcc7088b"


def test_toolchain_table_equals_the_oracle_map_on_every_kind_the_oracle_uses():
    drift = {
        kind: (sorted(classes), sorted(REQUIREMENTS[kind].classes) if kind in REQUIREMENTS else None)
        for kind, classes in _oracle_map()["requirements"].items()
        if kind not in REQUIREMENTS or frozenset(classes) != REQUIREMENTS[kind].classes
    }
    assert not drift, f"oracle kind -> (oracle classes, toolchain classes): {drift}"


def test_every_row_is_well_formed():
    for kind, requirement in REQUIREMENTS.items():
        assert requirement.kind == kind
        assert requirement.classes <= frozenset(CAPABILITY_CLASSES), kind
        assert requirement.placement in PLACEMENTS, kind
        assert requirement.work in WORK_RULES, kind
        assert requirement.reason.strip(), f"{kind} carries no named reason"


def test_host_only_kinds_declare_no_class_and_are_never_met():
    host_only = [r for r in REQUIREMENTS.values() if r.placement == HOST_ONLY]
    assert host_only, "the table names its host-only kinds"
    every_class = frozenset(CAPABILITY_CLASSES)
    for requirement in host_only:
        assert not requirement.classes
        assert not requirement.met_by(every_class)


def test_lane_classes_are_the_arithmetic_toggle_lane():
    assert LANE_CLASSES == frozenset("LAEPC")
    assert {r.kind for r in REQUIREMENTS.values() if r.is_lane} == {
        "clamp", "affine", "add", "pool", "maxp"}


def test_a_seeded_unknown_kind_refuses_by_name():
    with pytest.raises(UnknownOpKindError) as caught:
        requirement_of("frobnicate")
    assert isinstance(caught.value, ToolchainRefusal)
    assert isinstance(caught.value, KeyError)
    assert "frobnicate" in str(caught.value)


def _module_names(path: Path) -> str:
    return path.relative_to(SRC).as_posix()


def _class_letter_set(node: ast.AST) -> bool:
    """A literal that spells a set of capability classes: a str of class letters or a set/frozenset of them."""
    letters = set(CAPABILITY_CLASSES)
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return 1 <= len(node.value) <= len(letters) and set(node.value) <= letters and node.value.isupper()
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("set", "frozenset"):
        return len(node.args) == 1 and _class_letter_set(node.args[0])
    if isinstance(node, (ast.Set, ast.List, ast.Tuple)):
        return bool(node.elts) and all(
            isinstance(e, ast.Constant) and isinstance(e.value, str) and e.value in letters for e in node.elts)
    return False


def _returns_class_letters(function: ast.AST) -> bool:
    return any(isinstance(node, ast.Return) and node.value is not None and _class_letter_set(node.value)
               for node in ast.walk(function))


def _class_valued_entries(node: ast.AST) -> int:
    """Entries of a map literal whose value spells a class set: a dict display, ``dict(k=...)`` or a pair sequence."""
    if isinstance(node, ast.Dict):
        return sum(1 for value in node.values if value is not None and _class_letter_set(value))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "dict":
        return sum(1 for keyword in node.keywords if _class_letter_set(keyword.value))
    if isinstance(node, (ast.Tuple, ast.List)):
        return sum(1 for e in node.elts
                   if isinstance(e, (ast.Tuple, ast.List)) and len(e.elts) == 2 and _class_letter_set(e.elts[1]))
    return 0


def _second_maps(tree: ast.AST) -> list:
    """Map literals with two or more class-set values, and any function (whatever its name) returning a class set."""
    found = []
    for node in ast.walk(tree):
        if _class_valued_entries(node) >= 2:
            found.append(f"{type(node).__name__} map at line {node.lineno}")
        if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name not in SECOND_MAP_EXEMPTIONS and _returns_class_letters(node)):
            found.append(f"function {node.name} at line {node.lineno}")
    return found


def test_no_second_op_kind_to_class_map_exists_in_src():
    offenders = {}
    for path in sorted(SRC.rglob("*.py")):
        name = _module_names(path)
        if name.startswith("mapping/capability/"):
            continue
        found = _second_maps(ast.parse(path.read_text()))
        if found:
            offenders[name] = found
    assert not offenders, f"a second op-kind-to-class map outside {TABLE_MODULE}: {offenders}"


def test_the_second_map_scan_sees_the_oracle_shape():
    oracle_shape = ast.parse('REQ = {"mm": set(), "mmg": {"G"}, "dmm": {"D"}, "norm": {"R", "T", "X", "A"}}')
    assert _second_maps(oracle_shape)
    assert _second_maps(ast.parse("def op_requirement(m):\n    return 'G'\n"))
    assert _second_maps(ast.parse('KIND_CLASSES = dict(mmg="G", dmm="D", norm="RTXA")'))
    assert _second_maps(ast.parse(
        "def needs_class(m):\n    if isinstance(m, Conv):\n        return frozenset('G')\n    return frozenset()\n"))
    assert _second_maps(ast.parse('KIND_CLASSES = (("mmg", "G"), ("dmm", {"D"}))'))
    assert not _second_maps(ast.parse("def core_lowering_requirement(m):\n    return 'G'\n"))
    assert not _second_maps(ast.parse('NAMES = dict(mmg="grouped", dmm="dynamic")'))


def test_the_mac_carrying_kinds_are_the_products():
    assert {kind for kind, r in REQUIREMENTS.items() if r.carries_macs} == {"mm", "mmg", "mms", "dmm", "leaf"}
    assert all(r.carries_macs for r in REQUIREMENTS.values() if r.placement == CROSSBAR)


def test_only_declared_kinds_carry_a_stated_host_boundary():
    boundaries = {kind: r.host_boundary for kind, r in REQUIREMENTS.items() if r.host_boundary}
    assert boundaries == {"mmg": "grouped convolution on the host until N5"}
