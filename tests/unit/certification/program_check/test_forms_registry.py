"""[W15] The registry of program forms names every attribute it knows, each with the predicate that reads it or an
explicit entry that no executor reads it: a field a unit adds to a program dataclass fails by name, a registered form
no predicate reads fails by name, and every element class is scanned (every axon source, every I/O slice)."""

from __future__ import annotations

import copy
import dataclasses

import pytest

from mimarsinan.code_generation.cpp_chip_model import SpikeSource

from unit.certification.program_check import programs as P


def _add_dataclass_field(monkeypatch, cls, name: str) -> None:
    """``name`` becomes a dataclass field of ``cls`` for the test (as a unit's patch adds one)."""
    fields = dict(cls.__dataclass_fields__)
    added = dataclasses.field(default=None)
    added.name, added.type = name, "object"
    added._field_type = dataclasses._FIELD  # type: ignore[attr-defined]
    fields[name] = added
    monkeypatch.setattr(cls, "__dataclass_fields__", fields)


@pytest.mark.parametrize("cls_path,attribute", [
    ("mimarsinan.mapping.ir:IRGraph", "zz_unit_graph_field"),
    ("mimarsinan.mapping.packing.hybrid_types:HybridStage", "lane_owner"),
])
def test_a_dataclass_field_a_unit_adds_fails_by_name(monkeypatch, cls_path, attribute):
    import importlib

    module, name = cls_path.split(":")
    cls = getattr(importlib.import_module(module), name)
    program, ir = P.streamed_multipass()
    bad, ir = copy.deepcopy(program), copy.deepcopy(ir)
    _add_dataclass_field(monkeypatch, cls, attribute)
    if cls.__name__ == "IRGraph":
        setattr(ir, attribute, {})
    else:
        setattr(P.neural(bad)[0], attribute, 3)
    report = P.run(bad, ir, P.decl("streamed_lif"))
    assert report.theorem1.startswith("failed: unknown program form"), report.theorem1
    assert attribute in report.theorem1


def test_a_registered_form_no_predicate_reads_still_fails_by_name():
    from mimarsinan.certification.program_check import ProgramForm, register_program_form
    from mimarsinan.certification.program_check.forms import unregister_program_form

    program, ir = P.streamed_multipass()
    bad = copy.deepcopy(program)
    setattr(P.neural(bad)[0].hard_core_mapping.cores[0], "accepted_roles", ("operand",))
    register_program_form(ProgramForm("test_roles", "hard_core", frozenset({"accepted_roles"})))
    try:
        verdict = P.run(bad, ir, P.decl("streamed_lif")).theorem1
        assert verdict.startswith("failed: unread program form") and "hard_core.accepted_roles" in verdict, verdict
    finally:
        unregister_program_form("test_roles")


def test_a_registered_form_with_a_predicate_is_read_by_it():
    from mimarsinan.certification.program_check import ProgramForm, register_program_form
    from mimarsinan.certification.program_check.forms import unregister_program_form
    from mimarsinan.certification.program_check.report import Violation

    def roles_are_static(program, delta, iota, declaration):
        return [Violation("tiles", "operand_role", "core", "an operand core's weights are written per inference")
                for unit in delta.units for cv in unit.cores if getattr(cv.core, "accepted_roles", None)]

    program, ir = P.streamed_multipass()
    bad = copy.deepcopy(program)
    setattr(P.neural(bad)[0].hard_core_mapping.cores[0], "accepted_roles", ("operand",))
    register_program_form(ProgramForm("test_roles", "hard_core", frozenset({"accepted_roles"}), condition="tiles",
                                      predicate=roles_are_static))
    try:
        report = P.run(bad, ir, P.decl("streamed_lif"))
        assert report.theorem1.startswith("failed: tiles"), report.theorem1
        assert "operand_role" in {v.rule for v in report.results["tiles"].violations}
    finally:
        unregister_program_form("test_roles")


def test_an_axon_source_of_another_class_past_the_first_row_fails_by_name():
    class LaneSource(SpikeSource):
        pass

    program, ir = P.streamed_multipass()
    bad = copy.deepcopy(program)
    core = P.neural(bad)[0].hard_core_mapping.cores[0]
    row = len(core.axon_sources) - 1
    src = core.axon_sources[row]
    core.axon_sources[row] = LaneSource(src.core_, src.neuron_, is_input=src.is_input_, is_off=src.is_off_,
                                        is_always_on=src.is_always_on_)
    core._axon_source_spans = None
    verdict = P.run(bad, ir, P.decl("streamed_lif")).theorem1
    assert verdict.startswith("failed: unknown program form") and "LaneSource" in verdict, verdict


# ── the read-coverage ratchets ──────────────────────────────────────────────────────────────────────────────────
# The first holds each attribute's marker to the executors' reads: an attribute a reference executor or a shared
# accessor reads is registered with a predicate reader, ``label`` or ``identity`` (never ``unread`` or ``backend``), and
# one a backend reads is never ``unread``. The second holds each predicate reader to the check's reads: an attribute
# registered with a predicate reader is read by the check's own sources (its package without the registry of forms),
# by a name table the check reads by, or by a shared accessor the check imports or calls. Both cover the program,
# stage, core mapping, hard core and placement elements (``PROGRAM_ELEMENTS``) and match by attribute name (not by
# element or condition); a name is a read when the scanned code accesses it (attribute, ``getattr``, ``.get`` or a
# subscript), and a spelled name counts only inside a declared name table (``NAME_TABLES``). [U-CG] The third holds the
# markers to the check's reads: an attribute or record key the check reads is never registered ``unread`` or
# ``backend``; a placement's keys are matched through the record they are read from (``_placement_record_reads``).
import ast  # noqa: E402
from pathlib import Path  # noqa: E402

SRC = Path(__file__).resolve().parents[4] / "src" / "mimarsinan"
#: The modules that execute a program: the reference executors Theorem 1's observable is read on (the HCM and its stage
#: loop, the TTFS and value executors) and the backends tied to them by parity (nevresim's emit and export, SANA-FE,
#: Lava); and the accessors they call to read it (whole modules, or one function of a module).
REFERENCE_EXECUTORS = ("models/spiking/hybrid", "models/spiking/cycle_policy.py", "models/spiking/wire_semantics.py",
                       "chip_simulation/hybrid_run", "spiking/segment_boundary.py", "spiking/compute_boundary.py",
                       "chip_simulation/ttfs", "chip_simulation/value_run")
BACKEND_EXECUTORS = ("chip_simulation/simulation_runner", "chip_simulation/nevresim", "code_generation",
                     "mapping/export/chip_export.py", "chip_simulation/sanafe", "chip_simulation/lava_loihi")
ACCESSORS = ("mapping/packing/pass_timing.py", "mapping/packing/phase_channels.py:channels_of",
             "mapping/packing/phase_channels.py:stage_input_channels", "mapping/support/schedule/pass_carry.py", "mapping/latency/chip.py", "mapping/latency/core_graph.py",
             "mapping/support/spike_source_spans.py", "mapping/packing/executed_reads.py",
             "mapping/packing/softcore/hard_core.py:HardCore.get_axon_source_spans",
             "mapping/packing/softcore/hard_core.py:HardCore.get_core_matrix",
             "mapping/packing/softcore/hard_core_mapping.py:HardCoreMapping.get_output_source_spans")
#: The check's package; ``forms/`` (the registry table under test and its scanner) is no source of the check's reads.
CHECK_PACKAGE, CHECK_EXCLUDED = "certification/program_check", ("forms",)
#: The name tables the check reads attributes by (``getattr`` over the table's names), each with the function that
#: reads by it. Their string constants are the only spelled names the second ratchet counts as reads.
NAME_TABLES = {
    "mapping/platform/core_residency.py:CORE_SINGLETON_PROPERTIES": "certification/program_check/tiles.py:sweep",
    "certification/program_check/tiles.py:NODE_MAPS": "certification/program_check/tiles.py:wiring",
    "mapping/packing/executed_reads.py:MEMOS": "mapping/packing/executed_reads.py:stale_memos",
}
PROGRAM_ELEMENTS = ("program", "stage", "core_mapping", "hard_core", "placement")


def _constants() -> dict:
    """Module-level string constants of the source tree, by name (a name bound to two values is dropped)."""
    seen: dict = {}
    for path in SRC.rglob("*.py"):
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value,
                                                                                                      str):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        seen.setdefault(target.id, set()).add(node.value.value)
    return {k: next(iter(v)) for k, v in seen.items() if len(v) == 1}


def _child(node, part: str):
    """The class, function or module-level assignment named ``part`` directly under ``node``."""
    for child in ast.iter_child_nodes(node):
        targets = (child.targets if isinstance(child, ast.Assign) else
                   [child.target] if isinstance(child, ast.AnnAssign) else [])
        if getattr(child, "name", None) == part or any(isinstance(t, ast.Name) and t.id == part for t in targets):
            return child
    raise LookupError(f"{part} is not defined under {type(node).__name__}")


def _trees(spec: str):
    path, _, qualname = spec.partition(":")
    root = SRC / path
    files = [root] if root.suffix == ".py" else sorted(root.rglob("*.py"))
    for file in files:
        tree = ast.parse(file.read_text())
        if not qualname:
            yield tree
            continue
        node = tree
        for part in qualname.split("."):
            node = _child(node, part)
        yield node


def _check_sources() -> tuple:
    """The check's own sources: every module of its package outside ``CHECK_EXCLUDED``."""
    root = SRC / CHECK_PACKAGE
    return tuple(str(p.relative_to(SRC)) for p in sorted(root.rglob("*.py"))
                 if not set(p.relative_to(root).parts[:-1]) & set(CHECK_EXCLUDED))


def _check_accessors() -> tuple:
    """The shared accessors the check's sources import (the module) or call (the method)."""
    imported, called = set(), set()
    for spec in _check_sources():
        for tree in _trees(spec):
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module)
                    imported.update(f"{node.module}.{alias.name}" for alias in node.names)
                elif isinstance(node, ast.Attribute):
                    called.add(node.attr)
    out = []
    for spec in ACCESSORS:
        path, _, qualname = spec.partition(":")
        module = "mimarsinan." + path.removesuffix(".py").replace("/", ".")
        if (qualname.split(".")[-1] in called) if qualname else module in imported:
            out.append(spec)
    return tuple(out)


def _read_names(specs, *, named: bool = False) -> set:
    """Attribute names read in ``specs``: attribute access, ``getattr`` and ``.get`` with a literal (or a module
    constant), and subscripts; with ``named``, every string constant too (a table of names the code reads by)."""
    constants = _constants()
    out = set()
    for spec in specs:
        for tree in _trees(spec):
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute):
                    out.add(node.attr)
                elif isinstance(node, ast.Call) and node.args:
                    func = node.func
                    name = getattr(func, "id", None) or getattr(func, "attr", None)
                    arg = node.args[1] if name in ("getattr", "hasattr", "setattr") and len(node.args) > 1 else (
                        node.args[0] if name in ("get", "setdefault", "pop") else None)
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        out.add(arg.value)
                    elif isinstance(arg, ast.Name) and arg.id in constants:
                        out.add(constants[arg.id])
                elif isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) and isinstance(
                        node.slice.value, str):
                    out.add(node.slice.value)
                elif named and isinstance(node, ast.Constant) and isinstance(node.value, str):
                    out.add(node.value)
    return out


def _check_reads() -> set:
    return (_read_names(_check_sources()) | _read_names(tuple(NAME_TABLES), named=True)
            | _read_names(_check_accessors()))


def test_every_attribute_an_executor_reads_is_read_by_a_predicate():
    """A reference executor's read is registered with a predicate reader (or as a label or an identity read), never
    ``unread`` or ``backend``; a backend's read is never ``unread``."""
    from mimarsinan.certification.program_check.forms import BACKEND, LABEL, UNREAD, registered_attributes

    reference = _read_names(REFERENCE_EXECUTORS + ACCESSORS)
    backends = _read_names(BACKEND_EXECUTORS)
    known = registered_attributes()
    wrong = []
    for element in PROGRAM_ELEMENTS:
        for attribute, readers in known[element].items():
            assert readers and (len(readers) == 1 or not set(readers) & {UNREAD, LABEL, BACKEND}), (element,
                                                                                                    attribute)
            if attribute in reference and readers in ((UNREAD,), (BACKEND,)):
                wrong.append(f"{element}.{attribute} (a reference executor reads it)")
            elif attribute in backends and readers == (UNREAD,):
                wrong.append(f"{element}.{attribute} (a backend reads it)")
    assert not wrong, f"registered as read by no predicate, yet an executor reads them: {wrong}"


def test_every_attribute_a_predicate_claims_is_read_by_the_check():
    """An attribute registered with a predicate reader is read by the check's own sources (not by the registry table
    that names it), by a name table the check reads by, or by a shared accessor the check imports."""
    from mimarsinan.certification.program_check.forms import BACKEND, LABEL, UNREAD, registered_attributes

    checked = _check_reads()
    missing = [f"{element}.{attribute}" for element in PROGRAM_ELEMENTS
               for attribute, readers in registered_attributes()[element].items()
               if readers not in ((UNREAD,), (LABEL,), (BACKEND,)) and attribute not in checked]
    assert not missing, f"a predicate is said to read them, but the check never does: {missing}"


def _placement_record_reads() -> dict:
    """The keys the check's own sources read from a placement's record (``<placement>.record`` or a parameter named
    ``record``: ``.get``, ``.setdefault`` or ``.pop`` with a literal or a module constant, or a literal subscript), each
    with the places that read it. Matched through the record, not by name: an IR node carries attributes of the same
    names (``coalescing_role``, ``psum_role``) that the check reads off nodes."""
    constants = _constants()

    def is_record(expr) -> bool:
        return ((isinstance(expr, ast.Attribute) and expr.attr == "record")
                or (isinstance(expr, ast.Name) and expr.id == "record"))

    out: dict = {}
    for spec in _check_sources():
        for tree in _trees(spec):
            for node in ast.walk(tree):
                key = None
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.args
                        and node.func.attr in ("get", "setdefault", "pop") and is_record(node.func.value)):
                    arg = node.args[0]
                    key = arg.value if isinstance(arg, ast.Constant) else constants.get(getattr(arg, "id", ""))
                elif isinstance(node, ast.Subscript) and is_record(node.value) and isinstance(node.slice, ast.Constant):
                    key = node.slice.value
                if isinstance(key, str):
                    out.setdefault(key, []).append(f"{spec}:{node.lineno}")
    return out


def _unread_yet_checked(known, placement_reads: dict, checked: set) -> list:
    """The third ratchet's rule over ``known`` (``registered_attributes()``): each placement key read from a record, and
    each other element's attribute the check reads, registered ``unread`` or ``backend``."""
    from mimarsinan.certification.program_check.forms import BACKEND, UNREAD

    wrong = [f"placement.{key} (read at {', '.join(places)})" for key, places in sorted(placement_reads.items())
             if known["placement"].get(key) in ((UNREAD,), (BACKEND,))]
    return wrong + [f"{element}.{attribute}" for element in PROGRAM_ELEMENTS if element != "placement"
                    for attribute, readers in sorted(known[element].items())
                    if readers in ((UNREAD,), (BACKEND,)) and attribute in checked]


def test_every_attribute_the_check_reads_is_registered_with_a_reader():
    """The third direction: an attribute or record key the check reads is never registered ``unread`` or ``backend``
    (read by no predicate), so a predicate whose verdict depends on a key fails by name until the key names its
    condition. Placement keys are those read from a placement's record; the other elements' attributes match by name
    against the check's reads (its own sources, the name tables it reads by, the shared accessors it calls)."""
    from mimarsinan.certification.program_check.forms import registered_attributes

    placement_reads = _placement_record_reads()
    assert placement_reads, "no read of a placement's record found in the check's sources: the scan is vacuous"
    wrong = _unread_yet_checked(registered_attributes(), placement_reads, _check_reads())
    assert not wrong, f"registered as read by no predicate, yet the check reads them: {wrong}"


def test_the_third_ratchet_fails_a_check_read_registered_unread():
    """Self-test of the third ratchet: a placement key the check reads from a record, or a stage attribute it reads,
    registered ``unread``, fails by name."""
    from mimarsinan.certification.program_check.forms import UNREAD, registered_attributes

    placement_reads, checked = _placement_record_reads(), _check_reads()
    stage = registered_attributes()["stage"]
    for element, name in (("placement", sorted(placement_reads)[0]),
                          ("stage", sorted(a for a, r in stage.items() if r != (UNREAD,) and a in checked)[0])):
        known = registered_attributes()
        known[element][name] = (UNREAD,)
        wrong = _unread_yet_checked(known, placement_reads, checked)
        assert any(w.split(" ")[0] == f"{element}.{name}" for w in wrong), (element, name, wrong)


def _reads_by(table: str, reader: str) -> bool:
    """``reader`` names ``table`` (it reads attributes by the table's names) and is the check's own or called by it."""
    name, function = table.partition(":")[2], reader.partition(":")[2].split(".")[-1]
    if not any(isinstance(n, ast.Name) and n.id == name for tree in _trees(reader) for n in ast.walk(tree)):
        return False
    if reader.partition(":")[0] in _check_sources():
        return True
    return any(getattr(n, "name" if isinstance(n, ast.alias) else "attr", None) == function
               for spec in _check_sources() for tree in _trees(spec) for n in ast.walk(tree)
               if isinstance(n, (ast.alias, ast.Attribute)))


def test_every_name_table_is_read_by_the_check_through_its_reader():
    unread = [f"{table} by {reader}" for table, reader in NAME_TABLES.items() if not _reads_by(table, reader)]
    assert not unread, f"declared name tables the check does not read by: {unread}"
    assert not _reads_by("mapping/platform/core_residency.py:CORE_SINGLETON_PROPERTIES",
                         "certification/program_check/tiles.py:wiring")
    assert not _reads_by("mapping/packing/executed_reads.py:MEMOS", "mapping/packing/executed_reads.py:bank_read")


def _fabricated_tree(tmp_path, attribute: str):
    """A copy of the source tree (``.py`` files) whose base table registers ``stage.<attribute>`` with a predicate
    reader (``tiles``) and whose reference executor reads it, while no predicate does: the B2, B3 and M1 class."""
    import shutil

    def keep_python(directory, names):
        return [n for n in names
                if n == "__pycache__" or not (n.endswith(".py") or (Path(directory) / n).is_dir())]

    src = tmp_path / "mimarsinan"
    shutil.copytree(SRC, src, ignore=keep_python)
    base = src / "certification/program_check/forms/base.py"
    text = base.read_text()
    head = "STAGE: Dict[str, Readers] = {"
    assert head in text
    base.write_text(text.replace(head, f'{head}\n    "{attribute}": _TILES,', 1))
    with (src / "models/spiking/wire_semantics.py").open("a") as handle:
        handle.write(f"\n\ndef _fabricated_read(stage):\n    return stage.{attribute}\n")
    return src


def test_the_ratchets_fail_an_executor_read_attribute_no_predicate_reads(tmp_path, monkeypatch):
    """Self-test of the read-coverage ratchets: an attribute a reference executor reads, registered in the base table
    under a predicate's condition that no predicate reads, passes the first ratchet (it names a predicate reader) and
    fails the second by name; registered ``unread``, it fails the first. The base table is not a source of the check:
    naming an attribute there is no read."""
    import sys

    from mimarsinan.certification.program_check.forms import UNREAD
    from mimarsinan.certification.program_check.forms import base as B

    attribute = "zz_fabricated_executor_attribute"
    monkeypatch.setattr(sys.modules[__name__], "SRC", _fabricated_tree(tmp_path, attribute))
    assert attribute in _read_names(REFERENCE_EXECUTORS)
    monkeypatch.setitem(B.STAGE, attribute, ("tiles",))
    test_every_attribute_an_executor_reads_is_read_by_a_predicate()
    with pytest.raises(AssertionError, match=f"stage.{attribute}"):
        test_every_attribute_a_predicate_claims_is_read_by_the_check()
    monkeypatch.setitem(B.STAGE, attribute, (UNREAD,))
    with pytest.raises(AssertionError, match=f"stage.{attribute}"):
        test_every_attribute_an_executor_reads_is_read_by_a_predicate()
