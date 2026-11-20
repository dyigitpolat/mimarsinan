"""[U-HOST2] The ratchets the verify asked to widen and close (m-2, m-5).

m-2: no module of the toolchain reads the record's sealed host-wall fields except through the one per-inference
function (``stage_timing.host_wall``). The sealed fields are the latency decomposition's host-wall fields, read from
the schema itself. Over all of src (the one function's module and the schema excepted): no attribute read of a sealed
field, no string constant naming one (a serialized record's field read, a getattr), and getattr-family reads of a
constant name counted on their own. A sealed field that shares its name with a quantity cannot pass this scan, which
is why the record names its raw field apart (m-3).

m-5: every site that derives a physics from another by ``replace(..., constants=...)`` is in a closed census, each with
the rows it may add, none of them a host row; a new such site fails here until it is swept.
"""

from __future__ import annotations

import ast
import dataclasses
import re
from pathlib import Path

import pytest

from mimarsinan.deployment_record.platform_physics.constants import spec_for
from mimarsinan.deployment_record.platform_physics.hosts.rows import HOST
from mimarsinan.deployment_record.schema.physics import LatencyDecomposition

SRC = Path(__file__).resolve().parents[3] / "src" / "mimarsinan"
#: The one function's module and the schema: the only modules that may name a sealed host-wall field.
EXEMPT = ("chip_simulation/hybrid_run/stage_timing.py", "deployment_record/schema/")
#: The sealed host-wall fields, read from the schema (every latency field whose name begins with the wall's stem).
FIELDS = tuple(f.name for f in dataclasses.fields(LatencyDecomposition) if f.name.startswith("host_ops_s"))
_ATTR_READERS = {"getattr", "hasattr", "setattr", "delattr"}


def _modules():
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        if not rel.startswith(EXEMPT):
            yield rel, ast.parse(path.read_text(encoding="utf-8"))


def _token(field):
    return re.compile(rf"(?<![A-Za-z0-9_]){re.escape(field)}(?![A-Za-z0-9_])")


class TestM2TheWallRatchetOverAllOfSrc:
    def test_the_sealed_fields_are_read_from_the_schema(self):
        assert len(FIELDS) == 2, FIELDS

    @pytest.mark.parametrize("field", FIELDS)
    def test_no_module_reads_a_sealed_field_by_attribute(self, field):
        hits = [f"{rel}:{node.lineno}" for rel, tree in _modules() for node in ast.walk(tree)
                if isinstance(node, ast.Attribute) and node.attr == field and isinstance(node.ctx, ast.Load)]
        assert hits == []

    @pytest.mark.parametrize("field", FIELDS)
    def test_no_module_names_a_sealed_field_in_a_string(self, field):
        """A serialized record's raw field read (``record["timing"]["latency"][<field>]``), a getattr of it, or any
        other string naming it, in any module of src but the one function's and the schema's."""
        token = _token(field)
        hits = [f"{rel}:{node.lineno}" for rel, tree in _modules() for node in ast.walk(tree)
                if isinstance(node, ast.Constant) and isinstance(node.value, str) and token.search(node.value)]
        assert hits == []

    @pytest.mark.parametrize("field", FIELDS)
    def test_getattr_reads_of_a_constant_name_are_counted(self, field):
        """The getattr family with a constant attribute name that is a sealed field: counted on its own (zero)."""
        hits = [f"{rel}:{node.lineno}" for rel, tree in _modules() for node in ast.walk(tree)
                if isinstance(node, ast.Call) and getattr(node.func, "id", None) in _ATTR_READERS
                and len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) and node.args[1].value == field]
        assert hits == []


def _enclosing(tree):
    """Each call node with the name of the function (or class.method) it sits in."""
    out = []

    def walk(node, where):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                walk(child, f"{where}.{child.name}" if where else child.name)
            elif isinstance(child, ast.ClassDef):
                walk(child, f"{where}.{child.name}" if where else child.name)
            else:
                if isinstance(child, ast.Call):
                    out.append((where, child))
                walk(child, where)

    walk(tree, "")
    return out


def _is_constants_replace(call):
    func = call.func
    name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
    return name == "replace" and any(k.arg == "constants" for k in call.keywords)


#: (module, function) -> the name of the row set the site may add, in the module that declares it.
CENSUS = {
    ("deployment_record/platform_physics/converter.py", "resolve"): ("converter", "CONVERTER_KEYS"),
    ("deployment_record/platform_physics/profile.py", "PlatformPhysics.at_carry_buffer"): (
        "profile", "CARRY_BUFFER_KEYS"),
    ("deployment_record/platform_physics/resolve.py", "resolve_platform_physics"): ("lane_blocks", "LANE_ROWS"),
    ("deployment_record/platform_physics/resolve.py", "with_write_technology"): ("resolve", "WRITE_TECHNOLOGY_KEYS"),
}


class TestM5TheDerivationSitesAreACensus:
    def test_every_replace_with_constants_site_is_in_the_census(self):
        found = sorted((path.relative_to(SRC).as_posix(), where) for path in sorted(SRC.rglob("*.py"))
                       for where, call in _enclosing(ast.parse(path.read_text(encoding="utf-8")))
                       if _is_constants_replace(call))
        assert found == sorted(CENSUS)

    @pytest.mark.parametrize("site", sorted(CENSUS))
    def test_each_site_adds_rows_of_its_own_set_and_no_host_row(self, site):
        from mimarsinan.deployment_record.platform_physics import converter, lane_blocks, profile, resolve

        module, name = CENSUS[site]
        rows = getattr({"converter": converter, "profile": profile, "lane_blocks": lane_blocks,
                        "resolve": resolve}[module], name)
        assert rows and not [key for key in rows if spec_for(key).group == HOST], site
