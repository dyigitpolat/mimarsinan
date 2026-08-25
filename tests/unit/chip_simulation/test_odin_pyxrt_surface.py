"""[ODIN P7b] the fake pyxrt may never be richer than the real one.

FIELD EVENT, 2026-08-25. A U250 loaded the package's xclbin, the CU came up, and
the shipped driver died on its first line of device work:

    AttributeError: 'pyxrt.kernel' object has no attribute 'read_register'

The fake the driver had been developed against carried a ``read_register`` the
real binding does not have. Nothing in the suite could notice, because the fake
was the only thing the driver was ever measured against.

This module closes that hole from the other side: the REAL surface is committed
as data (``pyxrt_surface_xrt_2024_2.json``, transcribed from the pybind11 module
definition, with its provenance in the file), and the fake's public API must be
a SUBSET of it. A name the fake invents fails here instead of on a card.
"""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List, Set

import pytest

REPO = Path(__file__).resolve().parents[3]
PACKAGE = REPO / "scripts" / "hacc" / "package"
FAKE_PATH = PACKAGE / "host" / "fake_pyxrt_for_selftest.py"
DRIVER_PATH = PACKAGE / "host" / "odin_board_driver.py"
TRANSPORT_PATH = (
    REPO / "src" / "mimarsinan" / "chip_simulation" / "odin_fpga"
    / "xrt_transport.py")
SURFACE_PATH = Path(__file__).with_name("pyxrt_surface_xrt_2024_2.json")

#: Names the real binding does not bind. The whole field failure is one of them.
FORBIDDEN = ("read_register", "write_register")


@pytest.fixture(scope="module")
def surface() -> Dict[str, Any]:
    return json.loads(SURFACE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def fake() -> Any:
    spec = importlib.util.spec_from_file_location("surface_fake_pyxrt", FAKE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _module_level_names(path: Path) -> Set[str]:
    """What a module DEFINES at module level — imports deliberately excluded."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: Set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return {name for name in names if not name.startswith("_")}


def _public_members(owner: Any) -> Set[str]:
    return {name for name in vars(owner) if not name.startswith("_")}


def _nested(fake: Any, dotted: str) -> Any:
    owner = fake
    for part in dotted.split("."):
        owner = getattr(owner, part)
    return owner


class TestTheProvenanceIsCarriedWithTheData:
    def test_the_surface_names_where_it_was_read_from(self, surface):
        provenance = surface["provenance"]
        assert provenance["repository"] == "github.com/Xilinx/XRT"
        assert provenance["branch"] == "2024.2"
        assert "pyxrt.cpp" in provenance["source"]

    def test_the_field_failure_is_recorded_as_absent_on_purpose(self, surface):
        absent = surface["provenance"]["absent_on_purpose"]
        assert "kernel.read_register" in absent
        assert "kernel.write_register" in absent


class TestTheFakeIsASubsetOfTheRealBinding:
    def test_every_module_level_name_is_real_or_declared_as_ours(
        self, fake, surface,
    ):
        declared_ours = set(fake.FAKE_ONLY)
        defined = _module_level_names(FAKE_PATH)
        invented = defined - set(surface["module"]) - declared_ours
        assert not invented, (
            f"the fake invents module-level names the real pyxrt has no such "
            f"thing as: {sorted(invented)}")

    def test_the_declared_extras_really_are_only_harness_hooks(self, fake, surface):
        # FAKE_ONLY is the escape hatch; it must stay small and never smuggle a
        # device call in. Nothing in it may look like part of the binding.
        assert set(fake.FAKE_ONLY) & set(surface["module"]) == set()
        for name in fake.FAKE_ONLY:
            assert hasattr(fake, name), f"FAKE_ONLY names {name}, which is absent"

    @pytest.mark.parametrize("dotted", [
        "uuid", "device", "run", "kernel", "kernel.cu_access_mode",
        "bo", "bo.flags", "xclbin", "xclbin.xclbinkernel", "xclbin.xclbinmem",
        "xclBOSyncDirection", "ert_cmd_state",
    ])
    def test_every_class_member_the_fake_offers_exists_on_the_real_class(
        self, fake, surface, dotted,
    ):
        owner = _nested(fake, dotted)
        real = set(surface["classes"][dotted])
        invented = _public_members(owner) - real
        assert not invented, (
            f"pyxrt.{dotted} has no {sorted(invented)}; the fake is richer than "
            f"the binding, which is exactly how the 2026-08-25 field failure "
            f"was written")

    def test_the_fake_declares_no_class_the_real_binding_lacks(self, fake, surface):
        classes = {
            name for name in _module_level_names(FAKE_PATH)
            if isinstance(getattr(fake, name, None), type)
        }
        assert classes <= set(surface["module"])


class TestNoHostCodeReachesForARegister:
    @pytest.mark.parametrize("path", [FAKE_PATH, DRIVER_PATH, TRANSPORT_PATH])
    def test_the_forbidden_calls_appear_nowhere_but_in_prose(self, path):
        source = path.read_text(encoding="utf-8")
        for name in FORBIDDEN:
            # The name is allowed in a comment or a refusal message that EXPLAINS
            # the absence; what must never appear is a call or an attribute use.
            assert f".{name}(" not in source, f"{path.name} calls {name}"
            assert f"getattr(" + name not in source

    def test_the_fake_has_no_register_surface_at_all(self, fake):
        for owner in (fake.kernel, fake.device, fake.bo, fake.run):
            for name in FORBIDDEN:
                assert not hasattr(owner, name)


class TestTheHostSpeaksTheEnumsTheBindingBinds:
    def test_the_completion_state_the_driver_compares_against_is_real(
        self, fake, surface,
    ):
        assert "ERT_CMD_STATE_COMPLETED" in surface["classes"]["ert_cmd_state"]
        assert hasattr(fake.ert_cmd_state, "ERT_CMD_STATE_COMPLETED")

    def test_exclusive_access_is_reached_through_the_nested_enum(
        self, fake, surface,
    ):
        assert "cu_access_mode" in surface["classes"]["kernel"]
        assert fake.kernel.cu_access_mode.exclusive == fake.kernel.exclusive

    def test_buffer_flags_are_reached_through_the_nested_enum(self, fake, surface):
        assert "flags" in surface["classes"]["bo"]
        assert fake.bo.flags.normal == fake.bo.normal

    def test_the_driver_and_the_transport_use_those_spellings(self):
        for path in (DRIVER_PATH, TRANSPORT_PATH):
            source = path.read_text(encoding="utf-8")
            assert "kernel.cu_access_mode.exclusive" in source
            assert "bo.flags.normal" in source
            assert "ert_cmd_state.ERT_CMD_STATE_COMPLETED" in source


def test_the_surface_lists_every_class_it_names_at_module_level(surface):
    """A class table entry that no module-level name reaches is dead data."""
    top_level: List[str] = [
        name for name in surface["classes"] if "." not in name]
    assert set(top_level) <= set(surface["module"])
