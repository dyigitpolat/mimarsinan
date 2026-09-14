"""Every third-party seam has three places to look: the manifest, the patch dir, this test.

Each block skips cleanly when its optional extra is absent, so the contract is
checkable on a base install and enforced on a full one.
"""

from __future__ import annotations

import importlib.metadata as md
import multiprocessing
import subprocess
import sys
from pathlib import Path

import pytest

from mimarsinan.common.dependency_manifest import (
    declared_direct_reference,
    declared_extras,
    declared_pin,
    declared_requirement,
)

REPO = Path(__file__).resolve().parents[3]
SCRIPTS = REPO / "scripts"


def _installed(distribution: str) -> str | None:
    try:
        return md.version(distribution)
    except md.PackageNotFoundError:
        return None


def _require(distribution: str, extra: str) -> str:
    version = _installed(distribution)
    if version is None:
        pytest.skip(f"{distribution} is not installed (`uv sync --extra {extra}`)")
    return version


def _under_prefix(path: Path) -> bool:
    """True when ``path`` lives inside the running interpreter's environment."""
    roots = [Path(sys.prefix).resolve()]
    roots += [Path(p).resolve() for p in sys.path if p and Path(p).name == "site-packages"]
    resolved = path.resolve()
    return any(root == resolved or root in resolved.parents for root in roots)


class TestTheManifestIsTheSingleDeclaration:
    def test_the_manifest_exists_and_is_locked(self):
        assert (REPO / "pyproject.toml").is_file()
        assert (REPO / "uv.lock").is_file(), "uv.lock is the reproducibility record"

    def test_every_converted_inclusion_is_declared(self):
        extras = declared_extras()
        assert declared_direct_reference("spikingjelly"), "spikingjelly must be a pinned git URL"
        assert "lava-nc==0.10.0" in extras.get("loihi", []), extras
        assert "sanafe==2.1.1" in extras.get("sanafe", []), extras

    def test_the_gpl_backend_stays_optional(self):
        """SANA-FE is GPL-3.0: it may never be a base dependency of an MIT tree."""
        base = declared_requirement("sanafe")
        assert base in declared_extras()["sanafe"], (
            "sanafe must be declared only under the `sanafe` extra"
        )


class TestLava:
    def test_the_installed_version_is_the_declared_one(self):
        assert _require("lava-nc", "loihi") == declared_pin("lava-nc") == "0.10.0"

    def test_numpy_two_is_what_the_override_buys(self):
        _require("lava-nc", "loihi")
        import numpy

        assert int(numpy.__version__.split(".")[0]) >= 2, (
            "lava-nc's metadata pins numpy<2; [tool.uv] override-dependencies lifts it"
        )

    def test_the_namespace_has_exactly_one_portion(self):
        """The removed submodule made ``lava`` a two-portion namespace package."""
        _require("lava-nc", "loihi")
        import lava

        portions = [Path(p) for p in lava.__path__]
        assert len(portions) == 1, portions
        assert _under_prefix(portions[0]), portions[0]

    def test_the_start_method_monkeypatch_applies_itself(self):
        _require("lava-nc", "loihi")
        import torch.multiprocessing as torch_mp
        from mimarsinan.chip_simulation.lava_loihi.core_lava import _subtractive_lif_cls

        _subtractive_lif_cls()

        assert getattr(multiprocessing.set_start_method, "_mimarsinan_lava_safe", False)
        assert getattr(torch_mp.set_start_method, "_mimarsinan_lava_safe", False)
        # The contract the patch exists for: lava sets `fork` at import, so a
        # later set_start_method must be a no-op instead of a RuntimeError.
        established = multiprocessing.get_start_method(allow_none=True)
        assert established is not None, "importing lava must establish a start method"
        multiprocessing.set_start_method("spawn")
        multiprocessing.set_start_method("spawn")
        assert multiprocessing.get_start_method(allow_none=True) == established, (
            "the patch must leave the established start method alone"
        )

    def test_the_process_model_stays_discoverable(self):
        """Lava discovers ProcessModels by ``__module__``; it must be a top-level module."""
        _require("lava-nc", "loihi")
        from mimarsinan.chip_simulation.subtractive_lif import SubtractiveLIFReset

        assert SubtractiveLIFReset.__module__ == "mimarsinan.chip_simulation.subtractive_lif"


class TestSpikingjelly:
    def test_the_installed_commit_is_the_declared_commit(self):
        _require("spikingjelly", "")
        declared = declared_direct_reference("spikingjelly") or ""
        commit = declared.rsplit("@", 1)[-1]
        assert len(commit) == 40, declared

        direct_url = md.distribution("spikingjelly").read_text("direct_url.json") or ""
        lock = (REPO / "uv.lock").read_text()
        assert commit in direct_url or commit in lock, direct_url

    def test_it_is_imported_from_the_environment_not_the_tree(self):
        _require("spikingjelly", "")
        import spikingjelly

        assert spikingjelly.__file__ is not None
        assert _under_prefix(Path(spikingjelly.__file__).parent)

    def test_no_path_hack_survives(self):
        assert not [p for p in sys.path if p.rstrip("/").endswith("spikingjelly")], sys.path
        assert not (REPO / "spikingjelly").exists()

    def test_the_subclassed_surface_resolves(self):
        """``_LatticeIFNode`` overrides exactly these; a bump that moves one is a red gate."""
        _require("spikingjelly", "")
        from spikingjelly.activation_based import functional, neuron, surrogate

        for attribute in (
            "single_step_forward",
            "multi_step_forward",
            "neuronal_charge",
            "neuronal_fire",
            "neuronal_reset",
            "v_float_to_tensor",
        ):
            assert hasattr(neuron.IFNode, attribute), attribute
        assert functional.reset_net is not None
        assert surrogate.ATan is not None
        assert surrogate.atan_backward is not None
        assert surrogate.SurrogateFunctionBase is not None


class TestSanafe:
    def test_the_installed_version_is_guarded(self):
        version = _require("sanafe", "sanafe")
        from mimarsinan.chip_simulation.sanafe.arch_synth.spec import (
            _SUPPORTED_SANAFE_VERSIONS,
        )

        assert version in _SUPPORTED_SANAFE_VERSIONS

    def test_the_whole_plugin_set_is_built(self):
        _require("sanafe", "sanafe")
        sys.path.insert(0, str(SCRIPTS))
        try:
            from build_sanafe_plugins import expected_libraries
        finally:
            sys.path.pop(0)

        missing = [p.name for p in expected_libraries() if not p.is_file()]
        assert not missing, (
            f"missing SANA-FE plugins {missing}; run `python scripts/build_sanafe_plugins.py`"
        )

    def test_no_sanafe_source_is_in_the_tree(self):
        """GPL-3.0 headers are fetched into build/, never committed to this MIT tree."""
        assert not (REPO / "sana_fe").exists()


class TestNevresim:
    def test_the_resolved_root_is_a_nevresim_checkout(self):
        from mimarsinan.common.env import nevresim_root

        root = Path(nevresim_root())
        if not root.is_dir():
            pytest.skip(f"{root} is not initialised (`git submodule update --init nevresim`)")
        assert (root / "include").is_dir(), root
        assert (root / "CMakeLists.txt").is_file(), root

    def test_it_is_the_only_submodule(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPTS / "check_submodule_pins.py"), "--offline"],
            cwd=str(REPO),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr


class TestRemovedInclusions:
    @pytest.mark.parametrize("path", ["kedi", ".tmp_repo", "lava", "sana_fe", "spikingjelly"])
    def test_the_directory_is_gone(self, path):
        assert not (REPO / path).exists(), f"{path} was converted to a declared dependency"

    def test_the_index_carries_one_gitlink(self):
        result = subprocess.run(
            ["git", "ls-files", "--stage"],
            cwd=str(REPO),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            pytest.skip("not a git checkout")
        gitlinks = sorted(
            line.split("\t", 1)[1].strip()
            for line in result.stdout.splitlines()
            if line.startswith("160000")
        )
        assert gitlinks == ["nevresim"], gitlinks


class TestPatchHook:
    def test_the_patch_directory_is_documented(self):
        assert (REPO / "third_party" / "patches" / "README.md").is_file()

    def test_every_declared_patch_is_applied(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPTS / "apply_patches.py"), "--check"],
            cwd=str(REPO),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
