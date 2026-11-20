"""The one timing decision's version: each timing soma exports it, and a soma built without it is refused by name.

``timing_decision.hpp`` carries the decision's version (``nevresim::timing::abi``). Each SANA-FE soma that decides
through the header exports it (``mimarsinan_timing_decision_abi``). Before it stamps the executor's lattice record on a
timing soma, ``stamp_timing_lattice`` loads the soma's library once and refuses it by name, with the remedy, when the
symbol is absent or differs (``TimingDecisionVersionError``), so a stale plugin build can no longer read the stamp's
attributes as unknown and reproduce the older partings in silence. A custom architecture names its own somas, and
they are read the same way at its load.

The class swept, every compiled artifact the toolchain loads at run time whose source U-CSOMA edits:

- the SANA-FE plugins: the units whose compiled files hold the header are exactly the somas that export the version
  (the other four plugins compile no file the unit edits);
- nevresim's binaries: compiled from the generated main on every run, and a compile cache serves a binary only under
  a key that holds the include tree's bytes, so each nevresim file the unit edits moves the key.
"""

from __future__ import annotations

import ctypes
import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import numpy as np
import pytest

from . import gimple_census as gc

_VERSION_LINE = re.compile(r'^extern "C" int mimarsinan_timing_decision_abi\(\).*\n', re.M)


def _version_module():
    from mimarsinan.chip_simulation import timing_decision

    return timing_decision


def _soma_sources() -> Dict[str, str]:
    td = _version_module()
    return {model: f"{model}.cpp" for model in td.TIMING_SOMA_MODELS}


def _build(unit: gc.TranslationUnit, work: Path, text: str, name: str) -> Path:
    """The soma's shared library from ``text``, compiled with the command the plugins' CMake writes for it."""
    work.mkdir(parents=True, exist_ok=True)
    source = work / unit.source.name
    source.write_text(text)
    library = work / f"lib{name}.so"
    argv = [str(source) if a == str(unit.source) else ("-shared" if a == "-c" else a) for a in unit.argv]
    done = subprocess.run([*argv, "-o", str(library)], cwd=unit.directory, capture_output=True, text=True,
                          timeout=600)
    assert done.returncode == 0, done.stderr[-3000:]
    return library


@pytest.fixture(scope="module")
def somas(tmp_path_factory) -> Dict[str, Dict[str, Path]]:
    """For each timing soma: the library built from the tree's source, from that source without the version (the
    form of the set export's somas), and with another version."""
    work = tmp_path_factory.mktemp("somas")
    units = {u.source.name: u for u in gc.plugin_units(work)}
    out: Dict[str, Dict[str, Path]] = {}
    for model, source in _soma_sources().items():
        unit = units[source]
        text = unit.source.read_text()
        assert len(_VERSION_LINE.findall(text)) == 1, f"{source} exports no decision version"
        out[model] = {
            "built": _build(unit, work / model / "built", text, model),
            "without": _build(unit, work / model / "without", _VERSION_LINE.sub("", text), model),
            "other": _build(unit, work / model / "other", text.replace(
                "return timing::abi; }", "return timing::abi + 1; }"), model),
        }
    return out


def test_the_header_carries_the_toolchain_version(tmp_path):
    from mimarsinan.common.env import nevresim_root

    unit = tmp_path / "version.cpp"
    unit.write_text('#include "simulator/compute_policy/timing_decision.hpp"\n'
                    f"static_assert(nevresim::timing::abi == {_version_module().TIMING_DECISION_ABI});\n")
    done = subprocess.run([gc.gnu_compiler(), "-std=c++17", "-fsyntax-only", "-I",
                           str(Path(nevresim_root()) / "include"), str(unit)], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-2000:]


@pytest.mark.parametrize("model", ["mimarsinan_ttfs_quantized_soma", "mimarsinan_ttfs_cycle_soma"])
def test_each_timing_soma_exports_the_version_and_passes(somas, model):
    td = _version_module()
    library = ctypes.CDLL(str(somas[model]["built"]))
    exported = getattr(library, td.VERSION_SYMBOL)
    exported.restype = ctypes.c_int
    assert exported() == td.TIMING_DECISION_ABI
    td.require_timing_decision_version(str(somas[model]["built"]))


@pytest.mark.parametrize("form, words", [("without", "was built without the timing decision"),
                                         ("other", "was built with the timing decision")])
@pytest.mark.parametrize("model", ["mimarsinan_ttfs_quantized_soma", "mimarsinan_ttfs_cycle_soma"])
def test_a_soma_built_from_other_decision_sources_is_refused_by_name(somas, model, form, words):
    td = _version_module()
    with pytest.raises(td.TimingDecisionVersionError) as refused:
        td.require_timing_decision_version(str(somas[model][form]))
    message = str(refused.value)
    assert words in message and str(somas[model][form]) in message
    assert "rebuild with scripts/build_sanafe_plugins.py" in message


class _Neuron:
    def __init__(self) -> None:
        self.model_attributes: Dict[str, object] = {}

    def set_attributes(self, *, model_attributes=None, **_ignored) -> None:
        self.model_attributes.update(model_attributes or {})


def _one_core():
    core = SimpleNamespace(neurons_per_core=2, available_neurons=0, threshold=2.0, latency=0)
    return {0: [_Neuron(), _Neuron()]}, SimpleNamespace(cores=[core])


@pytest.mark.parametrize("preset, model", [(True, "mimarsinan_ttfs_quantized_soma"),
                                           (False, "mimarsinan_ttfs_cycle_soma")])
def test_the_stamp_loads_the_soma_once_and_refuses_a_stale_build_before_stamping(somas, monkeypatch, preset, model):
    from mimarsinan.chip_simulation.sanafe.net_synth import timing_lattice

    td = _version_module()
    record = [(np.ones((1, 2), dtype=bool), np.full((1, 2), 4.0))]
    asked: List[bool] = []
    loads: List[str] = []
    real_cdll = ctypes.CDLL
    monkeypatch.setattr(td.ctypes, "CDLL", lambda path, *a, **k: loads.append(path) or real_cdll(path, *a, **k))
    td._exported_version.cache_clear()

    def library(form):
        def resolve(p: bool) -> str:
            asked.append(p)
            return str(somas[model][form])
        return resolve

    monkeypatch.setattr(timing_lattice, "timing_soma_library", library("without"))
    groups, hcm = _one_core()
    with pytest.raises(td.TimingDecisionVersionError):
        timing_lattice.stamp_timing_lattice(groups, hcm, record, simulation_length=4, preset=preset)
    assert all(not n.model_attributes for n in groups[0]), "a stale soma was stamped"

    monkeypatch.setattr(timing_lattice, "timing_soma_library", library("built"))
    for _ in range(3):
        groups, hcm = _one_core()
        timing_lattice.stamp_timing_lattice(groups, hcm, record, simulation_length=4, preset=preset)
        assert all(n.model_attributes.get(timing_lattice.LATTICE_ATTRIBUTE) is True for n in groups[0])
    assert asked == [preset] * 4
    assert loads == [str(somas[model]["without"]), str(somas[model]["built"])], "the library is loaded once"


def test_the_stamp_resolves_the_toolchains_built_soma():
    from mimarsinan.chip_simulation.sanafe.arch_synth.spec import _plugin_path
    from mimarsinan.chip_simulation.sanafe.net_synth.timing_lattice import timing_soma_library

    assert timing_soma_library(True) == _plugin_path("ttfs_quantized_soma")
    assert timing_soma_library(False) == _plugin_path("ttfs_cycle_soma")


def test_a_custom_architecture_s_timing_somas_are_read_at_its_load(somas, tmp_path):
    from mimarsinan.chip_simulation.sanafe.arch_synth.build import build_architecture

    td = _version_module()

    def arch(form: str) -> Path:
        somas_yaml = "".join(
            f"            - name: soma_{i}\n              attributes:\n                plugin: {somas[m][form]}\n"
            f"                model: {m}\n" for i, m in enumerate(td.TIMING_SOMA_MODELS))
        path = tmp_path / f"{form}.yaml"
        path.write_text("architecture:\n  name: custom\n  tile:\n    - name: tile0\n      core:\n"
                        f"        - name: c0\n          soma:\n{somas_yaml}")
        return path

    assert td.timing_somas_of_arch(arch("built").read_text()) == [str(somas[m]["built"])
                                                                   for m in td.TIMING_SOMA_MODELS]
    td.require_arch_timing_somas(str(arch("built")))
    with pytest.raises(td.TimingDecisionVersionError):
        build_architecture(SimpleNamespace(), custom_arch_path=str(arch("without")))


def test_the_plugins_holding_the_header_are_exactly_the_version_checked_somas(tmp_path):
    header = (Path(__import__("mimarsinan.common.env", fromlist=["nevresim_root"]).nevresim_root()) / "include" /
              "simulator" / "compute_policy" / "timing_decision.hpp").resolve()
    holding = {u.source.stem for u in gc.plugin_units(tmp_path) if header in gc.dependencies(u)}
    assert holding == set(_version_module().TIMING_SOMA_MODELS)


@pytest.mark.parametrize("edited", [
    "simulator/compute_policy/timing_decision.hpp",
    "simulator/recording/output_format.hpp",
    "simulator/compute_policy/neuron/ttfs_quantized_neuron_compute.hpp",
])
def test_a_nevresim_binary_is_compiled_per_run_or_served_under_its_include_tree(edited, tmp_path):
    from mimarsinan.chip_simulation.nevresim.compile_cache import simulator_source_fingerprint
    from mimarsinan.chip_simulation.nevresim.nevresim_driver import NevresimDriver
    from mimarsinan.common.env import nevresim_root

    copy = tmp_path / "nevresim"
    shutil.copytree(Path(nevresim_root()) / "include", copy / "include")
    before = simulator_source_fingerprint(copy)
    path = copy / "include" / edited
    path.write_text(path.read_text() + "\n")
    assert simulator_source_fingerprint(copy) != before, f"a cached binary would outlive an edit of {edited}"
    driver = SimpleNamespace(compile_cache_dir=None)
    assert NevresimDriver._compile_cache_lookup(driver, 7, 0) is None, "a driver without a cache compiles per run"
