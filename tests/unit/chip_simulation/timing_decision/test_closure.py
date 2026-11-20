"""The closure of the compiled timing decision: a census over the compiled form of every timing translation unit.

Every translation unit the two backends build for a timing program is compiled as its backend compiles it and read in
the GNU compiler's GIMPLE (``gimple_census``): the SANA-FE plugins from the compile commands of their own CMake
configuration, and the nevresim main ``NevresimDriver`` generates for a one-core program of every timing program it
builds. In each, every toolchain function outside the one decision's namespace (``nevresim::timing``) that compares,
converts, rounds or clamps a floating value, or calls a library function or another toolchain function that does, must
be on the ruled-out list below with its reason, and each entry must be found where it says. The files whose functions
call the one decision are a finite list too, and the census fails on an unlisted caller. The census reads no file by
a name pattern: a new plugin source enters it by entering the build (a file of the plugins' directory that no built
unit compiles or includes fails), and a new compute, generator or helper enters it by being compiled into a program.

The census needs the GNU C++ compiler as the toolchain's C++20 compiler (``find_cpp20_compiler``; g++ 13.3 here, its
``-fdump-tree-ssa-lineno`` dump), cmake and SANA-FE's header tree (``MIMARSINAN_SANAFE_SRC`` or the tree a plugin
build fetched). Where one is missing it fails by name (``CensusUnavailable``), never a skip: the suite's profile counts
it as a failure, so the closure cannot pass on a machine that cannot read it. Its self-check holds the shapes the check
of U-CSOMA wrote on paper (p1, p2, p3, x1, n1, t1, k1, k2) and two more (m1, m2), each of which must turn it red.
"""

from __future__ import annotations

import fnmatch
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import pytest

from . import gimple_census as gc

REPO_ROOT = gc.REPO_ROOT
HEADER = REPO_ROOT / "nevresim" / "include" / "simulator" / "compute_policy" / "timing_decision.hpp"
QUANTIZED_NEURON = "nevresim/include/simulator/compute_policy/neuron/ttfs_quantized_neuron_compute.hpp"
QUANTIZED_SOMA = "mimarsinan_ttfs_quantized_soma.cpp"
CYCLE_SOMA = "mimarsinan_ttfs_cycle_soma.cpp"

#: The files whose functions call the one decision, with the units that compile them (an unlisted caller fails).
IN_FORCE: Dict[str, str] = {
    f"plugins/{QUANTIZED_SOMA}": f"plugin:{QUANTIZED_SOMA}",
    f"plugins/{CYCLE_SOMA}": f"plugin:{CYCLE_SOMA}",
    QUANTIZED_NEURON: "main:ttfs_quantized/*",
}


@dataclass(frozen=True)
class Ruled:
    units: str  # the units whose census finds it (a pattern over the unit labels)
    reason: str


_ANALYTIC = "the analytic continuous soma, whose float arithmetic is the program's own (no exact ratio is formed)"
_COMPARATOR = ("a generic comparator; in a timing program only the decision's namespace calls it, to read a policy's "
               "strictness from one exact tie (timing::strict_compare), and a site calling it is a 'calls' item")

#: The toolchain functions (``Name @ file``, all their items) or single items (``Name @ file | item``) that decide in
#: floating point outside the one decision, each with its reason and the units where the census finds it.
RULED_OUT: Dict[str, Ruled] = {
    "MimarsinanSoma::update @ plugins/mimarsinan_soma.cpp": Ruled(
        "plugin:mimarsinan_soma.cpp",
        "the LIF soma: its float membrane against its threshold is the LIF program's law; no timing program runs it"),
    "MimarsinanTtfsCascadeSoma::update @ plugins/mimarsinan_ttfs_cascade_soma.cpp": Ruled(
        "plugin:mimarsinan_ttfs_cascade_soma.cpp",
        "the cascaded count soma: an integer membrane held exactly in double below 2^53, compared with the "
        "threshold; no ratio is formed, and its program decodes counts"),
    "MimarsinanTtfsContinuousSoma::update @ plugins/mimarsinan_ttfs_continuous_soma.cpp": Ruled(
        "plugin:mimarsinan_ttfs_continuous_soma.cpp", _ANALYTIC),
    "MimarsinanTtfsCycleSoma::update @ plugins/mimarsinan_ttfs_cycle_soma.cpp | line else if "
    "(current_in.has_value() && current_in.value() != 0.0)": Ruled(
        f"plugin:{CYCLE_SOMA}",
        "the late arrival's trace guard: a current delivered after the window start is traced, never integrated, "
        "and no fire step reads it"),
    "nevresim::NeuronCompute::operator() @ "
    "nevresim/include/simulator/compute_policy/neuron/ttfs_analytical_neuron_compute.hpp": Ruled(
        "main:ttfs/*", _ANALYTIC),
    "nevresim::Neuron::compute @ nevresim/include/simulator/chip/neuron.hpp | calls nevresim::NeuronCompute::"
    "operator() @ nevresim/include/simulator/compute_policy/neuron/ttfs_analytical_neuron_compute.hpp": Ruled(
        "main:ttfs/*", "the chip's neuron dispatches the analytic continuous program's own compute"),
    "nevresim::InclusiveCompare::fires @ nevresim/include/simulator/compute_policy/compare_policy/"
    "inclusive_compare.hpp": Ruled("main:ttfs_quantized/*/<=/*", _COMPARATOR),
    "nevresim::StrictCompare::fires @ nevresim/include/simulator/compute_policy/compare_policy/"
    "strict_compare.hpp": Ruled("main:ttfs_quantized/*/</*", _COMPARATOR),
}

#: Sites every timing program includes and none compiles: a function of one entering a program fails the census
#: until its reason is read into ``RULED_OUT``.
DORMANT: Dict[str, str] = {
    "nevresim/include/simulator/spike_generation/ttfs_spike_generator.hpp": (
        "rounds a segment input the host holds in floating point (U-TIE's tie rule); on a level j/S the double error "
        "is below S 2^-52, far from a half; at a midpoint (2k+1)/(2S) the product is an exact half, and its rounding "
        "is U-TIE's tie rule (half away from zero), decided there"),
}


def _covered(unit: str, key: str, item: str) -> bool:
    for entry in (key, f"{key} | {item}"):
        ruled = RULED_OUT.get(entry)
        if ruled is not None and fnmatch.fnmatchcase(unit, ruled.units):
            return True
    return False


def uncovered(census: gc.Census) -> List[str]:
    """The census's items no ruled-out entry covers in this unit, with the compiled statements that gave them."""
    return [f"{key} | {item}  [{'; '.join(census.evidence.get(key, ()))[:400]}]"
            for key, items in census.items.items() for item in items if not _covered(census.unit, key, item)]


def stale(census: gc.Census) -> List[str]:
    """Ruled-out entries due in this unit that its census does not find."""
    found = {key for key in census.items} | {f"{k} | {i}" for k, items in census.items.items() for i in items}
    return [entry for entry, ruled in RULED_OUT.items()
            if fnmatch.fnmatchcase(census.unit, ruled.units) and entry not in found]


def callers_out_of_force(census: gc.Census) -> List[str]:
    """Files calling the one decision that the in-force list does not name for this unit, and listed ones absent."""
    due = {f for f, units in IN_FORCE.items() if fnmatch.fnmatchcase(census.unit, units)}
    return sorted(set(census.header_callers) ^ due)


def assert_closed(census: gc.Census, deps=()) -> None:
    assert census.unknown_types == (), f"{census.unit}: operand types the census does not classify: " \
                                       f"{census.unknown_types} (classify them in gimple_census.TYPEDEFS)"
    assert uncovered(census) == [], f"{census.unit} decides by itself outside the one decision"
    assert stale(census) == [], f"{census.unit}: ruled-out entries the census no longer finds"
    assert callers_out_of_force(census) == [], f"{census.unit}: the callers of the one decision moved"
    for site, _reason in DORMANT.items():
        if census.unit.startswith("main:"):
            assert any(str(d).endswith(site.split("/", 2)[2]) for d in deps), f"{site} is no longer included"
            assert site not in census.files, f"{site} is compiled into {census.unit}: read its reason into RULED_OUT"


# -------------------------------------------------------------------------------------------------- the census


@pytest.fixture(scope="module")
def plugin_build(tmp_path_factory) -> List[gc.TranslationUnit]:
    return gc.plugin_units(tmp_path_factory.mktemp("plugin_units"))


@pytest.fixture(scope="module")
def plugin_census(plugin_build, tmp_path_factory):
    work = tmp_path_factory.mktemp("plugin_dumps")
    censuses, deps = [], set()
    for unit in plugin_build:
        text, unit_deps = gc.compile_unit(unit, work)
        deps |= unit_deps
        censuses.append(gc.census_of_dump(text, unit=unit.label, roots=gc.roots()))
    return censuses, deps


def test_the_plugins_build_every_source_of_their_directory(plugin_build, plugin_census):
    _censuses, deps = plugin_census
    assert {u.source.name for u in plugin_build} >= {QUANTIZED_SOMA, CYCLE_SOMA}
    assert gc.outside_the_build(deps) == [], "a file of the plugins' directory that no built unit compiles"


def test_every_plugin_unit_decides_only_through_the_header(plugin_census):
    censuses, _deps = plugin_census
    for census in censuses:
        assert_closed(census)


@pytest.mark.parametrize("program", gc.timing_programs(), ids=gc.program_label)
def test_every_timing_main_decides_only_through_the_header(program, tmp_path):
    gen = gc.render_main(tmp_path, program)
    text, deps = gc.compile_main(gen)
    assert_closed(gc.census_of_dump(text, unit=gc.program_label(program), roots=gc.roots(generated=gen)), deps)


def test_the_census_reads_the_programs_its_lists_name():
    """The enumeration reads no fewer programs than the lists name: each in-force site and each ruled-out entry of a
    generated main is due in at least one program the census reads (a new program enters with no edit here)."""
    labels = [gc.program_label(p) for p in gc.timing_programs()]
    assert labels
    patterns = [u for u in IN_FORCE.values() if u.startswith("main:")]
    patterns += [r.units for r in RULED_OUT.values() if r.units.startswith("main:")]
    for pattern in patterns:
        assert any(fnmatch.fnmatchcase(label, pattern) for label in labels), f"no program read for {pattern}"


def test_the_header_is_the_one_function():
    assert HEADER.is_file()
    text = HEADER.read_text()
    for name in ("lattice_fire_step", "fire_step", "neuron_membrane"):
        assert re.search(rf"\b{name}\s*\(", text), name


# ------------------------------------------------------------------------------------------- the self-check


def _edit(path: Path, anchor: str, new: str) -> None:
    text = path.read_text()
    assert text.count(anchor) == 1, (path.name, anchor)
    path.write_text(text.replace(anchor, new))


_PRESET_END = "                        theta, s, false, timing::FloatPath::closed_form);\n            }\n"
_NEURON_END = "            initialized_ = true;\n"
_NEURON_INCLUDE = '#include "simulator/compute_policy/timing_decision.hpp"\n'


def _plugin_shape(source: str, edits: List[Tuple[str, str]]):
    def run(tmp: Path, units: List[gc.TranslationUnit]) -> List[str]:
        shape_dir = tmp / "plugins"
        shape_dir.mkdir()
        copy = shape_dir / source
        shutil.copy2(gc.PLUGIN_SOURCE / source, copy)
        for anchor, new in edits:
            _edit(copy, anchor, new)
        unit = next(u for u in units if u.source.name == source)
        text, _deps = gc.compile_unit(unit, tmp, source=copy)
        return uncovered(gc.census_of_dump(text, unit=unit.label, roots=gc.roots(plugins=shape_dir)))
    return run


def _nevresim_shape(edits: List[Tuple[str, str, str]], new_files: Dict[str, str]):
    def run(tmp: Path, _units) -> List[str]:
        from mimarsinan.common.env import nevresim_root

        root = tmp / "nevresim"
        shutil.copytree(Path(nevresim_root()) / "include", root / "include")
        for rel, text in new_files.items():
            (root / "include" / rel).write_text(text)
        for rel, anchor, new in edits:
            _edit(root / "include" / rel, anchor, new)
        program = ("ttfs_quantized", "runtime", "<=", "int")
        gen = gc.render_main(tmp, program)
        text, _deps = gc.compile_main(gen, nevresim=root)
        return uncovered(gc.census_of_dump(text, unit=gc.program_label(program),
                                           roots=gc.roots(nevresim=root, generated=gen)))
    return run


def _template_shape(tmp: Path, _units) -> List[str]:
    from mimarsinan.chip_simulation.nevresim import nevresim_driver

    program = ("ttfs_quantized", "compile_time", "<=", "int")
    original = nevresim_driver.main_cpp_template
    rounded = "            std::cout << std::round(i * 64.0) / 64.0 << ' ';\n"
    assert original.count("            std::cout << i << ' ';\n") == 1
    nevresim_driver.main_cpp_template = original.replace("            std::cout << i << ' ';\n", rounded)
    try:
        gen = gc.render_main(tmp, program)
    finally:
        nevresim_driver.main_cpp_template = original
    text, _deps = gc.compile_main(gen)
    return uncovered(gc.census_of_dump(text, unit=gc.program_label(program), roots=gc.roots(generated=gen)))


def _unbuilt_source_shape(tmp: Path, units) -> List[str]:
    copy = tmp / "plugins"
    shutil.copytree(gc.PLUGIN_SOURCE, copy)
    shutil.copy2(copy / QUANTIZED_SOMA, copy / "mimarsinan_ttfs_relay_soma.cpp")
    deps = set()
    for unit in units:
        deps |= gc.compile_unit(unit, tmp)[1]
    return gc.outside_the_build(deps, source_dir=copy)


def _new_built_soma_shape(tmp: Path, _units) -> List[str]:
    from mimarsinan.common.env import nevresim_root

    copy = tmp / "plugins"
    shutil.copytree(gc.PLUGIN_SOURCE, copy)
    relay = (copy / QUANTIZED_SOMA).read_text().replace("create_mimarsinan_ttfs_quantized_soma",
                                                        "create_mimarsinan_ttfs_relay_soma")
    (copy / "mimarsinan_ttfs_relay_soma.cpp").write_text(relay.replace(
        _PRESET_END, _PRESET_END.replace("            }\n", "") + "                if (preset_membrane_[neuron_address] "
        ">= theta) fire_step_[neuron_address] = 0;\n            }\n"))
    _edit(copy / "CMakeLists.txt", "mimarsinan_ttfs_cycle_soma mimarsinan_ttfs_cascade_soma)",
          "mimarsinan_ttfs_cycle_soma mimarsinan_ttfs_cascade_soma mimarsinan_ttfs_relay_soma)")
    units = gc.plugin_units(tmp, source_dir=copy, nevresim_include=Path(nevresim_root()) / "include")
    relay_unit = next(u for u in units if u.source.name == "mimarsinan_ttfs_relay_soma.cpp")
    text, _deps = gc.compile_unit(relay_unit, tmp)
    return uncovered(gc.census_of_dump(text, unit=relay_unit.label, roots=gc.roots(plugins=copy)))


_HELPER = ("#pragma once\n#include <cmath>\nnamespace nevresim {\n"
           "inline int ttfs_fire_helper(double v, double theta, int s)\n"
           "{ return static_cast<int>(std::ceil(s * (1.0 - v / theta))); }\n}\n")
_RELAY = ("#pragma once\nnamespace nevresim {\ninline int ttfs_relay_fire_step(double membrane, double threshold, int s)\n"
          "{\n    for (int step = 0; step < s; ++step) {\n        if (membrane >= threshold) return step;\n"
          "        membrane += threshold / s;\n    }\n    return -1;\n}\n}\n")

#: name -> (the shape, a text its finding must hold)
SHAPES: Dict[str, Tuple[Callable, str]] = {
    "p1_cycle_truncating_cast": (_plugin_shape(CYCLE_SOMA, [(
        "        fire_step_[neuron_address] = k_fire;\n",
        "        const int k_trunc = static_cast<int>(s * (1.0 - v / theta));\n"
        "        fire_step_[neuron_address] = lattice_[neuron_address] ? k_fire : k_trunc;\n")]), "k_trunc"),
    "p2_quantized_unspaced_compare": (_plugin_shape(QUANTIZED_SOMA, [(
        _PRESET_END, _PRESET_END.replace("            }\n", "")
        + "                if (preset_membrane_[neuron_address]>=theta) fire_step_[neuron_address] = 0;\n"
        "            }\n")]), ">=theta"),
    "p3_neuron_auto_compare": (_nevresim_shape([(QUANTIZED_NEURON.split("/", 2)[2], _NEURON_END,
                                                 "            const auto v0 = membrane.v;\n"
                                                 "            if (!membrane.lattice && v0 >= static_cast<double>("
                                                 "threshold_)) fire_step_ = 0;\n" + _NEURON_END)], {}), "v0 >="),
    "x1_helper_header": (_nevresim_shape(
        [(QUANTIZED_NEURON.split("/", 2)[2], _NEURON_INCLUDE,
          _NEURON_INCLUDE + '#include "simulator/compute_policy/ttfs_fire_helper.hpp"\n'),
         (QUANTIZED_NEURON.split("/", 2)[2], _NEURON_END,
          "            if (!membrane.lattice) fire_step_ = ttfs_fire_helper(membrane.v, theta, S);\n" + _NEURON_END)],
        {"simulator/compute_policy/ttfs_fire_helper.hpp": _HELPER}), "ttfs_fire_helper"),
    "n1_new_compute_name": (_nevresim_shape(
        [(QUANTIZED_NEURON.split("/", 2)[2], _NEURON_INCLUDE,
          _NEURON_INCLUDE + '#include "simulator/compute_policy/neuron/ttfs_relay_compute.hpp"\n'),
         (QUANTIZED_NEURON.split("/", 2)[2], _NEURON_END,
          "            if (!membrane.lattice) fire_step_ = ttfs_relay_fire_step(membrane.v, theta, S);\n"
          + _NEURON_END)],
        {"simulator/compute_policy/neuron/ttfs_relay_compute.hpp": _RELAY}), "ttfs_relay_fire_step"),
    "t1_template_round": (_template_shape, "std::round"),
    "k1_spaced_compare": (_plugin_shape(QUANTIZED_SOMA, [(
        _PRESET_END, _PRESET_END.replace("            }\n", "")
        + "                if (membrane_[neuron_address] >= theta) fire_step_[neuron_address] = 0;\n"
        "            }\n")]), "membrane_[neuron_address] >= theta"),
    "k2_new_soma_file": (_unbuilt_source_shape, "mimarsinan_ttfs_relay_soma.cpp"),
    "m1_macro_library_compare": (_plugin_shape(QUANTIZED_SOMA, [
        ("#include <vector>\n", "#include <vector>\n#include <functional>\n"
                                "#define MIMARSINAN_REACHES(v, t) std::less_equal<double>{}((t), (v))\n"),
        (_PRESET_END, _PRESET_END.replace("            }\n", "")
         + "                if (MIMARSINAN_REACHES(preset_membrane_[neuron_address], theta)) "
         "fire_step_[neuron_address] = 0;\n            }\n")]), "MIMARSINAN_REACHES"),
    "m2_new_soma_in_the_build": (_new_built_soma_shape, "MimarsinanTtfsQuantizedSoma"),
}


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_the_census_turns_red_on_a_site_that_decides_by_itself(shape, plugin_build, tmp_path):
    run, marker = SHAPES[shape]
    found = run(tmp_path, plugin_build)
    assert found, f"{shape}: the census reads a self-deciding site as closed"
    assert any(marker in line for line in found), (shape, found)
