"""[W33] Which backends read a stage input in the reading core's local time on this checkout.

The stage-input timing rule: a core whose window starts at latency L (times its policy's latency scale) reads cycle t
of every stage-input row at global cycle L + t, a delayed slice |d| cycles later still (U1 R2), as the HCM reads it
(``lif_step``, ``packed_cycle``). A backend reads that way either in mimarsinan's own code, present on every checkout
(``IN_TREE_CORE_LOCAL_BACKENDS``), or in a source tree outside mimarsinan, present only where that tree carries it
(``SOURCE_TREE_CORE_LOCAL_BACKENDS``): nevresim's generated simulator, whose read is W33's change to nevresim
(``nevresim/stage_input_read``). For such a backend one probe of the tree it runs decides, from the tree's sources:
every declared read site holds its text and none still holds the global-cycle read it replaced; then the C++20
compiler the backend builds with constant-evaluates the tree's own self-test in a ``static_assert`` (syntax only, about
0.4 s). A verdict is kept per include-tree fingerprint (``compile_cache.simulator_source_fingerprint``), so an edited
tree is probed again. The guard (``models/spiking/hybrid/carry.require_backend_replay``) asks only for a program with
a late reader or a delayed slice, and refuses it by name on a backend whose read is not established."""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

from mimarsinan.chip_simulation.nevresim.compile_cache import simulator_source_fingerprint
from mimarsinan.chip_simulation.nevresim.compile_nevresim import constexpr_options
from mimarsinan.chip_simulation.nevresim.stage_input_read import (
    SELF_TEST,
    STAGE_INPUT_READ_SITES,
    ReadSite,
    nevresim_tree,
)
from mimarsinan.common.build_utils import find_cpp20_compiler


@dataclass(frozen=True)
class StageInputRead:
    """Whether a backend reads every stage-input row in the reading core's local time, and what says so."""

    core_local: bool
    evidence: str


@dataclass(frozen=True)
class SourceTreeRead:
    """A read that lives in a source tree outside mimarsinan: the tree a run uses, the read sites under its
    ``include/``, its self-test (header, constexpr function) and the options of the backend's own builds."""

    tree: Callable[[], str]
    sites: Sequence[ReadSite]
    self_test: Tuple[str, str]
    compile_options: Callable[[str], List[str]]


#: Backends whose read is mimarsinan's own code, with where it is.
IN_TREE_CORE_LOCAL_BACKENDS: Mapping[str, str] = {
    "hcm": "models/spiking/hybrid/lif_step and executors/packed_cycle read cycle - latency",
    "sanafe": "chip_simulation/sanafe/runner/carry.present_inputs_in_core_time",
    "lava": "chip_simulation/lava_loihi/segment_runner.shift_rows_to_core_time",
}
#: Backends whose read lives in the source tree they run.
SOURCE_TREE_CORE_LOCAL_BACKENDS: Mapping[str, SourceTreeRead] = {
    "nevresim": SourceTreeRead(nevresim_tree, STAGE_INPUT_READ_SITES, SELF_TEST, constexpr_options),
}
SELF_TEST_TIMEOUT_S = 300.0
#: (backend, include-tree fingerprint) -> (established, what the probe found).
_VERDICTS: Dict[Tuple[str, str], Tuple[bool, str]] = {}


def stage_input_read(backend: str) -> StageInputRead:
    """Whether ``backend`` reads every stage-input row in core-local time on this checkout, with the evidence."""
    where = IN_TREE_CORE_LOCAL_BACKENDS.get(backend)
    if where is not None:
        return StageInputRead(True, f"{backend} reads in core-local time in mimarsinan's own code ({where})")
    source = SOURCE_TREE_CORE_LOCAL_BACKENDS.get(backend)
    if source is None:
        return StageInputRead(False, f"{backend} declares no stage-input read in core-local time")
    return probe_source_tree(backend, source, source.tree())


def core_local_replay_backends() -> FrozenSet[str]:
    """The backends that read a stage input in core-local time on this checkout."""
    declared = list(IN_TREE_CORE_LOCAL_BACKENDS) + list(SOURCE_TREE_CORE_LOCAL_BACKENDS)
    return frozenset(b for b in declared if stage_input_read(b).core_local)


def probe_source_tree(backend: str, source: SourceTreeRead, root) -> StageInputRead:
    """The probe of the tree at ``root``: the read sites, then the self-test, kept per include-tree fingerprint."""
    include = Path(root) / "include"
    if not include.is_dir():
        return StageInputRead(False, f"no {backend} include tree at {include}")
    key = (backend, simulator_source_fingerprint(root))
    if key not in _VERDICTS:
        missing = site_problems(include, source.sites)
        problem = None if missing else self_test_problem(include, source.self_test, source.compile_options)
        _VERDICTS[key] = (
            (False, "lack the core-local stage-input read: " + "; ".join(missing)) if missing
            else (False, f"hold every read site, but {problem}") if problem
            else (True, f"hold every read site, and the self-test {source.self_test[1]} holds"))
    established, found = _VERDICTS[key]
    return StageInputRead(established, f"the {backend} sources at {include} {found}")


def site_problems(include: Path, sites: Sequence[ReadSite]) -> List[str]:
    """Per read site, a missing file, a missing text, or a global-cycle read the site still holds."""
    out: List[str] = []
    for path, marker, replaced in sites:
        file = include / path
        if not file.is_file():
            out.append(f"{path} is missing")
            continue
        text = file.read_text(errors="replace")
        if marker not in text:
            out.append(f"{path} does not hold {marker!r}")
        out.extend(f"{path} still holds the global-cycle read {r!r}" for r in replaced if r in text)
    return out


def self_test_problem(include: Path, self_test: Tuple[str, str],
                      compile_options: Callable[[str], List[str]]) -> Optional[str]:
    """Why the tree's self-test does not hold under the backend's compiler, or None when it holds."""
    header, function = self_test
    compiler, family = find_cpp20_compiler()
    if compiler is None or family is None:
        return f"no C++20 compiler is found to evaluate the self-test {function}"
    with tempfile.TemporaryDirectory() as work:
        probe = Path(work) / "stage_input_read_probe.cpp"
        probe.write_text(f'#include "{header}"\nstatic_assert({function}());\n')
        cmd = [compiler, "-std=c++20", "-fsyntax-only", *compile_options(family), "-I", str(include), str(probe)]
        try:
            done = subprocess.run(cmd, capture_output=True, text=True, timeout=SELF_TEST_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            return f"the self-test {function} did not finish within {SELF_TEST_TIMEOUT_S:.0f} s ({compiler})"
    if done.returncode != 0:
        return f"the self-test {function} does not hold ({compiler}: {' '.join(done.stderr.split())[-300:]})"
    return None
