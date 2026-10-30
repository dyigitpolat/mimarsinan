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

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, FrozenSet, List, Mapping, Sequence, Tuple

from mimarsinan.chip_simulation.nevresim.compile_cache import simulator_source_fingerprint
from mimarsinan.chip_simulation.nevresim.compile_nevresim import constexpr_options
from mimarsinan.chip_simulation.nevresim.source_probe import self_test_problem, site_problems
from mimarsinan.chip_simulation.nevresim.stage_input_read import (
    SELF_TEST,
    STAGE_INPUT_READ_SITES,
    ReadSite,
    nevresim_tree,
)


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
