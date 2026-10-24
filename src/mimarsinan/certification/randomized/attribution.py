"""Root-causing a counterexample by ablation: the first decision layer whose undoing restores equality."""

from __future__ import annotations

import contextlib
from typing import Any, Callable, Dict, Iterator, List, Mapping, Tuple

import torch

from mimarsinan.certification.randomized.arms import ARMS
from mimarsinan.certification.randomized.instance import (
    ANALYTICAL_TTFS,
    STREAMED_LIF,
    SYNCHRONIZED_LIF,
    SYNCHRONIZED_TTFS,
    VALUE,
    WINDOWED_LIF,
)
from mimarsinan.certification.randomized.runner import (
    COUNTEREXAMPLE,
    RUN_ERROR,
    UNNAMED_FAILURES,
    Verdict,
    _run,
    compare,
    deployment,
    instance_for,
    reference_nodes,
)
from mimarsinan.certification.randomized.cost_draws import platform_constraints, settled
from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.latency.chip import ChipLatency
from mimarsinan.models.spiking.signal_spans import SpanFillPlan
from mimarsinan.spiking import spike_trains

#: The decision layers, outermost first: each ablation undoes one more of them. ``positions`` comes first
#: under a dithered encode (the one position-dependent read of a slice), ``walk`` under a timing-free
#: executor: there the walk order is the one thing ChipLatency's latencies decide; ``counts`` under the
#: synchronized LIF count executor, whose in-segment reads are rebuilt counts.
LAYERS = ("positions", "walk", "counts", "relabel", "geometry", "packing", "pass_cut")
#: Semantics whose executors walk a segment's cores in stored-latency order and are timing-free (one count,
#: level or value per core for any arrival: ``executors/sync_counts.py``, ``ttfs_segment``, the value walk).
WALK_ORDERED = (SYNCHRONIZED_LIF, ANALYTICAL_TTFS, SYNCHRONIZED_TTFS, VALUE)
#: The run-error message of a pass holding cached trains for some input slices and none for another, and the
#: rule it breaks per semantics (``u1_design.md`` section 3); under any other semantics it names no rule.
CACHED_TRAINS = "cached spike trains"
#: [MN8] A counterexample only the int64 golden sees: identity program and deployment agree and both depart from it.
GOLDEN_DEPARTURE = "golden_departure"
CACHED_TRAINS_RULE: Mapping[str, str] = {STREAMED_LIF: "mixed_input_pass", WINDOWED_LIF: "windowed_cached_trains"}
#: A rebuilt count within this distance of an integer is read as that integer by the ``counts`` ablation
#: (float64 rebuild dust is about 1e-15; a TTFS level ``k/S`` is at least ``1/S`` from any non-equal integer).
COUNT_DUST = 1e-9
#: Mechanism tags, each with the plan item that owns it (G5: plan section 21 item 1, rules R1 to R6).
MECHANISMS: Mapping[str, str] = {
    "u1_cut": "G5 pass projection (R1 pins, R2 delivery delays; u1_design.md section 3, class 'Pass timing'); met "
              "on the integrated tree (CL1), so the class arises where the H5-skew arm switches pins and delays off",
    "latency_order_walk": "R6 dependency-ordered walks: G5 for ttfs_segment.py and G2 for value_execution.py "
                          "(plan section 21 item 1), the synchronized LIF count walk by merge fix 75998141; met on "
                          "the integrated tree (CL1), so the class arises where the R6 arm switches the walks off",
    "retiming_dropped": "H0 semantics fidelity (sign-off P1 (a), CL1 fix round): segment_depth_groups "
                        "(mapping/layout/segmentation.py) levels a coalescing group as one unit; met on the "
                        "integrated tree, so the class (a windowed segment holding a coalescing group runs "
                        "streamed) arises where the H0 arm switches the leveling off",
    "mixed_input_pass": "G5 R4 per-slice input kinds (u1_design.md section 3, 'Crash: mixed-input pass "
                        "under VERBATIM'; spiking/segment_boundary.py raises on cached plus missing slices); fixed "
                        "on the integrated tree (CL1)",
    "position_dependent_encode": "G5 R4 fused channel positions (u1_design.md section 3, 'Position-dependent "
                                 "encodes'): the dithered comb rotates by the slice's channel index in the deployed "
                                 "stage input (chip_simulation/recording/spike_modes.py uniform_phase_offsets)",
    "windowed_cached_trains": "G5 R5 level units (u1_design.md section 3, 'Crash: windowed LIF, cached spike "
                              "trains'; mapping/packing/retimed_levels.py gives a single-level chunk no levels)",
    "count_rebuild": "H8 exact accumulation (sign-off P1 (a), CL1 fix round): the synchronized LIF count executor "
                     "reads each core's count off its window charge (wire_semantics.lif_window_count, exact on the "
                     "integer lattice); the base rebuilt it as stair*T/theta in float64, off the lattice for (T, "
                     "theta) beyond T in 1-8, so the class arises only on that base",
    "relabel": "finding for the conductor: the program depends on IR node ids or order",
    "golden_departure": "the identity program and the deployment agree and both depart from G2's int64 golden "
                        "(an executor off its integer readout law): a defect both programs share",
}


@contextlib.contextmanager
def dependency_walks() -> Iterator[None]:
    """Every stored-latency walk a dependency order, in-process and for the block only (no file edit):
    ChipLatency's ALAP raise (``_align_shiftable_cores``) off, so every producer keeps a latency below its
    consumers'. Under a timing-free executor this changes the walk order and nothing else."""
    saved = ChipLatency._align_shiftable_cores
    setattr(ChipLatency, "_align_shiftable_cores", lambda _self, current_max_latency: current_max_latency)
    try:
        yield
    finally:
        setattr(ChipLatency, "_align_shiftable_cores", saved)


@contextlib.contextmanager
def position_free_rotation() -> Iterator[None]:
    """The dithered comb's rotation made position-free, in-process and for the block only (no file edit):
    every channel takes one offset, so a slice's train no longer depends on where the stage input holds it
    (R4's position clause met by construction) while the trains stay rotated."""
    modes = spike_trains.spike_modes
    saved = modes.uniform_phase_offsets

    def one_offset(n_channels, simulation_length, device=None):
        """Channel 1's offset (the dithered comb's first nonzero rotation) for every channel."""
        return saved(2, simulation_length, device=device)[1:].expand(int(n_channels)).clone()

    setattr(modes, "uniform_phase_offsets", one_offset)
    try:
        yield
    finally:
        setattr(modes, "uniform_phase_offsets", saved)


@contextlib.contextmanager
def integer_counts() -> Iterator[None]:
    """Every count a core reads on the integer lattice, in-process and for the block only (no file edit): each
    filled axon signal within ``COUNT_DUST`` of an integer is read as that integer. Under the synchronized LIF
    count executor this undoes the float64 rebuild ``stair*T/theta`` (``sync_counts.py:56-59``) at every
    in-segment read and changes nothing else (segment inputs and always-on axons are integers already)."""
    saved = SpanFillPlan.apply

    def snapped(plan, out, *args, **kwargs):
        result = saved(plan, out, *args, **kwargs)
        nearest = torch.round(out)
        out.copy_(torch.where((out - nearest).abs() <= COUNT_DUST, nearest, out))
        return result

    setattr(SpanFillPlan, "apply", snapped)
    try:
        yield
    finally:
        setattr(SpanFillPlan, "apply", saved)


#: The executor ablations per semantics, tried alone and then together: each undoes one base violation of an
#: executor condition that no decision layer controls (R6's walk order; H8's rebuilt counts).
EXECUTOR_ABLATIONS: Mapping[str, Tuple[Tuple[str, str, Callable[[], contextlib.AbstractContextManager]], ...]] = {
    sem: (("walk", "latency_order_walk", dependency_walks),)
    + ((("counts", "count_rebuild", integer_counts),) if sem == SYNCHRONIZED_LIF else ())
    for sem in WALK_ORDERED}


def attribute(verdict: Verdict) -> Tuple[str, ...]:
    """Mechanism tags for a counterexample or a run error (:func:`counterexample_tags` for a counterexample). A
    crash on cached plus missing slices is R4's per-slice rule under streamed LIF and R5's level units under
    windowed LIF (``u1_design.md`` section 3); under any other semantics it is an unattributed run error."""
    if verdict.outcome == RUN_ERROR:
        if CACHED_TRAINS not in verdict.detail:
            return ("run_error",)
        return (CACHED_TRAINS_RULE.get(verdict.semantics, "run_error"),)
    if verdict.outcome != COUNTEREXAMPLE:
        return ()
    if not verdict.mismatches and verdict.golden_mismatches:
        return (GOLDEN_DEPARTURE,)
    arm = ARMS[verdict.arm]
    instance = instance_for(verdict.seed, arm)
    constraints = platform_constraints(verdict.declaration, hardware_bias=instance.hardware_bias)
    return counterexample_tags(instance, constraints, verdict.seed, arm)


def counterexample_tags(instance, constraints, seed, arm) -> Tuple[str, ...]:
    """Mechanism tags for a counterexample, by successive ablation of its decision layers.

    Under a dithered encode the slice positions are ablated first (a position-free rotation on the reference
    and the deployment alike): a counterexample that it removes is ``position_dependent_encode`` (R4). Under
    a timing-free walk-ordered executor the executor ablations come next, alone and then together:
    dependency-ordered walks (``latency_order_walk``, R6, with ``reference_walk`` and ``deployment_walk``
    saying which side walked out of order) and, under synchronized LIF, integer counts (``count_rebuild``,
    H8). The decision layers follow (relabelling, geometry, packing, the pass cut)."""
    if arm.phase_dither and _restored(instance, constraints, seed, arm, position_free_rotation):
        return ("layer:positions", "position_dependent_encode")
    ref_ir, ref_map, ref_nodes = reference_nodes(instance, arm)
    executor = _executor_tags(instance, constraints, seed, arm, ref_map)
    if executor:
        return executor
    with arm.engaged():
        layer = _first_restoring_layer(instance, constraints, seed, arm, ref_ir, ref_nodes)
        tags: List[str] = [f"layer:{layer}"]
        if layer == "geometry" and instance.semantics == WINDOWED_LIF:
            _ir, geo_map = deployment(instance, constraints, seed, arm, relabel=False, identity=True)
            if _levels(ref_map) > _levels(geo_map):
                tags.append("retiming_dropped")
        elif layer == "pass_cut":
            tags.append("u1_cut")
        elif layer == "relabel":
            tags.append("relabel")
    return tuple(tags)


def _restored(instance, constraints, seed, arm, *ablations) -> bool:
    """Whether the deployment equals the identity program with ``ablations`` applied to both runs."""
    try:
        with contextlib.ExitStack() as stack:
            for ablation in ablations:
                stack.enter_context(ablation())
            ref_ir, _ref_map, ref_nodes = reference_nodes(instance, arm)
            with arm.engaged():
                ir, mapping = deployment(instance, constraints, seed, arm)
                return compare(ref_ir, ref_nodes, ir, _run(mapping, instance, arm))[1] == 0
    except UNNAMED_FAILURES + (ToolchainRefusal,):
        return False


def _executor_tags(instance, constraints, seed, arm, ref_map) -> Tuple[str, ...]:
    """The first executor ablation (alone, then all together) that restores equality: its ``layer:`` tags and
    mechanisms, plus which side walked out of order when the walk is among them; else ()."""
    single = EXECUTOR_ABLATIONS.get(instance.semantics, ())
    trials = [(entry,) for entry in single] + ([tuple(single)] if len(single) > 1 else [])
    for trial in trials:
        if not _restored(instance, constraints, seed, arm, *(ablation for _l, _m, ablation in trial)):
            continue
        tags = [f"layer:{layer}" for layer, _m, _a in trial] + [mechanism for _l, mechanism, _a in trial]
        if any(layer == "walk" for layer, _m, _a in trial):
            tags += _walk_sides(instance, constraints, seed, arm, ref_map)
        return tuple(tags)
    return ()


def _walk_sides(instance, constraints, seed, arm, ref_map) -> List[str]:
    """``reference_walk`` and ``deployment_walk``: which program reads an in-segment producer at or after itself,
    on each mapping as run (a run's ChipLatency writes the latencies its walk reads, ``mapping/latency/chip.py``)."""
    try:
        with arm.engaged():
            _ir, dep_map = deployment(instance, constraints, seed, arm)
            _run(dep_map, instance, arm)
    except UNNAMED_FAILURES + (ToolchainRefusal,):
        return []
    return [name for name, mapping in (("reference_walk", ref_map), ("deployment_walk", dep_map))
            if _producer_at_or_after_consumer(mapping)]


def _first_restoring_layer(instance, constraints, seed, arm, ref_ir, ref_nodes) -> str:
    unscheduled = settled(dict(constraints, allow_scheduling=False,
                               cores=[dict(core, count=max(int(core["count"]), 4096)) for core in constraints["cores"]]))
    trials = (
        ("relabel", lambda: deployment(instance, constraints, seed, arm, relabel=False)),
        ("geometry", lambda: deployment(instance, constraints, seed, arm, relabel=False, identity=True)),
        ("packing", lambda: deployment(instance, unscheduled, seed, arm, relabel=False)),
    )
    for layer, build in trials:
        try:
            ir, mapping = build()
            _w, mismatches, _f = compare(ref_ir, ref_nodes, ir, _run(mapping, instance, arm))
        except UNNAMED_FAILURES + (ToolchainRefusal,):
            return f"{layer}_unbuildable"
        if layer == "relabel" and mismatches == 0:
            return "relabel"
        if layer in ("geometry", "packing") and mismatches:
            return layer
    return "pass_cut"


def _levels(mapping) -> int:
    return sum(len(getattr(s, "retimed_level_stages", None) or ()) for s in mapping.stages if s.kind == "neural")


def _producer_at_or_after_consumer(mapping) -> bool:
    """Whether some core reads an in-segment producer whose stored latency is not below its own."""
    for stage in mapping.stages:
        if stage.kind != "neural":
            continue
        cores = stage.hard_core_mapping.cores
        for core in cores:
            for source in getattr(core, "axon_sources", ()) or ():
                index = getattr(source, "core_", None)
                if index is None or any(getattr(source, flag, False) for flag in ("is_input_", "is_off_", "is_always_on_")):
                    continue
                producer = cores[int(index)] if 0 <= int(index) < len(cores) else None
                if producer is not None and producer is not core and producer.latency is not None \
                        and core.latency is not None and int(producer.latency) >= int(core.latency):
                    return True
    return False


def owner_of(tags: Tuple[str, ...]) -> Dict[str, Any]:
    named = [t for t in tags if t in MECHANISMS]
    return {"tags": list(tags), "owners": [MECHANISMS[t] for t in named]}
