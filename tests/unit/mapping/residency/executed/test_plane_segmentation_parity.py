"""[U1 W9, review round 2] Key absent, every plane plans the program the builder deploys.

The builder executes the IR-run segments (``partition_ir_graph``) and its capacity split cuts each at the specs'
depth segments (``segment_id``), as on the base. The stats plane (``compute_mapping_stats``: layout verification and
the search; ``verify_hardware_config``: the wizard preview) and the census plane (``collect_noc_fragments``: the
search's NoC, latency and programming terms) consume the same planner:

- given the walk's runs (``spec_runs``), each plane plans exactly the builder's passes, pass counts and syncs;
- as production calls them today (no runs) they group the specs by depth segment, which is the builder's program
  wherever the two segmentations group the specs alike (``_segmentations_agree``), and otherwise the named
  pre-existing divergence of the report, which the runs close once the callers pass them (integration edits).

Witness of the review's blocker: fuzz seed 0 at C 3, where a builder that planned each IR-run segment whole deployed
3 passes while both planes, as production calls them, planned 4."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import pytest

from mimarsinan.mapping.layout.layout_ir_mapping import LayoutIRMapping
from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType
from mimarsinan.mapping.noc import collect_noc_fragments
from mimarsinan.mapping.noc.fragments import walk_runs
from mimarsinan.mapping.noc.wire_census import census_of_walk
from mimarsinan.mapping.packing.hybrid_build_scheduled import PlannedPassPackingRefusal
from mimarsinan.mapping.support.residency_rules import ProgramGraph, spec_runs
from mimarsinan.mapping.support.schedule.pass_planner import plan_program_passes
from mimarsinan.mapping.verification.layout_verification_scheduling import compute_mapping_stats
from mimarsinan.mapping.verification.verifier.mapping_verifier_hw import (
    clear_verify_hardware_config_memo,
    verify_hardware_config,
)

from . import vehicles as V

BUDGETS = (2, 3, 4, 5, 6, 8)
#: Fuzz seeds of the sweep (the reviewers' generator, ``V.random_rep``).
SWEEP = range(0, 100)

Partition = List[Tuple[int, ...]]


def _walk(rep):
    walk = LayoutIRMapping(max_axons=V.MAX_AXONS, max_neurons=V.MAX_NEURONS, collect_wire_census=True)
    return walk, walk.collect_layout_softcores(rep)


def _segmentations_agree(specs, runs: Sequence[int]) -> bool:
    """The class predicate: the builder's segments (runs) and the depth segments group the specs alike."""
    by_run: Dict[int, set] = {}
    by_depth: Dict[int, set] = {}
    for i, (spec, run) in enumerate(zip(specs, runs)):
        by_run.setdefault(int(run), set()).add(i)
        by_depth.setdefault(int(spec.segment_id or 0), set()).add(i)
    return sorted(map(sorted, by_run.values())) == sorted(map(sorted, by_depth.values()))


def _builder(rep, count: int) -> Tuple[Partition, int]:
    """The deployed passes (layers per pass) and their syncs (passes - 1 per schedule segment)."""
    ir = V.ir_of(rep)
    hybrid = V.build(ir, count, None)
    per_segment: Dict[int, int] = {}
    for stage in V.neural_passes(hybrid):
        per_segment[int(stage.schedule_segment_index)] = per_segment.get(int(stage.schedule_segment_index), 0) + 1
    return V.built_pass_layers(hybrid, ir), sum(n - 1 for n in per_segment.values())


def _planes(rep, count: int, runs: Optional[Sequence[int]]) -> Dict[str, object]:
    """Each plane's answer for the vehicle at ``count`` cores, called as production calls it plus ``runs``."""
    walk, specs = _walk(rep)
    census = census_of_walk(walk)
    caps = V.capabilities()
    types = [LayoutHardCoreType(V.MAX_AXONS, V.MAX_NEURONS, count)]
    plan = plan_program_passes(specs, types, max_schedule_passes=caps.max_schedule_passes, runs=runs)
    stats, error = compute_mapping_stats(specs, types, runs=runs, **caps.layout_kwargs())
    assert error is None
    clear_verify_hardware_config_memo()
    preview = verify_hardware_config(specs, V.cores(count), runs=runs, **caps.layout_kwargs())["stats"]
    frag = collect_noc_fragments(softcores=specs, core_types=types, census=census, runs=runs, **caps.layout_kwargs())
    return dict(
        planner=[tuple(sorted({s.perceptron_index for s in chunk})) for chunk in plan.pass_lists],
        census=V.fragment_pass_layers(frag, specs),
        stats=(stats.schedule_pass_count, stats.schedule_sync_count),
        preview=(preview["schedule_pass_count"], preview["schedule_sync_count"]),
    )


def _assert_planes_deploy(rep, count: int, runs: Optional[Sequence[int]]) -> None:
    built, syncs = _builder(rep, count)
    planes = _planes(rep, count, runs)
    assert planes["planner"] == built
    assert planes["census"] == built
    assert planes["stats"] == (len(built), syncs)
    assert planes["preview"] == (len(built), syncs)


class TestTheBlockerWitness:
    """Fuzz seed 0: one builder segment (the encoder runs on the host) holding depth segments 0 and 1."""

    def test_fuzz_seed_0_at_3_cores_plans_the_deployed_passes_on_every_plane_as_production_calls_them(self):
        rep = V.random_rep(0)
        built, _ = _builder(rep, 3)
        planes = _planes(rep, 3, None)
        assert len(built) == 4
        assert planes["planner"] == planes["census"] == built
        assert planes["stats"][0] == planes["preview"][0] == len(built)

    def test_given_the_runs_its_syncs_are_the_builders_too(self):
        """As production calls them the planes count a sync per depth segment's passes (2, the base's count too);
        the builder syncs its one segment's 4 passes 3 times. The runs close it."""
        rep = V.random_rep(0)
        walk, _ = _walk(rep)
        _assert_planes_deploy(rep, 3, spec_runs(walk))


class TestTheRunsSeam:
    def test_the_walks_runs_are_the_program_graphs(self):
        """``spec_runs`` reads the walk's node records alone (no wire census) and equals the program graph's runs."""
        for seed in (0, 2, 5, 71):
            walk, specs = _walk(V.random_rep(seed))
            assert spec_runs(walk) == walk_runs(ProgramGraph.of_walk(walk), len(specs))

    @pytest.mark.parametrize("kind", ["edge_class", "intra_to_cross", "cores_only", "deep_mismatch",
                                      "moved_to_output", "shift_streamed", "two_hosts"])
    @pytest.mark.parametrize("count", [2, 3, 5, 8])
    def test_with_the_runs_every_plane_plans_the_builders_passes(self, kind, count):
        rep = V.branchy_repr(kind)
        walk, _ = _walk(rep)
        _assert_planes_deploy(rep, count, spec_runs(walk))


class TestTheFuzzSweep:
    """Key absent, over the fuzz seeds of the sweep at every budget: as production calls them, the planes plan the
    deployed program on every vehicle whose segmentations agree; on the others (the pre-existing class) they plan it
    given the runs. Both classes are populated, so neither clause is vacuous."""

    def test_every_plane_plans_the_deployed_program(self):
        agree = disagree = 0
        for seed in SWEEP:
            rep = V.random_rep(seed)
            walk, specs = _walk(rep)
            runs = spec_runs(walk)
            same = _segmentations_agree(specs, runs)
            for count in BUDGETS:
                try:
                    _builder(rep, count)
                except PlannedPassPackingRefusal:      # the residency-class refusal, not a plane question
                    continue
                _assert_planes_deploy(rep, count, None if same else runs)
                agree, disagree = agree + same, disagree + (not same)
        assert agree > 100 and disagree > 20
