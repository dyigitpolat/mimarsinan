"""[H2] The candidate sizes the carry of the program it just planned.

``schedule_policy`` is a search axis, and carry is precisely the cost that
axis moves — yet the candidate produced NO carry quantity, so the search
optimized that dimension blind. The census function is the record's own
(`carry_census_from_spans`); only the span producer differs, so the numbers
are the same numbers by construction — checked here against the DEPLOYED
program's census on the same graph.
"""

from __future__ import annotations

import pytest

from mimarsinan.mapping.noc import collect_noc_fragments
from mimarsinan.mapping.support.schedule.pass_carry import (
    carried_softcore_spans,
    carry_census_from_spans,
    pass_carry_census,
)
from mimarsinan.mapping.support.schedule.pass_cut import COLLAPSE, VERBATIM

from unit.mapping.bank_clustered_vehicles import (
    TWO_CORES,
    hard_core_types,
    softcores_of,
    token_graph,
    two_layer_dependency_graph,
)

NARROW_CHIP = [{"max_axons": 16, "max_neurons": 8, "count": 2}]
T = 16


def _pair_wires(graph):
    """Adjacency the wire census would record: distinct producer→consumer
    wires, read from the vehicles' own IR sources (node ids are positional)."""
    pairs = {}
    for consumer_index, core in enumerate(graph.get_neural_cores()):
        for source in core.input_sources.ravel():
            if source.node_id < 0:
                continue
            key = (int(source.node_id), consumer_index)
            pairs[key] = pairs.get(key, 0) + 1
    return pairs


def _candidate_census(graph, policy, transfer, cores):
    softcores = softcores_of(graph)
    noc = collect_noc_fragments(
        softcores=softcores, core_types=hard_core_types(cores), census=None,
        allow_scheduling=True, allow_neuron_splitting=False,
        allow_coalescing=False, schedule_policy=policy, max_schedule_passes=8,
    )
    spans = carried_softcore_spans(
        softcores, noc.pass_placements, _pair_wires(graph),
    )
    return carry_census_from_spans(
        spans, boundary_count=len(noc.pass_placements),
        timesteps=T, transfer=transfer,
    )


def _deployed_census(graph, policy, transfer, cores):
    from mimarsinan.mapping.packing.hybrid_build_pool import (
        build_hybrid_hard_core_mapping,
    )
    from mimarsinan.mapping.platform.mapping_structure import (
        ChipCapabilities,
        MappingStrategy,
    )

    hybrid = build_hybrid_hard_core_mapping(
        ir_graph=graph, cores_config=[dict(ct) for ct in cores],
        strategy=MappingStrategy.resolve(
            ChipCapabilities(allow_scheduling=True, schedule_policy=policy)
        ),
    )
    return pass_carry_census(hybrid.stages, T, transfer)


class TestTheCandidateMatchesTheDeployedCensus:
    @pytest.mark.parametrize("transfer", [VERBATIM, COLLAPSE])
    def test_a_dependency_cut_carries_the_same_bytes_both_planes(self, transfer):
        """Layer 1 consumes layer 0 across the pass cut: every figure —
        buffer, peak, both boundary directions — equals the deployed one."""
        graph = two_layer_dependency_graph(3, 4)
        candidate = _candidate_census(graph, "pool", transfer, NARROW_CHIP)
        deployed = _deployed_census(graph, "pool", transfer, NARROW_CHIP)
        assert candidate == deployed
        assert candidate["carried_wires"] > 0

    @pytest.mark.parametrize("policy", ["pool", "bank_clustered"])
    def test_an_independent_token_stream_carries_nothing_on_either_plane(
        self, policy,
    ):
        """The token graph's passes are data-independent: a KNOWN zero on
        both planes — the search must see 0, never refuse."""
        graph = token_graph(7)
        candidate = _candidate_census(graph, policy, VERBATIM, TWO_CORES)
        deployed = _deployed_census(graph, policy, VERBATIM, TWO_CORES)
        assert candidate == deployed
        assert candidate["carried_bytes"] == 0


class TestTheBoundaryRules:
    def test_a_cross_segment_wire_is_a_host_boundary_never_a_carry(self):
        """Two segments in two passes with adjacency between them: that wire
        collapses at the HOST boundary by design — counting it as carry would
        double-charge what the host re-encode already owns."""

        class _Spec:
            def __init__(self, segment):
                self.segment_id = segment
                self.output_count = 8

        spans = carried_softcore_spans(
            [_Spec(0), _Spec(1)],
            pass_placements=(((0, 0),), ((1, 0),)),
            pair_wires={(0, 1): 4},
        )
        assert spans == ()

    def test_no_wire_census_keeps_carry_absent_not_crashed(self):
        """An active set that never asked for the walk: adjacency is unknown,
        so the census is None — and the layout must still resolve."""
        from unit.search.test_candidate_fragments_live_path import (
            _candidate,
            _cfg,
            _problem,
        )

        problem = _problem(
            _cfg(), ["param_utilization_pct"], pass_transfer=COLLAPSE,
        )
        layout = problem.candidate_layout(_candidate(problem))
        assert layout.noc is not None and layout.noc.census is None
        assert not layout.view.quantities.has("carried_raster_bytes")


class TestTheLiveCandidateClaimsCarry:
    def test_the_quantities_ride_the_live_path_as_known_zeros(self):
        """The fixture MLP is single-pass: all four carry quantities present
        at 0 — an optimizer minimizing carry scores it best, not unknown."""
        from unit.search.test_candidate_fragments_live_path import (
            _candidate,
            _physics_cfg,
            _problem,
        )

        problem = _problem(
            _physics_cfg(), ["carry_peak_live_bytes", "carried_raster_bytes"],
            pass_transfer=COLLAPSE,
        )
        q = problem.candidate_layout(_candidate(problem)).view.quantities
        for key in ("carried_raster_bytes", "carry_peak_live_bytes",
                    "carry_out_bytes", "carry_in_bytes"):
            assert q.get(key).value == 0.0, key
            assert q.get(key).provenance == "static"

    def test_an_undeclared_discipline_keeps_carry_absent(self):
        """No transfer discipline, no census: the two disciplines are
        different computations, so no number may stand in."""
        from unit.search.test_candidate_fragments_live_path import (
            _candidate,
            _physics_cfg,
            _problem,
        )

        problem = _problem(
            _physics_cfg(), ["carry_peak_live_bytes"], pass_transfer=None,
        )
        q = problem.candidate_layout(_candidate(problem)).view.quantities
        assert not q.has("carried_raster_bytes")
