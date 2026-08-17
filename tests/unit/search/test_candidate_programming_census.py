"""[E2] What the candidate pays to PROGRAM the chip it just laid out.

Three multiplicands were absent at candidate time, so three priced terms were
silently zero for every searched candidate: per-core initialization
(``e_core_init`` x ``segment_cores``, ``t_core_init``), per-core programming
overhead (``e_core_program`` x ``reprogrammed_cores``) and the DMA payload
(``e_dma_per_byte`` x ``reprogrammed_bytes``).

The owner's accounting rule is the whole design here: a pass whose weights are
already resident sends NO payload and pays NO per-core programming — but it
still resets its cores' neuron state, so it DOES pay core init. Charging every
pass as a reprogram would have made bank-clustered scheduling — the composition
all 12 IMC presets declare — look identical to reprogramming from scratch.

The census is checked against the DEPLOYED record built from the same graph,
not against a re-derivation: same vehicle, same schedule, same numbers.
"""

from __future__ import annotations

import pytest

from mimarsinan.deployment_record.build.payload_sizes import params_bytes
from mimarsinan.mapping.noc import collect_noc_fragments
from mimarsinan.mapping.support.schedule.pass_planner import resident_passes
from mimarsinan.search.problems.joint.candidate_fragments import (
    candidate_programming_census,
)

from unit.mapping.bank_clustered_vehicles import (
    TWO_CORES,
    hard_core_types,
    softcores_of,
    token_graph,
)

WEIGHT_BITS = 4


def _census(graph=None, weight_bits: int = WEIGHT_BITS, *, unbanked=False):
    """The candidate's programming census under the composed schedule.

    ``unbanked`` strips the shared-bank identity, putting the same shapes on
    the capacity composition — the reprograms-every-pass contrast vehicle."""
    from dataclasses import replace

    graph = token_graph(7) if graph is None else graph
    softcores = softcores_of(graph)
    if unbanked:
        softcores = [replace(sc, bank_id=None) for sc in softcores]
    core_types = hard_core_types(TWO_CORES)
    noc = collect_noc_fragments(
        softcores=softcores, core_types=core_types, census=None,
        allow_scheduling=True, allow_neuron_splitting=False,
        allow_coalescing=False, max_schedule_passes=8,
    )
    return candidate_programming_census(noc, weight_bits=weight_bits)


def _deployed_census(graph=None, weight_bits: int = WEIGHT_BITS):
    """The same census read off the sealed record of the deployed program."""
    from mimarsinan.deployment_record.build.from_mapping import (
        schedule_record_from_mapping,
    )
    from mimarsinan.mapping.packing.hybrid_build_pool import (
        build_hybrid_hard_core_mapping,
    )
    from mimarsinan.mapping.platform.mapping_structure import (
        ChipCapabilities,
        MappingStrategy,
    )

    hybrid = build_hybrid_hard_core_mapping(
        ir_graph=token_graph(7) if graph is None else graph,
        cores_config=[dict(ct) for ct in TWO_CORES],
        strategy=MappingStrategy.resolve(
            ChipCapabilities(allow_scheduling=True)
        ),
    )
    record = schedule_record_from_mapping(
        hybrid, weight_bits=weight_bits, params_reloaded=0,
    )
    segments = list(record.segments())
    reprogrammed = [s for s in segments if s.programming == "reprogram"]
    return {
        "segment_cores": sum(len(s.cores) for s in segments),
        "reprogrammed_cores": sum(len(s.cores) for s in reprogrammed),
        "reprogrammed_bytes": sum(s.params_bytes for s in reprogrammed),
        "reprogram_passes": len(reprogrammed),
    }


class TestTheResidencyLawIsOneLaw:
    def test_a_bank_clustered_segment_programs_only_its_head_pass(self):
        assert resident_passes(4, policy_applied=True) == (False, True, True, True)

    def test_a_capacity_split_segment_reprograms_every_pass(self):
        """The pool composition places DIFFERENT weights each pass — nothing
        stays resident, so nothing may be credited as resident."""
        assert resident_passes(4, policy_applied=False) == (False,) * 4

    def test_a_single_pass_program_has_nothing_to_reuse(self):
        assert resident_passes(1, policy_applied=True) == (False,)

    def test_the_deployed_marker_reads_this_law(self):
        """``mark_bank_residency`` must not carry a second copy of the rule."""
        import inspect

        from mimarsinan.mapping.packing import schedule_bank_clustered

        assert "resident_passes" in inspect.getsource(schedule_bank_clustered)


class TestTheCandidateMatchesTheDeployedCensus:
    def test_every_programming_quantity_agrees_with_the_sealed_record(self):
        candidate = _census()
        deployed = _deployed_census()
        assert {
            "segment_cores": candidate.segment_cores,
            "reprogrammed_cores": candidate.reprogrammed_cores,
            "reprogrammed_bytes": candidate.reprogrammed_bytes,
            "reprogram_passes": candidate.reprogram_passes,
        } == deployed


class TestAllocationMeansTheSameThingOnBothSides:
    """``cores_allocated`` named the DECLARED chip's core count at candidate
    time and the mapping's allocated cores in the record — one name, two
    meanings, off by the whole chip (36 vs 3 on the study MLP). Nothing
    priced it yet, which is exactly why it could drift unnoticed."""

    def test_the_candidate_allocates_what_the_deployed_mapping_allocates(self):
        from mimarsinan.mapping.crossbar_utilization import (
            CrossbarUtilizationReport,
        )
        from mimarsinan.mapping.packing.hybrid_build_pool import (
            build_hybrid_hard_core_mapping,
        )
        from mimarsinan.mapping.platform.mapping_structure import (
            ChipCapabilities,
            MappingStrategy,
        )

        hybrid = build_hybrid_hard_core_mapping(
            ir_graph=token_graph(7),
            cores_config=[dict(ct) for ct in TWO_CORES],
            strategy=MappingStrategy.resolve(
                ChipCapabilities(allow_scheduling=True)
            ),
        )
        deployed = CrossbarUtilizationReport.from_hybrid_mapping(hybrid)
        assert _census().segment_cores == deployed.cores_allocated


class TestResidencyIsWorthMoney:
    def test_the_streamed_composition_pays_core_init_on_every_pass(self):
        """Resident or not, a pass resets its cores — core init is per PASS."""
        streamed = _census()
        assert streamed.segment_cores > streamed.reprogrammed_cores

    def test_the_streamed_composition_sends_one_pass_worth_of_payload(self):
        """The composition streams ONE shared bank: the head programs it and
        every later pass reuses it, so the payload does not scale with passes.
        The same shapes without the bank reprogram every pass — the cost the
        unified scheduler exists to avoid."""
        streamed = _census()
        assert streamed.reprogram_passes == 1
        assert streamed.reprogrammed_bytes < _census(unbanked=True).reprogrammed_bytes

    def test_the_capacity_composition_reprograms_every_pass(self):
        capacity = _census(unbanked=True)
        assert capacity.reprogrammed_cores == capacity.segment_cores
        assert capacity.reprogram_passes == len(capacity.pass_cores)


class TestThePayloadIsSizedByTheSharedRule:
    def test_bytes_come_from_the_payload_ssot_at_the_declared_width(self):
        census = _census()
        assert census.reprogrammed_bytes == sum(
            params_bytes(cells, WEIGHT_BITS) for cells in census.reprogrammed_cells
        )

    def test_a_wider_weight_moves_the_payload_and_nothing_else(self):
        narrow, wide = _census(weight_bits=4), _census(weight_bits=8)
        assert wide.reprogrammed_bytes > narrow.reprogrammed_bytes
        assert wide.segment_cores == narrow.segment_cores
        assert wide.reprogrammed_cores == narrow.reprogrammed_cores

    def test_an_undeclared_width_drops_the_payload_and_keeps_the_counts(self):
        """A width nobody declared cannot size bytes — so the payload is
        ABSENT (its term refuses by name) while the core counts, which need no
        width, still count. Losing the whole census here would cost a
        candidate its core-init term over an unrelated declaration."""
        census = _census(weight_bits=None)
        assert census.reprogrammed_bytes is None
        assert census.segment_cores == _census().segment_cores
        assert census.reprogrammed_cores == _census().reprogrammed_cores


class TestAsMappedCells:
    """[R1] The event model's multiplicand: replicas really fire."""

    def test_committed_cells_match_the_deployed_crossbar(self):
        """The token graph replicates ONE shared bank across instances — the
        as-mapped figure counts every replica, exactly as the record's
        crossbar does. The logical census would count the bank once."""
        from mimarsinan.mapping.crossbar_utilization import (
            CrossbarUtilizationReport,
        )
        from mimarsinan.mapping.packing.hybrid_build_pool import (
            build_hybrid_hard_core_mapping,
        )
        from mimarsinan.mapping.platform.mapping_structure import (
            ChipCapabilities,
            MappingStrategy,
        )

        hybrid = build_hybrid_hard_core_mapping(
            ir_graph=token_graph(7),
            cores_config=[dict(ct) for ct in TWO_CORES],
            strategy=MappingStrategy.resolve(
                ChipCapabilities(allow_scheduling=True)
            ),
        )
        deployed = CrossbarUtilizationReport.from_hybrid_mapping(hybrid)
        assert _census().committed_cells == deployed.cells_used

    def test_modeled_events_scale_with_replication(self):
        """Seven replicated instances must model ~7x one instance's events —
        the H5 finding (record-plane ~108x the logical model on offload
        LeNet5) made structural."""
        from mimarsinan.deployment_record.quantities.from_candidate import (
            CandidateQuantityContext,
            from_candidate,
        )

        def _events(cells):
            quantities = from_candidate(
                layout=None, chip_param_capacity=None, total_params=None,
                host_side_segment_count=None,
                context=CandidateQuantityContext(
                    timesteps=4, activity_factor=0.05, cells_committed=cells,
                ),
            )
            return quantities.get("synaptic_events").value

        one = _census().committed_cells // 7
        assert _events(7 * one) == pytest.approx(7 * _events(one))

    def test_the_replicated_shape_is_priced_at_its_replicas_not_its_logic(self):
        """The discriminating case: logical census 100, as-mapped 700 (one
        bank, seven firing replicas). The model must multiply the replicas —
        the logical shortcut is exactly the ~108x LeNet5 understatement."""
        from mimarsinan.deployment_record.quantities.from_candidate import (
            CandidateQuantityContext,
            from_candidate,
        )

        quantities = from_candidate(
            layout=None, chip_param_capacity=None, total_params=None,
            host_side_segment_count=None,
            context=CandidateQuantityContext(
                timesteps=4, activity_factor=0.05,
                onchip_macs=100, cells_committed=700,
            ),
        )
        assert quantities.get("synaptic_events").value == pytest.approx(
            700 * 4 * 0.05
        )
