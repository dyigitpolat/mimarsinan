"""[H3] One estimand per name: utilization is ALLOCATED-based on both planes.

The fork this closes, measured on the loihi study run: the candidate reported
5.354% (used cells over the WHOLE declared chip, idle cores padded in) while
the record reported 10.709% (used cells over ALLOCATED cores) — the same axis
name, an exact 2.00x apart, and fidelity dutifully reported the split without
anyone noticing the denominators differed. The catalog's own text ("share of
allocated crossbar cells") sides with the record, so the candidate now
measures the record's estimand and the whole-chip signal gets its own name:
``chip_occupancy_pct``.
"""

from __future__ import annotations

import pytest

from mimarsinan.mapping.crossbar_utilization import CrossbarUtilizationReport
from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType
from mimarsinan.mapping.packing.hybrid_build_pool import (
    build_hybrid_hard_core_mapping,
)
from mimarsinan.mapping.platform.mapping_structure import (
    ChipCapabilities,
    MappingStrategy,
)
from mimarsinan.mapping.verification.layout_verification_scheduling import (
    compute_mapping_stats,
)

from unit.mapping.bank_clustered_vehicles import (
    hard_core_types,
    softcores_of,
    token_graph,
)

#: More cores than the program needs, so idle cores exist to fork on.
ROOMY_CHIP = [{"max_axons": 32, "max_neurons": 32, "count": 8}]


def _both_planes(chip=ROOMY_CHIP):
    graph = token_graph(7)
    stats, error = compute_mapping_stats(
        softcores=softcores_of(graph),
        core_types=hard_core_types(chip),
        allow_scheduling=True,
    )
    assert error is None
    hybrid = build_hybrid_hard_core_mapping(
        ir_graph=graph, cores_config=[dict(ct) for ct in chip],
        strategy=MappingStrategy.resolve(
            ChipCapabilities(allow_scheduling=True)
        ),
    )
    return stats, CrossbarUtilizationReport.from_hybrid_mapping(hybrid)


class TestOneEstimandPerName:
    def test_utilization_matches_the_deployed_crossbar_occupancy(self):
        stats, crossbar = _both_planes()
        assert stats.mapped_params_pct == pytest.approx(
            crossbar.cell_occupancy * 100.0, rel=1e-6,
        )

    def test_wastage_matches_the_deployed_utilization_complement(self):
        stats, crossbar = _both_planes()
        assert stats.total_wasted_neurons_pct == pytest.approx(
            (1.0 - crossbar.neuron_utilization) * 100.0, rel=1e-6,
        )
        assert stats.total_wasted_axons_pct == pytest.approx(
            (1.0 - crossbar.axon_utilization) * 100.0, rel=1e-6,
        )


class TestTheChipSignalKeepsItsOwnName:
    def test_chip_occupancy_divides_by_the_declared_chip(self):
        stats, crossbar = _both_planes()
        declared_cells = sum(
            ct["max_axons"] * ct["max_neurons"] * ct["count"] for ct in ROOMY_CHIP
        )
        assert stats.chip_occupancy_pct == pytest.approx(
            100.0 * crossbar.cells_used / declared_cells, rel=1e-6,
        )

    def test_the_two_figures_really_differ_on_a_roomy_chip(self):
        """Idle cores exist here, so the fork would be visible: occupancy is
        strictly below utilization — the 2x-class gap the fidelity zip showed."""
        stats, _ = _both_planes()
        assert stats.chip_occupancy_pct < stats.mapped_params_pct


class TestTheRecordMirrorDividesByTheDeclaredChip:
    def test_mirror_occupancy_equals_the_candidates_estimand(self):
        """[R1/R0] The emission computed the mirror's stats WITHOUT the
        declared chip, so record-plane occupancy degenerated to utilization
        (measured occupancy == measured utilization == 15.6379 on the sealed
        loihi run). With the declaration threaded, both planes divide the
        same committed cells by the same declared capacity."""
        from mimarsinan.mapping.verification.layout_verification_hybrid import (
            stats_dict_from_hybrid_mapping,
        )

        graph = token_graph(7)
        stats, _ = _both_planes()
        hybrid = build_hybrid_hard_core_mapping(
            ir_graph=graph, cores_config=[dict(ct) for ct in ROOMY_CHIP],
            strategy=MappingStrategy.resolve(
                ChipCapabilities(allow_scheduling=True)
            ),
        )
        mirror = stats_dict_from_hybrid_mapping(
            hybrid, core_types=hard_core_types(ROOMY_CHIP),
        )
        assert mirror["chip_occupancy_pct"] == pytest.approx(
            stats.chip_occupancy_pct, rel=1e-6,
        )
        assert mirror["chip_occupancy_pct"] < mirror["mapped_params_pct"]

    def test_an_undeclared_chip_keeps_the_old_degenerate_reading(self):
        """Without the declaration the mirror can only divide by what it
        sees — stated, and exactly why the emission now threads it."""
        from mimarsinan.mapping.verification.layout_verification_hybrid import (
            stats_dict_from_hybrid_mapping,
        )

        hybrid = build_hybrid_hard_core_mapping(
            ir_graph=token_graph(7),
            cores_config=[dict(ct) for ct in ROOMY_CHIP],
            strategy=MappingStrategy.resolve(
                ChipCapabilities(allow_scheduling=True)
            ),
        )
        mirror = stats_dict_from_hybrid_mapping(hybrid)
        assert mirror["chip_occupancy_pct"] == pytest.approx(
            mirror["mapped_params_pct"], rel=1e-6,
        )


class TestMultiPassAggregation:
    #: Two cores stream seven token instances in four passes — the multi-pass
    #: aggregation vehicle (the roomy chip composes a single resident pass).
    TIGHT_CHIP = [{"max_axons": 32, "max_neurons": 32, "count": 2}]

    def test_a_scheduled_programs_utilization_aggregates_every_pass(self):
        """[R5] The deepcnn witness gated 33.3% (one pass's figure) against a
        measured 40.8% (all passes). The candidate sums committed and
        allocated over EVERY pass — the record's own aggregation."""
        stats, crossbar = _both_planes(chip=self.TIGHT_CHIP)
        assert stats.schedule_pass_count == 4
        assert stats.mapped_params_pct == pytest.approx(
            crossbar.cell_occupancy * 100.0, rel=1e-6,
        )

    def test_the_single_resident_pass_aggregates_too(self):
        stats, crossbar = _both_planes()
        assert stats.mapped_params_pct == pytest.approx(
            crossbar.cell_occupancy * 100.0, rel=1e-6,
        )
