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


def _both_planes(policy="pool"):
    graph = token_graph(7)
    stats, error = compute_mapping_stats(
        softcores=softcores_of(graph),
        core_types=hard_core_types(ROOMY_CHIP),
        allow_scheduling=True, schedule_policy=policy,
    )
    assert error is None
    hybrid = build_hybrid_hard_core_mapping(
        ir_graph=graph, cores_config=[dict(ct) for ct in ROOMY_CHIP],
        strategy=MappingStrategy.resolve(
            ChipCapabilities(allow_scheduling=True, schedule_policy=policy)
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
