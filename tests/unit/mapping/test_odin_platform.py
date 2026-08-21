"""[ODIN4] The stock-ODIN platform declaration and the physical-expansion accounting.

Plan Sec.5.2 / Sec.14 J2-6 and J2-10: the platform is registered through the
NON-literature seam with its own provenance, and the row-pair expansion factor
derived from `weight_sign_granularity` feeds a PHYSICAL twin of the crossbar
ledger. The logical ledger is a model quantity and stays untouched: a signed-cell
substrate's 2x is a substrate-cost multiplier, never a re-interpretation of
`cells_used`.
"""

import json
import os

import pytest

from mimarsinan.mapping.crossbar_utilization import (
    CoreOccupancy,
    CrossbarUtilizationReport,
)
from mimarsinan.mapping.packing.softcore import HardCore
from mimarsinan.mapping.platform.imc_platforms import (
    get_imc_platform,
    imc_platform_names,
)
from mimarsinan.chip_simulation.soma_axes import physical_row_expansion

ODIN_PLATFORM = "odin_stock_core"

PHYSICS_PROFILE = os.path.join(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ),
    "src", "mimarsinan", "deployment_record", "platform_physics", "profiles",
    "odin.json",
)


class TestTheExpansionFactorIsDerivedFromTheDeclaration:
    def test_per_synapse_signs_cost_one_physical_row_per_logical_slot(self):
        assert physical_row_expansion("per_synapse") == 1

    def test_per_axon_signs_cost_the_excitatory_inhibitory_pair(self):
        assert physical_row_expansion("per_axon") == 2

    def test_an_undeclared_granularity_is_the_framework_default_of_one(self):
        assert physical_row_expansion(None) == 1

    def test_an_unknown_granularity_is_refused_rather_than_defaulted(self):
        with pytest.raises(ValueError, match="per_axon"):
            physical_row_expansion("per_core")


class TestThePhysicalTwinIsDefaultOneByteIdentical:
    def _occupancy(self, **kwargs):
        return CoreOccupancy(
            axons_used=100, axons_physical=128,
            neurons_used=200, neurons_physical=256,
            unusable_space=0, **kwargs,
        )

    def test_the_default_expansion_leaves_every_physical_figure_unchanged(self):
        occ = self._occupancy()
        assert occ.sign_expansion == 1
        assert occ.rows_physical == occ.axons_physical
        assert occ.cells_physical_expanded == occ.cells_physical

    def test_the_logical_ledger_never_moves_with_the_expansion(self):
        plain = self._occupancy()
        expanded = self._occupancy(sign_expansion=2)
        assert expanded.cells_used == plain.cells_used
        assert expanded.axons_used == plain.axons_used
        assert expanded.occupancy == plain.occupancy

    def test_the_physical_twin_multiplies_by_the_declared_factor(self):
        occ = self._occupancy(sign_expansion=2)
        assert occ.rows_physical == 256
        assert occ.cells_physical_expanded == 256 * 256

    def test_the_flat_record_still_has_exactly_its_fourteen_keys(self):
        report = CrossbarUtilizationReport.from_hard_cores(
            [HardCore(128, 256)], weight_bits=4, sign_expansion=2
        )
        assert len(report.to_dict()) == 14
        assert "sign_expansion" not in report.to_dict()

    def test_the_report_aggregates_the_physical_twin(self):
        report = CrossbarUtilizationReport.from_hard_cores(
            [HardCore(128, 256), HardCore(128, 256)],
            weight_bits=4, sign_expansion=2,
        )
        assert report.cells_physical == 2 * 128 * 256
        assert report.cells_physical_expanded == 2 * 256 * 256

    def test_an_expansion_below_one_is_refused(self):
        with pytest.raises(ValueError, match="sign_expansion"):
            CoreOccupancy(
                axons_used=1, axons_physical=1, neurons_used=1,
                neurons_physical=1, unusable_space=0, sign_expansion=0,
            )


class TestTheStockOdinPlatformDeclaration:
    def test_it_is_registered_under_its_own_name(self):
        assert ODIN_PLATFORM in imc_platform_names()

    def test_the_geometry_is_the_logical_twin_of_the_physical_crossbar(self):
        platform = get_imc_platform(ODIN_PLATFORM)
        assert len(platform.cores) == 1
        core = platform.cores[0]
        assert (int(core["max_axons"]), int(core["max_neurons"])) == (128, 256)
        assert int(core["count"]) == 1
        assert core["has_bias"] is False

    def test_the_weight_width_is_the_four_bit_synapse(self):
        assert get_imc_platform(ODIN_PLATFORM).weight_bits == 4

    def test_the_capabilities_declare_the_per_axon_sign_and_the_membrane_width(self):
        constraints = get_imc_platform(ODIN_PLATFORM).to_platform_constraints()
        assert constraints["weight_sign_granularity"] == "per_axon"
        assert constraints["membrane_bits"] == 8
        assert constraints["allow_coalescing"] is False
        assert constraints["allow_neuron_splitting"] is False

    def test_the_provenance_cites_both_the_paper_and_the_upstream_repo(self):
        provenance = get_imc_platform(ODIN_PLATFORM).provenance
        assert "Frenkel" in provenance
        assert "ChFrenkel/ODIN" in provenance

    def test_the_eligibility_is_the_honest_single_core_class(self):
        # One core on the die (population 1) is exactly the degenerate case the
        # curated table quarantines; the geometry is additionally a LOGICAL twin
        # of the quoted 256x256, so a headline claim would be doubly wrong.
        assert get_imc_platform(ODIN_PLATFORM).claim_eligibility == "quarantined"

    def test_the_twelve_literature_platforms_are_untouched(self):
        from unit.mapping.test_imc_platform_registry_content import (
            EXPECTED_GEOMETRY,
        )

        assert ODIN_PLATFORM not in EXPECTED_GEOMETRY
        assert len(EXPECTED_GEOMETRY) == 12


class TestTheAreaAndCellPin:
    def test_declared_geometry_times_expansion_is_the_published_64k_synapses(self):
        platform = get_imc_platform(ODIN_PLATFORM)
        constraints = platform.to_platform_constraints()
        expansion = physical_row_expansion(constraints["weight_sign_granularity"])
        assert platform.total_cells * expansion == 65536

    def test_the_physical_twin_of_a_fully_packed_core_is_the_same_64k(self):
        core = HardCore(128, 256, has_bias_capability=False)
        core.available_axons = 0
        core.available_neurons = 0
        report = CrossbarUtilizationReport.from_hard_cores(
            [core], weight_bits=4, sign_expansion=2
        )
        assert report.cells_physical_expanded == 65536

    def test_the_area_profile_prices_that_one_core_at_86400_um2(self):
        with open(PHYSICS_PROFILE, encoding="utf-8") as handle:
            profile = json.load(handle)
        assert profile["name"] == "odin"
        assert profile["constants"]["area_per_core_total"]["nominal"] == 86400.0
        assert "64k-synapse" in profile["validity"]["array_size_assumed"]

    def test_the_per_cell_area_times_the_expanded_cells_stays_inside_the_core(self):
        with open(PHYSICS_PROFILE, encoding="utf-8") as handle:
            profile = json.load(handle)
        per_cell = profile["constants"]["area_per_cell"]["nominal"]
        total = profile["constants"]["area_per_core_total"]["nominal"]
        platform = get_imc_platform(ODIN_PLATFORM)
        expanded = platform.total_cells * physical_row_expansion(
            platform.to_platform_constraints()["weight_sign_granularity"]
        )
        assert expanded * per_cell < total
