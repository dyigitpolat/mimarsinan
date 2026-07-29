"""[IMC] Crossbar occupancy accounting, elimination ablation, and platform geometry.

The IMC study reports resources, not accuracy alone: a row or column is only
reclaimed when eliminated entirely, so the deliverable is how many crossbars
were allocated and how full they are.
"""

import pytest

from mimarsinan.mapping.crossbar_utilization import (
    CoreOccupancy,
    CrossbarUtilizationReport,
)
from mimarsinan.mapping.platform.imc_platforms import (
    IMCPlatform,
    get_imc_platform,
    imc_platform_names,
)
from mimarsinan.mapping.pruning.graph.pruning_propagation import (
    compute_propagated_pruned_rows_cols,
)
import numpy as np


class _FakeHardCore:
    """Minimal HardCore surface the report reads (geometry + remaining capacity)."""

    def __init__(self, axons, neurons, used_axons, used_neurons, unusable=0):
        self.axons_per_core = axons
        self.neurons_per_core = neurons
        self.available_axons = axons - used_axons
        self.available_neurons = neurons - used_neurons
        self.unusable_space = unusable


class TestCoreOccupancy:
    def test_used_and_physical_cells(self):
        core = CoreOccupancy(
            axons_used=64, axons_physical=128,
            neurons_used=32, neurons_physical=128, unusable_space=0,
        )
        assert core.cells_physical == 128 * 128
        assert core.cells_used == 64 * 32
        assert core.occupancy == pytest.approx((64 * 32) / (128 * 128))

    def test_empty_core_has_zero_occupancy_not_a_zero_division(self):
        core = CoreOccupancy(0, 0, 0, 0, 0)
        assert core.occupancy == 0.0


class TestReportAggregates:
    def _report(self):
        return CrossbarUtilizationReport.from_hard_cores(
            [
                _FakeHardCore(128, 128, used_axons=128, used_neurons=128),
                _FakeHardCore(128, 128, used_axons=32, used_neurons=16, unusable=7),
            ],
            weight_bits=4,
        )

    def test_core_count_and_axis_totals(self):
        r = self._report()
        assert r.cores_allocated == 2
        assert r.axons_used == 128 + 32
        assert r.axons_physical == 256
        assert r.neurons_used == 128 + 16
        assert r.neurons_physical == 256

    def test_axis_utilizations(self):
        r = self._report()
        assert r.axon_utilization == pytest.approx(160 / 256)
        assert r.neuron_utilization == pytest.approx(144 / 256)

    def test_cell_occupancy_is_the_crossbar_fill_rate(self):
        r = self._report()
        used = 128 * 128 + 32 * 16
        assert r.cells_used == used
        assert r.cell_occupancy == pytest.approx(used / (2 * 128 * 128))

    def test_unusable_space_is_summed(self):
        assert self._report().unusable_space == 7

    def test_energy_proxies_scale_with_weight_bits(self):
        r = self._report()
        assert r.macs == r.cells_used
        assert r.programming_bits == r.cells_used * 4
        wide = CrossbarUtilizationReport.from_hard_cores(
            [_FakeHardCore(128, 128, 128, 128)], weight_bits=8
        )
        narrow = CrossbarUtilizationReport.from_hard_cores(
            [_FakeHardCore(128, 128, 128, 128)], weight_bits=4
        )
        assert wide.programming_bits == 2 * narrow.programming_bits

    def test_undeclared_weight_bits_leaves_programming_bits_unknown(self):
        r = CrossbarUtilizationReport.from_hard_cores(
            [_FakeHardCore(8, 8, 4, 4)], weight_bits=None
        )
        assert r.programming_bits is None

    def test_report_serializes_flat_for_harvesting(self):
        d = self._report().to_dict()
        assert d["cores_allocated"] == 2
        assert d["cell_occupancy"] == pytest.approx(self._report().cell_occupancy)
        assert all(not isinstance(v, (list, dict)) for v in d.values())

    def test_empty_mapping_is_reportable(self):
        r = CrossbarUtilizationReport.from_hard_cores([], weight_bits=8)
        assert r.cores_allocated == 0
        assert r.cell_occupancy == 0.0


class TestEliminationAblation:
    """The single-layer baseline: identical seeds, propagation disabled."""

    def _matrix(self):
        # col 2 is seeded dead; row 1 feeds ONLY col 2, so propagation kills it.
        return np.array(
            [[1.0, 1.0, 0.0],
             [0.0, 0.0, 1.0],
             [1.0, 1.0, 0.0]],
            dtype=np.float64,
        )

    def test_propagation_kills_the_orphaned_row(self):
        rows, cols = compute_propagated_pruned_rows_cols(
            self._matrix(), initial_zero_rows=set(), initial_zero_cols={2},
        )
        assert 1 in rows, "row feeding only dead columns must cascade"
        assert 2 in cols

    def test_ablation_returns_exactly_the_seeds(self):
        rows, cols = compute_propagated_pruned_rows_cols(
            self._matrix(), initial_zero_rows=set(), initial_zero_cols={2},
            propagate=False,
        )
        assert rows == set(), "propagation disabled must not cascade"
        assert cols == {2}

    def test_ablation_is_a_subset_of_the_cascade(self):
        seeded = dict(initial_zero_rows=set(), initial_zero_cols={2})
        base_r, base_c = compute_propagated_pruned_rows_cols(
            self._matrix(), propagate=False, **seeded
        )
        casc_r, casc_c = compute_propagated_pruned_rows_cols(
            self._matrix(), **seeded
        )
        assert base_r <= casc_r and base_c <= casc_c

    def test_exemptions_hold_under_ablation(self):
        _, cols = compute_propagated_pruned_rows_cols(
            self._matrix(), initial_zero_rows=set(), initial_zero_cols={2},
            exempt_cols={2}, propagate=False,
        )
        assert 2 not in cols


class TestIMCPlatformRegistry:
    def test_platform_is_geometry_plus_capabilities(self):
        p = IMCPlatform(
            name="demo", cores=({"max_axons": 128, "max_neurons": 128,
                                 "count": 4, "has_bias": True},),
            weight_bits=4, provenance="placeholder",
        )
        constraints = p.to_platform_constraints()
        assert constraints["cores"][0]["max_axons"] == 128
        assert constraints["weight_bits"] == 4

    def test_heterogeneous_core_types_are_expressible(self):
        p = IMCPlatform(
            name="het",
            cores=({"max_axons": 256, "max_neurons": 256, "count": 2},
                   {"max_axons": 64, "max_neurons": 64, "count": 8}),
            weight_bits=8, provenance="placeholder",
        )
        assert p.total_cores == 10
        assert p.total_cells == 2 * 256 * 256 + 8 * 64 * 64

    def test_heterogeneous_geometry_round_trips_with_types_distinct(self):
        # Core types must survive as SEPARATE entries: collapsing them to a
        # single max-geometry would silently invent capacity that no tile has.
        p = IMCPlatform(
            name="het2",
            cores=({"max_axons": 512, "max_neurons": 256, "count": 2},
                   {"max_axons": 64, "max_neurons": 128, "count": 16}),
            weight_bits=4, provenance="placeholder",
        )
        cores = p.to_platform_constraints()["cores"]
        assert len(cores) == 2
        assert (cores[0]["max_axons"], cores[0]["max_neurons"]) == (512, 256)
        assert (cores[1]["max_axons"], cores[1]["max_neurons"]) == (64, 128)
        assert cores[1]["count"] == 16

    def test_non_square_crossbars_are_expressible(self):
        p = IMCPlatform(
            name="tall", cores=({"max_axons": 1024, "max_neurons": 256, "count": 4},),
            weight_bits=8, provenance="placeholder",
        )
        assert p.total_cells == 4 * 1024 * 256

    def test_a_heterogeneous_platform_is_registered(self):
        het = [n for n in imc_platform_names()
               if len(get_imc_platform(n).cores) > 1]
        assert het, "registry must exercise the heterogeneous case"

    def test_heterogeneous_platform_feeds_the_real_mapping_resolver(self):
        # Two consumers with DIFFERENT semantics, both fed by the same core list:
        # the tiling resolver bounds a softcore by the LARGEST tile, while the
        # packer honours per-type populations from `cores` verbatim.
        from mimarsinan.mapping.platform.platform_constraints import (
            resolve_platform_mapping_params,
        )
        p = IMCPlatform(
            name="het3",
            cores=({"max_axons": 256, "max_neurons": 256, "count": 2},
                   {"max_axons": 64, "max_neurons": 64, "count": 8}),
            weight_bits=8, provenance="placeholder",
        )
        constraints = p.to_platform_constraints()
        params = resolve_platform_mapping_params(constraints["cores"])
        assert params.effective_max_axons == 256
        assert params.effective_max_neurons == 256
        # ...and the small tiles are still individually visible to the packer.
        assert [ct["count"] for ct in constraints["cores"]] == [2, 8]

    def test_malformed_core_type_fails_loud(self):
        for bad in ({"max_neurons": 8, "count": 1},              # missing axons
                    {"max_axons": 0, "max_neurons": 8, "count": 1},   # zero dim
                    {"max_axons": 8, "max_neurons": 8, "count": 0}):  # zero count
            with pytest.raises((KeyError, ValueError)):
                IMCPlatform(name="bad", cores=(bad,), weight_bits=8,
                            provenance="placeholder").validate()

    def test_registry_round_trips_by_name(self):
        for name in imc_platform_names():
            assert get_imc_platform(name).name == name

    def test_unknown_platform_fails_loud_with_the_known_names(self):
        with pytest.raises(KeyError) as excinfo:
            get_imc_platform("no_such_chip")
        assert "no_such_chip" in str(excinfo.value)

    def test_every_registered_platform_declares_provenance(self):
        # Geometries must be traceable; placeholders must SAY they are placeholders.
        for name in imc_platform_names():
            assert get_imc_platform(name).provenance.strip()
