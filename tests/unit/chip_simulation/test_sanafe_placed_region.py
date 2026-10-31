"""[U-TT T-1] The simulated SANA-FE chip is sized by the cores the program places, not by the declared capacity.

Every executed neural unit is loaded from core 0 of a fresh chip (``runner.neural_stage``), so a chip only ever holds
the widest unit's cores. The declared floorplan stays what the records carry (name, mesh, tile coordinates) and the
declared capacity stays the admission bound; the chip the builder defines is the declared floorplan's region holding
the placed cores (whole columns of the declared mesh height, each tile at its declared coordinates). These tests read
the YAML the builder hands ``load_arch`` (a fake SANA-FE), so they need no SANA-FE; the real-SANA-FE identity and
memory tests are ``tests/integration/test_sanafe_placed_region.py``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from mimarsinan.chip_simulation.sanafe import arch_synth
from mimarsinan.chip_simulation.sanafe.arch_synth import build_architecture, derive_arch_spec
from mimarsinan.chip_simulation.sanafe.noc_geometry import xy_of_tile


def _core(axons=3, neurons=2):
    return SimpleNamespace(axons_per_core=axons, neurons_per_core=neurons)


def _stage(n_cores, kind="neural", levels=None, pass_index=None):
    hcm = SimpleNamespace(cores=[_core() for _ in range(n_cores)]) if kind == "neural" else None
    return SimpleNamespace(kind=kind, hard_core_mapping=hcm, compute_op=None, input_map=[],
                           schedule_pass_index=pass_index, retimed_level_stages=levels)


def _mapping(*stages):
    return SimpleNamespace(
        stages=list(stages),
        get_neural_segments=lambda: [s.hard_core_mapping for s in stages if s.kind == "neural" and s.hard_core_mapping])


def _rendered_yaml(monkeypatch, spec) -> str:
    """The YAML ``build_architecture`` hands SANA-FE's ``load_arch`` for ``spec``."""
    fake = MagicMock()
    fake.load_arch.return_value = MagicMock()
    monkeypatch.setattr(arch_synth, "_sanafe", lambda: fake)
    build_architecture(spec)
    with open(fake.load_arch.call_args.args[0]) as handle:
        return handle.read()


def _defined_cores(yaml: str) -> int:
    return yaml.count("          attributes:\n            buffer_position: soma")


def _defined_tiles(yaml: str) -> int:
    return yaml.count("    - name: tile")


class TestTheChipFollowsTheProgram:
    """The memory-scaling defect (D3): a discovery budget declares 200,000 cores and the chip was built at that size,
    about 7.6 MiB of chip and 1.1 MiB of architecture per declared core (the C5 probe ran out of its 120G limit)."""

    def test_a_declared_capacity_far_above_the_placed_cores_builds_the_placed_region(self, monkeypatch):
        mapping = _mapping(_stage(3))
        spec = derive_arch_spec(mapping, preset_name="loihi", declared_core_capacity=4000)
        assert spec.total_cores == 4000 and spec.packed_cores == 3      # the declared floorplan is unchanged
        yaml = _rendered_yaml(monkeypatch, spec)
        assert _defined_tiles(yaml) == 1                                 # 3 cores at 4 per tile: one tile
        assert _defined_cores(yaml) == 4

    @pytest.mark.parametrize("capacity", [40, 400, 4000])
    def test_the_chip_size_does_not_scale_with_the_declared_capacity(self, monkeypatch, capacity):
        mapping = _mapping(_stage(7), _stage(0, kind="compute"), _stage(5))
        spec = derive_arch_spec(mapping, preset_name="loihi", declared_core_capacity=capacity)
        yaml = _rendered_yaml(monkeypatch, spec)
        # The widest unit holds 7 cores: 2 tiles of 4, at most one column of the declared mesh.
        assert _defined_cores(yaml) <= 8, (capacity, _defined_cores(yaml))
        assert _defined_cores(yaml) >= 7

    def test_serial_units_are_not_summed(self, monkeypatch):
        # Eight 60-core units on a 360-core platform: each is loaded from core 0, so the chip holds 60 cores at once.
        mapping = _mapping(*[_stage(60) for _ in range(8)])
        spec = derive_arch_spec(mapping, preset_name="loihi", declared_core_capacity=360)
        assert spec.floorplan_replicas == 2 and spec.total_cores >= 480   # the record's floorplan, as before
        yaml = _rendered_yaml(monkeypatch, spec)
        assert 60 <= _defined_cores(yaml) < 120

    def test_a_retimed_level_stage_counts_as_an_executed_unit(self, monkeypatch):
        levels = [_stage(9), _stage(2)]
        mapping = _mapping(_stage(4, levels=levels))
        spec = derive_arch_spec(mapping, preset_name="loihi", declared_core_capacity=400)
        yaml = _rendered_yaml(monkeypatch, spec)
        assert _defined_cores(yaml) >= 9

    def test_the_whole_floorplan_is_built_where_the_program_fills_it(self, monkeypatch):
        mapping = _mapping(_stage(40))
        spec = derive_arch_spec(mapping, preset_name="loihi", declared_core_capacity=40)
        yaml = _rendered_yaml(monkeypatch, spec)
        assert _defined_tiles(yaml) == spec.n_tiles == 10
        assert f"width: {spec.mesh_width}" in yaml and f"height: {spec.mesh_height}" in yaml

    def test_the_declared_capacity_stays_the_admission_bound(self):
        with pytest.raises(ValueError, match="declared platform capacity"):
            derive_arch_spec(_mapping(_stage(10)), preset_name="loihi", declared_core_capacity=8)


# The region's geometry: read through the new helper (``placed_region``), imported in each test so the tests above
# run on a tree without it.
FLOORPLANS = [
    # (declared capacity, cores per tile, grid rows, grid cols, unit widths)
    (96, 0, 0, 0, (28, 12)),       # 6x4 mesh: 7 tiles -> 2 columns of 4
    (96, 0, 0, 0, (12, 6)),        # 3 tiles -> one partial column of 3
    (48, 2, 3, 8, (28, 12)),       # explicit grid 8x3 at 2 per tile: 14 tiles -> 5 columns of 3
    (32, 0, 0, 0, (28, 12)),       # 4x2 grid, 40 packed over 32 slots: 2 replicas (4x4); 7 tiles -> 2 columns
    (0, 0, 0, 0, (28, 12)),        # no declared platform: the packed-count mesh (7 per tile, 3x2); 4 tiles -> 2x2
    (40, 0, 0, 0, (40,)),          # the program fills the floorplan: the region is the floorplan
]


def _spec_of(capacity, cpt, rows, cols, widths):
    mapping = _mapping(*[_stage(w) for w in widths])
    return derive_arch_spec(mapping, preset_name="loihi", declared_core_capacity=capacity, cores_per_tile=cpt,
                            tile_grid_rows=rows, tile_grid_cols=cols), max(widths)


class TestTheRegionGeometry:
    @pytest.mark.parametrize("floorplan", FLOORPLANS)
    def test_every_placed_tile_keeps_its_declared_coordinates(self, floorplan):
        from mimarsinan.chip_simulation.sanafe.arch_synth.placed_region import placed_region

        spec, widest = _spec_of(*floorplan)
        region = placed_region(spec)
        cpt = spec.cores_per_tile_resolved
        needed = -(-widest // cpt)
        assert spec.placed_cores == widest
        assert needed <= region.n_tiles <= spec.n_tiles
        assert region.mesh_width * region.mesh_height == region.n_tiles      # a full rectangle: no phantom tile
        assert region.n_cores_per_tile == spec.n_cores_per_tile[:region.n_tiles]
        for tile in range(needed):
            assert xy_of_tile(tile, region.mesh_height) == xy_of_tile(tile, spec.mesh_height), (floorplan, tile)

    @pytest.mark.parametrize("floorplan", FLOORPLANS)
    def test_the_region_holds_every_tile_an_xy_route_between_placed_tiles_walks(self, floorplan):
        from mimarsinan.chip_simulation.sanafe.arch_synth.placed_region import placed_region
        from mimarsinan.chip_simulation.sanafe.noc_geometry import xy_route_edges

        spec, widest = _spec_of(*floorplan)
        region = placed_region(spec)
        needed = -(-widest // spec.cores_per_tile_resolved)
        cells = {xy_of_tile(t, region.mesh_height) for t in range(region.n_tiles)}
        placed = [xy_of_tile(t, spec.mesh_height) for t in range(needed)]
        for src in placed:
            for dst in placed:
                for _fx, _fy, tx, ty in xy_route_edges(src, dst):
                    assert (tx, ty) in cells, (floorplan, src, dst)

    def test_the_record_fields_stay_the_declared_floorplans(self):
        from mimarsinan.chip_simulation.sanafe.arch_synth.placed_region import placed_region

        spec, _ = _spec_of(96, 0, 0, 0, (28, 12))
        region = placed_region(spec)
        assert region.name == spec.name == "mimarsinan_loihi_96core"
        assert (region.cores_per_tile_resolved, region.axons_per_core, region.neurons_per_core) == (
            spec.cores_per_tile_resolved, spec.axons_per_core, spec.neurons_per_core)
        assert (spec.mesh_width, spec.mesh_height, spec.n_tiles) == (6, 4, 24)   # the derivation is untouched

    def test_a_spec_without_a_placed_count_is_built_whole(self):
        from mimarsinan.chip_simulation.sanafe.arch_synth import ArchSpec
        from mimarsinan.chip_simulation.sanafe.arch_synth.placed_region import placed_region
        from mimarsinan.chip_simulation.sanafe.presets import LOIHI_PRESET

        spec = ArchSpec(name="t", n_tiles=4, n_cores_per_tile=[2, 2, 2, 2], axons_per_core=4, neurons_per_core=2,
                        preset=LOIHI_PRESET, mesh_width=2, mesh_height=2, cores_per_tile_resolved=2)
        assert placed_region(spec) is spec


class TestTheRunnerRecordsTheDeclaredFloorplan:
    def test_the_geometry_and_the_arch_name_are_the_declared_floorplans(self, monkeypatch):
        import mimarsinan.chip_simulation.sanafe.runner as runner_mod
        from mimarsinan.chip_simulation.behavior_config import NeuralBehaviorConfig
        from mimarsinan.chip_simulation.sanafe.runner import SanafeRunner

        built = []
        monkeypatch.setattr(runner_mod, "build_architecture", lambda spec, **_kw: built.append(spec) or object())
        behavior = NeuralBehaviorConfig(spiking_mode="lif", firing_mode="Default", thresholding_mode="<",
                                        spike_generation_mode="Uniform")
        runner = SanafeRunner(mapping=_mapping(_stage(3)), simulation_length=4, behavior=behavior,
                              declared_core_capacity=4000)
        runner._ensure_arch()
        geom = runner._arch_geometry
        assert geom is not None and geom.width * geom.height == 1000      # the declared 40x25 tile mesh
        assert runner._arch_name == "mimarsinan_loihi_4000core"
        assert built and built[0].total_cores == 4000                       # the builder takes the declared spec
