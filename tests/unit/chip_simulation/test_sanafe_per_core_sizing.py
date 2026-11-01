"""[U-SF] Each core of the SANA-FE chip is sized by the cores the program places on it, not by the program's widest core.

SANA-FE 2.1.1 gives every declared input slot of a core (one ``inputs[k]`` soma unit per axon) its own hardware unit,
about 9.3 KiB of chip and 1.5 KiB of architecture each, and every core a timestep buffer of ``max_neurons_supported``
entries (56 B each). The builder sized every core of the chip at the program's widest core: on C5's T-LIF program
(cores of 256, 512, 768 and 1,280 axons) a bare chip of the placed region (1,600 cores, all at 1,280 axons) held
19,363 MiB and the SANA-FE step stopped at its 16G class (phase C group 1, ``c5probe``).

Every executed unit (a neural stage or each of its retimed level stages) is loaded from core 0 of a fresh chip
(``runner.neural_stage``), so core ``p`` of the chip holds core ``p`` of each unit in turn. The chip now declares core
``p`` at the largest axons and neurons any unit places there; a core no unit places on declares one input slot. These
tests read the YAML the builder hands ``load_arch`` (a fake SANA-FE); the real-SANA-FE identity and memory tests are
``tests/integration/test_sanafe_per_core_sizing.py``.
"""

from __future__ import annotations

import re
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from mimarsinan.chip_simulation.sanafe import arch_synth
from mimarsinan.chip_simulation.sanafe.arch_synth import build_architecture, derive_arch_spec
from mimarsinan.chip_simulation.sanafe.noc_geometry import tile_and_local_of_core


def _core(axons, neurons=256):
    return SimpleNamespace(axons_per_core=axons, neurons_per_core=neurons)


def _stage(shapes, levels=None):
    """A neural stage whose cores have the (axons, neurons) ``shapes``; ``levels``: its retimed level stages' shapes."""
    hcm = SimpleNamespace(cores=[_core(a, n) for a, n in shapes])
    level_stages = [SimpleNamespace(kind="neural", hard_core_mapping=SimpleNamespace(cores=[_core(a, n) for a, n in s]))
                    for s in levels] if levels else None
    return SimpleNamespace(kind="neural", hard_core_mapping=hcm, compute_op=None, input_map=[],
                           schedule_pass_index=None, retimed_level_stages=level_stages)


def _compute():
    return SimpleNamespace(kind="compute", hard_core_mapping=None, compute_op=None, input_map=[],
                           schedule_pass_index=None, retimed_level_stages=None)


def _mapping(*stages):
    return SimpleNamespace(
        stages=list(stages),
        get_neural_segments=lambda: [s.hard_core_mapping for s in stages if s.kind == "neural"])


def _rendered_yaml(monkeypatch, spec) -> str:
    """The YAML ``build_architecture`` hands SANA-FE's ``load_arch`` for ``spec``."""
    fake = MagicMock()
    fake.load_arch.return_value = MagicMock()
    monkeypatch.setattr(arch_synth, "_sanafe", lambda: fake)
    build_architecture(spec)
    with open(fake.load_arch.call_args.args[0]) as handle:
        return handle.read()


_CORE = re.compile(r"^        - name: (t(\d+)_c(\d+))$", re.M)


def _declared_cores(yaml: str) -> dict:
    """Per core block (tile, local core): its input slots (the ``inputs`` soma range) and ``max_neurons_supported``."""
    starts = [(m.start(), int(m.group(2)), int(m.group(3))) for m in _CORE.finditer(yaml)] + [(len(yaml), -1, -1)]
    cores = {}
    for (begin, tile, local), (end, _t, _l) in zip(starts, starts[1:]):
        block = yaml[begin:end]
        rng = re.search(r"^            - name: inputs(?:\[(\d+)\.\.(\d+)\])?$", block, re.M)
        assert rng is not None, block[:400]
        slots = int(rng.group(2)) - int(rng.group(1)) + 1 if rng.group(1) is not None else 1
        neurons = int(re.search(r"max_neurons_supported: (\d+)", block).group(1))
        cores[(tile, local)] = {"slots": slots, "max_neurons": neurons, "ranged": rng.group(1) is not None}
    return cores


def _at(spec, position):
    """The (tile, local core) a unit's core ``position`` lands on: the net builder's own rule (sequential tile fill)."""
    return tile_and_local_of_core(position, spec.cores_per_tile_resolved)


# The C5 probe's core classes in small: one unit of mixed axon counts on a declared 16-core platform (4 per tile).
MIXED = [(256, 256), (512, 256), (1280, 256), (256, 64), (768, 128)]


class TestEachCoreDeclaresWhatTheProgramPlacesOnIt:
    def test_a_unit_of_mixed_core_classes_declares_each_cores_own_axons(self, monkeypatch):
        spec = derive_arch_spec(_mapping(_stage(MIXED)), preset_name="loihi", declared_core_capacity=16)
        cores = _declared_cores(_rendered_yaml(monkeypatch, spec))
        got = [cores[_at(spec, p)]["slots"] for p in range(len(MIXED))]
        assert got == [a for a, _n in MIXED], got         # R: [1280] * 5, every core at the widest

    def test_each_cores_neuron_capacity_follows_its_own_axons_and_neurons(self, monkeypatch):
        spec = derive_arch_spec(_mapping(_stage(MIXED)), preset_name="loihi", declared_core_capacity=16)
        cores = _declared_cores(_rendered_yaml(monkeypatch, spec))
        got = [cores[_at(spec, p)]["max_neurons"] for p in range(len(MIXED))]
        assert got == [2 * (a + n) for a, n in MIXED], got   # R: 8192 on every core (a floor no core needs)

    def test_a_core_declares_the_largest_core_any_unit_places_on_it(self, monkeypatch):
        units = (_stage([(256, 256)] * 4), _compute(), _stage([(1280, 128), (512, 256)]))
        spec = derive_arch_spec(_mapping(*units), preset_name="loihi", declared_core_capacity=16)
        cores = _declared_cores(_rendered_yaml(monkeypatch, spec))
        assert [cores[_at(spec, p)]["slots"] for p in range(4)] == [1280, 512, 256, 256]
        assert [cores[_at(spec, p)]["max_neurons"] for p in range(4)] == [
            2 * (1280 + 256), 2 * (512 + 256), 2 * (256 + 256), 2 * (256 + 256)]

    def test_a_retimed_level_stage_sizes_the_cores_it_places(self, monkeypatch):
        stage = _stage([(64, 32)] * 3, levels=[[(64, 32)], [(1024, 32), (64, 32)]])
        spec = derive_arch_spec(_mapping(stage), preset_name="loihi", declared_core_capacity=16)
        cores = _declared_cores(_rendered_yaml(monkeypatch, spec))
        assert [cores[_at(spec, p)]["slots"] for p in range(3)] == [1024, 64, 64]

    def test_a_core_no_unit_places_on_declares_the_smallest_size(self, monkeypatch):
        spec = derive_arch_spec(_mapping(_stage([(1280, 256)] * 3)), preset_name="loihi", declared_core_capacity=16)
        cores = _declared_cores(_rendered_yaml(monkeypatch, spec))
        assert len(cores) == 4                           # the placed region: one tile of 4, core 3 holds nothing
        assert cores[_at(spec, 3)] == {"slots": 1, "max_neurons": 2, "ranged": True}   # R: 1,280 slots

    def test_the_spec_carries_the_per_core_sizes(self):
        units = (_stage([(256, 256)] * 4), _stage([(1280, 128), (512, 256)]))
        spec = derive_arch_spec(_mapping(*units), preset_name="loihi", declared_core_capacity=16)
        assert spec.core_sizes == ((1280, 256), (512, 256), (256, 256), (256, 256))

    def test_every_input_slot_is_a_range_so_a_one_axon_core_binds_inputs_0(self, monkeypatch):
        """The net builder binds input axon ``a`` to ``inputs[a]``; a one-slot range renders as ``inputs``, a name it
        never binds (on R a program whose widest core has one axon fails to load its network)."""
        spec = derive_arch_spec(_mapping(_stage([(1, 1), (4, 2)])), preset_name="loihi", declared_core_capacity=16)
        cores = _declared_cores(_rendered_yaml(monkeypatch, spec))
        assert cores[_at(spec, 0)] == {"slots": 1, "max_neurons": 4, "ranged": True}


class TestWhatStaysTheDeclaredFloorplans:
    """Guards: the record-facing fields, the region and the tile layout are unchanged."""

    def test_the_widest_core_and_the_name_stay_on_the_spec(self):
        spec = derive_arch_spec(_mapping(_stage(MIXED)), preset_name="loihi", declared_core_capacity=16)
        assert (spec.axons_per_core, spec.neurons_per_core, spec.name) == (1280, 256, "mimarsinan_loihi_16core")
        assert spec.placed_cores == len(MIXED)

    def test_the_region_and_its_tiles_are_unchanged(self, monkeypatch):
        spec = derive_arch_spec(_mapping(_stage(MIXED)), preset_name="loihi", declared_core_capacity=16)
        yaml = _rendered_yaml(monkeypatch, spec)
        cores = _declared_cores(yaml)
        assert yaml.count("    - name: tile") == 2 and len(cores) == 8   # 5 placed cores: 2 tiles of 4
        assert sorted(cores) == [(t, c) for t in range(2) for c in range(4)]

    def test_a_hand_built_spec_without_per_core_sizes_declares_every_core_at_its_size(self, monkeypatch):
        from mimarsinan.chip_simulation.sanafe.arch_synth import ArchSpec
        from mimarsinan.chip_simulation.sanafe.presets import LOIHI_PRESET

        spec = ArchSpec(name="t", n_tiles=2, n_cores_per_tile=[2, 2], axons_per_core=7, neurons_per_core=5,
                        preset=LOIHI_PRESET, dendrite_plugin_path="/tmp/d.so", soma_plugin_path="/tmp/s.so",
                        mesh_width=2, mesh_height=1, cores_per_tile_resolved=2)
        cores = _declared_cores(_rendered_yaml(monkeypatch, spec))
        assert len(cores) == 4 and all(c["slots"] == 7 for c in cores.values())

    @pytest.mark.parametrize("capacity, cpt, rows, cols", [(96, 0, 0, 0), (48, 2, 3, 8), (32, 0, 0, 0), (0, 0, 0, 0)])
    def test_each_units_core_lands_on_the_core_sized_for_it_on_every_floorplan(self, monkeypatch, capacity, cpt, rows,
                                                                                 cols):
        """The size table is read through the net builder's own placement rule, on every regime of the derivation."""
        shapes = [(8 + 8 * (p % 5), 4 + p % 3) for p in range(28)]
        spec = derive_arch_spec(_mapping(_stage(shapes), _stage(shapes[:12][::-1])), preset_name="loihi",
                                declared_core_capacity=capacity, cores_per_tile=cpt, tile_grid_rows=rows,
                                tile_grid_cols=cols)
        cores = _declared_cores(_rendered_yaml(monkeypatch, spec))
        for p, (axons, _neurons) in enumerate(shapes):
            need = max(axons, shapes[:12][::-1][p][0]) if p < 12 else axons
            assert cores[_at(spec, p)]["slots"] == need, (capacity, p)
