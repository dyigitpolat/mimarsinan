"""ArchSpec derivation + SANA-FE Architecture YAML synthesis.

These tests pin two concerns:

1. ``derive_arch_spec`` is pure-Python — it walks a (possibly fake)
   ``HybridHardCoreMapping`` and produces a deterministic ``ArchSpec``.
2. ``build_architecture`` renders a YAML matching the spec and calls
   ``sanafe.load_arch`` — the YAML rendering is what we pin here
   directly (text contents are inspected) because the SANA-FE-touching
   half is covered by the slow integration test.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from mimarsinan.chip_simulation.sanafe import arch_synth
from mimarsinan.chip_simulation.sanafe.arch_synth import (
    ArchSpec,
    _render_arch_yaml,
    build_architecture,
    derive_arch_spec,
)
from mimarsinan.chip_simulation.sanafe.presets import (
    LOIHI_PRESET,
    PRESETS,
    SOMA_INPUT_RANGE_NAME,
    SOMA_LIF_NAME,
    SYNAPSE_NAME,
    TRUENORTH_PRESET,
)


# ---------------------------------------------------------------------------
# Lightweight fakes (no real HCM construction — exercise the geometry walk only)
# ---------------------------------------------------------------------------


def _fake_hard_core(axons_per_core: int, neurons_per_core: int):
    return SimpleNamespace(
        axons_per_core=axons_per_core,
        neurons_per_core=neurons_per_core,
    )


def _fake_hcm(*core_shapes: tuple[int, int]):
    cores = [_fake_hard_core(a, n) for a, n in core_shapes]
    return SimpleNamespace(
        cores=cores,
        axons_per_core=(max(c.axons_per_core for c in cores) if cores else 0),
        neurons_per_core=(max(c.neurons_per_core for c in cores) if cores else 0),
    )


def _fake_stage(kind: str, hcm=None, input_map=None):
    return SimpleNamespace(
        kind=kind, hard_core_mapping=hcm, compute_op=None,
        input_map=input_map or [],
    )


def _fake_mapping(*stages):
    return SimpleNamespace(
        stages=list(stages),
        get_neural_segments=lambda: [s.hard_core_mapping for s in stages
                                     if s.kind == "neural" and s.hard_core_mapping],
    )


def _io_slice(node_id=-2, offset=0, size=0):
    return SimpleNamespace(node_id=node_id, offset=offset, size=size)


# ---------------------------------------------------------------------------
# derive_arch_spec — pure Python
# ---------------------------------------------------------------------------


def test_derive_arch_spec_single_segment_single_core_returns_one_tile():
    mapping = _fake_mapping(_fake_stage("neural", _fake_hcm((4, 2))))
    spec = derive_arch_spec(mapping, preset_name="loihi")
    assert isinstance(spec, ArchSpec)
    assert spec.n_tiles == 1
    assert spec.n_cores_per_tile == [1]
    assert spec.axons_per_core == 4
    assert spec.neurons_per_core == 2


def test_derive_arch_spec_axons_neurons_use_hcm_geometry_max_across_segments():
    seg_a = _fake_hcm((3, 5))
    seg_b = _fake_hcm((7, 2), (6, 4))
    mapping = _fake_mapping(
        _fake_stage("neural", seg_a),
        _fake_stage("compute"),
        _fake_stage("neural", seg_b),
    )
    spec = derive_arch_spec(mapping, preset_name="loihi")
    assert sum(spec.n_cores_per_tile) == 3
    assert spec.axons_per_core == 7
    assert spec.neurons_per_core == 5


def test_derive_arch_spec_default_packs_all_cores_into_one_tile():
    mapping = _fake_mapping(
        _fake_stage("neural", _fake_hcm((3, 2), (3, 2), (3, 2), (3, 2))),
    )
    spec = derive_arch_spec(mapping, preset_name="loihi")
    assert spec.n_tiles == 1
    assert spec.n_cores_per_tile == [4]


def test_derive_arch_spec_cores_per_tile_splits_evenly():
    mapping = _fake_mapping(
        _fake_stage("neural", _fake_hcm((3, 2), (3, 2), (3, 2), (3, 2), (3, 2))),
    )
    spec = derive_arch_spec(mapping, preset_name="loihi", cores_per_tile=2)
    assert spec.n_tiles == 3
    assert spec.n_cores_per_tile == [2, 2, 1]


def test_derive_arch_spec_rejects_unknown_preset():
    mapping = _fake_mapping(_fake_stage("neural", _fake_hcm((2, 1))))
    with pytest.raises(ValueError, match="unknown.*preset"):
        derive_arch_spec(mapping, preset_name="silicon-dreams")


def test_derive_arch_spec_rejects_mapping_with_no_neural_cores():
    mapping = _fake_mapping(_fake_stage("compute"))
    with pytest.raises(ValueError, match="no neural"):
        derive_arch_spec(mapping, preset_name="loihi")


def test_derive_arch_spec_attaches_preset_dict():
    mapping = _fake_mapping(_fake_stage("neural", _fake_hcm((2, 1))))
    spec_loihi = derive_arch_spec(mapping, preset_name="loihi")
    spec_truenorth = derive_arch_spec(mapping, preset_name="truenorth")
    assert spec_loihi.preset == LOIHI_PRESET
    assert spec_truenorth.preset == TRUENORTH_PRESET


def test_derive_arch_spec_name_includes_preset_and_core_count():
    mapping = _fake_mapping(_fake_stage("neural", _fake_hcm((4, 2), (4, 2))))
    spec = derive_arch_spec(mapping, preset_name="loihi")
    assert "loihi" in spec.name.lower()
    assert "2" in spec.name


# ---------------------------------------------------------------------------
# YAML rendering
# ---------------------------------------------------------------------------


def _spec(n_tiles=1, n_cores_per_tile=None, axons=4, neurons=2,
          preset=None,
          dendrite_plugin_path="/tmp/fake_dendrite.so",
          soma_plugin_path="/tmp/fake_soma.so"):
    return ArchSpec(
        name="t", n_tiles=n_tiles,
        n_cores_per_tile=n_cores_per_tile or [1],
        axons_per_core=axons, neurons_per_core=neurons,
        preset=preset or LOIHI_PRESET,
        dendrite_plugin_path=dendrite_plugin_path,
        soma_plugin_path=soma_plugin_path,
    )


def test_render_arch_yaml_includes_required_hardware_unit_names():
    """The YAML must reference the names net_synth binds against."""
    yaml = _render_arch_yaml(_spec())
    assert SYNAPSE_NAME in yaml
    assert SOMA_LIF_NAME in yaml
    assert SOMA_INPUT_RANGE_NAME in yaml      # rendered as inputs[0..N-1]


def test_render_arch_yaml_carries_preset_synapse_energy():
    yaml = _render_arch_yaml(_spec(preset=LOIHI_PRESET))
    # Synapse energy from the preset must appear verbatim.
    assert f"energy_process_spike: {LOIHI_PRESET['synapse_energy_j']}" in yaml
    assert f"latency_process_spike: {LOIHI_PRESET['synapse_latency_s']}" in yaml


def test_render_arch_yaml_carries_preset_tile_hop_energy():
    yaml = _render_arch_yaml(_spec(preset=LOIHI_PRESET))
    assert f"energy_north_hop: {LOIHI_PRESET['tile_hop_energy_j']}" in yaml
    assert f"energy_east_hop: {LOIHI_PRESET['tile_hop_energy_j']}" in yaml


def test_render_arch_yaml_distinct_presets_produce_distinct_yamls():
    y1 = _render_arch_yaml(_spec(preset=LOIHI_PRESET))
    y2 = _render_arch_yaml(_spec(preset=TRUENORTH_PRESET))
    assert y1 != y2


def test_render_arch_yaml_input_soma_range_sized_to_per_core_axons():
    """Each core's ``inputs[0..N-1]`` is sized to its axon capacity.

    No global input host — every core hosts its own input neurons (see
    ``sanafe_per_core_input_neurons`` memory).
    """
    yaml = _render_arch_yaml(_spec(axons=7))
    assert f"{SOMA_INPUT_RANGE_NAME}[0..6]" in yaml


def test_render_arch_yaml_includes_plugin_references():
    """The arch YAML loads our two mimarsinan plugins for dendrite + soma."""
    yaml = _render_arch_yaml(_spec(
        dendrite_plugin_path="/abs/path/libmimarsinan_dendrite.so",
        soma_plugin_path="/abs/path/libmimarsinan_soma.so",
    ))
    assert "/abs/path/libmimarsinan_dendrite.so" in yaml
    assert "/abs/path/libmimarsinan_soma.so" in yaml
    assert "model: mimarsinan_dendrite" in yaml
    assert "model: mimarsinan_soma" in yaml


def test_derive_arch_spec_resolves_plugin_paths():
    """When called with the real bootstrap_sanafe.sh artifacts in place,
    derive_arch_spec produces a spec with non-empty plugin paths."""
    mapping = _fake_mapping(_fake_stage("neural", _fake_hcm((4, 2))))
    spec = derive_arch_spec(mapping, preset_name="loihi")
    assert spec.dendrite_plugin_path.endswith("libmimarsinan_dendrite.so")
    assert spec.soma_plugin_path.endswith("libmimarsinan_soma.so")


def test_render_arch_yaml_multi_tile_emits_one_block_per_tile():
    yaml = _render_arch_yaml(_spec(n_tiles=3, n_cores_per_tile=[2, 2, 1]))
    assert yaml.count("- name: tile") == 3
    # One core block per core (no `name[0..N-1]` range shorthand on cores —
    # SANA-FE 2.1.1's shorthand mangles the inner `inputs[0..K]` soma range).
    assert yaml.count("- name: t0_c") == 2
    assert yaml.count("- name: t1_c") == 2
    assert yaml.count("- name: t2_c") == 1


def test_render_arch_yaml_topology_width_matches_n_tiles():
    yaml = _render_arch_yaml(_spec(n_tiles=4, n_cores_per_tile=[1, 1, 1, 1]))
    assert "width: 4" in yaml


# ---------------------------------------------------------------------------
# build_architecture — touches sanafe via the lazy accessor
# ---------------------------------------------------------------------------


def test_build_architecture_calls_load_arch_with_synthesized_yaml(monkeypatch, tmp_path):
    fake_sanafe = MagicMock()
    fake_arch = MagicMock()
    fake_arch.tiles = [MagicMock()]
    fake_arch.tiles[0].cores = [MagicMock()]
    fake_sanafe.load_arch.return_value = fake_arch
    monkeypatch.setattr(arch_synth, "_sanafe", lambda: fake_sanafe)

    spec = _spec()
    arch = build_architecture(spec)

    fake_sanafe.load_arch.assert_called_once()
    yaml_path = fake_sanafe.load_arch.call_args.args[0]
    # The path passed must point at an actual file SANA-FE could load.
    import os
    assert os.path.isfile(yaml_path)
    with open(yaml_path) as f:
        content = f.read()
    assert SYNAPSE_NAME in content
    assert SOMA_LIF_NAME in content
    assert arch is fake_arch


def test_build_architecture_custom_yaml_path_loads_external_arch(monkeypatch, tmp_path):
    yaml_path = tmp_path / "custom_arch.yaml"
    yaml_path.write_text("architecture: {}\n")

    fake_sanafe = MagicMock()
    fake_arch = MagicMock()
    fake_arch.tiles = [MagicMock()]
    fake_arch.tiles[0].cores = [MagicMock()]
    fake_sanafe.load_arch.return_value = fake_arch
    monkeypatch.setattr(arch_synth, "_sanafe", lambda: fake_sanafe)

    spec = _spec()
    arch = build_architecture(spec, custom_arch_path=str(yaml_path))

    fake_sanafe.load_arch.assert_called_once_with(str(yaml_path))
    assert arch is fake_arch


def test_build_architecture_custom_yaml_validates_enough_cores(monkeypatch, tmp_path):
    yaml_path = tmp_path / "small_arch.yaml"
    yaml_path.write_text("architecture: {}\n")

    fake_sanafe = MagicMock()
    fake_arch = MagicMock()
    fake_arch.tiles = [MagicMock()]
    fake_arch.tiles[0].cores = [MagicMock()]
    fake_sanafe.load_arch.return_value = fake_arch
    monkeypatch.setattr(arch_synth, "_sanafe", lambda: fake_sanafe)

    spec = _spec(n_cores_per_tile=[3])
    with pytest.raises(ValueError, match="cores"):
        build_architecture(spec, custom_arch_path=str(yaml_path))


def test_build_architecture_custom_yaml_path_missing_file_raises(monkeypatch, tmp_path):
    missing = tmp_path / "absent.yaml"
    monkeypatch.setattr(arch_synth, "_sanafe", lambda: MagicMock())
    with pytest.raises(FileNotFoundError):
        build_architecture(_spec(), custom_arch_path=str(missing))


# ---------------------------------------------------------------------------
# Preset registry sanity
# ---------------------------------------------------------------------------


def test_presets_registry_includes_loihi_and_truenorth():
    assert "loihi" in PRESETS
    assert "truenorth" in PRESETS


def test_loihi_preset_has_all_required_keys():
    required = {
        "tile_hop_energy_j", "tile_hop_latency_s",
        "axon_in_energy_j", "axon_in_latency_s",
        "axon_out_energy_j", "axon_out_latency_s",
        "synapse_energy_j", "synapse_latency_s",
        "dendrite_energy_j", "dendrite_latency_s",
        "soma_access_energy_j", "soma_access_latency_s",
        "soma_update_energy_j", "soma_update_latency_s",
        "soma_spike_out_energy_j", "soma_spike_out_latency_s",
    }
    assert required <= set(LOIHI_PRESET.keys())


def test_loihi_preset_values_are_non_negative_floats():
    for key, val in LOIHI_PRESET.items():
        assert isinstance(val, float), f"{key} is not a float"
        assert val >= 0.0, f"{key} should be non-negative (got {val})"
