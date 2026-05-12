"""ArchSpec derivation + SANA-FE Architecture construction.

These tests pin two concerns:

1. ``derive_arch_spec`` is pure-Python — it walks a (possibly fake)
   ``HybridHardCoreMapping`` and produces a deterministic ``ArchSpec``.
2. ``build_architecture`` is the only function that touches the SANA-FE
   Python package; the lazy ``_sanafe()`` accessor is monkey-patched
   here so the test suite stays runnable without SANA-FE installed.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from mimarsinan.chip_simulation.sanafe import arch_synth
from mimarsinan.chip_simulation.sanafe.arch_synth import (
    ArchSpec,
    build_architecture,
    derive_arch_spec,
)
from mimarsinan.chip_simulation.sanafe.presets import (
    LOIHI_PRESET,
    PRESETS,
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
    """Build a fake HardCoreMapping with the given (axons, neurons) per core."""
    cores = [_fake_hard_core(a, n) for a, n in core_shapes]
    return SimpleNamespace(
        cores=cores,
        axons_per_core=(max(c.axons_per_core for c in cores) if cores else 0),
        neurons_per_core=(max(c.neurons_per_core for c in cores) if cores else 0),
    )


def _fake_stage(kind: str, hcm=None):
    return SimpleNamespace(
        kind=kind,
        hard_core_mapping=hcm,
        compute_op=None,
    )


def _fake_mapping(*stages):
    return SimpleNamespace(
        stages=list(stages),
        get_neural_segments=lambda: [s.hard_core_mapping for s in stages
                                     if s.kind == "neural" and s.hard_core_mapping],
    )


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
    """axons_per_core / neurons_per_core take the max across every neural HCM."""
    seg_a = _fake_hcm((3, 5))
    seg_b = _fake_hcm((7, 2), (6, 4))
    mapping = _fake_mapping(
        _fake_stage("neural", seg_a),
        _fake_stage("compute"),
        _fake_stage("neural", seg_b),
    )
    spec = derive_arch_spec(mapping, preset_name="loihi")

    # Total cores across all neural segments: 1 + 2 = 3
    assert sum(spec.n_cores_per_tile) == 3
    # max axons across {3, 7, 6} = 7; max neurons across {5, 2, 4} = 5
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
    """``cores_per_tile`` lets the user bin cores into multiple SANA-FE tiles."""
    mapping = _fake_mapping(
        _fake_stage("neural", _fake_hcm((3, 2), (3, 2), (3, 2), (3, 2), (3, 2))),
    )
    spec = derive_arch_spec(mapping, preset_name="loihi", cores_per_tile=2)
    assert spec.n_tiles == 3                  # 5 cores / 2 = 3 tiles (2+2+1)
    assert spec.n_cores_per_tile == [2, 2, 1]


def test_derive_arch_spec_rejects_unknown_preset():
    mapping = _fake_mapping(_fake_stage("neural", _fake_hcm((2, 1))))
    with pytest.raises(ValueError, match="unknown.*preset"):
        derive_arch_spec(mapping, preset_name="silicon-dreams")


def test_derive_arch_spec_rejects_mapping_with_no_neural_cores():
    """A pure-compute mapping shouldn't crash silently — surface it clearly."""
    mapping = _fake_mapping(_fake_stage("compute"))
    with pytest.raises(ValueError, match="no neural"):
        derive_arch_spec(mapping, preset_name="loihi")


def test_derive_arch_spec_attaches_preset_dict():
    """The chosen preset's per-event energy/latency dict is on the spec."""
    mapping = _fake_mapping(_fake_stage("neural", _fake_hcm((2, 1))))
    spec_loihi = derive_arch_spec(mapping, preset_name="loihi")
    spec_truenorth = derive_arch_spec(mapping, preset_name="truenorth")

    assert spec_loihi.preset == LOIHI_PRESET
    assert spec_truenorth.preset == TRUENORTH_PRESET
    assert spec_loihi.preset is not spec_truenorth.preset


def test_derive_arch_spec_name_includes_preset_and_core_count():
    mapping = _fake_mapping(_fake_stage("neural", _fake_hcm((4, 2), (4, 2))))
    spec = derive_arch_spec(mapping, preset_name="loihi")
    assert "loihi" in spec.name.lower()
    assert "2" in spec.name        # 2 cores total


# ---------------------------------------------------------------------------
# build_architecture — touches sanafe via the lazy accessor
# ---------------------------------------------------------------------------


def test_build_architecture_invokes_sanafe_with_spec_name(monkeypatch):
    fake_sanafe = MagicMock()
    fake_arch = MagicMock()
    fake_arch.tiles = [MagicMock()]
    fake_sanafe.Architecture.return_value = fake_arch
    monkeypatch.setattr(arch_synth, "_sanafe", lambda: fake_sanafe)

    spec = ArchSpec(
        name="mimarsinan_loihi_1core",
        n_tiles=1, n_cores_per_tile=[1],
        axons_per_core=2, neurons_per_core=1,
        preset=LOIHI_PRESET,
    )
    arch = build_architecture(spec)

    fake_sanafe.Architecture.assert_called_once()
    call = fake_sanafe.Architecture.call_args
    assert call.args[0] == "mimarsinan_loihi_1core" or call.kwargs.get("name") == "mimarsinan_loihi_1core"
    assert arch is fake_arch


def test_build_architecture_creates_one_tile_one_core_for_single_core_spec(monkeypatch):
    fake_sanafe = MagicMock()
    fake_arch = MagicMock()
    fake_sanafe.Architecture.return_value = fake_arch
    monkeypatch.setattr(arch_synth, "_sanafe", lambda: fake_sanafe)

    spec = ArchSpec(
        name="mimarsinan_loihi_1core",
        n_tiles=1, n_cores_per_tile=[1],
        axons_per_core=2, neurons_per_core=1,
        preset=LOIHI_PRESET,
    )
    build_architecture(spec)

    assert fake_arch.create_tile.call_count == 1
    assert fake_arch.create_core.call_count == 1


def test_build_architecture_creates_correct_tile_and_core_counts(monkeypatch):
    fake_sanafe = MagicMock()
    fake_arch = MagicMock()
    fake_sanafe.Architecture.return_value = fake_arch
    monkeypatch.setattr(arch_synth, "_sanafe", lambda: fake_sanafe)

    spec = ArchSpec(
        name="multi",
        n_tiles=3, n_cores_per_tile=[2, 2, 1],
        axons_per_core=8, neurons_per_core=4,
        preset=LOIHI_PRESET,
    )
    build_architecture(spec)
    assert fake_arch.create_tile.call_count == 3
    assert fake_arch.create_core.call_count == 5


def test_build_architecture_loihi_preset_applies_energy_constants(monkeypatch):
    """Per-event energies from LOIHI_PRESET should appear in create_core call kwargs."""
    fake_sanafe = MagicMock()
    fake_arch = MagicMock()
    fake_sanafe.Architecture.return_value = fake_arch
    monkeypatch.setattr(arch_synth, "_sanafe", lambda: fake_sanafe)

    spec = ArchSpec(
        name="t", n_tiles=1, n_cores_per_tile=[1],
        axons_per_core=2, neurons_per_core=1,
        preset=LOIHI_PRESET,
    )
    build_architecture(spec)

    # Each create_core call carries the preset's per-event numbers.
    flat = []
    for call in fake_arch.create_core.call_args_list:
        flat.append({**call.kwargs})
    assert len(flat) == 1
    kwargs = flat[0]
    # The numbers must come from LOIHI_PRESET, not hard-coded inside build_architecture.
    assert kwargs["synapse_energy_j"] == LOIHI_PRESET["synapse_energy_j"]
    assert kwargs["soma_energy_j"] == LOIHI_PRESET["soma_energy_j"]
    assert kwargs["soma_latency_s"] == LOIHI_PRESET["soma_latency_s"]


def test_build_architecture_custom_yaml_path_loads_external_arch(monkeypatch, tmp_path):
    """When ``custom_arch_path`` is given, ``sanafe.load_arch`` is used instead."""
    yaml_path = tmp_path / "custom_arch.yaml"
    yaml_path.write_text("architecture: {}\n")

    fake_sanafe = MagicMock()
    fake_arch = MagicMock()
    fake_arch.tiles = [MagicMock()]
    fake_arch.tiles[0].cores = [MagicMock()]
    fake_sanafe.load_arch.return_value = fake_arch
    monkeypatch.setattr(arch_synth, "_sanafe", lambda: fake_sanafe)

    spec = ArchSpec(
        name="custom", n_tiles=1, n_cores_per_tile=[1],
        axons_per_core=4, neurons_per_core=2,
        preset=LOIHI_PRESET,
    )
    arch = build_architecture(spec, custom_arch_path=str(yaml_path))

    fake_sanafe.load_arch.assert_called_once_with(str(yaml_path))
    fake_sanafe.Architecture.assert_not_called()
    assert arch is fake_arch


def test_build_architecture_custom_yaml_validates_enough_cores(monkeypatch, tmp_path):
    """A custom YAML with fewer total cores than the spec needs must raise."""
    yaml_path = tmp_path / "small_arch.yaml"
    yaml_path.write_text("architecture: {}\n")

    fake_sanafe = MagicMock()
    fake_arch = MagicMock()
    fake_arch.tiles = [MagicMock()]
    fake_arch.tiles[0].cores = [MagicMock()]   # 1 core total
    fake_sanafe.load_arch.return_value = fake_arch
    monkeypatch.setattr(arch_synth, "_sanafe", lambda: fake_sanafe)

    spec = ArchSpec(
        name="t", n_tiles=1, n_cores_per_tile=[3],   # needs 3 cores
        axons_per_core=4, neurons_per_core=2,
        preset=LOIHI_PRESET,
    )
    with pytest.raises(ValueError, match="cores"):
        build_architecture(spec, custom_arch_path=str(yaml_path))


def test_build_architecture_custom_yaml_path_missing_file_raises(monkeypatch, tmp_path):
    missing = tmp_path / "absent.yaml"
    monkeypatch.setattr(arch_synth, "_sanafe", lambda: MagicMock())
    spec = ArchSpec(
        name="t", n_tiles=1, n_cores_per_tile=[1],
        axons_per_core=2, neurons_per_core=1,
        preset=LOIHI_PRESET,
    )
    with pytest.raises(FileNotFoundError):
        build_architecture(spec, custom_arch_path=str(missing))


# ---------------------------------------------------------------------------
# Preset registry sanity
# ---------------------------------------------------------------------------


def test_presets_registry_includes_loihi_and_truenorth():
    assert "loihi" in PRESETS
    assert "truenorth" in PRESETS


def test_loihi_preset_has_all_required_keys():
    required = {
        "synapse_energy_j", "dendrite_energy_j", "soma_energy_j", "network_energy_j",
        "synapse_latency_s", "soma_latency_s", "network_latency_s",
    }
    assert required <= set(LOIHI_PRESET.keys())


def test_loihi_preset_values_are_positive_floats():
    for key, val in LOIHI_PRESET.items():
        assert isinstance(val, float), f"{key} is not a float"
        assert val > 0.0, f"{key} should be positive (got {val})"
