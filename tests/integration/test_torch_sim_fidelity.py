"""Torch-NF == deployed-HCM-sim fidelity lock across deployment modes × mapping configs.

Locks the strongest valid torch↔sim invariant for every (deployment mode, mapping
config) cell over real converted models exercising neuron split, the partial-sum
mapping (axon fuse) and sync-point segmentation (a mid-graph LayerNorm ComputeOp).
The per-cell granularity and the reason each bounded cell cannot be bit-exact live
in ``_torch_sim_fidelity``; this file pins which cells hold which invariant and
proves each config actually packs the way its name claims (no vacuous "identity in
disguise" passes).

- bit-exact (``lif``/``ttfs_cycle_based``/``ttfs_quantized`` × identity|neuron_split|
  axon_fuse, on single-core, multi-core and sync-point models): float64 ``atol=0`` +
  LIF per-neuron ``k==k``. ``axon_fuse`` is the partial-sum mapping — a wide fan-in
  consumes N hard cores fused into one wider crossbar, so it computes the full sum
  once and is bit-exact (not a re-fired spike accumulation).
- bounded: ``ttfs`` continuous — the analytical real-valued NF sits ~half a step
  off its S-step quantized sim, so the value residual stays inside a documented
  quantization-step budget, exercised on a single-on-chip-layer model so it does
  not compound across layers.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from integration._torch_sim_fidelity import (
    assert_config_triggered,
    assert_torch_sim_fidelity,
    build_torch_and_hcm,
    mapping_configs,
    mapping_structure,
)

T = 8
BIT_EXACT_MODES = ["lif", "ttfs_cycle_based", "ttfs_quantized"]
ALL_MODES = BIT_EXACT_MODES + ["ttfs"]
LOSSLESS_CONFIGS = ["identity", "neuron_split", "axon_fuse"]
INPUT_SHAPE = (16,)
NUM_CLASSES = 10

# split_neurons=8 tiles every on-chip layer (widths 10/20 > 8) while keeping axons
# wide; fuse_core_axons=16 < every fan-in (>=21) so wide layers fuse N cores into one.
_CFGS = mapping_configs(wide_dim=64, split_neurons=8, fuse_core_axons=16)


class _OneOnChip(nn.Module):
    """input -> fc1 (encoding host) -> ReLU -> fc2 (the single on-chip layer).

    One on-chip neural core, so a bounded cell's residual cannot compound across
    layers — the budget the harness asserts is the per-layer quantization residual.
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 24)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(24, 10)
        self.act2 = nn.ReLU()

    def forward(self, x):
        return self.act2(self.fc2(self.act1(self.fc1(x))))


class _TwoOnChip(nn.Module):
    """One neural segment, two on-chip layers (no mid-graph ComputeOp)."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 24)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(24, 20)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(20, 10)
        self.act3 = nn.ReLU()

    def forward(self, x):
        return self.act3(self.fc3(self.act2(self.fc2(self.act1(self.fc1(x))))))


class _SyncPoint(nn.Module):
    """Two on-chip segments split by a LayerNorm sync-point (mid-graph ComputeOp)."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 24)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(24, 20)
        self.act2 = nn.ReLU()
        self.ln = nn.LayerNorm(20)
        self.fc3 = nn.Linear(20, 10)
        self.act3 = nn.ReLU()

    def forward(self, x):
        x = self.act2(self.fc2(self.act1(self.fc1(x))))
        return self.act3(self.fc3(self.ln(x)))


_MODELS = {"one_on_chip": _OneOnChip, "two_on_chip": _TwoOnChip, "sync_point": _SyncPoint}


def _samples(n, seed):
    torch.manual_seed(seed)
    return torch.rand(n, 16)


def _build(model_key, mode, config):
    torch.manual_seed(0)
    return build_torch_and_hcm(
        _MODELS[model_key](), INPUT_SHAPE, NUM_CLASSES,
        spiking_mode=mode, config=config, T=T,
    )


def _run_cell(model_key, mode, config, *, n_samples=24, seed=11):
    flow, hcm, hybrid, nodes = _build(model_key, mode, config)
    assert_config_triggered(hybrid, config.name)
    return assert_torch_sim_fidelity(
        flow, hcm, hybrid, nodes, _samples(n_samples, seed),
        spiking_mode=mode, config_name=config.name, T=T,
    )


# --- bit-exact lock: identity, neuron_split & axon_fuse; every mode; every model ---

@pytest.mark.parametrize("model_key", list(_MODELS))
@pytest.mark.parametrize("mode", BIT_EXACT_MODES)
@pytest.mark.parametrize("config_name", LOSSLESS_CONFIGS)
def test_bit_exact_torch_equals_sim(model_key, mode, config_name):
    """torch NF == deployed sim, float64 ``atol=0`` (LIF also per-neuron k==k),
    across single-core, multi-core and sync-point models. ``axon_fuse`` is the
    partial-sum mapping: a wide fan-in fused into one wider crossbar stays exact."""
    result = _run_cell(model_key, mode, _CFGS[config_name])
    assert result["granularity"] == "bit_exact"
    assert result["out_max_abs"] == 0.0
    if mode == "lif":
        assert result.get("per_neuron_perceptrons", 0) > 0


def test_axon_fuse_builds_one_wider_hard_core():
    """The partial-sum mapping consumes N hard cores of one type and registers a
    single bigger hard core (Σ component axons) — pin that it fuses, not splits."""
    _, _, hybrid, _ = _build("one_on_chip", "lif", _CFGS["axon_fuse"])
    s = mapping_structure(hybrid)
    assert s["fused_cores"] > 0 and s["psum_partials"] == 0
    fused_widths = [
        sum(core.fused_component_axons)
        for stage in hybrid.stages if stage.hard_core_mapping is not None
        for core in stage.hard_core_mapping.cores
        if getattr(core, "fused_component_axons", None)
    ]
    # fc2 fan-in (24 + bias) > the 16-axon core, so it fuses two 16-axon cores.
    assert fused_widths and all(w > 16 for w in fused_widths)


# --- bounded lock: continuous ttfs (real-valued NF vs S-step quantized sim) ---

@pytest.mark.parametrize("config_name", LOSSLESS_CONFIGS)
def test_continuous_ttfs_value_residual_bounded(config_name):
    """Continuous TTFS sits ~half a quantization step off its own quantized
    deployment everywhere — never bit-exact, but bounded (single on-chip layer)."""
    result = _run_cell("one_on_chip", "ttfs", _CFGS[config_name], n_samples=32, seed=5)
    assert result["granularity"] == "bounded"
    assert result["out_max_abs"] <= result["value_bound"]


# --- documentation guards: the headline claims cannot silently weaken ---

def test_bit_exact_cells_are_truly_zero():
    """Every bit-exact cell really measures max|Δ|==0 (not merely small) so a
    sub-tolerance drift cannot pass — checked on the sync-point model."""
    for mode in BIT_EXACT_MODES:
        for config_name in LOSSLESS_CONFIGS:
            result = _run_cell("sync_point", mode, _CFGS[config_name], n_samples=16, seed=7)
            assert result["out_max_abs"] == 0.0, (mode, config_name, result["out_max_abs"])


def test_wide_fan_in_is_never_a_firing_partial_sum():
    """The lossy firing partial-sum decomposition was removed: a wide fan-in fuses
    into one wider crossbar (bit-exact), never emitting firing ``partial_*`` cores."""
    for model_key in _MODELS:
        _, _, hybrid, _ = _build(model_key, "lif", _CFGS["axon_fuse"])
        assert mapping_structure(hybrid)["psum_partials"] == 0


def test_wide_fan_in_without_coalescing_is_unmappable():
    """Coalescing is a chip capability (inter-core membrane transfer). With it off,
    a wide fan-in is unmappable and fails loudly — never a silent lossy mapping."""
    from mimarsinan.mapping.platform.mapping_structure import WideFanInUnsupportedError
    from integration._torch_sim_fidelity import MappingConfig

    no_coalesce = MappingConfig(
        "axon_fuse", ir_max_axons=16, ir_max_neurons=64,
        core_max_axons=16, core_max_neurons=64,
        allow_neuron_splitting=True, allow_coalescing=False,
    )
    with pytest.raises(WideFanInUnsupportedError):
        _build("one_on_chip", "lif", no_coalesce)
