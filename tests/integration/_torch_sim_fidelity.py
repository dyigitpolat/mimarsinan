"""Torch-NF vs deployed-HCM-sim fidelity harness across modes and mapping configs.

The lock: for a torch model converted to the IR and packed into a hybrid hard-core
mapping (HCM), the torch-side neuromorphic forward (NF) must agree with the
deployed spiking sim at the *strongest granularity the (mode, mapping-config) pair
admits*. The harness builds both sides from one model and picks the assertion:

- **bit-exact** (``atol=0`` float64 output; for LIF additionally per-neuron
  ``k == k`` spike counts) for the configs that compose losslessly. Holds for
  ``lif``, ``ttfs_cycle_based`` (cascaded) and ``ttfs_quantized``:
    * ``identity`` — one un-split core per perceptron.
    * ``neuron_split`` — a core's neurons tiled across hard cores.
    * ``axon_fuse`` — **the partial-sum mapping** (coalescing): a fan-in wider
      than one hard core consumes N hard cores of the same type and registers them
      to the sim as a single *bigger* hard core (``axons_per_core`` = Σ component
      axons, one wide crossbar). The full weighted sum is computed once, so it is
      bit-exact — the partial sums are membrane potentials in one merged core,
      never re-fired spikes. Coalescing is a chip capability (inter-core membrane
      transfer); ``allow_coalescing=False`` makes a wide fan-in unmappable
      (``WideFanInUnsupportedError``), not lossy — the spike-domain firing
      partial-sum fallback was removed.
- **bounded** (the value residual is locked to a documented quantization-step
  budget) for the cell that *cannot* be bit-exact by construction:
    * ``ttfs`` (continuous) — the analytical NF carries a real-valued TTFS while
      the sim quantizes to S steps, so the value sits ~half a step off its own
      quantized deployment everywhere.

  This bounded cell's *decision* (argmax) is preserved only on a trained,
  scale-calibrated model — that is the production ``nf_scm_parity`` gate's job;
  uncalibrated argmax ties make it unstable in a unit test, so the harness locks
  the value residual and only reports the decision agreement.

A mid-graph ComputeOp (LayerNorm) makes a model multi-segment ("sync-point
segmentation"): the deployed sim decodes each segment output to a rate, runs the
host op, and re-encodes for the next segment, and the NF mirrors that — so the
invariants above hold *across* the sync boundary too.

Neuron splitting needs the hard core wide enough in axons to hold the fan-in but
narrower than the layer in neurons; axon fusing needs the hard-core axon budget
below the fan-in (so the packer fuses cores). ``mapping_configs`` sizes these from
the model and ``assert_config_triggered`` fails loudly if a config did not actually
pack the way its name claims — so no cell can vacuously pass as "identity in
disguise".

So the harness does not pretend every (mode, config) is bit-exact — it asserts the
true invariant for each cell and documents why the bounded cells are bounded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.torch_mapping.encoding_layers import mark_encoding_layers
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import build_hybrid_hard_core_mapping
from mimarsinan.mapping.platform.mapping_structure import MappingStrategy
from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow
from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.models.nn.activations.ttfs_spiking import TTFSActivation
from mimarsinan.mapping.support.bias_compensation import calibration_forward_for_mode
from mimarsinan.spiking.chip_aligned_nf import chip_aligned_segment_forward

from integration._split_reassembly import (
    hcm_per_perceptron_counts,
    torch_lif_node_counts,
)

# Modes whose torch NF and deployed sim share value-domain semantics exactly
# (every neuron's value is bit-identical) when the packing composes losslessly.
BIT_EXACT_MODES: frozenset[str] = frozenset({"lif", "ttfs_cycle_based", "ttfs_quantized"})

# Mapping configs whose packing composes losslessly (no spike-domain re-quantization).
# ``axon_fuse`` is the partial-sum mapping (N cores fused into one wider crossbar).
LOSSLESS_CONFIGS: frozenset[str] = frozenset({"identity", "neuron_split", "axon_fuse"})

# Bounded-cell value budgets, expressed in quantization steps (value bound = steps/T).
# Measured on single-on-chip-layer models (no downstream compounding) across seeds,
# with margin; a genuine regression pushes the residual toward full scale (T/T) and
# trips these. The decision-level agreement of a *trained, scale-calibrated* model is
# the production ``nf_scm_parity`` gate's job — uncalibrated argmax ties make it
# unstable here, so the harness locks the value residual instead and only reports the
# decision agreement.
_CONTINUOUS_STEP_BUDGET = 2.0     # continuous TTFS ~half a step off its quantized sim


@dataclass(frozen=True)
class MappingConfig:
    """One packing knob preset.

    ``ir_max_axons`` < a layer's fan-in forces psum (axon-tile) split at the IR
    level. A hard core ``core_max_neurons`` < a layer's width with
    ``allow_neuron_splitting`` forces neuron tiling — but only if
    ``core_max_axons`` still holds the fan-in (else the softcore fits no core and
    packing fails), so the two budgets are independent.
    """

    name: str
    ir_max_axons: int
    ir_max_neurons: int
    core_max_axons: int
    core_max_neurons: int
    allow_neuron_splitting: bool
    allow_coalescing: bool


def mapping_configs(*, wide_dim: int, split_neurons: int, fuse_core_axons: int) -> dict[str, MappingConfig]:
    """Build the canonical (bit-exact-intended) configs sized for a given model.

    ``wide_dim`` is a generous budget (no splitting); ``split_neurons`` is a hard-core
    neuron budget below the widest on-chip layer (forces neuron tiling, axons wide);
    ``fuse_core_axons`` is a hard-core axon budget below the widest fan-in so the
    packer fuses N such cores into one wider crossbar (the partial-sum mapping). All
    three are the deployable lossless configs.
    """
    return {
        "identity": MappingConfig("identity", wide_dim, wide_dim, wide_dim, wide_dim, False, False),
        "neuron_split": MappingConfig(
            "neuron_split", wide_dim, wide_dim, wide_dim, split_neurons, True, False),
        "axon_fuse": MappingConfig(
            "axon_fuse", wide_dim, wide_dim, fuse_core_axons, wide_dim, True, True),
    }


def mapping_structure(hybrid) -> dict[str, int]:
    """Count the packing fingerprints of a hybrid mapping: neural segments, hard
    cores, neuron-split fragments, firing psum partial cores, and fused (combined)
    hard cores."""
    neural_segments = hard_cores = split_frags = psum_partials = fused_cores = 0
    for stage in hybrid.stages:
        if stage.hard_core_mapping is None:
            continue
        neural_segments += 1
        mapping = stage.hard_core_mapping
        placements = mapping.soft_core_placements_per_hard_core
        hard_cores += len(placements)
        for core in mapping.cores:
            if getattr(core, "fused_component_axons", None):
                fused_cores += 1
        for core_placements in placements:
            for pl in core_placements:
                if pl.get("split_group_id") is not None:
                    split_frags += 1
                if pl.get("psum_role") in ("partial_pos", "partial_neg"):
                    psum_partials += 1
    return {
        "neural_segments": neural_segments,
        "hard_cores": hard_cores,
        "split_frags": split_frags,
        "psum_partials": psum_partials,
        "fused_cores": fused_cores,
    }


def assert_config_triggered(hybrid, config_name: str) -> dict[str, int]:
    """Fail loudly if a config did not pack the way its name claims (a cell that
    silently degenerates to identity would pass vacuously)."""
    s = mapping_structure(hybrid)
    if config_name == "neuron_split":
        assert s["split_frags"] > 0, (
            f"neuron_split did not tile any core (split_frags=0); the model's on-chip "
            f"layers are narrower than the hard-core neuron budget — widen them"
        )
    elif config_name == "axon_fuse":
        assert s["fused_cores"] > 0, (
            f"axon_fuse did not fuse any cores (fused_cores=0); no layer's fan-in "
            f"exceeds the hard-core axon budget — lower fuse_core_axons"
        )
        assert s["psum_partials"] == 0, (
            f"axon_fuse unexpectedly emitted firing psum partials: {s} (coalescing off?)"
        )
    elif config_name == "identity":
        assert s["split_frags"] == 0 and s["psum_partials"] == 0 and s["fused_cores"] == 0, (
            f"identity config unexpectedly split/fused the mapping: {s}"
        )
    return s


def _install_activations(flow, spiking_mode: str, T: int) -> dict[int, nn.Module]:
    nodes: dict[int, nn.Module] = {}
    if spiking_mode == "lif":
        for i, p in enumerate(flow.get_perceptrons()):
            lif = LIFActivation(T=T, activation_scale=torch.tensor(1.0), thresholding_mode="<=")
            p.base_activation = lif
            p.activation = lif
            nodes[i] = lif
        return nodes
    for p in flow.get_perceptrons():
        p.set_activation(TTFSActivation(
            T=T, activation_scale=p.activation_scale,
            input_scale=p.input_activation_scale, bias=p.layer.bias,
            thresholding_mode="<=", encoding=getattr(p, "is_encoding_layer", False),
        ))
    return nodes


def build_torch_and_hcm(
    torch_model: nn.Module,
    input_shape,
    num_classes: int,
    *,
    spiking_mode: str,
    config: MappingConfig,
    T: int,
    ttfs_cycle_schedule: str = "cascaded",
    device: str = "cpu",
):
    """Convert ``torch_model`` to NF + a packed HCM sim for one (mode, config).

    Returns ``(flow, hcm, hybrid, nodes)``: ``flow`` is the torch NF (float32 for
    LIF, float64 otherwise), ``hcm`` the deployed ``SpikingHybridCoreFlow``,
    ``hybrid`` the mapping (for per-neuron reassembly), and ``nodes`` the LIF
    activation map (empty for TTFS).
    """
    flow = convert_torch_model(torch_model, input_shape, num_classes, device=device)
    flow.eval()
    repr_ = flow.get_mapper_repr()
    mark_encoding_layers(repr_)
    nodes = _install_activations(flow, spiking_mode, T)
    if spiking_mode != "lif":
        flow = flow.double()
    repr_.assign_perceptron_indices()

    if spiking_mode == "lif":
        firing, spike = "Default", "Uniform"
        flow_kwargs = dict(spiking_mode="lif", cycle_accurate_lif_forward=True)
    else:
        firing, spike = "TTFS", "TTFS"
        flow_kwargs = dict(spiking_mode=spiking_mode, ttfs_cycle_schedule=ttfs_cycle_schedule)

    ir = IRMapping(
        q_max=127.0, firing_mode=firing,
        max_axons=config.ir_max_axons, max_neurons=config.ir_max_neurons,
        allow_coalescing=config.allow_coalescing,
    ).map(repr_)
    hybrid = build_hybrid_hard_core_mapping(
        ir_graph=ir,
        cores_config=[{
            "max_axons": config.core_max_axons,
            "max_neurons": config.core_max_neurons,
            "count": 4000,
        }],
        strategy=MappingStrategy.from_permissions(
            allow_neuron_splitting=config.allow_neuron_splitting,
            allow_coalescing=config.allow_coalescing,
        ),
    )
    hcm = SpikingHybridCoreFlow(
        input_shape, hybrid, simulation_length=T, preprocessor=nn.Identity(),
        firing_mode=firing, spike_mode=spike, thresholding_mode="<=", **flow_kwargs,
    )
    return flow, hcm, hybrid, nodes


def _torch_nf(flow, x, spiking_mode, T):
    if spiking_mode == "lif":
        return chip_aligned_segment_forward(flow, x, T)
    return calibration_forward_for_mode(spiking_mode)(flow, x, T)


def _lif_per_neuron_counts(flow, hcm, hybrid, nodes, sample, T):
    """Torch NF node counts and reassembled HCM counts, per on-chip perceptron."""
    torch_counts = torch_lif_node_counts(
        lambda: chip_aligned_segment_forward(flow, sample, T), nodes, sample, T)
    with torch.no_grad():
        _, record = hcm.forward_with_recording(sample, sample_index=0)
    hcm_counts = hcm_per_perceptron_counts(record, hybrid)
    pairs = [(pi, torch_counts[pi], hcm_counts[pi]) for pi in torch_counts if pi in hcm_counts]
    assert pairs, "no on-chip perceptron compared (assign_perceptron_indices run?)"
    for pi, t, h in pairs:
        assert t.shape == h.shape, f"perceptron {pi} shape {t.shape} vs HCM {h.shape}"
    return pairs


def _assert_per_neuron_lif_exact(flow, hcm, hybrid, nodes, sample, T):
    """LIF per-neuron k == k: the strongest lock — packing reassembly bit-exact."""
    pairs = _lif_per_neuron_counts(flow, hcm, hybrid, nodes, sample, T)
    for pi, t, h in pairs:
        assert np.array_equal(t, h), (
            f"LIF per-neuron mismatch perceptron {pi}: "
            f"{int((t != h).sum())}/{t.size} neurons differ "
            f"(torch_total={int(t.sum())} hcm_total={int(h.sum())})"
        )
    return len(pairs)


def assert_torch_sim_fidelity(
    flow,
    hcm,
    hybrid,
    nodes,
    samples: torch.Tensor,
    *,
    spiking_mode: str,
    config_name: str,
    T: int,
) -> dict:
    """Lock torch NF == deployed HCM sim for ``samples`` at the strongest valid
    granularity for ``(spiking_mode, config_name)``.

    Bit-exact cells (``identity``/``neuron_split``/``axon_fuse`` × ``lif``/
    ``ttfs_cycle_based``/``ttfs_quantized``) assert ``atol=0`` output (+ LIF
    per-neuron ``k==k``) on any model. The bounded cell (``ttfs`` continuous,
    any config) asserts the value residual stays inside a documented
    quantization-step budget — it is not bit-exact by construction and is
    exercised on a single-on-chip-layer model so the residual does not compound.
    Returns measured residuals; raises ``AssertionError`` on divergence."""
    dtype = torch.float32 if spiking_mode == "lif" else torch.float64
    x = samples.to(dtype)
    with torch.no_grad():
        nf = _torch_nf(flow, x, spiking_mode, T).double()
        hc = hcm(x).double() / float(T)

    out_max_abs = float((nf - hc).abs().max())
    decision_agreement = float((nf.argmax(dim=1) == hc.argmax(dim=1)).double().mean())
    result = {
        "out_max_abs": out_max_abs,
        "decision_agreement": decision_agreement,
        "granularity": None,
    }

    if config_name in LOSSLESS_CONFIGS and spiking_mode in BIT_EXACT_MODES:
        result["granularity"] = "bit_exact"
        assert out_max_abs == 0.0, (
            f"{spiking_mode}/{config_name}: expected bit-exact torch==sim, "
            f"max|Δ|={out_max_abs} (a fidelity regression)"
        )
        if spiking_mode == "lif" and nodes:
            result["per_neuron_perceptrons"] = _assert_per_neuron_lif_exact(
                flow, hcm, hybrid, nodes, samples[:1].to(dtype), T)
        return result

    # Bounded cell: continuous TTFS (real-valued NF vs S-step quantized sim) is
    # inherently not bit-exact. Lock the value residual to its step budget.
    result["granularity"] = "bounded"
    assert np.isfinite(out_max_abs), (
        f"{spiking_mode}/{config_name}: non-finite value residual (NaN/inf in the sim)"
    )
    budget = _CONTINUOUS_STEP_BUDGET / float(T)
    result["value_bound"] = budget
    assert out_max_abs <= budget, (
        f"{spiking_mode}/{config_name}: continuous-TTFS residual {out_max_abs:.4f} > "
        f"budget {budget:.4f} — it should sit ~0.5/T off its own quantized deployment"
    )
    return result
