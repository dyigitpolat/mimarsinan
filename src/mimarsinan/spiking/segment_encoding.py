"""Boundary spike-train encoding shared by SCM/HCM/SANA-FE/Lava/Nevresim.

Centralises *how* a host-side ComputeOp output becomes a per-cycle ``(T, B, D)``
spike train for the next neural segment. The semantics match a NF-style cycle
loop: the wrapped Perceptron / structural module runs T times in single-step
mode on uniform-encoded raw inputs (or on an upstream spike train pulled from
``state_buffer_spikes``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict

import torch
import torch.nn as nn

from mimarsinan.mapping.hybrid_hardcore_mapping import (
    HybridHardCoreMapping,
    HybridStage,
)
from mimarsinan.mapping.ir import ComputeOp, IRSource
from mimarsinan.models.activations import LIFActivation
from mimarsinan.spiking.lif_utils import unwrap_lif_activation


class BoundaryKind(Enum):
    RAW_INPUT = "raw_input"
    ENCODING_LIF_PERCEPTRON = "encoding_lif_perceptron"
    ENCODING_SPLIT_HOST = "encoding_split_host"
    STRUCTURAL_PASSTHROUGH = "structural_passthrough"
    LEGACY_RATE = "legacy_rate"


@dataclass
class SegmentEncodingConfig:
    simulation_length: int
    spiking_mode: str
    cycle_accurate: bool
    spike_mode: str = "Uniform"
    thresholding_mode: str = "<="
    firing_mode: str = "Default"
    compute_dtype: torch.dtype = torch.float64

    @property
    def use_cycle_accurate_trains(self) -> bool:
        return self.spiking_mode == "lif" and self.cycle_accurate


@dataclass
class BoundaryLifCache:
    """Ephemeral ``LIFActivation`` instances for structural-op spike emission."""

    _cache: Dict[tuple, LIFActivation] = field(default_factory=dict)

    def get(
        self,
        *,
        T: int,
        activation_scale,
        thresholding_mode: str,
        firing_mode: str,
    ) -> LIFActivation:
        scale_key = float(
            activation_scale.item() if hasattr(activation_scale, "item")
            else activation_scale
        )
        key = (int(T), scale_key, str(thresholding_mode), str(firing_mode))
        lif = self._cache.get(key)
        if lif is not None:
            return lif
        from mimarsinan.spiking.lif_utils import boundary_lif_activation

        lif = boundary_lif_activation(
            T=int(T),
            activation_scale=activation_scale,
            thresholding_mode=thresholding_mode,
            firing_mode=firing_mode,
        )
        self._cache[key] = lif
        return lif


def _resolve_lif_perceptron(module) -> nn.Module | None:
    """Return inner ``Perceptron`` with ``LIFActivation`` for encoding spike trains."""
    if module is None:
        return None
    if hasattr(module, "perceptron") and module is not getattr(module, "perceptron"):
        return None
    candidate = module
    if not hasattr(candidate, "forward_spiking"):
        return None
    activation = getattr(candidate, "activation", None)
    if unwrap_lif_activation(activation) is not None:
        return candidate
    return None


def _is_structural_compute_module(module) -> bool:
    """Bare ``Conv``/``Linear``/``Sequential`` whose first child is one of those."""
    if module is None:
        return False
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        return True
    if isinstance(module, nn.Sequential) and len(module) > 0:
        return isinstance(module[0], (nn.Linear, nn.Conv1d, nn.Conv2d))
    return False


_STRUCTURAL_PASSTHROUGH_OP_TYPES = frozenset({
    "mean", "flatten", "identity", "dropout", "select",
    "add", "add_constant", "concat_constant", "layer_norm", "gelu",
    "max_pool2d", "avg_pool2d", "adaptive_avg_pool2d",
    "multi_head_attention", "linear",
})


def classify_encoding_boundary(
    op: ComputeOp,
    hybrid_mapping: HybridHardCoreMapping,
    config: SegmentEncodingConfig,
) -> BoundaryKind:
    """Classify how this ComputeOp's output should produce spike trains for downstream segments."""
    if not config.use_cycle_accurate_trains:
        return BoundaryKind.LEGACY_RATE
    if not isinstance(op, ComputeOp):
        return BoundaryKind.STRUCTURAL_PASSTHROUGH

    if op.op_type == "module":
        module = (op.params or {}).get("module")
        if _resolve_lif_perceptron(module) is not None:
            return BoundaryKind.ENCODING_LIF_PERCEPTRON
        if _is_structural_compute_module(module):
            return BoundaryKind.ENCODING_SPLIT_HOST
        return BoundaryKind.STRUCTURAL_PASSTHROUGH

    if op.op_type in _STRUCTURAL_PASSTHROUGH_OP_TYPES:
        return BoundaryKind.STRUCTURAL_PASSTHROUGH
    return BoundaryKind.STRUCTURAL_PASSTHROUGH


def _gather_op_input_train(
    op: ComputeOp,
    state_buffer: Dict[int, torch.Tensor],
    state_buffer_spikes: Dict[int, torch.Tensor],
    T: int,
    config: SegmentEncodingConfig,
) -> torch.Tensor | None:
    """Assemble ``(T, B, in_size)`` input train for ``op``: cached trains take precedence."""
    sources = op.input_sources.flatten()
    if len(sources) == 0:
        return None

    raw_input = state_buffer.get(-2)
    sample_batch = None
    if raw_input is not None:
        sample_batch = int(raw_input.shape[0])
    else:
        for src in sources:
            if isinstance(src, IRSource) and src.node_id >= 0:
                t = state_buffer_spikes.get(int(src.node_id))
                if t is None:
                    rate = state_buffer.get(int(src.node_id))
                    if rate is None:
                        continue
                    sample_batch = int(rate.shape[0])
                    break
                sample_batch = int(t.shape[1])
                break
    if sample_batch is None:
        return None

    in_size = len(sources)
    device = (raw_input.device if raw_input is not None
              else next(iter(state_buffer_spikes.values())).device)
    out = torch.zeros(T, sample_batch, in_size, dtype=config.compute_dtype, device=device)

    from mimarsinan.spiking.spike_trains import uniform_spike_train

    for idx, src in enumerate(sources):
        if not isinstance(src, IRSource):
            continue
        if src.is_off():
            continue
        if src.is_always_on():
            out[:, :, idx] = 1.0
            continue
        if src.is_input():
            if raw_input is None:
                return None
            rate_slice = raw_input[:, src.index].clamp(0.0, 1.0)
            spikes = uniform_spike_train(rate_slice, T).to(config.compute_dtype)
            out[:, :, idx] = spikes
            continue
        cached = state_buffer_spikes.get(int(src.node_id))
        if cached is not None:
            out[:, :, idx] = cached[:, :, src.index].to(config.compute_dtype)
            continue
        rate = state_buffer.get(int(src.node_id))
        if rate is None:
            return None
        rate_slice = rate[:, src.index].clamp(0.0, 1.0).to(config.compute_dtype)
        spikes = uniform_spike_train(rate_slice, T).to(config.compute_dtype)
        out[:, :, idx] = spikes

    return out


def _run_perceptron_single_step_T(
    perceptron,
    input_train: torch.Tensor,
    op: ComputeOp,
    config: SegmentEncodingConfig,
) -> torch.Tensor:
    """``T`` single-step forwards through a Perceptron; returns ``(T, B, D)`` spikes."""
    from spikingjelly.activation_based import functional

    lif = unwrap_lif_activation(perceptron.activation)
    assert lif is not None, "_run_perceptron_single_step_T requires reachable LIFActivation"

    T = input_train.shape[0]
    B = input_train.shape[1]

    was_ca = lif._cycle_accurate_mode
    lif.set_cycle_accurate(True)
    functional.reset_net(lif.if_node)
    try:
        outputs = []
        for t in range(T):
            inp = input_train[t]
            if op.input_shape is not None:
                inp = inp.reshape(B, *op.input_shape)
            inp_f = inp.to(torch.float32)
            out_t = perceptron(inp_f)
            outputs.append(out_t)
        stacked = torch.stack(outputs, dim=0)
    finally:
        lif.set_cycle_accurate(was_ca)

    return stacked.reshape(T, B, -1).to(config.compute_dtype)


def _run_structural_module_single_step_T(
    module: nn.Module,
    input_train: torch.Tensor,
    op: ComputeOp,
    config: SegmentEncodingConfig,
    lif_cache: BoundaryLifCache,
    hybrid_mapping: HybridHardCoreMapping,
) -> torch.Tensor:
    """``T`` cycles of structural ``module`` wrapped in an ephemeral LIF; ``(T, B, D)``."""
    from spikingjelly.activation_based import functional

    T = input_train.shape[0]
    B = input_train.shape[1]
    scale = float(hybrid_mapping.node_activation_scales.get(int(op.id), 1.0))
    lif = lif_cache.get(
        T=T,
        activation_scale=torch.tensor(scale),
        thresholding_mode=config.thresholding_mode,
        firing_mode=config.firing_mode,
    )
    lif.set_cycle_accurate(True)
    functional.reset_net(lif.if_node)
    outputs = []
    try:
        for t in range(T):
            inp = input_train[t]
            if op.input_shape is not None:
                inp = inp.reshape(B, *op.input_shape)
            inp_f = inp.to(torch.float32)
            with torch.no_grad():
                pre = module(inp_f)
            out_t = lif(pre.reshape(B, -1))
            outputs.append(out_t)
        stacked = torch.stack(outputs, dim=0)
    finally:
        lif.set_cycle_accurate(False)

    return stacked.reshape(T, B, -1).to(config.compute_dtype)


def emit_compute_spike_train(
    *,
    op: ComputeOp,
    state_buffer: Dict[int, torch.Tensor],
    state_buffer_spikes: Dict[int, torch.Tensor],
    config: SegmentEncodingConfig,
    hybrid_mapping: HybridHardCoreMapping,
    lif_cache: BoundaryLifCache,
) -> torch.Tensor | None:
    """Return ``(T, B, D)`` spike train for this ComputeOp boundary, or ``None`` if N/A."""
    kind = classify_encoding_boundary(op, hybrid_mapping, config)
    if kind in (BoundaryKind.LEGACY_RATE, BoundaryKind.STRUCTURAL_PASSTHROUGH):
        return None

    T = config.simulation_length
    input_train = _gather_op_input_train(
        op, state_buffer, state_buffer_spikes, T, config,
    )
    if input_train is None:
        return None

    module = (op.params or {}).get("module") if op.params else None
    if kind == BoundaryKind.ENCODING_LIF_PERCEPTRON:
        perceptron = _resolve_lif_perceptron(module)
        assert perceptron is not None
        return _run_perceptron_single_step_T(perceptron, input_train, op, config)

    if kind == BoundaryKind.ENCODING_SPLIT_HOST:
        assert module is not None
        return _run_structural_module_single_step_T(
            module, input_train, op, config, lif_cache, hybrid_mapping,
        )
    return None


def build_segment_input_spike_train(
    stage: HybridStage,
    seg_input_rates_clamped: torch.Tensor,
    state_buffer_spikes: Dict[int, torch.Tensor],
    *,
    config: SegmentEncodingConfig,
    hybrid_mapping: HybridHardCoreMapping,
    lif_cache: BoundaryLifCache,
    T: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """``(T, B, in_size)`` neural-segment input. Cached trains take precedence; otherwise
    raw-input slices get uniform encoding and any missing non-raw slice is a hard error
    (under cycle-accurate). In legacy rate mode, falls back to ``rates_to_spike_train``.
    """
    from mimarsinan.spiking.spike_trains import rates_to_spike_train, uniform_spike_train

    in_size = seg_input_rates_clamped.shape[1]
    spike_train = torch.zeros(
        T, batch_size, in_size, device=device, dtype=config.compute_dtype,
    )

    filled_ranges: list[tuple[int, int]] = []
    missing_slices: list[tuple[int, int, int, int]] = []  # (node_id, offset, size, slice_index)
    for slice_idx, s in enumerate(stage.input_map):
        train = state_buffer_spikes.get(int(s.node_id))
        if train is None:
            missing_slices.append((int(s.node_id), int(s.offset), int(s.size), slice_idx))
            continue
        spike_train[:, :, s.offset : s.offset + s.size] = (
            train[:, :, : s.size].to(config.compute_dtype)
        )
        filled_ranges.append((int(s.offset), int(s.offset) + int(s.size)))

    if not config.use_cycle_accurate_trains:
        if not filled_ranges:
            return rates_to_spike_train(
                seg_input_rates_clamped,
                T,
                spike_mode=config.spike_mode,
                log_fallback=True,
            ).to(config.compute_dtype)
        if not missing_slices:
            return spike_train
        encoded = rates_to_spike_train(
            seg_input_rates_clamped,
            T,
            spike_mode=config.spike_mode,
            log_fallback=False,
        ).to(config.compute_dtype)
        for lo, hi in filled_ranges:
            encoded[:, :, lo:hi] = spike_train[:, :, lo:hi]
        return encoded

    # Cycle-accurate path.
    only_raw_input = (
        len(stage.input_map) == 1
        and int(stage.input_map[0].node_id) == -2
    )
    if not filled_ranges and only_raw_input:
        return uniform_spike_train(
            seg_input_rates_clamped, T,
        ).to(config.compute_dtype)

    non_raw_missing = [m for m in missing_slices if m[0] != -2]
    raw_missing = [m for m in missing_slices if m[0] == -2]

    if non_raw_missing:
        if filled_ranges:
            raise ValueError(
                f"build_segment_input_spike_train: stage {stage.name!r} has cached spike "
                f"trains for some inputs but is missing spike train(s) for node_id(s) "
                f"{[m[0] for m in non_raw_missing]}. Every non-raw input slice must have a "
                f"cached train (cycle-accurate parity)."
            )
        # No cached trains at all and inputs are non-raw — legacy uniform fallback with warning.
        import logging
        logging.getLogger("mimarsinan.spiking.segment_encoding").warning(
            "build_segment_input_spike_train: cycle-accurate stage %r has no spike trains "
            "for non-raw inputs %s; falling back to uniform rate encoding.",
            stage.name, [m[0] for m in non_raw_missing],
        )
        return uniform_spike_train(seg_input_rates_clamped, T).to(config.compute_dtype)

    if raw_missing:
        # Uniform-encode the raw slices, overlay cached trains.
        encoded = uniform_spike_train(seg_input_rates_clamped, T).to(config.compute_dtype)
        for lo, hi in filled_ranges:
            encoded[:, :, lo:hi] = spike_train[:, :, lo:hi]
        return encoded

    return spike_train
