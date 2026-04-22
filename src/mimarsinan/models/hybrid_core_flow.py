"""
SpikingHybridCoreFlow: Spiking simulation for HybridHardCoreMapping.

Supports both rate-coded (Default/Novena) and Time-to-First-Spike (TTFS)
firing modes.  Handles skip connections / residual paths via a global
state buffer keyed by original IR node_id.

TTFS mode implements the B1-model from:
  Stanojevic et al., "High-performance deep spiking neural networks with
  0.3 spikes per neuron", Nature Communications 15, 6793 (2024).
  https://www.nature.com/articles/s41467-024-51110-5

Each layer's neuron dynamics have two phases:

  Phase 1 (t < t_min):  Accumulate incoming spikes.
      V_i(t_min) = Σ_j W_ij · x_j   (matmul; bias via always-on axon)

  Phase 2 (t_min ≤ t ≤ t_max):  Constant ramp (B=1 → +θ/S per step).
      Neuron fires when V_i reaches threshold θ_i.
      Output activation  x_i = (S − k_fire) / S.

Two deployment modes for TTFS (selected via ``spiking_mode``):

  * **ttfs** (continuous / event-based) — exact analytical computation,
    no time-step discretisation.  Equivalent to ``clamp(ReLU(W @ x + b) / θ, 0, 1)``.
    Outputs clamped to ``[0, 1]`` per core (hardware TTFS fires at most once).
    Inputs are not clamped; weight matrices normalize ComputeOp sources.
  * **ttfs_quantized** (analytical quantised) — closed-form computation
    that matches the cycle-based simulation exactly but runs in O(N_cores)
    instead of O(max_latency * S * N_cores).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from mimarsinan.mapping.chip_latency import ChipLatency
from mimarsinan.mapping.hybrid_hardcore_mapping import (
    HybridHardCoreMapping,
    HybridStage,
    SegmentIOSlice,
)
from mimarsinan.mapping.ir import ComputeOp, IRSource

from mimarsinan.mapping.spike_source_spans import SpikeSourceSpan, compress_spike_sources


# Precision for on-chip spiking compute.
#
# The C++ nevresim simulator uses:
#   * int-weight rate-coded path: exact integer arithmetic
#     (weight_t = int, MembranePotential<weight_t> = int, spike_t = int_fast8_t).
#   * TTFS paths: IEEE-754 64-bit float (signal_t = double).
#
# PyTorch defaulting to float32 here introduces ~1e-7 roundoff per operation,
# which accumulates and — most importantly — flips ``ceil`` / threshold
# comparisons for neurons whose membrane potential lands near θ, producing
# per-sample prediction differences vs the C++ simulator.  Using float64
# matches C++'s double for TTFS and exactly represents integer sums up to
# 2**53 for rate-coded mode (far larger than any practical axon × weight
# range), restoring bit-for-bit equivalence at the cost of ~2× compute
# time on GPU.  These flows run once per pipeline so the cost is acceptable.
_COMPUTE_DTYPE = torch.float64


class SpikingHybridCoreFlow(nn.Module):
    """
    Execute a HybridHardCoreMapping using a global state buffer.

    State buffer (``Dict[int, Tensor]``) keyed by original IR node_id.
    Neural segment I/O is described by ``SegmentIOSlice`` metadata on
    each ``HybridStage``.  ComputeOps use their ``input_sources`` to
    gather directly from the state buffer.
    """

    _TTFS_FIRING_MODES = {"TTFS"}
    _TTFS_SPIKING_MODES = {"ttfs", "ttfs_quantized"}

    def __init__(
        self,
        input_shape,
        hybrid_mapping: HybridHardCoreMapping,
        simulation_length: int,
        preprocessor: nn.Module | None = None,
        firing_mode: str = "Default",
        spike_mode: str = "Uniform",
        thresholding_mode: str = "<",
        spiking_mode: str = "rate",
    ):
        super().__init__()

        self.input_shape = input_shape
        self.hybrid_mapping = hybrid_mapping
        self.simulation_length = int(simulation_length)
        self.preprocessor = preprocessor if preprocessor is not None else nn.Identity()

        self.firing_mode = firing_mode
        self.spike_mode = spike_mode
        self.thresholding_mode = thresholding_mode
        self.spiking_mode = spiking_mode

        assert firing_mode in ["Default", "Novena", "TTFS"]
        assert spike_mode in ["Stochastic", "Deterministic", "FrontLoaded", "Uniform", "TTFS"]
        assert thresholding_mode in ["<", "<="]

        # Single-segment tensor cache.  Each neural segment's hw-core
        # matrices / thresholds / hardware_bias are uploaded to device
        # lazily and evicted as soon as a different segment is requested.
        # Why only one at a time: a ViT-scale mapping has 12 segments
        # each holding ~200 hw cores (~18 MB per core in float64) —
        # caching all of them simultaneously was ~40 GB of GPU residency
        # and caused OOM even though no single segment's weights are
        # that large.  Segments execute sequentially in the forward loop,
        # so a size-1 cache still eliminates redundant re-uploads within
        # a single segment (across cycles, batches of the same shape).
        self._segment_tensor_cache: Dict[int, dict] = {}
        self._segment_tensor_cache_key: int | None = None

    # ---------------------------------------------------------------------
    # Spike generation (rate-coded modes)
    # ---------------------------------------------------------------------
    def to_stochastic_spikes(self, tensor: torch.Tensor) -> torch.Tensor:
        return (torch.rand(tensor.shape, device=tensor.device) < tensor).float()

    def to_front_loaded_spikes(self, tensor: torch.Tensor, cycle: int) -> torch.Tensor:
        return (torch.round(tensor * self.simulation_length) > cycle).float()

    def to_deterministic_spikes(self, tensor: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        return (tensor > threshold).float()

    def to_uniform_spikes(self, tensor: torch.Tensor, cycle: int) -> torch.Tensor:
        T = self.simulation_length
        N = torch.round(tensor * T).to(torch.long)

        mask = (N != 0) & (N != T) & (cycle < T)
        N_safe = torch.clamp(N, min=1)
        spacing = T / N_safe.float()

        result = mask & (torch.floor(cycle / spacing) < N_safe) & (torch.floor(cycle % spacing) == 0)
        result = result.float()
        result[N == T] = 1.0
        return result

    def to_spikes(self, tensor: torch.Tensor, cycle: int) -> torch.Tensor:
        if self.spike_mode == "Stochastic":
            return self.to_stochastic_spikes(tensor)
        if self.spike_mode == "Deterministic":
            return self.to_deterministic_spikes(tensor)
        if self.spike_mode == "FrontLoaded":
            return self.to_front_loaded_spikes(tensor, cycle)
        if self.spike_mode == "Uniform":
            return self.to_uniform_spikes(tensor, cycle)
        raise ValueError("Invalid spike mode: " + str(self.spike_mode))

    # ---------------------------------------------------------------------
    # State-buffer helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _assemble_segment_input(
        input_map: list[SegmentIOSlice],
        state_buffer: Dict[int, torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build a segment's composite input tensor from the state buffer."""
        total_size = max((s.offset + s.size for s in input_map), default=0)
        inp = torch.zeros(batch_size, total_size, device=device, dtype=_COMPUTE_DTYPE)
        for s in input_map:
            buf = state_buffer[s.node_id]
            inp[:, s.offset : s.offset + s.size] = buf[:, :s.size].to(_COMPUTE_DTYPE)
        return inp

    @staticmethod
    def _store_segment_output(
        output_map: list[SegmentIOSlice],
        state_buffer: Dict[int, torch.Tensor],
        output_tensor: torch.Tensor,
    ) -> None:
        """Parse a segment's output tensor into the state buffer."""
        for s in output_map:
            state_buffer[s.node_id] = output_tensor[:, s.offset : s.offset + s.size]

    def _gather_final_output(
        self,
        state_buffer: Dict[int, torch.Tensor],
        original_input: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Assemble the network's final output from the state buffer."""
        output_sources = self.hybrid_mapping.output_sources.flatten()
        out = torch.zeros(batch_size, len(output_sources), device=device, dtype=_COMPUTE_DTYPE)
        for idx, src in enumerate(output_sources):
            if not isinstance(src, IRSource):
                continue
            if src.is_off():
                continue
            elif src.is_input():
                out[:, idx] = original_input[:, src.index].to(_COMPUTE_DTYPE)
            elif src.is_always_on():
                out[:, idx] = 1.0
            else:
                out[:, idx] = state_buffer[src.node_id][:, src.index].to(_COMPUTE_DTYPE)
        return out

    # ---------------------------------------------------------------------
    # Segment execution helpers (unchanged internal mechanics)
    # ---------------------------------------------------------------------
    def _fill_signal_tensor_from_spans(
        self,
        out: torch.Tensor,
        *,
        input_spikes: torch.Tensor,
        buffers: list[torch.Tensor],
        spans: list[SpikeSourceSpan],
        cycle: int = -1,
    ) -> None:
        out.zero_()
        for sp in spans:
            d0 = int(sp.dst_start)
            d1 = int(sp.dst_end)
            if sp.kind == "off":
                continue
            if sp.kind == "on":
                out[:, d0:d1].fill_(1.0)
                continue
            if sp.kind == "input":
                out[:, d0:d1] = input_spikes[:, int(sp.src_start):int(sp.src_end)]
                continue
            out[:, d0:d1] = buffers[int(sp.src_core)][:, int(sp.src_start):int(sp.src_end)]

    def _get_segment_tensors(self, stage: HybridStage, device: torch.device) -> dict:
        """Return cached (axon_spans, output_spans, core_params, thresholds,
        hw_biases) for ``stage``; building and uploading to device on
        first access.  Weights are immutable post-build so this cache is
        safe — the huge CPU→GPU transfer is paid once per stage, not per
        batch.
        """
        mapping = stage.hard_core_mapping
        assert mapping is not None
        key = id(stage)
        cached = self._segment_tensor_cache.get(key)
        if cached is not None and cached.get("device") == device:
            return cached

        # Evict any previously-cached segment before uploading the new
        # one — keep GPU residency bounded to a single segment's worth
        # of weights.  Drop hard references so the allocator can reclaim.
        if self._segment_tensor_cache_key is not None:
            prev = self._segment_tensor_cache.pop(
                self._segment_tensor_cache_key, None,
            )
            if prev is not None:
                prev.clear()
            self._segment_tensor_cache_key = None
            if device.type == "cuda":
                torch.cuda.empty_cache()

        cores = mapping.cores
        output_sources = mapping.output_sources

        # Occupied sub-rectangle per hw core.  Greedy packing fills
        # contiguous axon/neuron offsets starting at (0, 0); axons beyond
        # ``used_axons`` are off-sources (pure padding added at the end
        # of packing), and neurons beyond ``used_neurons`` are zero
        # columns that no output span references.  Simulating only the
        # occupied slice skips the zero-padded region entirely.
        axon_spans = []
        for c in cores:
            if hasattr(c, "get_axon_source_spans"):
                axon_spans.append(c.get_axon_source_spans())
            else:
                axon_spans.append(compress_spike_sources(c.axon_sources))
        if hasattr(mapping, "get_output_source_spans"):
            output_spans = mapping.get_output_source_spans()
        else:
            output_spans = compress_spike_sources(list(output_sources.flatten()))

        core_params = []
        hw_biases = []
        for core in cores:
            ua = max(int(core.axons_per_core - core.available_axons), 1)
            un = max(int(core.neurons_per_core - core.available_neurons), 1)
            mat = core.core_matrix[:ua, :un]
            core_params.append(
                torch.tensor(mat.T, dtype=_COMPUTE_DTYPE, device=device)
            )
            bias = getattr(core, "hardware_bias", None)
            if bias is None:
                hw_biases.append(None)
            else:
                hw_biases.append(
                    torch.tensor(bias[:un], dtype=_COMPUTE_DTYPE, device=device)
                )
        thresholds = [
            torch.tensor(float(core.threshold), dtype=_COMPUTE_DTYPE, device=device)
            for core in cores
        ]

        cached = dict(
            device=device,
            cores=cores,
            output_sources=output_sources,
            axon_spans=axon_spans,
            output_spans=output_spans,
            core_params=core_params,
            thresholds=thresholds,
            hw_biases=hw_biases,
        )
        self._segment_tensor_cache[key] = cached
        self._segment_tensor_cache_key = key
        return cached

    def _run_neural_segment_rate(
        self,
        stage: HybridStage,
        *,
        input_spike_train: torch.Tensor,
    ) -> torch.Tensor:
        """Rate-coded neural segment execution.  Returns spike counts (B, out_dim)."""
        mapping = stage.hard_core_mapping
        assert mapping is not None

        T = self.simulation_length
        assert input_spike_train.shape[0] == T

        batch_size = input_spike_train.shape[1]
        device = input_spike_train.device
        input_size = input_spike_train.shape[2]

        latency = ChipLatency(mapping).calculate()
        cycles = int(latency) + T

        seg = self._get_segment_tensors(stage, device)
        cores = seg["cores"]
        output_sources = seg["output_sources"]
        axon_spans = seg["axon_spans"]
        output_spans = seg["output_spans"]
        core_params = seg["core_params"]
        thresholds = seg["thresholds"]
        hw_biases = seg["hw_biases"]

        ops = {"<": torch.lt, "<=": torch.le}

        # Buffers / memb / input_signals sized to each core's occupied
        # rectangle (frozen post-packing; read directly from the core).
        buffers = [
            torch.zeros(batch_size, max(int(c.neurons_per_core - c.available_neurons), 1),
                        device=device, dtype=_COMPUTE_DTYPE)
            for c in cores
        ]
        memb = [
            torch.zeros(batch_size, max(int(c.neurons_per_core - c.available_neurons), 1),
                        device=device, dtype=_COMPUTE_DTYPE)
            for c in cores
        ]

        output_counts = torch.zeros(batch_size, len(output_sources), device=device, dtype=_COMPUTE_DTYPE)

        zeros_in = torch.zeros(batch_size, input_size, device=device, dtype=_COMPUTE_DTYPE)
        input_signals = [
            torch.zeros(batch_size, max(int(c.axons_per_core - c.available_axons), 1),
                        device=device, dtype=_COMPUTE_DTYPE)
            for c in cores
        ]

        # Spike train arrives as float32 from spike generators; cast to compute dtype once.
        input_spike_train = input_spike_train.to(_COMPUTE_DTYPE)

        for cycle in range(cycles):
            input_spikes = input_spike_train[cycle] if cycle < T else zeros_in

            for core_idx, core in enumerate(cores):
                self._fill_signal_tensor_from_spans(
                    input_signals[core_idx],
                    input_spikes=input_spikes,
                    buffers=buffers,
                    spans=axon_spans[core_idx],
                    cycle=cycle,
                )

            for core_idx, core in enumerate(cores):
                if core.latency is None:
                    continue
                if not (cycle >= core.latency and cycle < T + core.latency):
                    continue

                memb_i = memb[core_idx]
                memb_i += torch.matmul(core_params[core_idx], input_signals[core_idx].T).T
                # Hardware-bias: add bias every cycle (matches always-on axon semantics).
                if hw_biases[core_idx] is not None:
                    memb_i += hw_biases[core_idx]

                fired = ops[self.thresholding_mode](thresholds[core_idx], memb_i)
                buffers[core_idx] = fired.to(_COMPUTE_DTYPE)

                if self.firing_mode == "Novena":
                    memb_i[fired] = 0.0
                elif self.firing_mode == "Default":
                    memb_i[fired] -= thresholds[core_idx]

            for sp in output_spans:
                d0 = int(sp.dst_start)
                d1 = int(sp.dst_end)
                if sp.kind == "off":
                    continue
                if sp.kind == "on":
                    output_counts[:, d0:d1] += 1.0
                    continue
                if sp.kind == "input":
                    output_counts[:, d0:d1] += input_spikes[:, int(sp.src_start):int(sp.src_end)]
                    continue
                output_counts[:, d0:d1] += buffers[int(sp.src_core)][:, int(sp.src_start):int(sp.src_end)]

        return output_counts

    def _run_neural_segment_ttfs(
        self,
        stage: HybridStage,
        *,
        input_activations: torch.Tensor,
        quantized: bool = False,
    ) -> torch.Tensor:
        """
        TTFS neural segment execution.

        When ``quantized=False`` (default): continuous analytical
        ``clamp(relu(W @ x + b) / θ, 0, 1)`` per core.  Outputs clamped to
        [0, 1] to match hardware TTFS.  Inputs are NOT clamped; weight matrices
        incorporate ``per_input_scales`` normalization for ComputeOp sources.

        When ``quantized=True``: analytical closed-form computation that
        matches the cycle-based simulation exactly::

            V = W @ x
            k_fire = ceil(S * (1 - V / θ))
            if k_fire < S: activation = (S - clamp(k_fire, 0, S-1)) / S
            else:          activation = 0  (neuron never fires)

        Both modes are O(N_cores) — one matmul + element-wise ops per core.
        """
        mapping = stage.hard_core_mapping
        assert mapping is not None

        S = self.simulation_length
        batch_size = input_activations.shape[0]
        device = input_activations.device

        seg = self._get_segment_tensors(stage, device)
        cores = seg["cores"]
        output_sources = seg["output_sources"]
        axon_spans = seg["axon_spans"]
        output_spans = seg["output_spans"]
        core_params = seg["core_params"]
        thresholds = seg["thresholds"]
        hw_biases = seg["hw_biases"]

        # Cast inputs to compute dtype to match C++ double precision.
        input_activations = input_activations.to(_COMPUTE_DTYPE)

        # Buffers / input_signals sized to each core's occupied rectangle
        # (frozen post-packing; read directly from the core).
        buffers = [
            torch.zeros(batch_size, max(int(c.neurons_per_core - c.available_neurons), 1),
                        device=device, dtype=_COMPUTE_DTYPE)
            for c in cores
        ]
        input_signals = [
            torch.zeros(batch_size, max(int(c.axons_per_core - c.available_axons), 1),
                        device=device, dtype=_COMPUTE_DTYPE)
            for c in cores
        ]

        topo_order = sorted(range(len(cores)), key=lambda i: cores[i].latency or 0)
        for ci in topo_order:
            self._fill_signal_tensor_from_spans(
                input_signals[ci],
                input_spikes=input_activations,
                buffers=buffers,
                spans=axon_spans[ci],
                cycle=0,
            )
            V = torch.matmul(core_params[ci], input_signals[ci].T).T
            # Hardware-bias: add dedicated bias register.
            if hw_biases[ci] is not None:
                V = V + hw_biases[ci]

            if quantized:
                safe_thresh = thresholds[ci].clamp(min=1e-12)
                k_fire_raw = torch.ceil(S * (1.0 - V / safe_thresh))
                fires = k_fire_raw < S
                k_fire = k_fire_raw.clamp(0, S - 1)
                buffers[ci] = torch.where(
                    fires, (S - k_fire) / S, torch.zeros_like(k_fire)
                )
            else:
                out = F.relu(V)
                # TTFS: a neuron fires at most once → output rate ∈ [0, 1].
                # Hardware naturally clamps (V > θ fires immediately → rate 1).
                buffers[ci] = (out / thresholds[ci]).clamp(0.0, 1.0)

        output = torch.zeros(batch_size, len(output_sources), device=device, dtype=_COMPUTE_DTYPE)
        for sp in output_spans:
            d0 = int(sp.dst_start)
            d1 = int(sp.dst_end)
            if sp.kind == "off":
                continue
            if sp.kind == "on":
                output[:, d0:d1] = 1.0
                continue
            if sp.kind == "input":
                output[:, d0:d1] = input_activations[:, int(sp.src_start):int(sp.src_end)]
                continue
            output[:, d0:d1] = buffers[int(sp.src_core)][:, int(sp.src_start):int(sp.src_end)]

        return output

    # ---------------------------------------------------------------------
    # Public forward
    # ---------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.preprocessor(x)
        x = x.view(x.shape[0], -1)

        if self.spiking_mode in self._TTFS_SPIKING_MODES:
            return self._forward_ttfs(x)

        return self._forward_rate(x)

    # ---------------------------------------------------------------------
    # TTFS forward (state-buffer driven)
    # ---------------------------------------------------------------------
    def _forward_ttfs(self, x: torch.Tensor) -> torch.Tensor:
        """TTFS forward pass using the global state buffer.

        ComputeOps receive TTFS-normalised inputs ([0, 1]) from the state
        buffer but wrap modules whose bias was never divided by
        activation_scale.  We rescale around execution using
        ``node_activation_scales`` — same fix as SpikingUnifiedCoreFlow.
        """
        T = self.simulation_length
        batch_size = x.shape[0]
        device = x.device
        quantized = self.spiking_mode == "ttfs_quantized"

        # Cast input to compute dtype so the state buffer is uniformly typed.
        x_compute = x.to(_COMPUTE_DTYPE)
        state_buffer: Dict[int, torch.Tensor] = {-2: x_compute}
        out_scales = getattr(self.hybrid_mapping, "node_activation_scales", {})
        in_scales = getattr(self.hybrid_mapping, "node_input_activation_scales", out_scales)

        for stage in self.hybrid_mapping.stages:
            if stage.kind == "neural":
                seg_input = self._assemble_segment_input(
                    stage.input_map, state_buffer, batch_size, device
                )
                seg_output = self._run_neural_segment_ttfs(
                    stage, input_activations=seg_input, quantized=quantized
                )
                self._store_segment_output(stage.output_map, state_buffer, seg_output)

            elif stage.kind == "compute":
                op = stage.compute_op
                assert op is not None
                in_scale = in_scales.get(op.id, 1.0)
                out_scale = out_scales.get(op.id, 1.0)
                # Host-side compute ops execute a Python torch module.  C++
                # SimulationRunner._execute_compute_op_np runs them in float32;
                # match that here so PyTorch and C++ agree bit-for-bit.
                #
                # Memory: never materialise a float32 copy of the entire
                # state_buffer — that duplicated every prior segment's
                # output per compute-op (2× state memory for each of ~2430
                # ops on cifar_vit → OOM).  ``gather_inputs`` only reads
                # the specific sources this op needs, so casting the
                # gathered result (O(op_inputs)) is sufficient.
                gathered = op.gather_inputs(x, state_buffer)
                gathered = gathered.to(torch.float32)
                if abs(in_scale - 1.0) > 1e-9:
                    gathered = gathered * in_scale
                result = op.execute_on_gathered(gathered)
                if abs(out_scale - 1.0) > 1e-9:
                    result = result / out_scale
                state_buffer[op.id] = result.to(_COMPUTE_DTYPE)

            else:
                raise ValueError(f"Invalid hybrid stage kind: {stage.kind}")

        final = self._gather_final_output(state_buffer, x_compute, batch_size, device)
        return final.to(torch.float32) * float(T)

    # ---------------------------------------------------------------------
    # Rate-coded forward (state-buffer driven)
    # ---------------------------------------------------------------------
    def _forward_rate(self, x: torch.Tensor) -> torch.Tensor:
        """Rate-coded forward pass using the global state buffer."""
        batch_size = x.shape[0]
        device = x.device
        T = self.simulation_length

        x_compute = x.to(_COMPUTE_DTYPE)
        state_buffer: Dict[int, torch.Tensor] = {-2: x_compute}

        for stage in self.hybrid_mapping.stages:
            if stage.kind == "neural":
                seg_input_rates = self._assemble_segment_input(
                    stage.input_map, state_buffer, batch_size, device
                )
                seg_input_rates_clamped = seg_input_rates.clamp(0.0, 1.0)
                spike_train = torch.zeros(
                    T, batch_size, seg_input_rates_clamped.shape[1], device=device,
                    dtype=_COMPUTE_DTYPE,
                )
                for cycle in range(T):
                    spike_train[cycle] = self.to_spikes(seg_input_rates_clamped, cycle).to(_COMPUTE_DTYPE)

                counts = self._run_neural_segment_rate(
                    stage, input_spike_train=spike_train
                )
                seg_output_rates = counts / float(T)
                self._store_segment_output(stage.output_map, state_buffer, seg_output_rates)

            elif stage.kind == "compute":
                op = stage.compute_op
                assert op is not None
                # Match C++ path: compute ops run in float32.  Avoid a
                # whole-state_buffer float32 copy (see _forward_ttfs note)
                # — gather first, cast only what's consumed.
                gathered = op.gather_inputs(x, state_buffer)
                gathered = gathered.to(torch.float32)
                result = op.execute_on_gathered(gathered)
                state_buffer[op.id] = result.to(_COMPUTE_DTYPE)

            else:
                raise ValueError(f"Invalid hybrid stage kind: {stage.kind}")

        final = self._gather_final_output(state_buffer, x_compute, batch_size, device)
        return final.to(torch.float32) * float(T)
