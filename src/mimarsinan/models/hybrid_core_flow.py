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
    no time-step discretisation.  Equivalent to ``ReLU(W @ x + b) / θ``.
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
        preprocessor: nn.Module,
        firing_mode: str = "Default",
        spike_mode: str = "Uniform",
        thresholding_mode: str = "<",
        spiking_mode: str = "rate",
    ):
        super().__init__()

        self.input_shape = input_shape
        self.hybrid_mapping = hybrid_mapping
        self.simulation_length = int(simulation_length)
        self.preprocessor = preprocessor

        self.firing_mode = firing_mode
        self.spike_mode = spike_mode
        self.thresholding_mode = thresholding_mode
        self.spiking_mode = spiking_mode

        assert firing_mode in ["Default", "Novena", "TTFS"]
        assert spike_mode in ["Stochastic", "Deterministic", "FrontLoaded", "Uniform", "TTFS"]
        assert thresholding_mode in ["<", "<="]

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
        inp = torch.zeros(batch_size, total_size, device=device)
        for s in input_map:
            buf = state_buffer[s.node_id]
            inp[:, s.offset : s.offset + s.size] = buf[:, :s.size]
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
        out = torch.zeros(batch_size, len(output_sources), device=device)
        for idx, src in enumerate(output_sources):
            if not isinstance(src, IRSource):
                continue
            if src.is_off():
                continue
            elif src.is_input():
                out[:, idx] = original_input[:, src.index]
            elif src.is_always_on():
                out[:, idx] = 1.0
            else:
                out[:, idx] = state_buffer[src.node_id][:, src.index]
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

        cores = mapping.cores
        output_sources = mapping.output_sources

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

        core_params = [
            torch.tensor(core.core_matrix.T, dtype=torch.float32, device=device)
            for core in cores
        ]
        thresholds = [
            torch.tensor(float(core.threshold), dtype=torch.float32, device=device)
            for core in cores
        ]

        ops = {"<": torch.lt, "<=": torch.le}

        buffers = [torch.zeros(batch_size, core.get_output_count(), device=device) for core in cores]
        memb = [torch.zeros(batch_size, core.get_output_count(), device=device) for core in cores]

        output_counts = torch.zeros(batch_size, len(output_sources), device=device)

        zeros_in = torch.zeros(batch_size, input_size, device=device)
        input_signals = [
            torch.zeros(batch_size, core.get_input_count(), device=device) for core in cores
        ]

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

                fired = ops[self.thresholding_mode](thresholds[core_idx], memb_i)
                buffers[core_idx] = fired.float()

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
        ``relu(W @ x + b) / θ`` per core.

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

        cores = mapping.cores
        output_sources = mapping.output_sources

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

        core_params = [
            torch.tensor(core.core_matrix.T, dtype=torch.float32, device=device)
            for core in cores
        ]
        thresholds = [
            torch.tensor(float(core.threshold), dtype=torch.float32, device=device)
            for core in cores
        ]

        buffers = [
            torch.zeros(batch_size, core.get_output_count(), device=device)
            for core in cores
        ]
        input_signals = [
            torch.zeros(batch_size, core.get_input_count(), device=device)
            for core in cores
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
                buffers[ci] = out / thresholds[ci]

        output = torch.zeros(batch_size, len(output_sources), device=device)
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
        """TTFS forward pass using the global state buffer."""
        T = self.simulation_length
        batch_size = x.shape[0]
        device = x.device
        quantized = self.spiking_mode == "ttfs_quantized"

        state_buffer: Dict[int, torch.Tensor] = {-2: x}

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
                result = op.execute(x, state_buffer)
                state_buffer[op.id] = result

            else:
                raise ValueError(f"Invalid hybrid stage kind: {stage.kind}")

        final = self._gather_final_output(state_buffer, x, batch_size, device)
        return final * float(T)

    # ---------------------------------------------------------------------
    # Rate-coded forward (state-buffer driven)
    # ---------------------------------------------------------------------
    def _forward_rate(self, x: torch.Tensor) -> torch.Tensor:
        """Rate-coded forward pass using the global state buffer."""
        batch_size = x.shape[0]
        device = x.device
        T = self.simulation_length

        state_buffer: Dict[int, torch.Tensor] = {-2: x}

        for stage in self.hybrid_mapping.stages:
            if stage.kind == "neural":
                seg_input_rates = self._assemble_segment_input(
                    stage.input_map, state_buffer, batch_size, device
                )
                seg_input_rates_clamped = seg_input_rates.clamp(0.0, 1.0)
                spike_train = torch.zeros(
                    T, batch_size, seg_input_rates_clamped.shape[1], device=device
                )
                for cycle in range(T):
                    spike_train[cycle] = self.to_spikes(seg_input_rates_clamped, cycle)

                counts = self._run_neural_segment_rate(
                    stage, input_spike_train=spike_train
                )
                seg_output_rates = counts / float(T)
                self._store_segment_output(stage.output_map, state_buffer, seg_output_rates)

            elif stage.kind == "compute":
                op = stage.compute_op
                assert op is not None
                result = op.execute(x, state_buffer)
                state_buffer[op.id] = result

            else:
                raise ValueError(f"Invalid hybrid stage kind: {stage.kind}")

        final = self._gather_final_output(state_buffer, x, batch_size, device)
        return final * float(T)
