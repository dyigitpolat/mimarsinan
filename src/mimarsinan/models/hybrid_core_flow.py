"""
SpikingHybridCoreFlow: Spiking simulation for HybridHardCoreMapping.

Supports both rate-coded (Default/Novena) and Time-to-First-Spike (TTFS)
firing modes.

TTFS mode implements the ReLU-equivalent single-spike coding from:
  Stanojevic et al., "High-performance deep spiking neural networks with
  0.3 spikes per neuron", Nature Communications 15, 6793 (2024).
  https://www.nature.com/articles/s41467-024-51110-5

In TTFS mode:
  - Each neuron fires at most once (fire-once semantics, no membrane reset).
  - Input activations are encoded as spike times: higher activation → earlier spike.
  - Output is decoded as (T - spike_time), giving ReLU-equivalent activations.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from mimarsinan.mapping.chip_latency import ChipLatency
from mimarsinan.mapping.hybrid_hardcore_mapping import HybridHardCoreMapping, HybridStage
from mimarsinan.mapping.ir import ComputeOp

from mimarsinan.mapping.spike_source_spans import SpikeSourceSpan, compress_spike_sources


class SpikingHybridCoreFlow(nn.Module):
    """
    Execute a HybridHardCoreMapping:

    neural segment (HardCoreMapping) -> ComputeOp sync barrier -> neural -> ...

    Semantics:
    - Neural segment runs on-chip (spiking), producing spike counts for its stage outputs.
    - ComputeOp is a sync barrier: spike counts -> rates, apply op, respike to a spike train.
    - Next neural segment consumes the respiked spike train as its input buffer.

    When firing_mode == "TTFS":
    - Each neuron fires at most once (fire-once, no membrane reset).
    - Inputs are encoded as single spikes at time T*(1-activation).
    - Outputs are decoded as (T - spike_time) for ReLU equivalence.
    """

    def __init__(
        self,
        input_shape,
        hybrid_mapping: HybridHardCoreMapping,
        simulation_length: int,
        preprocessor: nn.Module,
        firing_mode: str = "Default",
        spike_mode: str = "Uniform",
        thresholding_mode: str = "<",
    ):
        super().__init__()

        self.input_shape = input_shape
        self.hybrid_mapping = hybrid_mapping
        self.simulation_length = int(simulation_length)
        self.preprocessor = preprocessor

        self.firing_mode = firing_mode
        self.spike_mode = spike_mode
        self.thresholding_mode = thresholding_mode

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
    # TTFS spike generation: single spike per neuron
    # ---------------------------------------------------------------------
    def _ttfs_encode_input(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Encode activations [0,1] as a single-spike train (T, B, N).

        Higher activation → earlier spike time.
        spike_time = round(T * (1 - activation)).
        activation=1 → spike at t=0, activation=0 → spike at t=T (i.e. never within [0,T-1]).
        """
        T = self.simulation_length
        # spike_time ∈ [0, T]: 0 = immediate, T = never fires
        spike_times = torch.round(T * (1.0 - activations.clamp(0.0, 1.0))).long()
        spike_train = torch.zeros(T, *activations.shape, device=activations.device)
        for cycle in range(T):
            spike_train[cycle] = (spike_times == cycle).float()
        return spike_train

    def _ttfs_respike_rates(self, rates: torch.Tensor) -> torch.Tensor:
        """
        Re-encode rate-space activations [0,1] into TTFS single-spike trains.

        Used at ComputeOp sync barriers when in TTFS mode.
        """
        return self._ttfs_encode_input(rates)

    # ---------------------------------------------------------------------
    # Segment execution
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
                # TTFS: always-on (bias) source fires only once at cycle 0.
                if self.firing_mode == "TTFS" and cycle != 0:
                    continue
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
        input_spike_train: torch.Tensor,  # (T, B, in)
    ) -> torch.Tensor:
        """
        Rate-coded neural segment execution (Default / Novena firing modes).

        Returns spike counts (B, out_dim).
        """
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

    def _fill_activation_from_spike_source_spans(
        self,
        out: torch.Tensor,
        *,
        input_activations: torch.Tensor,
        core_activations: list[torch.Tensor],
        spans: list[SpikeSourceSpan],
    ) -> None:
        """
        Fill `out` (B, N) with *activations* from spike-source spans.

        Used by the analytical TTFS forward pass. Always-on → 1.0 (bias).
        """
        out.zero_()
        for sp in spans:
            d0 = int(sp.dst_start)
            d1 = int(sp.dst_end)
            if sp.kind == "off":
                continue
            if sp.kind == "on":
                out[:, d0:d1] = 1.0  # bias
                continue
            if sp.kind == "input":
                out[:, d0:d1] = input_activations[:, int(sp.src_start):int(sp.src_end)]
                continue
            out[:, d0:d1] = core_activations[int(sp.src_core)][:, int(sp.src_start):int(sp.src_end)]

    def _run_neural_segment_ttfs(
        self,
        stage: HybridStage,
        *,
        input_activations: torch.Tensor,  # (B, in) — continuous activations
    ) -> torch.Tensor:
        """
        TTFS neural segment: analytical ReLU-equivalent computation.

        The IBM TTFS model (Stanojevic et al., 2024) computes spike times
        analytically as a linear function of input spike times, NOT via
        integrate-and-fire membrane dynamics. This is equivalent to:

            output = relu(W_quantized @ input + bias) / threshold

        where threshold = parameter_scale restores the original float scale.

        Returns: (B, out_dim) tensor of activations (not spike counts).
        """
        mapping = stage.hard_core_mapping
        assert mapping is not None

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

        # Process cores in topological order (by latency).
        # Cores with lower latency are upstream of cores with higher latency.
        core_activations = [
            torch.zeros(batch_size, core.get_output_count(), device=device) for core in cores
        ]
        input_signals = [
            torch.zeros(batch_size, core.get_input_count(), device=device) for core in cores
        ]

        # Sort cores by latency for correct ordering
        ordered = sorted(range(len(cores)), key=lambda i: cores[i].latency or 0)

        for core_idx in ordered:
            core = cores[core_idx]

            self._fill_activation_from_spike_source_spans(
                input_signals[core_idx],
                input_activations=input_activations,
                core_activations=core_activations,
                spans=axon_spans[core_idx],
            )

            # Analytical forward: matmul + ReLU + normalize by threshold
            out = torch.matmul(core_params[core_idx], input_signals[core_idx].T).T
            out = F.relu(out)
            out = out / thresholds[core_idx]

            core_activations[core_idx] = out

        # Gather output activations
        output = torch.zeros(batch_size, len(output_sources), device=device)
        self._fill_activation_from_spike_source_spans(
            output,
            input_activations=input_activations,
            core_activations=core_activations,
            spans=output_spans,
        )

        return output

    def _run_neural_segment(
        self,
        stage: HybridStage,
        *,
        input_spike_train: torch.Tensor,
    ) -> torch.Tensor:
        """Dispatch to rate-coded segment execution."""
        return self._run_neural_segment_rate(stage, input_spike_train=input_spike_train)

    # ---------------------------------------------------------------------
    # ComputeOp execution (rate space) + respike
    # ---------------------------------------------------------------------
    def _execute_compute_op_rates(self, op: ComputeOp, in_rates: torch.Tensor) -> torch.Tensor:
        """
        in_rates: (B, N) in [0,1]
        returns: (B, M) in [0,1]
        """
        batch_size = in_rates.shape[0]

        if op.input_shape is not None:
            x = in_rates.view(batch_size, *op.input_shape)
        else:
            x = in_rates

        if op.op_type == "max_pool2d":
            y = F.max_pool2d(
                x,
                kernel_size=op.params.get("kernel_size", 2),
                stride=op.params.get("stride", None),
                padding=op.params.get("padding", 0),
            )
        elif op.op_type == "avg_pool2d":
            y = F.avg_pool2d(
                x,
                kernel_size=op.params.get("kernel_size", 2),
                stride=op.params.get("stride", None),
                padding=op.params.get("padding", 0),
            )
        elif op.op_type == "adaptive_avg_pool2d":
            y = F.adaptive_avg_pool2d(x, op.params.get("output_size", (1, 1)))
        elif op.op_type == "flatten":
            y = x
        elif op.op_type == "identity":
            y = x
        else:
            raise NotImplementedError(f"ComputeOp '{op.op_type}' not implemented in SpikingHybridCoreFlow")

        return y.view(batch_size, -1).clamp(0.0, 1.0)

    def _respike_rates(self, rates: torch.Tensor) -> torch.Tensor:
        """
        rates: (B, N) in [0,1]
        returns: spike_train (T, B, N)
        """
        if self.firing_mode == "TTFS":
            return self._ttfs_respike_rates(rates)
        T = self.simulation_length
        batch_size, n = rates.shape
        out_train = torch.zeros(T, batch_size, n, device=rates.device)
        for cycle in range(T):
            out_train[cycle] = self.to_spikes(rates, cycle)
        return out_train

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.preprocessor(x)
        x = x.view(x.shape[0], -1)

        # TTFS: analytical forward pass (no spike trains needed)
        if self.firing_mode == "TTFS":
            return self._forward_ttfs_analytical(x)

        batch_size = x.shape[0]
        device = x.device
        T = self.simulation_length

        # Generate initial input spike train (T, B, in).
        input_spike_train = torch.zeros(T, batch_size, x.shape[1], device=device)
        for cycle in range(T):
            input_spike_train[cycle] = self.to_spikes(x, cycle)

        current_spike_train: torch.Tensor | None = input_spike_train
        current_rates: torch.Tensor | None = None

        for stage in self.hybrid_mapping.stages:
            if stage.kind == "neural":
                assert current_spike_train is not None, "Neural stage requires a spike train input."
                counts = self._run_neural_segment(stage, input_spike_train=current_spike_train)
                current_rates = counts / float(T)
                current_spike_train = None
                continue

            if stage.kind == "compute":
                op = stage.compute_op
                assert op is not None

                if current_rates is None:
                    assert current_spike_train is not None
                    current_rates = current_spike_train.float().mean(dim=0)

                y_rates = self._execute_compute_op_rates(op, current_rates)
                current_spike_train = self._respike_rates(y_rates)
                current_rates = None
                continue

            raise ValueError(f"Invalid hybrid stage kind: {stage.kind}")

        # Final output: return spike counts (B, out_dim), matching other spiking flows.
        if current_rates is not None:
            return current_rates * float(T)
        assert current_spike_train is not None
        return current_spike_train.sum(dim=0)

    def _forward_ttfs_analytical(self, x: torch.Tensor) -> torch.Tensor:
        """
        TTFS analytical forward pass: no spike trains, no cycles.

        Processes each stage with continuous activations:
        - Neural segment: relu(W @ input) / threshold per core
        - Compute op: pooling / identity on activations

        Returns: (B, out_dim) activations (ReLU-equivalent).
        """
        current_activations = x  # (B, in)

        for stage in self.hybrid_mapping.stages:
            if stage.kind == "neural":
                current_activations = self._run_neural_segment_ttfs(
                    stage, input_activations=current_activations
                )
                continue

            if stage.kind == "compute":
                op = stage.compute_op
                assert op is not None
                current_activations = self._execute_compute_op_rates(op, current_activations)
                continue

            raise ValueError(f"Invalid hybrid stage kind: {stage.kind}")

        return current_activations
