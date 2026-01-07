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

        assert firing_mode in ["Default", "Novena"]
        assert spike_mode in ["Stochastic", "Deterministic", "FrontLoaded", "Uniform"]
        assert thresholding_mode in ["<", "<="]

    # ---------------------------------------------------------------------
    # Spike generation (match SpikingCoreFlow / SpikingUnifiedCoreFlow)
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
    # Segment execution
    # ---------------------------------------------------------------------
    def _get_signal_tensor(
        self,
        *,
        input_spikes: torch.Tensor,
        buffers: list[torch.Tensor],
        sources,
    ) -> torch.Tensor:
        """
        sources: list[SpikeSource] or np.ndarray[SpikeSource] (from HardCoreMapping)
        returns: (B, len(sources)) tensor of spikes for this cycle.
        """
        batch_size = input_spikes.shape[0]
        device = input_spikes.device
        out = torch.empty(batch_size, len(sources), device=device)
        for idx, src in enumerate(sources):
            if src.is_input_:
                out[:, idx] = input_spikes[:, src.neuron_]
            elif src.is_off_:
                out[:, idx] = 0.0
            elif src.is_always_on_:
                out[:, idx] = 1.0
            else:
                out[:, idx] = buffers[src.core_][:, src.neuron_]
        return out

    def _fill_signal_tensor_from_spans(
        self,
        out: torch.Tensor,
        *,
        input_spikes: torch.Tensor,
        buffers: list[torch.Tensor],
        spans: list[SpikeSourceSpan],
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

    def _run_neural_segment(
        self,
        stage: HybridStage,
        *,
        input_spike_train: torch.Tensor,  # (T, B, in)
    ) -> torch.Tensor:
        mapping = stage.hard_core_mapping
        assert mapping is not None

        T = self.simulation_length
        assert input_spike_train.shape[0] == T

        batch_size = input_spike_train.shape[1]
        device = input_spike_train.device
        input_size = input_spike_train.shape[2]

        # Compute/update per-core latency for this segment.
        latency = ChipLatency(mapping).calculate()
        cycles = int(latency) + T

        cores = mapping.cores
        output_sources = mapping.output_sources

        # Precompute span views (per core + outputs) for this mapping (cheap cache on objects)
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

        # Pre-pack core weights and thresholds.
        core_params = [
            torch.tensor(core.core_matrix.T, dtype=torch.float32, device=device)
            for core in cores
        ]  # (neurons, axons)
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

            # Gather all core inputs from previous-cycle buffers (sync update).
            for core_idx, core in enumerate(cores):
                self._fill_signal_tensor_from_spans(
                    input_signals[core_idx],
                    input_spikes=input_spikes,
                    buffers=buffers,
                    spans=axon_spans[core_idx],
                )

            # Update all cores.
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

            # Accumulate stage outputs (spike counts).
            # Range-based output gather (adds into output_counts)
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
        T = self.simulation_length
        batch_size, n = rates.shape
        out_train = torch.zeros(T, batch_size, n, device=rates.device)
        for cycle in range(T):
            out_train[cycle] = self.to_spikes(rates, cycle)
        return out_train

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.preprocessor(x)
        x = x.view(x.shape[0], -1)

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


