"""
UnifiedCoreFlow: Simulation that supports both neural cores and compute ops.

This module extends CoreFlow to handle IRGraph with mixed NeuralCore and ComputeOp nodes,
enabling simulation of networks that include non-neural operations like pooling.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from mimarsinan.mapping.ir import ComputeOp, IRGraph, IRNode, IRSource, NeuralCore
from mimarsinan.models.layers import (
    ClampDecorator,
    QuantizeDecorator,
    TransformedActivation,
)


class UnifiedCoreFlow(nn.Module):
    """
    Execute an IRGraph that may contain both NeuralCore and ComputeOp nodes.

    This is a generalization of CoreFlow that can simulate:
    1. Crossbar-based neural cores (NeuralCore)
    2. Non-neural compute operations (ComputeOp) like pooling

    Execution proceeds in topological order (nodes are assumed pre-sorted).
    """

    def __init__(
        self,
        input_shape,
        ir_graph: IRGraph,
        Tq: int,
        preprocessor: nn.Module,
    ):
        super().__init__()

        self.input_shape = input_shape
        self.ir_graph = ir_graph
        self.nodes = ir_graph.nodes
        self.output_sources = ir_graph.output_sources
        self.Tq = Tq
        self.preprocessor = preprocessor

        # Register neural core parameters
        self.neural_core_params = nn.ParameterList()
        self.neural_core_ids = []  # Map param index -> node id
        self.activations = nn.ModuleList()

        for node in self.nodes:
            if isinstance(node, NeuralCore):
                # Store transposed weight matrix as a parameter
                weight = torch.tensor(
                    node.core_matrix.T,
                    dtype=torch.float32,
                )
                self.neural_core_params.append(nn.Parameter(weight, requires_grad=False))
                self.neural_core_ids.append(node.id)

                # Build activation for this core
                activation = TransformedActivation(
                    nn.ReLU(),
                    [
                        ClampDecorator(torch.tensor(0.0), node.activation_scale),
                        QuantizeDecorator(torch.tensor(Tq), node.activation_scale),
                    ],
                )
                self.activations.append(activation)

        # Build ID -> param index map
        self._id_to_param_idx = {
            nid: idx for idx, nid in enumerate(self.neural_core_ids)
        }

        # Compute cycles (simplified: 1 cycle per layer depth)
        # In practice, you'd compute latency based on chip characteristics.
        self.cycles = self._compute_cycles()

    def _compute_cycles(self) -> int:
        """Compute the number of simulation cycles needed."""
        # Simple heuristic: max depth of the graph + 1
        # A proper implementation would use ChipLatency.
        return len(self.nodes) + 1

    def _gather_signal(
        self,
        source: IRSource,
        input_flat: torch.Tensor,
        buffers: Dict[int, torch.Tensor],
    ) -> torch.Tensor:
        """Gather a single signal from a source."""
        batch_size = input_flat.shape[0]

        if source.is_off():
            return torch.zeros(batch_size, device=input_flat.device)
        elif source.is_input():
            return input_flat[:, source.index]
        elif source.is_always_on():
            return torch.ones(batch_size, device=input_flat.device)
        else:
            return buffers[source.node_id][:, source.index]

    def _gather_signals(
        self,
        sources: List[IRSource],
        input_flat: torch.Tensor,
        buffers: Dict[int, torch.Tensor],
    ) -> torch.Tensor:
        """Gather multiple signals into a tensor."""
        batch_size = input_flat.shape[0]
        result = torch.zeros(batch_size, len(sources), device=input_flat.device)

        for idx, src in enumerate(sources):
            result[:, idx] = self._gather_signal(src, input_flat, buffers)

        return result

    def _execute_neural_core(
        self,
        node: NeuralCore,
        input_flat: torch.Tensor,
        buffers: Dict[int, torch.Tensor],
    ) -> torch.Tensor:
        """Execute a neural core (crossbar computation)."""
        # Gather inputs
        sources = list(node.input_sources.flatten())
        inputs = self._gather_signals(sources, input_flat, buffers)

        # Get weight matrix
        param_idx = self._id_to_param_idx[node.id]
        weight = self.neural_core_params[param_idx]

        # Compute: weight @ inputs.T -> (neurons, batch) -> transpose -> (batch, neurons)
        out = torch.matmul(weight, inputs.T).T

        # Apply activation
        out = self.activations[param_idx](out)

        return out

    def _execute_compute_op(
        self,
        node: ComputeOp,
        input_flat: torch.Tensor,
        buffers: Dict[int, torch.Tensor],
    ) -> torch.Tensor:
        """Execute a compute operation."""
        # Gather inputs
        sources = list(node.input_sources.flatten())
        inputs = self._gather_signals(sources, input_flat, buffers)

        # Reshape if spatial operation
        if node.input_shape is not None and len(node.input_shape) >= 2:
            x = inputs.view(inputs.shape[0], *node.input_shape)
        else:
            x = inputs

        # Execute the operation
        if node.op_type == "max_pool2d":
            y = F.max_pool2d(
                x,
                kernel_size=node.params.get("kernel_size", 2),
                stride=node.params.get("stride", None),
                padding=node.params.get("padding", 0),
            )
        elif node.op_type == "avg_pool2d":
            y = F.avg_pool2d(
                x,
                kernel_size=node.params.get("kernel_size", 2),
                stride=node.params.get("stride", None),
                padding=node.params.get("padding", 0),
            )
        elif node.op_type == "adaptive_avg_pool2d":
            y = F.adaptive_avg_pool2d(x, node.params.get("output_size", (1, 1)))
        elif node.op_type == "flatten":
            y = x
        elif node.op_type == "identity":
            y = x
        else:
            raise NotImplementedError(f"ComputeOp type '{node.op_type}' not implemented")

        # Flatten output to (B, N)
        return y.view(y.shape[0], -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Execute the IR graph.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        x = self.preprocessor(x)
        x = x.view(x.shape[0], -1)  # Flatten to (B, input_size)

        # Buffer to store each node's output
        buffers: Dict[int, torch.Tensor] = {}

        # Execute nodes in order
        for node in self.nodes:
            if isinstance(node, NeuralCore):
                buffers[node.id] = self._execute_neural_core(node, x, buffers)
            elif isinstance(node, ComputeOp):
                buffers[node.id] = self._execute_compute_op(node, x, buffers)
            else:
                raise TypeError(f"Unknown node type: {type(node)}")

        # Gather outputs
        output_sources = list(self.output_sources.flatten())
        output = self._gather_signals(output_sources, x, buffers)

        return output


class SpikingUnifiedCoreFlow(nn.Module):
    """
    Spiking version of UnifiedCoreFlow.

    This handles spike-based simulation with membrane potential dynamics.
    Non-neural operations (ComputeOp) act as synchronization points where
    spike counts are converted to rates, the operation is applied, and
    rates are converted back to spikes.
    """

    def __init__(
        self,
        input_shape,
        ir_graph: IRGraph,
        simulation_length: int,
        preprocessor: nn.Module,
        firing_mode: str = "Default",
        spike_mode: str = "Uniform",
        thresholding_mode: str = "<",
    ):
        super().__init__()

        self.input_shape = input_shape
        self.ir_graph = ir_graph
        self.nodes = ir_graph.nodes
        self.output_sources = ir_graph.output_sources
        self.simulation_length = simulation_length
        self.preprocessor = preprocessor
        self.firing_mode = firing_mode
        self.spike_mode = spike_mode
        self.thresholding_mode = thresholding_mode

        assert firing_mode in ["Default", "Novena"]
        assert spike_mode in ["Stochastic", "Deterministic", "FrontLoaded", "Uniform"]
        assert thresholding_mode in ["<", "<="]

        # Register neural core parameters
        self.neural_core_params = nn.ParameterList()
        self.thresholds = nn.ParameterList()
        self.neural_core_ids = []

        for node in self.nodes:
            if isinstance(node, NeuralCore):
                weight = torch.tensor(node.core_matrix.T, dtype=torch.float32)
                self.neural_core_params.append(nn.Parameter(weight, requires_grad=False))
                self.thresholds.append(
                    nn.Parameter(torch.tensor(node.threshold, dtype=torch.float32), requires_grad=False)
                )
                self.neural_core_ids.append(node.id)

        self._id_to_param_idx = {nid: idx for idx, nid in enumerate(self.neural_core_ids)}

        # Identify synchronization points (ComputeOps that break the spiking flow)
        self._sync_points = [i for i, n in enumerate(self.nodes) if isinstance(n, ComputeOp)]

    # ---------------------------------------------------------------------
    # Spike generation (must match SpikingCoreFlow semantics)
    # ---------------------------------------------------------------------
    def to_stochastic_spikes(self, tensor: torch.Tensor) -> torch.Tensor:
        return (torch.rand(tensor.shape, device=tensor.device) < tensor).float()

    def to_front_loaded_spikes(self, tensor: torch.Tensor, cycle: int) -> torch.Tensor:
        return (torch.round(tensor * self.simulation_length) > cycle).float()

    def to_deterministic_spikes(self, tensor: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        return (tensor > threshold).float()

    def to_uniform_spikes(self, tensor: torch.Tensor, cycle: int) -> torch.Tensor:
        T = self.simulation_length

        # Compute N for all elements in the tensor at once
        N = torch.round(tensor * T).to(torch.long)

        # Create a mask for edge cases
        mask = (N != 0) & (N != T) & (cycle < T)

        # Avoid divide-by-zero by clamping N
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
    # IRSource helpers
    # ---------------------------------------------------------------------
    def _get_spike_train_for_source(
        self,
        spike_train_cache: Dict[int, torch.Tensor],
        input_spike_train: torch.Tensor,
        batch_size: int,
        device: torch.device,
        src: IRSource,
        cycle: int,
    ) -> torch.Tensor:
        """
        Return spike vector (B,) for the given source at the given cycle.
        """
        if src.is_input():
            return input_spike_train[cycle][:, src.index]
        if src.is_off():
            return torch.zeros(batch_size, device=device)
        if src.is_always_on():
            return torch.ones(batch_size, device=device)
        return spike_train_cache[src.node_id][cycle][:, src.index]

    def _get_signal_tensor(
        self,
        spike_train_cache: Dict[int, torch.Tensor],
        input_spike_train: torch.Tensor,
        batch_size: int,
        device: torch.device,
        sources: torch.Tensor | list[IRSource],
        cycle: int,
    ) -> torch.Tensor:
        """
        Return (B, len(sources)) spikes for the given cycle.
        """
        if not isinstance(sources, list):
            sources = list(sources)
        signal_tensor = torch.empty(batch_size, len(sources), device=device)
        for idx, src in enumerate(sources):
            signal_tensor[:, idx] = self._get_spike_train_for_source(
                spike_train_cache, input_spike_train, batch_size, device, src, cycle
            )
        return signal_tensor

    def _rates_for_sources(
        self,
        spike_train_cache: Dict[int, torch.Tensor],
        input_spike_train: torch.Tensor,
        batch_size: int,
        device: torch.device,
        sources: list[IRSource],
    ) -> torch.Tensor:
        """
        Convert a list of sources into a (B, len(sources)) rate tensor in [0,1]
        by averaging spikes over the full simulation window (sync barrier).
        """
        T = self.simulation_length
        rates = torch.empty(batch_size, len(sources), device=device)
        for idx, src in enumerate(sources):
            if src.is_input():
                rates[:, idx] = input_spike_train[:, :, src.index].float().mean(dim=0)
            elif src.is_off():
                rates[:, idx] = 0.0
            elif src.is_always_on():
                rates[:, idx] = 1.0
            else:
                rates[:, idx] = spike_train_cache[src.node_id][:, :, src.index].float().mean(dim=0)
        return rates

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Execute spiking simulation over a unified IRGraph (NeuralCore + ComputeOp).

        Key invariant:
        - NeuralCore produces a spike *train* (T, B, out) using LIF integration.
        - ComputeOp is a *sync barrier*: it consumes upstream spike trains, converts
          to rates, applies the op in rate space, then regenerates a new spike train
          for downstream nodes using the same spike generation mode as inputs.
        """
        x = self.preprocessor(x)
        x = x.view(x.shape[0], -1)

        batch_size = x.shape[0]
        device = x.device

        T = self.simulation_length

        # Generate input spike train (T, B, in)
        input_spike_train = torch.zeros(T, batch_size, x.shape[1], device=device)
        for cycle in range(T):
            input_spike_train[cycle] = self.to_spikes(x, cycle)

        # Compute spike trains for all nodes in topological order.
        # spike_train_cache[node_id] = (T, B, out_dim)
        spike_train_cache: Dict[int, torch.Tensor] = {}

        ops = {"<": torch.lt, "<=": torch.le}

        for node in self.nodes:
            if isinstance(node, NeuralCore):
                param_idx = self._id_to_param_idx[node.id]
                weight = self.neural_core_params[param_idx]  # (neurons, axons)
                threshold = self.thresholds[param_idx]

                sources = list(node.input_sources.flatten())
                out_dim = int(node.core_matrix.shape[1])

                memb = torch.zeros(batch_size, out_dim, device=device)
                out_train = torch.zeros(T, batch_size, out_dim, device=device)

                for cycle in range(T):
                    inp = self._get_signal_tensor(
                        spike_train_cache, input_spike_train, batch_size, device, sources, cycle
                    )  # (B, axons)

                    memb += torch.matmul(weight, inp.T).T
                    fired = ops[self.thresholding_mode](threshold, memb)
                    out_train[cycle] = fired.float()

                    if self.firing_mode == "Novena":
                        memb[fired] = 0.0
                    elif self.firing_mode == "Default":
                        memb[fired] -= threshold

                spike_train_cache[node.id] = out_train

            elif isinstance(node, ComputeOp):
                # SYNC BARRIER: convert upstream spikes -> rates, apply op, respike.
                sources = list(node.input_sources.flatten())
                in_rates = self._rates_for_sources(
                    spike_train_cache, input_spike_train, batch_size, device, sources
                )  # (B, N)

                if node.input_shape is not None:
                    x_rates = in_rates.view(batch_size, *node.input_shape)
                else:
                    x_rates = in_rates

                if node.op_type == "max_pool2d":
                    y_rates = F.max_pool2d(
                        x_rates,
                        kernel_size=node.params.get("kernel_size", 2),
                        stride=node.params.get("stride", None),
                        padding=node.params.get("padding", 0),
                    )
                elif node.op_type == "avg_pool2d":
                    y_rates = F.avg_pool2d(
                        x_rates,
                        kernel_size=node.params.get("kernel_size", 2),
                        stride=node.params.get("stride", None),
                        padding=node.params.get("padding", 0),
                    )
                elif node.op_type == "adaptive_avg_pool2d":
                    y_rates = F.adaptive_avg_pool2d(x_rates, node.params.get("output_size", (1, 1)))
                elif node.op_type == "flatten":
                    y_rates = x_rates
                elif node.op_type == "identity":
                    y_rates = x_rates
                else:
                    raise NotImplementedError(
                        f"ComputeOp '{node.op_type}' not implemented in SpikingUnifiedCoreFlow"
                    )

                y_rates = y_rates.view(batch_size, -1).clamp(0.0, 1.0)

                out_train = torch.zeros(T, batch_size, y_rates.shape[1], device=device)
                for cycle in range(T):
                    out_train[cycle] = self.to_spikes(y_rates, cycle)

                spike_train_cache[node.id] = out_train
            else:
                raise TypeError(f"Unknown node type: {type(node)}")

        # Gather output spike *counts* (B, out_dim) by summing spikes over time.
        output_sources = list(self.output_sources.flatten())
        output_signals = torch.zeros(batch_size, len(output_sources), device=device)
        for cycle in range(T):
            output_signals += self._get_signal_tensor(
                spike_train_cache, input_spike_train, batch_size, device, output_sources, cycle
            )

        self.total_spikes = torch.sum(output_signals).item()
        return output_signals

    # Legacy helpers removed: the unified implementation above operates on full
    # spike trains and handles ComputeOps as explicit sync barriers.


