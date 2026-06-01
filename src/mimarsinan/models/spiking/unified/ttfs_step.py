"""Unified IRGraph TTFS forward steps."""

from __future__ import annotations

from typing import Dict

import torch

from mimarsinan.mapping.ir import ComputeOp, NeuralCore
from mimarsinan.models.spiking.ttfs_activation import ttfs_activation_from_type
from mimarsinan.models.spiking.ttfs_kernels import ttfs_quantized_activation


class UnifiedTtfsStepMixin:
    """Analytical TTFS paths (continuous and quantized)."""

    def _forward_ttfs(self, x: torch.Tensor) -> torch.Tensor:
        x = self.preprocessor(x)
        x = x.view(x.shape[0], -1)

        if self.spiking_mode != "ttfs":
            return self._forward_ttfs_quantized(x)
        return self._forward_ttfs_continuous(x)

    def _forward_ttfs_continuous(self, x: torch.Tensor) -> torch.Tensor:
        """Analytical TTFS: clamp(relu(Wx+b)/θ, 0, 1) per core; outputs only clamped."""
        batch_size = x.shape[0]
        device = x.device
        compute_dtype = self._compute_dtype

        x_compute = x.to(compute_dtype)

        activation_cache: Dict[int, torch.Tensor] = {}
        self._last_core_spike_counts: Dict[int, float] = {}

        for node_idx, node in enumerate(self.nodes):
            if isinstance(node, NeuralCore):
                weight = self._get_weight(node).to(compute_dtype)
                threshold = self._get_threshold(node).to(compute_dtype)
                hw_bias = self._get_hw_bias(node)

                spans = self._input_spans[int(node.id)]
                in_dim = int(len(node.input_sources.flatten()))
                inp = torch.zeros(batch_size, in_dim, device=device, dtype=compute_dtype)
                self._fill_activation_from_ir_spans(
                    inp, x=x_compute, activation_cache=activation_cache, spans=spans
                )

                out = torch.matmul(weight, inp.T).T
                if hw_bias is not None:
                    out = out + hw_bias.to(compute_dtype)

                act_fn = ttfs_activation_from_type(node.activation_type)
                out = act_fn(out)
                out = out / threshold
                out = out.clamp(0.0, 1.0)

                activation_cache[node.id] = out
                self._last_core_spike_counts[node.id] = 0.0

            elif isinstance(node, ComputeOp):
                activation_cache[node.id] = self._execute_compute_op_ttfs(
                    node, x, batch_size, device, activation_cache
                )
            else:
                raise TypeError(f"Unknown node type: {type(node)}")

            for released_id in self._release_at_step.get(node_idx, ()):
                activation_cache.pop(released_id, None)

        output_sources = list(self.output_sources.flatten())
        output_signals = torch.zeros(batch_size, len(output_sources), device=device, dtype=compute_dtype)
        self._fill_activation_from_ir_spans(
            output_signals, x=x_compute, activation_cache=activation_cache, spans=self._output_spans
        )

        self.total_spikes = 0.0
        return output_signals.to(torch.float32)

    def _forward_ttfs_quantized(self, x: torch.Tensor) -> torch.Tensor:
        """Closed-form ttfs_quantized: k_fire = ceil(S*(1-V/θ)); O(cores) not O(latency*S*cores)."""
        batch_size = x.shape[0]
        device = x.device
        S = self.simulation_length
        compute_dtype = self._compute_dtype
        x_compute = x.to(compute_dtype)

        activation_cache: Dict[int, torch.Tensor] = {}
        self._last_core_spike_counts: Dict[int, float] = {}

        for node_idx, node in enumerate(self.nodes):
            if isinstance(node, NeuralCore):
                weight = self._get_weight(node).to(compute_dtype)
                threshold = self._get_threshold(node).to(compute_dtype)
                hw_bias = self._get_hw_bias(node)

                spans = self._input_spans[int(node.id)]
                in_dim = int(len(node.input_sources.flatten()))
                inp = torch.zeros(batch_size, in_dim, device=device, dtype=compute_dtype)
                self._fill_activation_from_ir_spans(
                    inp, x=x_compute, activation_cache=activation_cache, spans=spans
                )

                V = torch.matmul(weight, inp.T).T
                if hw_bias is not None:
                    V = V + hw_bias.to(compute_dtype)
                from mimarsinan.models.spiking.ttfs_kernels import ttfs_quantized_activation

                activation_cache[node.id] = ttfs_quantized_activation(V, threshold, S)
                self._last_core_spike_counts[node.id] = 0.0

            elif isinstance(node, ComputeOp):
                activation_cache[node.id] = self._execute_compute_op_ttfs(
                    node, x, batch_size, device, activation_cache
                )
            else:
                raise TypeError(f"Unknown node type: {type(node)}")

            for released_id in self._release_at_step.get(node_idx, ()):
                activation_cache.pop(released_id, None)

        output_sources = list(self.output_sources.flatten())
        output_signals = torch.zeros(
            batch_size, len(output_sources), device=device, dtype=compute_dtype
        )
        self._fill_activation_from_ir_spans(
            output_signals, x=x_compute, activation_cache=activation_cache, spans=self._output_spans
        )

        self.total_spikes = 0.0
        return output_signals
