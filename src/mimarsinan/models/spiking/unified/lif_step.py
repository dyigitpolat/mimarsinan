"""Unified IRGraph LIF forward step."""

from __future__ import annotations

from typing import Dict

import torch

from mimarsinan.mapping.ir import ComputeOp, NeuralCore
from mimarsinan.models.spiking.cycle_policy import cycle_neuron_policy


class UnifiedLifStepMixin:
    """Rate-coded LIF simulation over flat IRGraph nodes."""

    def _forward_lif(self, x: torch.Tensor) -> torch.Tensor:
        """LIF (rate-coded integrate-and-fire) forward pass (Default / Novena)."""
        x = self.preprocessor(x)
        x = x.view(x.shape[0], -1)

        batch_size = x.shape[0]
        device = x.device

        T = self.simulation_length
        policy = cycle_neuron_policy(
            self.spiking_mode, self.ttfs_cycle_schedule, self.firing_mode,
        )

        input_spike_train = torch.zeros(T, batch_size, x.shape[1], device=device)
        for cycle in range(T):
            input_spike_train[cycle] = self.to_spikes(x, cycle)

        spike_train_cache: Dict[int, torch.Tensor] = {}
        self._last_core_spike_counts: Dict[int, float] = {}

        for node_idx, node in enumerate(self.nodes):
            if isinstance(node, NeuralCore):
                weight = self._get_weight(node).to(torch.float32)  # (neurons, axons)
                threshold = self._get_threshold(node)
                hw_bias = self._get_hw_bias(node)

                spans = self._input_spans[int(node.id)]
                in_dim = int(len(node.input_sources.flatten()))
                out_dim = self._id_to_out_dim[node.id]

                state = policy.make_state(batch_size, out_dim, device, torch.float32)
                out_train = torch.zeros(T, batch_size, out_dim, device=device)
                inp = torch.zeros(batch_size, in_dim, device=device)
                total_spikes = 0.0

                for cycle in range(T):
                    self._fill_signal_tensor_from_spans(
                        inp,
                        spike_train_cache=spike_train_cache,
                        input_spike_train=input_spike_train,
                        batch_size=batch_size,
                        device=device,
                        spans=spans,
                        cycle=cycle,
                    )

                    spikes = policy.step(
                        state,
                        weight,
                        inp,
                        threshold,
                        hw_bias=hw_bias,
                        thresholding_mode=self.thresholding_mode,
                    )
                    out_train[cycle] = spikes
                    total_spikes += spikes.sum().item()

                spike_train_cache[node.id] = out_train
                self._last_core_spike_counts[node.id] = total_spikes / (batch_size * out_dim * T + 1e-9)

            elif isinstance(node, ComputeOp):
                spans = self._input_spans[int(node.id)]
                in_dim = int(len(node.input_sources.flatten()))
                in_rates = torch.zeros(batch_size, in_dim, device=device)
                self._fill_rate_tensor_from_spans(
                    in_rates,
                    spike_train_cache=spike_train_cache,
                    input_spike_train=input_spike_train,
                    spans=spans,
                )

                in_scale = self._ttfs_node_input_scale.get(node.id, 1.0)
                out_scale = self._ttfs_node_output_scale.get(node.id, 1.0)
                if in_scale != 1.0:
                    module_in = in_rates * in_scale
                else:
                    module_in = in_rates
                y = node.execute_on_gathered(module_in)
                y = y.view(batch_size, -1)
                if out_scale != 1.0:
                    y = y / out_scale
                y_rates = y.clamp(0.0, 1.0)

                out_train = torch.zeros(T, batch_size, y_rates.shape[1], device=device)
                for cycle in range(T):
                    out_train[cycle] = self.to_spikes(y_rates, cycle)

                spike_train_cache[node.id] = out_train
            else:
                raise TypeError(f"Unknown node type: {type(node)}")

            for released_id in self._release_at_step.get(node_idx, ()):
                spike_train_cache.pop(released_id, None)

        output_sources = list(self.output_sources.flatten())
        output_signals = torch.zeros(batch_size, len(output_sources), device=device)
        for cycle in range(T):
            for sp in self._output_spans:
                d0 = int(sp.dst_start)
                d1 = int(sp.dst_end)
                if sp.kind == "off":
                    continue
                if sp.kind == "on":
                    output_signals[:, d0:d1] += 1.0
                    continue
                if sp.kind == "input":
                    output_signals[:, d0:d1] += input_spike_train[cycle][:, int(sp.src_start):int(sp.src_end)]
                    continue
                output_signals[:, d0:d1] += spike_train_cache[int(sp.src_node_id)][cycle][:, int(sp.src_start):int(sp.src_end)]

        self.total_spikes = torch.sum(output_signals).item()
        return output_signals
