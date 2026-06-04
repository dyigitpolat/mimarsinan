"""Unified IRGraph stage I/O helpers (weights, spans, contracts)."""

from __future__ import annotations

from typing import Dict

import torch

from mimarsinan.mapping.ir import ComputeOp, IRGraph, NeuralCore
from mimarsinan.mapping.support.ir_source_spans import IRSourceSpan
from mimarsinan.models.spiking.signal_spans import fill_signal_from_spans
from mimarsinan.models.spiking.spiking_config import TTFS_SPIKING_MODES


class UnifiedStageIOMixin:
    """Weight/threshold accessors and IR span fill helpers."""

    _TTFS_SPIKING_MODES = TTFS_SPIKING_MODES

    def _get_weight(self, node: NeuralCore) -> torch.Tensor:
        """(neurons, axons) float32 weights; bank-backed cores slice ``_bank_params``."""
        bank_info = self._id_to_bank.get(node.id)
        if bank_info is not None:
            bank_key, row_slice = bank_info
            w = self._bank_params[bank_key]  # (neurons_full, axons)
            if row_slice is not None:
                start, end = row_slice
                w = w[start:end, :]
            return w

        return self.neural_core_params[self._id_to_owned_param[node.id]]

    def _get_threshold(self, node: NeuralCore) -> torch.Tensor:
        """Return the scalar threshold tensor for *node*."""
        return self._thresholds_packed[self._threshold_idx_cache[node.id]]

    def _set_threshold(self, node: NeuralCore, value: float) -> None:
        """Overwrite the threshold for *node* in the packed buffer."""
        self._thresholds_packed[self._threshold_idx_cache[node.id]] = float(value)

    def _get_hw_bias(self, node: NeuralCore) -> torch.Tensor | None:
        """Return the hardware-bias vector for *node*, or None if unset."""
        span = self._hw_bias_spans.get(node.id)
        if span is None:
            return None
        offset, length = span
        return self._hw_bias_packed[offset:offset + length]

    def _ttfs_encode_input(self, activations: torch.Tensor) -> torch.Tensor:
        """Encode activations [0,1] → single-spike train (T, B, N)."""
        T = self.simulation_length
        spike_times = torch.round(T * (1.0 - activations.clamp(0.0, 1.0))).long()
        spike_train = torch.zeros(T, *activations.shape, device=activations.device)
        for cycle in range(T):
            spike_train[cycle] = (spike_times == cycle).float()
        return spike_train

    def _fill_signal_tensor_from_spans(
        self,
        out: torch.Tensor,
        *,
        spike_train_cache: Dict[int, torch.Tensor],
        input_spike_train: torch.Tensor,
        batch_size: int,
        device: torch.device,
        spans: list[IRSourceSpan],
        cycle: int,
    ) -> None:
        """Fill ``out`` (B, N) from compressed IRSource spans for the given cycle."""
        ttfs = self.spiking_mode in self._TTFS_SPIKING_MODES

        def _on_always_on(d0: int, d1: int) -> None:
            # Single-spike TTFS bias = one spike at the core's local window start.
            # Each node runs its own [0, T) loop over materialized upstream trains,
            # so cycle 0 IS the local window start here (no shared latency-gated loop
            # as in the hybrid flow, where it must fire at cycle == core.latency).
            if ttfs and cycle != 0:
                return
            out[:, d0:d1].fill_(1.0)

        fill_signal_from_spans(
            out,
            spans,
            read_input=lambda sp: out.__setitem__(
                (slice(None), slice(int(sp.dst_start), int(sp.dst_end))),
                input_spike_train[cycle][:, int(sp.src_start) : int(sp.src_end)],
            ),
            read_upstream=lambda sp: out.__setitem__(
                (slice(None), slice(int(sp.dst_start), int(sp.dst_end))),
                spike_train_cache[int(sp.src_node_id)][cycle][
                    :, int(sp.src_start) : int(sp.src_end)
                ],
            ),
            on_always_on=_on_always_on,
            cycle=cycle,
        )

    def _fill_rate_tensor_from_spans(
        self,
        out_rates: torch.Tensor,
        *,
        spike_train_cache: Dict[int, torch.Tensor],
        input_spike_train: torch.Tensor,
        spans: list[IRSourceSpan],
    ) -> None:
        """
        Fill `out_rates` (B, N) from compressed IRSource spans by averaging spikes over T.
        """
        out_rates.zero_()
        for sp in spans:
            d0 = int(sp.dst_start)
            d1 = int(sp.dst_end)
            if sp.kind == "off":
                continue
            if sp.kind == "on":
                out_rates[:, d0:d1].fill_(1.0)
                continue
            if sp.kind == "input":
                out_rates[:, d0:d1] = input_spike_train[:, :, int(sp.src_start):int(sp.src_end)].float().mean(dim=0)
                continue
            out_rates[:, d0:d1] = spike_train_cache[int(sp.src_node_id)][:, :, int(sp.src_start):int(sp.src_end)].float().mean(dim=0)

    def _fill_activation_from_ir_spans(
        self,
        out: torch.Tensor,
        *,
        x: torch.Tensor,
        activation_cache: Dict[int, torch.Tensor],
        spans: list[IRSourceSpan],
    ) -> None:
        """Fill activation tensor from IR spans (TTFS analytical path)."""
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
                out[:, d0:d1] = x[:, int(sp.src_start):int(sp.src_end)]
                continue
            out[:, d0:d1] = activation_cache[int(sp.src_node_id)][:, int(sp.src_start):int(sp.src_end)]

    def _assert_mapping_contracts(self, ir_graph: IRGraph) -> None:
        """Fail fast if pruned IR has mismatched axon/neuron counts vs weight matrix."""
        from mimarsinan.mapping.ir import IRSource

        for node in self.nodes:
            if not isinstance(node, NeuralCore):
                continue
            try:
                weight = self._get_weight(node)
            except Exception:
                continue
            n_neurons, n_axons = weight.shape[0], weight.shape[1]
            n_src = int(len(node.input_sources.flatten()))
            if n_src != n_axons:
                raise ValueError(
                    f"Mapping contract violated: core id={node.id} "
                    f"len(input_sources)={n_src} != weight axons={n_axons}. "
                    "Check pruning/compaction left sources and matrix rows aligned."
                )
            try:
                out_count = node.get_output_count()
                if out_count != n_neurons:
                    raise ValueError(
                        f"Mapping contract violated: core id={node.id} "
                        f"get_output_count()={out_count} != weight neurons={n_neurons}. "
                        "Check pruning/compaction left matrix columns and output count aligned."
                    )
            except ValueError:
                pass
        if ir_graph.output_sources.size:
            flat = ir_graph.output_sources.flatten()
            for i, src in enumerate(flat):
                if not isinstance(src, IRSource) or src.node_id < 0:
                    continue
                node = next(
                    (n for n in ir_graph.nodes if getattr(n, "id", None) == src.node_id),
                    None,
                )
                if node is None:
                    raise ValueError(
                        f"Mapping contract violated: output_sources[{i}] references "
                        f"node_id={src.node_id} which is not in the graph."
                    )
                if isinstance(node, NeuralCore):
                    try:
                        out_count = node.get_output_count()
                    except ValueError:
                        out_count = self._get_weight(node).shape[0]
                    if src.index < 0 or src.index >= out_count:
                        raise ValueError(
                            f"Mapping contract violated: output_sources[{i}] "
                            f"node_id={src.node_id} index={src.index} out of range [0, {out_count})."
                        )

    def _execute_compute_op_ttfs(
        self,
        node: ComputeOp,
        x: torch.Tensor,
        batch_size: int,
        device: torch.device,
        activation_cache: Dict[int, torch.Tensor],
    ) -> torch.Tensor:
        """TTFS ComputeOp: gather in float32, rescale in/out, store in compute_dtype."""
        spans = self._input_spans[int(node.id)]
        in_dim = int(len(node.input_sources.flatten()))
        inp = torch.zeros(batch_size, in_dim, device=device, dtype=torch.float32)
        self._fill_activation_from_ir_spans(
            inp, x=x, activation_cache=activation_cache, spans=spans,
        )

        in_scale = self._ttfs_node_input_scale.get(node.id, 1.0)
        out_scale = self._ttfs_node_output_scale.get(node.id, 1.0)
        if abs(in_scale - 1.0) > 1e-9:
            inp = inp * in_scale

        out = node.execute_on_gathered(inp)

        if abs(out_scale - 1.0) > 1e-9:
            out = out / out_scale

        return out.to(self._compute_dtype)
