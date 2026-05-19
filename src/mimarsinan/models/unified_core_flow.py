"""Spiking simulator for unified IRGraph (NeuralCore + ComputeOp); LIF and TTFS modes."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mimarsinan.mapping.ir import ComputeOp, IRGraph, NeuralCore, WeightBank
from mimarsinan.mapping.ir_source_spans import IRSourceSpan, compress_ir_sources


# float64 default: match nevresim signal_t; optional float32 via compute_dtype.
_COMPUTE_DTYPE = torch.float64


def _ttfs_activation_from_type(activation_type: str | None):
    """Map IR activation_type (compound strings use the base name before ' + ') to torch.nn.functional."""
    if activation_type is None or (isinstance(activation_type, str) and activation_type.strip() in ("", "ReLU")):
        return F.relu
    base = activation_type.split(" + ")[0].strip()
    name_map = {
        "LeakyReLU": "leaky_relu",
        "LeakyGradReLU": "relu",  # LeakyGradReLU.forward is pure ReLU; leaky only in backward
        "ReLU": "relu",
        "GELU": "gelu",
        "Identity": "identity",
    }
    f_name = name_map.get(base, "relu")
    if f_name == "identity":
        return lambda x: x
    try:
        return getattr(F, f_name)
    except AttributeError:
        return F.relu


class SpikingUnifiedCoreFlow(nn.Module):
    """Flat IRGraph spiking sim: LIF/TTFS cores, ComputeOp sync barriers, shared WeightBank params."""

    _TTFS_SPIKING_MODES = {"ttfs", "ttfs_quantized"}

    def __init__(
        self,
        input_shape,
        ir_graph: IRGraph,
        simulation_length: int,
        preprocessor: nn.Module | None = None,
        firing_mode: str = "Default",
        spike_mode: str = "Uniform",
        thresholding_mode: str = "<=",
        spiking_mode: str = "lif",
        compute_dtype: torch.dtype = _COMPUTE_DTYPE,
    ):
        super().__init__()

        self.input_shape = input_shape
        self.ir_graph = ir_graph
        self.nodes = ir_graph.nodes
        self.output_sources = ir_graph.output_sources
        self.simulation_length = simulation_length
        self.preprocessor = preprocessor if preprocessor is not None else nn.Identity()
        self.firing_mode = firing_mode
        self.spike_mode = spike_mode
        self.thresholding_mode = thresholding_mode
        self.spiking_mode = spiking_mode
        self._compute_dtype = compute_dtype

        assert firing_mode in ["Default", "Novena", "TTFS"]
        assert spike_mode in ["Stochastic", "Deterministic", "FrontLoaded", "Uniform", "TTFS"]
        assert thresholding_mode in ["<", "<="]

        self._bank_params = nn.ParameterDict()
        for bank_id, bank in ir_graph.weight_banks.items():
            w = torch.tensor(bank.core_matrix.T, dtype=torch.float32)
            self._bank_params[str(bank_id)] = nn.Parameter(w, requires_grad=False)

        self.neural_core_ids: list[int] = []
        self._id_to_bank: Dict[int, tuple[str, tuple[int, int] | None]] = {}
        self._id_to_owned_param: Dict[int, int] = {}
        self.neural_core_params = nn.ParameterList()

        for node in self.nodes:
            if not isinstance(node, NeuralCore):
                continue
            self.neural_core_ids.append(node.id)
            if node.has_weight_bank():
                self._id_to_bank[node.id] = (
                    str(node.weight_bank_id),
                    node.weight_row_slice,
                )
                continue
            weight = torch.tensor(node.core_matrix.T, dtype=torch.float32)
            self.neural_core_params.append(nn.Parameter(weight, requires_grad=False))
            self._id_to_owned_param[node.id] = len(self.neural_core_params) - 1

        threshold_vals: list[float] = []
        self._threshold_idx_cache: Dict[int, int] = {}
        for node in self.nodes:
            if not isinstance(node, NeuralCore):
                continue
            t = node.threshold
            threshold_vals.append(float(t.item()) if hasattr(t, "item") else float(t))
            self._threshold_idx_cache[node.id] = len(threshold_vals) - 1
        self.register_buffer(
            "_thresholds_packed",
            torch.tensor(threshold_vals, dtype=_COMPUTE_DTYPE)
            if threshold_vals
            else torch.empty(0, dtype=_COMPUTE_DTYPE),
        )

        hw_chunks: list[torch.Tensor] = []
        self._hw_bias_spans: Dict[int, tuple[int, int]] = {}
        hw_offset = 0
        for node in self.nodes:
            if not isinstance(node, NeuralCore) or node.hardware_bias is None:
                continue
            hb = torch.tensor(node.hardware_bias, dtype=_COMPUTE_DTYPE).reshape(-1)
            hw_chunks.append(hb)
            self._hw_bias_spans[node.id] = (hw_offset, hb.numel())
            hw_offset += hb.numel()
        self.register_buffer(
            "_hw_bias_packed",
            torch.cat(hw_chunks) if hw_chunks else torch.empty(0, dtype=_COMPUTE_DTYPE),
        )

        self._id_to_out_dim: Dict[int, int] = {}
        for node in self.nodes:
            if isinstance(node, NeuralCore):
                self._id_to_out_dim[node.id] = node.get_output_count() if node.core_matrix is not None else (
                    (node.weight_row_slice[1] - node.weight_row_slice[0]) if node.weight_row_slice else
                    ir_graph.weight_banks[node.weight_bank_id].core_matrix.shape[1]
                )

        self._sync_points = [i for i, n in enumerate(self.nodes) if isinstance(n, ComputeOp)]

        self._input_spans: Dict[int, list[IRSourceSpan]] = {}
        for node in self.nodes:
            flat = list(node.input_sources.flatten())
            self._input_spans[int(node.id)] = compress_ir_sources(flat)
        self._output_spans: list[IRSourceSpan] = compress_ir_sources(list(self.output_sources.flatten()))

        self._release_at_step: Dict[int, list[int]] = {}
        consumed_by_output: set[int] = set()
        for sp in self._output_spans:
            if sp.kind == "node":
                consumed_by_output.add(int(sp.src_node_id))
        last_reader: Dict[int, int] = {}
        for reader_idx, node in enumerate(self.nodes):
            for sp in self._input_spans[int(node.id)]:
                if sp.kind != "node":
                    continue
                last_reader[int(sp.src_node_id)] = reader_idx
        for src_id, idx in last_reader.items():
            if src_id in consumed_by_output:
                continue
            self._release_at_step.setdefault(idx, []).append(src_id)

        self._assert_mapping_contracts(ir_graph)

        from mimarsinan.mapping.hybrid_hardcore_mapping import (
            _perceptron_wrapped_activation_scale,
        )
        self._ttfs_node_output_scale: Dict[int, float] = {}
        self._ttfs_node_input_scale: Dict[int, float] = {}
        for node in self.nodes:
            if isinstance(node, NeuralCore):
                s = float(
                    node.activation_scale.item()
                    if hasattr(node.activation_scale, "item")
                    else node.activation_scale
                )
                self._ttfs_node_output_scale[node.id] = s
                self._ttfs_node_input_scale[node.id] = s
            elif isinstance(node, ComputeOp):
                module = (node.params or {}).get("module")
                wrapped_scale = _perceptron_wrapped_activation_scale(module)
                src_scales: list[float] = []
                all_raw_inputs = True
                for src in node.input_sources.flatten():
                    src_id = int(src.node_id) if hasattr(src, "node_id") else int(src)
                    if src_id >= 0:
                        all_raw_inputs = False
                        src_scales.append(self._ttfs_node_output_scale.get(src_id, 1.0))

                if all_raw_inputs:
                    self._ttfs_node_input_scale[node.id] = 1.0
                else:
                    self._ttfs_node_input_scale[node.id] = (
                        sum(src_scales) / len(src_scales) if src_scales else 1.0
                    )

                if wrapped_scale is not None:
                    self._ttfs_node_output_scale[node.id] = wrapped_scale
                else:
                    self._ttfs_node_output_scale[node.id] = self._ttfs_node_input_scale[node.id]

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
        """
        Fill `out` (B, N) from compressed IRSource spans for the given cycle.
        """
        out.zero_()
        for sp in spans:
            d0 = int(sp.dst_start)
            d1 = int(sp.dst_end)
            if sp.kind == "off":
                continue
            if sp.kind == "on":
                if self.spiking_mode in self._TTFS_SPIKING_MODES and cycle != 0:
                    continue  # TTFS always-on fires once at cycle 0 only
                out[:, d0:d1].fill_(1.0)
                continue
            if sp.kind == "input":
                out[:, d0:d1] = input_spike_train[cycle][:, int(sp.src_start):int(sp.src_end)]
                continue
            out[:, d0:d1] = spike_train_cache[int(sp.src_node_id)][cycle][:, int(sp.src_start):int(sp.src_end)]

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """LIF spike-train sim or analytical TTFS; ComputeOps are rate-space sync barriers."""
        try:
            if self.spiking_mode in self._TTFS_SPIKING_MODES:
                return self._forward_ttfs(x)
            return self._forward_lif(x)
        finally:
            if isinstance(x, torch.Tensor) and x.is_cuda:
                torch.cuda.empty_cache()

    def _forward_lif(self, x: torch.Tensor) -> torch.Tensor:
        """LIF (rate-coded integrate-and-fire) forward pass (Default / Novena)."""
        x = self.preprocessor(x)
        x = x.view(x.shape[0], -1)

        batch_size = x.shape[0]
        device = x.device

        T = self.simulation_length

        input_spike_train = torch.zeros(T, batch_size, x.shape[1], device=device)
        for cycle in range(T):
            input_spike_train[cycle] = self.to_spikes(x, cycle)

        spike_train_cache: Dict[int, torch.Tensor] = {}
        self._last_core_spike_counts: Dict[int, float] = {}

        ops = {"<": torch.lt, "<=": torch.le}

        for node_idx, node in enumerate(self.nodes):
            if isinstance(node, NeuralCore):
                weight = self._get_weight(node).to(torch.float32)  # (neurons, axons)
                threshold = self._get_threshold(node)
                hw_bias = self._get_hw_bias(node)

                spans = self._input_spans[int(node.id)]
                in_dim = int(len(node.input_sources.flatten()))
                out_dim = self._id_to_out_dim[node.id]

                memb = torch.zeros(batch_size, out_dim, device=device)
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

                    contribution = torch.matmul(weight, inp.T).T
                    if hw_bias is not None:
                        contribution = contribution + hw_bias
                    memb += contribution
                    fired = ops[self.thresholding_mode](threshold, memb)
                    out_train[cycle] = fired.float()
                    total_spikes += fired.float().sum().item()

                    if self.firing_mode == "Novena":
                        memb[fired] = 0.0
                    elif self.firing_mode == "Default":
                        memb[fired] -= threshold

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

    def _forward_ttfs(self, x: torch.Tensor) -> torch.Tensor:
        x = self.preprocessor(x)
        x = x.view(x.shape[0], -1)

        if self.spiking_mode == "ttfs_quantized":
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

                act_fn = _ttfs_activation_from_type(node.activation_type)
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
                safe_thresh = threshold.clamp(min=1e-12)
                k_fire_raw = torch.ceil(S * (1.0 - V / safe_thresh))
                fires = k_fire_raw < S
                k_fire = k_fire_raw.clamp(0, S - 1)
                activation_cache[node.id] = torch.where(
                    fires, (S - k_fire) / S, torch.zeros_like(k_fire)
                )
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

    def get_core_spike_rates(self) -> list[float]:
        """Per-core mean spike rate after forward (graph order)."""
        if not hasattr(self, '_last_core_spike_counts'):
            raise RuntimeError("get_core_spike_rates called before forward pass")
        
        rates = []
        for node in self.nodes:
            if isinstance(node, NeuralCore):
                rates.append(self._last_core_spike_counts.get(node.id, 0.0))
        return rates

    def get_cores(self) -> list[NeuralCore]:
        """Return list of neural cores in graph order."""
        return [n for n in self.nodes if isinstance(n, NeuralCore)]

    def refresh_thresholds(self) -> None:
        """Sync node.threshold edits into the packed threshold buffer."""
        for node in self.nodes:
            if isinstance(node, NeuralCore):
                t = node.threshold
                self._set_threshold(node, float(t.item()) if hasattr(t, "item") else float(t))
