"""
UnifiedCoreFlow: Simulation that supports both neural cores and compute ops.

This module provides a spiking simulator for the unified IRGraph (NeuralCore + ComputeOp),
including correct sync-barrier semantics for ComputeOps (rate -> op -> respike).

Supports both rate-coded (Default/Novena) and Time-to-First-Spike (TTFS)
firing modes.

TTFS mode implements the B1-model from:
  Stanojevic et al., "High-performance deep spiking neural networks with
  0.3 spikes per neuron", Nature Communications 15, 6793 (2024).
  https://www.nature.com/articles/s41467-024-51110-5

Two TTFS deployment modes (selected via ``spiking_mode``):

  * **ttfs** (continuous) — analytical ``clamp(ReLU(W @ x + b) / θ, 0, 1)``
    per NeuralCore; outputs clamped to ``[0, 1]`` to match hardware TTFS
    (neurons fire at most once).  Inputs are not clamped because weight
    matrices already incorporate ``per_input_scales`` normalization.
  * **ttfs_quantized** (analytical quantised) — closed-form computation
    that yields the exact same output as the cycle-based simulation
    but in O(N_cores) instead of O(max_latency * S * N_cores):

      V = W @ x
      k_fire = ceil(S * (1 - V / θ))
      k_fire = clamp(k_fire, 0, S-1)
      activation = (S - k_fire) / S
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mimarsinan.mapping.ir import ComputeOp, IRGraph, NeuralCore, WeightBank
from mimarsinan.mapping.ir_source_spans import IRSourceSpan, compress_ir_sources


# Default precision for on-chip spiking compute.  The C++ nevresim simulator
# uses ``signal_t = double`` on TTFS paths; float64 matches that bit-for-bit
# but is ~2× the memory and ~2× slower on GPU.  For validation / sim-level
# checks (SoftCoreMappingStep, CoreFlow Tuning) float32 suffices — the
# ``ceil(S * (1 - V/θ))`` boundary flips only on a tiny fraction of neurons
# whose V sits within one ULP of θ.  Callers that need bit-exact C++ match
# (e.g. final deployment verification) pass ``compute_dtype=torch.float64``.
_COMPUTE_DTYPE = torch.float32


def _ttfs_activation_from_type(activation_type: str | None):
    """Resolve activation function for TTFS from IR activation_type string.

    activation_type may be a compound string from TransformedActivation, e.g.
    "LeakyReLU + ClampDecorator, QuantizeDecorator". We use only the base name
    (before " + ") and map to torch.nn.functional: LeakyReLU -> leaky_relu,
    ReLU -> relu, GELU -> gelu. Falls back to relu if unknown or lookup fails.
    """
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
    """
    Spiking version of UnifiedCoreFlow.

    This handles spike-based simulation with membrane potential dynamics.
    Non-neural operations (ComputeOp) act as synchronization points where
    spike counts are converted to rates, the operation is applied, and
    rates are converted back to spikes.

    **Shared-weight optimisation:** When the IR graph contains
    ``WeightBank``s (e.g. from conv layers), a single ``nn.Parameter`` is
    registered per bank instead of per core.  Bank-backed cores look up
    their weight via ``_bank_params`` instead of ``neural_core_params``,
    avoiding O(h_out * w_out) memory duplication.
    """

    _TTFS_SPIKING_MODES = {"ttfs", "ttfs_quantized"}

    def __init__(
        self,
        input_shape,
        ir_graph: IRGraph,
        simulation_length: int,
        preprocessor: nn.Module | None = None,
        firing_mode: str = "Default",
        spike_mode: str = "Uniform",
        thresholding_mode: str = "<",
        spiking_mode: str = "rate",
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
        # Compute dtype is per-instance so callers can pick float32 (default, fast)
        # or float64 (bit-exact match to C++ nevresim for deployment verification).
        self._compute_dtype = compute_dtype

        assert firing_mode in ["Default", "Novena", "TTFS"]
        assert spike_mode in ["Stochastic", "Deterministic", "FrontLoaded", "Uniform", "TTFS"]
        assert thresholding_mode in ["<", "<="]

        # --- Weight bank parameters (shared across many cores) ----------
        # One Parameter per bank: already memory-efficient (conv kernels).
        self._bank_params = nn.ParameterDict()
        for bank_id, bank in ir_graph.weight_banks.items():
            # Stored as (neurons, axons) for matmul convenience
            w = torch.tensor(bank.core_matrix.T, dtype=torch.float32)
            self._bank_params[str(bank_id)] = nn.Parameter(w, requires_grad=False)

        # --- Packed owned-core weights ------------------------------------
        # Instead of a nn.ParameterList with O(#owned_cores) entries (each
        # carrying its own tensor metadata + separate .to(device) kernel), we
        # concatenate every owned core's weight (as (neurons, axons)) into a
        # single flat buffer per source dtype. Int8-quantized weights stay in
        # int8 on GPU (~8× smaller than float32) and are upcast to
        # ``compute_dtype`` only inside ``_get_weight`` for the active core.
        self.neural_core_ids: list[int] = []
        self._id_to_bank: Dict[int, tuple[str, tuple[int, int] | None]] = {}
        # node_id -> (kind, offset, length, neurons, axons); kind in {"int8","int16","f32"}.
        self._owned_spans: Dict[int, tuple[str, int, int, int, int]] = {}

        int_chunks: dict[str, list[torch.Tensor]] = {"int8": [], "int16": []}
        int_offsets: dict[str, int] = {"int8": 0, "int16": 0}
        f32_chunks: list[torch.Tensor] = []
        f32_offset = 0

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

            mat = node.core_matrix  # (axons, neurons)
            # Store transposed so matmul is weight @ inp.T directly; ascontiguousarray
            # avoids the per-core np.zeros+fill pattern the old code had.
            mat_T = np.ascontiguousarray(mat.T)  # (neurons, axons)
            n_neurons, n_axons = mat_T.shape
            length = n_neurons * n_axons
            dtype = mat_T.dtype
            if dtype == np.int8:
                int_chunks["int8"].append(torch.from_numpy(mat_T.reshape(-1)))
                self._owned_spans[node.id] = ("int8", int_offsets["int8"], length, n_neurons, n_axons)
                int_offsets["int8"] += length
            elif dtype == np.int16:
                int_chunks["int16"].append(torch.from_numpy(mat_T.reshape(-1)))
                self._owned_spans[node.id] = ("int16", int_offsets["int16"], length, n_neurons, n_axons)
                int_offsets["int16"] += length
            else:
                f32_chunks.append(torch.from_numpy(mat_T.astype(np.float32, copy=False).reshape(-1)))
                self._owned_spans[node.id] = ("f32", f32_offset, length, n_neurons, n_axons)
                f32_offset += length

        # Register each dtype's packed buffer (or a zero-length placeholder) so
        # ``.to(device)`` moves the whole corpus in one kernel launch.
        self.register_buffer(
            "_owned_w_int8",
            torch.cat(int_chunks["int8"]) if int_chunks["int8"] else torch.empty(0, dtype=torch.int8),
        )
        self.register_buffer(
            "_owned_w_int16",
            torch.cat(int_chunks["int16"]) if int_chunks["int16"] else torch.empty(0, dtype=torch.int16),
        )
        self.register_buffer(
            "_owned_w_f32",
            torch.cat(f32_chunks) if f32_chunks else torch.empty(0, dtype=torch.float32),
        )

        # --- Packed thresholds (1-D float32 over all NeuralCores) ---------
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
            torch.tensor(threshold_vals, dtype=torch.float32)
            if threshold_vals
            else torch.empty(0, dtype=torch.float32),
        )

        # --- Packed hardware biases ---------------------------------------
        hw_chunks: list[torch.Tensor] = []
        self._hw_bias_spans: Dict[int, tuple[int, int]] = {}
        hw_offset = 0
        for node in self.nodes:
            if not isinstance(node, NeuralCore) or node.hardware_bias is None:
                continue
            hb = torch.tensor(node.hardware_bias, dtype=torch.float32).reshape(-1)
            hw_chunks.append(hb)
            self._hw_bias_spans[node.id] = (hw_offset, hb.numel())
            hw_offset += hb.numel()
        self.register_buffer(
            "_hw_bias_packed",
            torch.cat(hw_chunks) if hw_chunks else torch.empty(0, dtype=torch.float32),
        )

        # Precompute output dims for each neural core (avoids needing graph at forward time)
        self._id_to_out_dim: Dict[int, int] = {}
        for node in self.nodes:
            if isinstance(node, NeuralCore):
                self._id_to_out_dim[node.id] = node.get_output_count() if node.core_matrix is not None else (
                    (node.weight_row_slice[1] - node.weight_row_slice[0]) if node.weight_row_slice else
                    ir_graph.weight_banks[node.weight_bank_id].core_matrix.shape[1]
                )

        # Identify synchronization points (ComputeOps that break the spiking flow)
        self._sync_points = [i for i, n in enumerate(self.nodes) if isinstance(n, ComputeOp)]

        # Precompute range-compressed source spans for faster gather.
        self._input_spans: Dict[int, list[IRSourceSpan]] = {}
        for node in self.nodes:
            flat = list(node.input_sources.flatten())
            self._input_spans[int(node.id)] = compress_ir_sources(flat)
        self._output_spans: list[IRSourceSpan] = compress_ir_sources(list(self.output_sources.flatten()))

        # Release schedule: after executing self.nodes[i], these node_ids' cached
        # outputs are no longer needed and can be dropped from the forward-pass
        # activation/spike cache. Cuts per-forward GPU peak from O(all node
        # outputs) to O(graph-frontier).
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
                # Output gather runs after every node; never release early.
                continue
            self._release_at_step.setdefault(idx, []).append(src_id)

        # Mapping contracts: after pruning, each core must satisfy axons/neurons vs sources.
        self._assert_mapping_contracts(ir_graph)

        # TTFS activation-scale bookkeeping: each node's output in TTFS mode is
        # normalised to [0, 1] by division with activation_scale inside the
        # effective-weight formula.  ComputeOps, however, wrap the *original*
        # PyTorch module (un-scaled weights + bias).  To keep the bias term
        # correct we must re-scale their input back to training range before
        # execution and normalise the output back afterwards.
        # TTFS ComputeOp scaling.  Two distinct scales per ComputeOp:
        #
        #   ``_ttfs_node_input_scale[op.id]``  — factor to multiply the
        #     gathered input by to bring it into training range before
        #     running the op's module.  NeuralCore-source inputs arrive
        #     normalised to [0, 1] via ``(S - k_fire)/S`` so they must be
        #     rescaled by the source activation_scale.  Raw-input sources
        #     (encoding-layer ops whose sources are the original model input)
        #     are already in training range and MUST NOT be rescaled.
        #
        #   ``_ttfs_node_output_scale[op.id]`` — factor to divide the module
        #     output by to bring it back into TTFS [0, 1] range for downstream
        #     NeuralCores whose ``W_eff`` assumes ``per_input_scales == this
        #     op's output scale``.  For Perceptron-wrapped ops the output
        #     scale is the Perceptron's own activation_scale (set by clamp
        #     adaptation); for generic ops it's the average of source scales
        #     (``compute_per_source_scales`` uses the same average for the
        #     downstream ``per_input_scales``).
        #
        # Before this fix both scales were equal (``_ttfs_node_output_scale``
        # only), which produced wrong results for encoding-layer perceptrons
        # wrapped as ``ComputeOp(module)``: their raw input was spuriously
        # multiplied by activation_scale, distorting the module's forward.
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
                # Neural cores don't use the input-rescale path.
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

                # Input rescale: no-op when all sources are raw input
                # (encoding path); otherwise average of source scales.
                if all_raw_inputs:
                    self._ttfs_node_input_scale[node.id] = 1.0
                else:
                    self._ttfs_node_input_scale[node.id] = (
                        sum(src_scales) / len(src_scales) if src_scales else 1.0
                    )

                # Output normalise: Perceptron-wrapped ops use the wrapped
                # activation_scale so state_buffer values sit in [0, 1],
                # matching what a NeuralCore would emit.  Generic ops use
                # the same average as the input rescale (preserves prior
                # behaviour for add/mean/etc.).
                if wrapped_scale is not None:
                    self._ttfs_node_output_scale[node.id] = wrapped_scale
                else:
                    self._ttfs_node_output_scale[node.id] = self._ttfs_node_input_scale[node.id]

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

    # -----------------------------------------------------------------
    # Weight resolution
    # -----------------------------------------------------------------
    def _get_weight(self, node: NeuralCore) -> torch.Tensor:
        """Return the (neurons, axons) weight tensor for *node*.

        Bank-backed cores slice from ``_bank_params``.  Owned-weight cores
        view a slice of the packed per-dtype buffer; int{8,16} storage stays
        in its native dtype on GPU (smaller) — callers upcast to
        ``self._compute_dtype`` before matmul.
        """
        bank_info = self._id_to_bank.get(node.id)
        if bank_info is not None:
            bank_key, row_slice = bank_info
            w = self._bank_params[bank_key]  # (neurons_full, axons)
            if row_slice is not None:
                start, end = row_slice
                w = w[start:end, :]
            return w

        kind, offset, length, n_neurons, n_axons = self._owned_spans[node.id]
        if kind == "int8":
            buf = self._owned_w_int8
        elif kind == "int16":
            buf = self._owned_w_int16
        else:
            buf = self._owned_w_f32
        return buf[offset:offset + length].view(n_neurons, n_axons)

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

    # -----------------------------------------------------------------
    # TTFS helpers
    # -----------------------------------------------------------------
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
                # TTFS: always-on (bias) source fires only once at cycle 0.
                # In rate-coded mode it fires every cycle (correct since inputs
                # also produce spikes every cycle, so everything scales by T).
                if self.spiking_mode in self._TTFS_SPIKING_MODES and cycle != 0:
                    continue
                out[:, d0:d1].fill_(1.0)
                continue
            if sp.kind == "input":
                out[:, d0:d1] = input_spike_train[cycle][:, int(sp.src_start):int(sp.src_end)]
                continue
            # node
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
        """
        Execute spiking simulation over a unified IRGraph (NeuralCore + ComputeOp).

        Key invariant:
        - NeuralCore produces a spike *train* (T, B, out) using LIF integration.
        - ComputeOp is a *sync barrier*: it consumes upstream spike trains, converts
          to rates, applies the op in rate space, then regenerates a new spike train
          for downstream nodes using the same spike generation mode as inputs.

        TTFS modes:
        - **TTFS** (continuous): analytical ``relu(W @ x + b) / θ``.
        - **TTFS_Quantized**: true cycle-based simulation (Phase 1 + Phase 2
          time-stepping with fire-once semantics).
        """
        if self.spiking_mode in self._TTFS_SPIKING_MODES:
            return self._forward_ttfs(x)
        return self._forward_rate(x)

    def _forward_rate(self, x: torch.Tensor) -> torch.Tensor:
        """Rate-coded forward pass (Default / Novena)."""
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
                    # Hardware-bias: add bias every cycle (matches always-on axon semantics)
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

                y_rates = node.execute_on_gathered(in_rates)
                y_rates = y_rates.view(batch_size, -1).clamp(0.0, 1.0)

                out_train = torch.zeros(T, batch_size, y_rates.shape[1], device=device)
                for cycle in range(T):
                    out_train[cycle] = self.to_spikes(y_rates, cycle)

                spike_train_cache[node.id] = out_train
            else:
                raise TypeError(f"Unknown node type: {type(node)}")

            # Free spike trains whose last consumer was this step.
            for released_id in self._release_at_step.get(node_idx, ()):
                spike_train_cache.pop(released_id, None)

        # Gather output spike *counts* (B, out_dim) by summing spikes over time.
        output_sources = list(self.output_sources.flatten())
        output_signals = torch.zeros(batch_size, len(output_sources), device=device)
        for cycle in range(T):
            # Range-based output gather (adds into output_signals)
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

    # -----------------------------------------------------------------
    # TTFS analytical helpers
    # -----------------------------------------------------------------
    def _fill_activation_from_ir_spans(
        self,
        out: torch.Tensor,
        *,
        x: torch.Tensor,
        activation_cache: Dict[int, torch.Tensor],
        spans: list[IRSourceSpan],
    ) -> None:
        """
        Fill `out` (B, N) with *activations* (not spikes) from IR source spans.

        Used by the analytical TTFS forward pass. The always-on source
        produces activation 1.0 (bias).
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
                out[:, d0:d1] = x[:, int(sp.src_start):int(sp.src_end)]
                continue
            # Node output
            out[:, d0:d1] = activation_cache[int(sp.src_node_id)][:, int(sp.src_start):int(sp.src_end)]

    def _forward_ttfs(self, x: torch.Tensor) -> torch.Tensor:
        """
        TTFS forward pass — dispatches to continuous or quantized.
        """
        x = self.preprocessor(x)
        x = x.view(x.shape[0], -1)

        if self.spiking_mode == "ttfs_quantized":
            return self._forward_ttfs_quantized(x)
        return self._forward_ttfs_continuous(x)

    # -----------------------------------------------------------------
    # TTFS continuous (analytical)
    # -----------------------------------------------------------------
    def _forward_ttfs_continuous(self, x: torch.Tensor) -> torch.Tensor:
        """
        Analytical TTFS continuous: ``clamp(relu(W @ x + b) / θ, 0, 1)`` per core.

        Outputs are clamped to [0, 1] because TTFS neurons fire at most once —
        V > θ fires immediately (rate 1), matching hardware behavior.

        Inputs are NOT clamped: weight matrices already incorporate
        ``per_input_scales`` (from ``compute_per_source_scales``) that normalize
        ComputeOp outputs to the expected range.  Clamping inputs would corrupt
        models with ComputeOp→NeuralCore paths (e.g. MLP-Mixer Identity layers).

        Single-pass over nodes in topological order.  ComputeOps are
        applied directly on activations (host-side ops preserve signed values;
        final output is read unclamped for argmax).
        """
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
                # Hardware-bias: add dedicated bias register
                if hw_bias is not None:
                    out = out + hw_bias.to(compute_dtype)

                # Apply activation: resolve from activation_type (may be compound
                # string e.g. "LeakyReLU + ClampDecorator, QuantizeDecorator").
                act_fn = _ttfs_activation_from_type(node.activation_type)
                out = act_fn(out)

                out = out / threshold

                # TTFS: a neuron fires at most once, so its output rate is in [0, 1].
                # Hardware naturally clamps (V > θ fires immediately → rate 1).
                # The analytical formula can exceed 1; clamp to match hardware.
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

    # -----------------------------------------------------------------
    # TTFS quantized (analytical closed-form)
    # -----------------------------------------------------------------
    def _forward_ttfs_quantized(self, x: torch.Tensor) -> torch.Tensor:
        """
        Analytical TTFS quantized forward pass.

        Produces the **exact** same output as the cycle-based simulation
        but in O(N_cores) — one matmul + element-wise ops per core —
        instead of O(max_latency * S * N_cores).

        For each NeuralCore::

            V = W @ x                                 (initial charge)
            k_fire = ceil(S * (1 - V / θ))            (analytical fire step)
            k_fire = clamp(k_fire, 0, S-1)
            activation = (S - k_fire) / S

        ComputeOps are applied directly on activations (same as continuous).
        """
        batch_size = x.shape[0]
        device = x.device
        S = self.simulation_length
        compute_dtype = self._compute_dtype

        # For bit-exact match to the C++ ``double`` simulator pass
        # ``compute_dtype=torch.float64`` at construction; the float32 default
        # flips ``ceil(S*(1-V/θ))`` on a handful of boundary neurons per batch
        # which is acceptable for SCM validation but not for deployment ship checks.
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
                # Hardware-bias: add dedicated bias register
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
        output_signals = torch.zeros(batch_size, len(output_sources), device=device)
        self._fill_activation_from_ir_spans(
            output_signals, x=x, activation_cache=activation_cache, spans=self._output_spans
        )

        self.total_spikes = 0.0
        return output_signals

    def _assert_mapping_contracts(self, ir_graph: IRGraph) -> None:
        """
        Verify dimension and index contracts after pruning (Contract 2).
        Fail fast with a clear error if any core has mismatched axons/neurons vs sources.
        Uses the same weight tensor the flow uses (so bank-backed cores are handled without get_output_count).
        """
        from mimarsinan.mapping.ir import IRSource

        for node in self.nodes:
            if not isinstance(node, NeuralCore):
                continue
            try:
                weight = self._get_weight(node)
            except Exception:
                continue
            # weight is (neurons, axons) for matmul
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
                # Bank-backed core without weight_row_slice: use weight shape as truth
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

    # -----------------------------------------------------------------
    # TTFS ComputeOp helper (shared by continuous + quantized)
    # -----------------------------------------------------------------
    def _execute_compute_op_ttfs(
        self,
        node: ComputeOp,
        x: torch.Tensor,
        batch_size: int,
        device: torch.device,
        activation_cache: Dict[int, torch.Tensor],
    ) -> torch.Tensor:
        """Execute a ComputeOp in activation space (TTFS modes).

        NeuralCore outputs are normalised to [0, 1] via the effective-weight
        formula (``W_eff = per_input_scales * W / activation_scale``).
        ComputeOps wrap the *original* module whose bias was never divided by
        ``activation_scale``.  To keep the bias correct we must:

        1. Rescale the gathered input from [0, 1] back to training range.
        2. Execute the module.
        3. Normalise the output back to TTFS range.

        For scale-equivariant ops (pooling, reshape, mean) the rescale is a
        no-op.  For ``nn.Linear`` (with bias) the rescale fixes a systematic
        offset of ``b * (1 - 1/s)`` per output element.
        """
        spans = self._input_spans[int(node.id)]
        in_dim = int(len(node.input_sources.flatten()))
        inp = torch.zeros(batch_size, in_dim, device=device)
        self._fill_activation_from_ir_spans(
            inp, x=x, activation_cache=activation_cache, spans=spans
        )

        in_scale = self._ttfs_node_input_scale.get(node.id, 1.0)
        out_scale = self._ttfs_node_output_scale.get(node.id, 1.0)
        if abs(in_scale - 1.0) > 1e-9:
            inp = inp * in_scale

        out = node.execute_on_gathered(inp)

        if abs(out_scale - 1.0) > 1e-9:
            out = out / out_scale

        return out

    def get_core_spike_rates(self) -> list[float]:
        """
        Get the average firing rate for each neural core.
        
        Must be called after a forward pass. Returns a list of rates (one per neural core)
        in the order they appear in the graph.
        """
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
        """
        Sync thresholds from ir_graph.nodes to the packed threshold buffer.

        Call this after modifying node.threshold directly.
        """
        for node in self.nodes:
            if isinstance(node, NeuralCore):
                t = node.threshold
                self._set_threshold(node, float(t.item()) if hasattr(t, "item") else float(t))


class StableSpikingUnifiedCoreFlow(SpikingUnifiedCoreFlow):
    """
    Stable (deterministic) version of SpikingUnifiedCoreFlow.
    
    Uses deterministic/front-loaded spike generation for consistent spike rates
    that can be used as tuning targets for the regular spiking flow.

    For TTFS mode, the stable flow is identical to the regular TTFS flow
    (since TTFS is inherently deterministic: single-spike encoding).
    """

    def __init__(
        self,
        input_shape,
        ir_graph: IRGraph,
        simulation_length: int,
        preprocessor: nn.Module | None = None,
        firing_mode: str = "Default",
        thresholding_mode: str = "<",
        spiking_mode: str = "rate",
    ):
        # Force deterministic spike mode for stability
        super().__init__(
            input_shape,
            ir_graph,
            simulation_length,
            preprocessor,
            firing_mode,
            spike_mode="Uniform",  # Uniform is deterministic and stable
            thresholding_mode=thresholding_mode,
            spiking_mode=spiking_mode,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Execute stable spiking simulation and track per-core spike counts.

        For TTFS modes, delegates to the parent TTFS forward (already deterministic).
        """
        if self.spiking_mode in self._TTFS_SPIKING_MODES:
            return self._forward_ttfs(x)
        return self._forward_stable_rate(x)

    def _forward_stable_rate(self, x: torch.Tensor) -> torch.Tensor:
        """Rate-coded stable forward pass."""
        x = self.preprocessor(x)
        x = x.view(x.shape[0], -1)

        batch_size = x.shape[0]
        device = x.device

        T = self.simulation_length

        # Generate input spike train (T, B, in)
        input_spike_train = torch.zeros(T, batch_size, x.shape[1], device=device)
        for cycle in range(T):
            input_spike_train[cycle] = self.to_spikes(x, cycle)

        spike_train_cache: Dict[int, torch.Tensor] = {}
        self._last_core_spike_counts = {}

        ops = {"<": torch.lt, "<=": torch.le}

        for node_idx, node in enumerate(self.nodes):
            if isinstance(node, NeuralCore):
                weight = self._get_weight(node).to(torch.float32)
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

                y_rates = node.execute_on_gathered(in_rates)
                y_rates = y_rates.view(batch_size, -1).clamp(0.0, 1.0)

                out_train = torch.zeros(T, batch_size, y_rates.shape[1], device=device)
                for cycle in range(T):
                    out_train[cycle] = self.to_spikes(y_rates, cycle)

                spike_train_cache[node.id] = out_train
            else:
                raise TypeError(f"Unknown node type: {type(node)}")

            for released_id in self._release_at_step.get(node_idx, ()):
                spike_train_cache.pop(released_id, None)

        # Gather output spike counts
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


