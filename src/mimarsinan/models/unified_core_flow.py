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

  * **ttfs** (continuous) — exact analytical ``ReLU(W @ x + b) / θ``.
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

import torch
import torch.nn as nn
import torch.nn.functional as F

from mimarsinan.mapping.ir import ComputeOp, IRGraph, NeuralCore, WeightBank
from mimarsinan.mapping.ir_source_spans import IRSourceSpan, compress_ir_sources


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
        preprocessor: nn.Module,
        firing_mode: str = "Default",
        spike_mode: str = "Uniform",
        thresholding_mode: str = "<",
        spiking_mode: str = "rate",
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
        self.spiking_mode = spiking_mode

        assert firing_mode in ["Default", "Novena", "TTFS"]
        assert spike_mode in ["Stochastic", "Deterministic", "FrontLoaded", "Uniform", "TTFS"]
        assert thresholding_mode in ["<", "<="]

        # --- Weight bank parameters (shared across many cores) ----------
        self._bank_params = nn.ParameterDict()
        for bank_id, bank in ir_graph.weight_banks.items():
            # Stored as (neurons, axons) for matmul convenience
            w = torch.tensor(bank.core_matrix.T, dtype=torch.float32)
            self._bank_params[str(bank_id)] = nn.Parameter(w, requires_grad=False)

        # --- Per-core parameters (only for cores that OWN their weights) -
        self.neural_core_params = nn.ParameterList()
        self.thresholds = nn.ParameterList()
        self.neural_core_ids = []

        self._id_to_param_idx: Dict[int, int] = {}
        self._id_to_bank: Dict[int, tuple[str, tuple[int, int] | None]] = {}

        for node in self.nodes:
            if isinstance(node, NeuralCore):
                self.neural_core_ids.append(node.id)
                self.thresholds.append(
                    nn.Parameter(torch.tensor(node.threshold, dtype=torch.float32), requires_grad=False)
                )
                thresh_idx = len(self.thresholds) - 1

                if node.has_weight_bank():
                    self._id_to_bank[node.id] = (
                        str(node.weight_bank_id),
                        node.weight_row_slice,
                    )
                    self._id_to_param_idx[node.id] = thresh_idx
                else:
                    weight = torch.tensor(node.core_matrix.T, dtype=torch.float32)
                    self.neural_core_params.append(nn.Parameter(weight, requires_grad=False))
                    owned_idx = len(self.neural_core_params) - 1
                    self._id_to_param_idx[node.id] = thresh_idx
                    self._id_to_bank[node.id] = None  # type: ignore[assignment]

        # Build fast lookup: node_id -> (owned_param_index | None)
        self._id_to_owned_param: Dict[int, int] = {}
        owned_counter = 0
        for node in self.nodes:
            if isinstance(node, NeuralCore) and not node.has_weight_bank():
                self._id_to_owned_param[node.id] = owned_counter
                owned_counter += 1

        # Build threshold index cache (sequential order of neural cores)
        self._threshold_idx_cache: Dict[int, int] = {}
        thresh_counter = 0
        for node in self.nodes:
            if isinstance(node, NeuralCore):
                self._threshold_idx_cache[node.id] = thresh_counter
                thresh_counter += 1

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

        For owned-weight cores this indexes ``neural_core_params``.
        For bank-backed cores this slices from ``_bank_params``.
        """
        bank_info = self._id_to_bank.get(node.id)
        if bank_info is not None:
            bank_key, row_slice = bank_info
            w = self._bank_params[bank_key]  # (neurons_full, axons)
            if row_slice is not None:
                start, end = row_slice
                w = w[start:end, :]
            return w

        owned_idx = self._id_to_owned_param[node.id]
        return self.neural_core_params[owned_idx]

    def _get_threshold_idx(self, node: NeuralCore) -> int:
        """Return the index into ``self.thresholds`` for *node*."""
        # Thresholds are appended in node order; use a sequential scan cache.
        return self._threshold_idx_cache[node.id]

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

        for node in self.nodes:
            if isinstance(node, NeuralCore):
                weight = self._get_weight(node)  # (neurons, axons)
                t_idx = self._get_threshold_idx(node)
                threshold = self.thresholds[t_idx]

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

                    memb += torch.matmul(weight, inp.T).T
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
        Exact analytical TTFS: ``relu(W @ x + b) / θ`` per core.

        Single-pass over nodes in topological order.  ComputeOps are
        applied directly on activations.
        """
        batch_size = x.shape[0]
        device = x.device

        activation_cache: Dict[int, torch.Tensor] = {}
        self._last_core_spike_counts: Dict[int, float] = {}

        for node in self.nodes:
            if isinstance(node, NeuralCore):
                weight = self._get_weight(node)
                t_idx = self._get_threshold_idx(node)
                threshold = self.thresholds[t_idx]

                spans = self._input_spans[int(node.id)]
                in_dim = int(len(node.input_sources.flatten()))
                inp = torch.zeros(batch_size, in_dim, device=device)
                self._fill_activation_from_ir_spans(
                    inp, x=x, activation_cache=activation_cache, spans=spans
                )

                out = torch.matmul(weight, inp.T).T
                out = F.relu(out)
                out = out / threshold

                activation_cache[node.id] = out
                self._last_core_spike_counts[node.id] = 0.0

            elif isinstance(node, ComputeOp):
                activation_cache[node.id] = self._execute_compute_op_ttfs(
                    node, x, batch_size, device, activation_cache
                )
            else:
                raise TypeError(f"Unknown node type: {type(node)}")

        output_sources = list(self.output_sources.flatten())
        output_signals = torch.zeros(batch_size, len(output_sources), device=device)
        self._fill_activation_from_ir_spans(
            output_signals, x=x, activation_cache=activation_cache, spans=self._output_spans
        )

        self.total_spikes = 0.0
        return output_signals

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

        activation_cache: Dict[int, torch.Tensor] = {}
        self._last_core_spike_counts: Dict[int, float] = {}

        for node in self.nodes:
            if isinstance(node, NeuralCore):
                weight = self._get_weight(node)
                t_idx = self._get_threshold_idx(node)
                threshold = self.thresholds[t_idx]

                spans = self._input_spans[int(node.id)]
                in_dim = int(len(node.input_sources.flatten()))
                inp = torch.zeros(batch_size, in_dim, device=device)
                self._fill_activation_from_ir_spans(
                    inp, x=x, activation_cache=activation_cache, spans=spans
                )

                V = torch.matmul(weight, inp.T).T
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

        output_sources = list(self.output_sources.flatten())
        output_signals = torch.zeros(batch_size, len(output_sources), device=device)
        self._fill_activation_from_ir_spans(
            output_signals, x=x, activation_cache=activation_cache, spans=self._output_spans
        )

        self.total_spikes = 0.0
        return output_signals

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
        """Execute a ComputeOp in activation space (TTFS modes)."""
        spans = self._input_spans[int(node.id)]
        in_dim = int(len(node.input_sources.flatten()))
        inp = torch.zeros(batch_size, in_dim, device=device)
        self._fill_activation_from_ir_spans(
            inp, x=x, activation_cache=activation_cache, spans=spans
        )
        return node.execute_on_gathered(inp)

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
        Sync thresholds from ir_graph.nodes to the registered parameters.
        
        Call this after modifying node.threshold directly.
        """
        for node in self.nodes:
            if isinstance(node, NeuralCore):
                t_idx = self._get_threshold_idx(node)
                self.thresholds[t_idx].data.fill_(float(node.threshold))


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
        preprocessor: nn.Module,
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

        for node in self.nodes:
            if isinstance(node, NeuralCore):
                weight = self._get_weight(node)
                t_idx = self._get_threshold_idx(node)
                threshold = self.thresholds[t_idx]

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

                    memb += torch.matmul(weight, inp.T).T
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


