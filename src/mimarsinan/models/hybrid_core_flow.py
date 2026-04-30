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
    no time-step discretisation.  Equivalent to ``clamp(ReLU(W @ x + b) / θ, 0, 1)``.
    Outputs clamped to ``[0, 1]`` per core (hardware TTFS fires at most once).
    Inputs are not clamped; weight matrices normalize ComputeOp sources.
  * **ttfs_quantized** (analytical quantised) — closed-form computation
    that matches the cycle-based simulation exactly but runs in O(N_cores)
    instead of O(max_latency * S * N_cores).
"""

from __future__ import annotations

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


# Precision for on-chip spiking compute.
#
# The C++ nevresim simulator uses:
#   * int-weight rate-coded path: exact integer arithmetic
#     (weight_t = int, MembranePotential<weight_t> = int, spike_t = int_fast8_t).
#   * TTFS paths: IEEE-754 64-bit float (signal_t = double).
#
# PyTorch defaulting to float32 here introduces ~1e-7 roundoff per operation,
# which accumulates and — most importantly — flips ``ceil`` / threshold
# comparisons for neurons whose membrane potential lands near θ, producing
# per-sample prediction differences vs the C++ simulator.  Using float64
# matches C++'s double for TTFS and exactly represents integer sums up to
# 2**53 for rate-coded mode (far larger than any practical axon × weight
# range), restoring bit-for-bit equivalence at the cost of ~2× compute
# time on GPU.  These flows run once per pipeline so the cost is acceptable.
_COMPUTE_DTYPE = torch.float64


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
        preprocessor: nn.Module | None = None,
        firing_mode: str = "Default",
        spike_mode: str = "Uniform",
        thresholding_mode: str = "<",
        spiking_mode: str = "lif",
    ):
        super().__init__()

        self.input_shape = input_shape
        self.hybrid_mapping = hybrid_mapping
        self.simulation_length = int(simulation_length)
        self.preprocessor = preprocessor if preprocessor is not None else nn.Identity()

        self.firing_mode = firing_mode
        self.spike_mode = spike_mode
        self.thresholding_mode = thresholding_mode
        self.spiking_mode = spiking_mode

        assert firing_mode in ["Default", "Novena", "TTFS"]
        assert spike_mode in ["Stochastic", "Deterministic", "FrontLoaded", "Uniform", "TTFS"]
        assert thresholding_mode in ["<", "<="]

        # Segment tensor cache — **single-segment LRU**.
        #
        # Rationale: for ViT-scale mappings (cifar_vit = 12 neural
        # segments × ~3 GB of float64 weights each) a "cache all"
        # strategy needs 40+ GB of permanent GPU residency before any
        # activations exist — OOMs on realistic budgets.  A 1-segment
        # cache keeps active weight residency bounded to the largest
        # single segment's size.  Segments execute sequentially inside
        # a single forward so intra-segment cache hits (cycles, output
        # spans) still skip redundant uploads.
        #
        # To bound CUDA allocator fragmentation caused by evict/upload
        # round-robin across 12 segments × N batches, we also call
        # ``torch.cuda.empty_cache()`` ONCE per ``forward`` (in
        # ``_release_cuda_blocks_after_forward``) — not per-stage.  One
        # sync per batch is cheap vs. many GB of reserved-but-unused
        # blocks the allocator would otherwise hold.
        self._segment_tensor_cache: Dict[int, dict] = {}
        self._segment_tensor_cache_key: int | None = None

    def _build_consumer_counts(self) -> Dict[int, int]:
        """Return ``{node_id: number_of_downstream_reads}`` for the mapping.

        A node is consumed by:
          * every NEURAL stage's ``input_map`` entry pointing at it;
          * every COMPUTE op whose ``input_sources`` references it;
          * the final ``output_sources`` of the HybridHardCoreMapping
            (``_gather_final_output`` reads those at the end).

        The counts are used to prune ``state_buffer`` entries as soon as
        their last consumer has run — crucial for ViT-scale mappings
        where state_buffer would otherwise retain ~2400 intermediate
        compute-op tensors through the whole forward (GB of growth per
        batch within a single forward pass).
        """
        cached = getattr(self.hybrid_mapping, "_consumer_counts_cache", None)
        if cached is not None:
            return cached

        counts: Dict[int, int] = {}

        def _bump(nid: int) -> None:
            counts[nid] = counts.get(nid, 0) + 1

        for stage in self.hybrid_mapping.stages:
            if stage.kind == "neural":
                for s in stage.input_map:
                    if s.node_id is not None and int(s.node_id) >= 0:
                        _bump(int(s.node_id))
            elif stage.kind == "compute":
                op = stage.compute_op
                assert op is not None
                for src in op.input_sources.flatten():
                    if isinstance(src, IRSource) and src.node_id >= 0:
                        _bump(int(src.node_id))

        for src in self.hybrid_mapping.output_sources.flatten():
            if isinstance(src, IRSource) and src.node_id >= 0:
                _bump(int(src.node_id))

        try:
            self.hybrid_mapping._consumer_counts_cache = counts
        except (AttributeError, TypeError):
            pass
        return counts

    @staticmethod
    def _decref_consumers(
        state_buffer: Dict[int, torch.Tensor],
        remaining: Dict[int, int],
        src_ids,
    ) -> None:
        """Decrement ``remaining[nid]`` for each source; drop state_buffer
        entries whose refcount hits 0.  ``src_ids`` is an iterable of
        already-resolved node_ids (ints).
        """
        for nid in src_ids:
            if nid < 0:
                continue
            r = remaining.get(nid)
            if r is None:
                continue
            r -= 1
            if r <= 0:
                remaining.pop(nid, None)
                state_buffer.pop(nid, None)
            else:
                remaining[nid] = r

    def _evict_segment_cache(self) -> None:
        """Drop the currently cached segment's GPU tensors.

        Explicitly clears every list / dict in the cached payload so no
        stale tensor refs keep CUDA blocks pinned.  Does NOT call
        ``empty_cache`` — that's deferred to one call per forward.
        """
        prev_key = self._segment_tensor_cache_key
        if prev_key is None:
            return
        prev = self._segment_tensor_cache.pop(prev_key, None)
        self._segment_tensor_cache_key = None
        if prev is None:
            return
        for k in ("core_params", "hw_biases", "thresholds"):
            v = prev.get(k)
            if v is not None:
                v.clear()
        bt = prev.get("bank_tensors")
        if bt is not None:
            bt.clear()
        prev.clear()

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
        inp = torch.zeros(batch_size, total_size, device=device, dtype=_COMPUTE_DTYPE)
        for s in input_map:
            buf = state_buffer[s.node_id]
            inp[:, s.offset : s.offset + s.size] = buf[:, :s.size].to(_COMPUTE_DTYPE)
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
        out = torch.zeros(batch_size, len(output_sources), device=device, dtype=_COMPUTE_DTYPE)
        for idx, src in enumerate(output_sources):
            if not isinstance(src, IRSource):
                continue
            if src.is_off():
                continue
            elif src.is_input():
                out[:, idx] = original_input[:, src.index].to(_COMPUTE_DTYPE)
            elif src.is_always_on():
                out[:, idx] = 1.0
            else:
                out[:, idx] = state_buffer[src.node_id][:, src.index].to(_COMPUTE_DTYPE)
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

    def _get_segment_tensors(self, stage: HybridStage, device: torch.device) -> dict:
        """Return cached (axon_spans, output_spans, core_params, thresholds,
        hw_biases) for ``stage``; building and uploading to device on
        first access.  Weights are immutable post-build so this cache is
        safe — the huge CPU→GPU transfer is paid once per stage, not per
        batch.
        """
        mapping = stage.hard_core_mapping
        assert mapping is not None
        key = id(stage)
        cached = self._segment_tensor_cache.get(key)
        if cached is not None and cached.get("device") == device:
            return cached

        # Evict prior segment before uploading this one — keeps GPU
        # residency bounded to one segment's weights (see __init__).
        self._evict_segment_cache()

        cores = mapping.cores
        output_sources = mapping.output_sources
        weight_banks = getattr(mapping, "weight_banks", None) or {}
        placements_per_core = getattr(
            mapping, "soft_core_placements_per_hard_core", []
        )

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

        # --- Per-hw-core weight resolution -------------------------------
        # Goal: ONE matmul per hw core per cycle (the pre-refactor fast
        # path), plus bank dedup across hw cores when possible.
        #
        # Strategy per hw core:
        #   * Zero-copy bank view: when the hw core's placements form a
        #     single bank-backed rectangle that exactly covers the
        #     occupied (used_ax × used_neu) space, we reference the bank
        #     tensor via a view (no CPU→GPU copy beyond the bank itself).
        #     Multiple hw cores pointing at the same bank share memory.
        #   * Dense tile: otherwise upload ``core.core_matrix[:used_ax, :used_neu].T``
        #     as a single tensor — identical to the pre-refactor path.
        #
        # Per-placement ``_accumulate_placements_into`` is kept ONLY for
        # the rare multi-bank-per-hw-core case (never observed in
        # practice but supported for completeness).  Falls back to the
        # blitted dense matrix (safe + simple) instead.
        bank_tensors: dict[int, torch.Tensor] = {}

        def _ensure_bank_tensor(bid: int) -> torch.Tensor:
            t = bank_tensors.get(bid)
            if t is not None:
                return t
            bank_mat = weight_banks.get(int(bid))
            if bank_mat is None:
                raise KeyError(
                    f"HardCoreMapping references bank_id={bid} but "
                    f"mapping.weight_banks does not contain it — "
                    f"ir_graph_to_soft_core_mapping must propagate the bank."
                )
            # Upload once in (out_features, in_features).  Slicing then
            # yields a (neurons, axons) matmul-ready matrix.
            t = torch.tensor(bank_mat.T, dtype=_COMPUTE_DTYPE, device=device)
            bank_tensors[int(bid)] = t
            return t

        core_params: list[torch.Tensor] = []
        hw_biases = []
        for core_idx, core in enumerate(cores):
            used_ax = max(int(core.axons_per_core - core.available_axons), 1)
            used_neu = max(int(core.neurons_per_core - core.available_neurons), 1)

            placement_dicts = (
                placements_per_core[core_idx]
                if core_idx < len(placements_per_core)
                else []
            )

            core_weight: torch.Tensor | None = None

            # Fast-path: a single bank-backed placement that exactly
            # covers the hw core's occupied rectangle → zero-copy view.
            if len(placement_dicts) == 1:
                pd = placement_dicts[0]
                bid = pd.get("weight_bank_id")
                ao = int(pd.get("axon_offset", 0))
                ne_off = int(pd.get("neuron_offset", 0))
                a = int(pd.get("axons", 0))
                n = int(pd.get("neurons", 0))
                if (
                    bid is not None
                    and ao == 0 and ne_off == 0
                    and a == used_ax and n == used_neu
                ):
                    bank_t = _ensure_bank_tensor(int(bid))
                    ba0, ba1 = pd.get("bank_axon_range") or (0, a)
                    bn0, bn1 = pd.get("bank_neuron_range") or (0, n)
                    # (neurons, axons) view — no copy.
                    core_weight = bank_t[int(bn0):int(bn1), int(ba0):int(ba1)]

            # Default path: upload the blitted dense matrix slice — same
            # single-tensor single-matmul behaviour as pre-refactor.
            if core_weight is None:
                tile = core.core_matrix[:used_ax, :used_neu]
                core_weight = torch.tensor(
                    tile.T, dtype=_COMPUTE_DTYPE, device=device,
                )

            core_params.append(core_weight)

            bias = getattr(core, "hardware_bias", None)
            if bias is None:
                hw_biases.append(None)
            else:
                hw_biases.append(
                    torch.tensor(
                        bias[:used_neu], dtype=_COMPUTE_DTYPE, device=device,
                    )
                )

        thresholds = [
            torch.tensor(float(core.threshold), dtype=_COMPUTE_DTYPE, device=device)
            for core in cores
        ]

        cached = dict(
            device=device,
            cores=cores,
            output_sources=output_sources,
            axon_spans=axon_spans,
            output_spans=output_spans,
            bank_tensors=bank_tensors,
            core_params=core_params,
            thresholds=thresholds,
            hw_biases=hw_biases,
        )
        self._segment_tensor_cache[key] = cached
        self._segment_tensor_cache_key = key
        return cached

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

        seg = self._get_segment_tensors(stage, device)
        cores = seg["cores"]
        output_sources = seg["output_sources"]
        axon_spans = seg["axon_spans"]
        output_spans = seg["output_spans"]
        core_params = seg["core_params"]
        thresholds = seg["thresholds"]
        hw_biases = seg["hw_biases"]

        ops = {"<": torch.lt, "<=": torch.le}

        # Buffers / memb / input_signals sized to each core's occupied
        # rectangle (frozen post-packing; read directly from the core).
        buffers = [
            torch.zeros(batch_size, max(int(c.neurons_per_core - c.available_neurons), 1),
                        device=device, dtype=_COMPUTE_DTYPE)
            for c in cores
        ]
        memb = [
            torch.zeros(batch_size, max(int(c.neurons_per_core - c.available_neurons), 1),
                        device=device, dtype=_COMPUTE_DTYPE)
            for c in cores
        ]

        output_counts = torch.zeros(batch_size, len(output_sources), device=device, dtype=_COMPUTE_DTYPE)

        zeros_in = torch.zeros(batch_size, input_size, device=device, dtype=_COMPUTE_DTYPE)
        input_signals = [
            torch.zeros(batch_size, max(int(c.axons_per_core - c.available_axons), 1),
                        device=device, dtype=_COMPUTE_DTYPE)
            for c in cores
        ]

        # Spike train arrives as float32 from spike generators; cast to compute dtype once.
        input_spike_train = input_spike_train.to(_COMPUTE_DTYPE)

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
                # Hardware-bias: add bias every cycle (matches always-on axon semantics).
                if hw_biases[core_idx] is not None:
                    memb_i += hw_biases[core_idx]

                fired = ops[self.thresholding_mode](thresholds[core_idx], memb_i)
                buffers[core_idx] = fired.to(_COMPUTE_DTYPE)

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
        ``clamp(relu(W @ x + b) / θ, 0, 1)`` per core.  Outputs clamped to
        [0, 1] to match hardware TTFS.  Inputs are NOT clamped; weight matrices
        incorporate ``per_input_scales`` normalization for ComputeOp sources.

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

        seg = self._get_segment_tensors(stage, device)
        cores = seg["cores"]
        output_sources = seg["output_sources"]
        axon_spans = seg["axon_spans"]
        output_spans = seg["output_spans"]
        core_params = seg["core_params"]
        thresholds = seg["thresholds"]
        hw_biases = seg["hw_biases"]

        # Cast inputs to compute dtype to match C++ double precision.
        input_activations = input_activations.to(_COMPUTE_DTYPE)

        # Buffers / input_signals sized to each core's occupied rectangle
        # (frozen post-packing; read directly from the core).
        buffers = [
            torch.zeros(batch_size, max(int(c.neurons_per_core - c.available_neurons), 1),
                        device=device, dtype=_COMPUTE_DTYPE)
            for c in cores
        ]
        input_signals = [
            torch.zeros(batch_size, max(int(c.axons_per_core - c.available_axons), 1),
                        device=device, dtype=_COMPUTE_DTYPE)
            for c in cores
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
            # Hardware-bias: add dedicated bias register.
            if hw_biases[ci] is not None:
                V = V + hw_biases[ci]

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
                # TTFS: a neuron fires at most once → output rate ∈ [0, 1].
                # Hardware naturally clamps (V > θ fires immediately → rate 1).
                buffers[ci] = (out / thresholds[ci]).clamp(0.0, 1.0)

        output = torch.zeros(batch_size, len(output_sources), device=device, dtype=_COMPUTE_DTYPE)
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
        try:
            x = self.preprocessor(x)
            x = x.view(x.shape[0], -1)

            if self.spiking_mode in self._TTFS_SPIKING_MODES:
                return self._forward_ttfs(x)

            return self._forward_rate(x)
        finally:
            # Evict the segment cached after the last stage AND release
            # per-forward transient blocks back to the driver.  Without
            # this, reserved CUDA memory climbs across batches (segments
            # and per-forward buffers churn through allocator blocks of
            # varying sizes, fragmenting the pool).  One sync per batch
            # is cheap vs. the alternative (tens of GB of reserved-but-
            # unused memory accumulating).
            self._evict_segment_cache()
            if isinstance(x, torch.Tensor) and x.is_cuda:
                torch.cuda.empty_cache()

    # ---------------------------------------------------------------------
    # TTFS forward (state-buffer driven)
    # ---------------------------------------------------------------------
    def _forward_ttfs(self, x: torch.Tensor) -> torch.Tensor:
        """TTFS forward pass using the global state buffer.

        ComputeOps receive TTFS-normalised inputs ([0, 1]) from the state
        buffer but wrap modules whose bias was never divided by
        activation_scale.  We rescale around execution using
        ``node_activation_scales`` — same fix as SpikingUnifiedCoreFlow.
        """
        T = self.simulation_length
        batch_size = x.shape[0]
        device = x.device
        quantized = self.spiking_mode == "ttfs_quantized"

        # Cast input to compute dtype so the state buffer is uniformly typed.
        x_compute = x.to(_COMPUTE_DTYPE)
        state_buffer: Dict[int, torch.Tensor] = {-2: x_compute}
        out_scales = getattr(self.hybrid_mapping, "node_activation_scales", {})
        in_scales = getattr(self.hybrid_mapping, "node_input_activation_scales", out_scales)

        # Pre-computed downstream-consumer counts for every node_id.
        # During the forward we decrement per consumption and drop
        # state_buffer entries whose refcount hits 0 so intermediate
        # tensors die as soon as possible — prevents GB-scale growth
        # within a single forward on long compute-op chains (ViT).
        remaining = dict(self._build_consumer_counts())

        for stage in self.hybrid_mapping.stages:
            if stage.kind == "neural":
                seg_input = self._assemble_segment_input(
                    stage.input_map, state_buffer, batch_size, device
                )
                seg_output = self._run_neural_segment_ttfs(
                    stage, input_activations=seg_input, quantized=quantized
                )
                self._store_segment_output(stage.output_map, state_buffer, seg_output)
                # Release every state_buffer entry that fed this segment
                # and has no remaining consumers downstream.
                self._decref_consumers(
                    state_buffer, remaining,
                    (int(s.node_id) for s in stage.input_map),
                )

            elif stage.kind == "compute":
                op = stage.compute_op
                assert op is not None
                in_scale = in_scales.get(op.id, 1.0)
                out_scale = out_scales.get(op.id, 1.0)
                # Host-side compute ops execute a Python torch module.  C++
                # SimulationRunner._execute_compute_op_np runs them in float32;
                # match that here so PyTorch and C++ agree bit-for-bit.
                #
                # Memory: never materialise a float32 copy of the entire
                # state_buffer — ``gather_inputs`` only reads the specific
                # sources this op needs, so casting the gathered result
                # (O(op_inputs)) is sufficient.
                gathered = op.gather_inputs(x, state_buffer)
                gathered = gathered.to(torch.float32)
                if abs(in_scale - 1.0) > 1e-9:
                    gathered = gathered * in_scale
                result = op.execute_on_gathered(gathered)
                if abs(out_scale - 1.0) > 1e-9:
                    result = result / out_scale
                state_buffer[op.id] = result.to(_COMPUTE_DTYPE)
                # Drop any source whose last consumer was this op.
                self._decref_consumers(
                    state_buffer, remaining,
                    (int(src.node_id) for src in op.input_sources.flatten()
                     if isinstance(src, IRSource) and src.node_id >= 0),
                )

            else:
                raise ValueError(f"Invalid hybrid stage kind: {stage.kind}")

        final = self._gather_final_output(state_buffer, x_compute, batch_size, device)
        return final.to(torch.float32) * float(T)

    # ---------------------------------------------------------------------
    # Rate-coded forward (state-buffer driven)
    # ---------------------------------------------------------------------
    def _forward_rate(self, x: torch.Tensor) -> torch.Tensor:
        """Rate-coded forward pass using the global state buffer."""
        batch_size = x.shape[0]
        device = x.device
        T = self.simulation_length

        x_compute = x.to(_COMPUTE_DTYPE)
        state_buffer: Dict[int, torch.Tensor] = {-2: x_compute}
        out_scales = getattr(self.hybrid_mapping, "node_activation_scales", {})
        in_scales = getattr(self.hybrid_mapping, "node_input_activation_scales", out_scales)

        # See _forward_ttfs for the rationale; same refcount-prune
        # strategy prevents state_buffer growth within a forward.
        remaining = dict(self._build_consumer_counts())

        for stage in self.hybrid_mapping.stages:
            if stage.kind == "neural":
                seg_input_rates = self._assemble_segment_input(
                    stage.input_map, state_buffer, batch_size, device
                )
                seg_input_rates_clamped = seg_input_rates.clamp(0.0, 1.0)
                spike_train = torch.zeros(
                    T, batch_size, seg_input_rates_clamped.shape[1], device=device,
                    dtype=_COMPUTE_DTYPE,
                )
                for cycle in range(T):
                    spike_train[cycle] = self.to_spikes(seg_input_rates_clamped, cycle).to(_COMPUTE_DTYPE)

                counts = self._run_neural_segment_rate(
                    stage, input_spike_train=spike_train
                )
                seg_output_rates = counts / float(T)
                self._store_segment_output(stage.output_map, state_buffer, seg_output_rates)
                self._decref_consumers(
                    state_buffer, remaining,
                    (int(s.node_id) for s in stage.input_map),
                )

            elif stage.kind == "compute":
                op = stage.compute_op
                assert op is not None
                in_scale = in_scales.get(op.id, 1.0)
                out_scale = out_scales.get(op.id, 1.0)
                # Match TTFS path: rescale inputs into training range before
                # running the host module, and rescale outputs back into
                # [0, 1] rate range so the next neural segment's input
                # clamp is a no-op (not a data-loss truncation).
                gathered = op.gather_inputs(x, state_buffer)
                gathered = gathered.to(torch.float32)
                if abs(in_scale - 1.0) > 1e-9:
                    gathered = gathered * in_scale
                result = op.execute_on_gathered(gathered)
                if abs(out_scale - 1.0) > 1e-9:
                    result = result / out_scale
                state_buffer[op.id] = result.to(_COMPUTE_DTYPE)
                self._decref_consumers(
                    state_buffer, remaining,
                    (int(src.node_id) for src in op.input_sources.flatten()
                     if isinstance(src, IRSource) and src.node_id >= 0),
                )

            else:
                raise ValueError(f"Invalid hybrid stage kind: {stage.kind}")

        final = self._gather_final_output(state_buffer, x_compute, batch_size, device)
        return final.to(torch.float32) * float(T)
