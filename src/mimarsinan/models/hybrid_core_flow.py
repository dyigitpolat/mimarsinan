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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mimarsinan.chip_simulation.hybrid_execution import (
    assemble_segment_input_torch,
    decref_consumers,
    execute_compute_op_torch,
    gather_final_output_torch,
    store_segment_output_torch,
)
from mimarsinan.chip_simulation.spike_recorder import (
    CoreSpikeCounts,
    RunRecord,
    SegmentSpikeRecord,
)
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

        # Set by ``forward_with_recording``; consumed by ``_forward_rate``
        # and ``_run_neural_segment_rate`` to populate per-segment /
        # per-core spike-count records for HCM↔Loihi parity diffing.
        # ``None`` (the normal case) makes recording a single-branch
        # check that has no measurable cost on the hot path.
        self._recorder: RunRecord | None = None

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
        decref_consumers(state_buffer, remaining, src_ids)

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
    # State-buffer helpers (thin wrappers around hybrid_execution)
    # ---------------------------------------------------------------------
    @staticmethod
    def _assemble_segment_input(
        input_map: list[SegmentIOSlice],
        state_buffer: Dict[int, torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        return assemble_segment_input_torch(
            input_map, state_buffer, batch_size, device, _COMPUTE_DTYPE,
        )

    @staticmethod
    def _store_segment_output(
        output_map: list[SegmentIOSlice],
        state_buffer: Dict[int, torch.Tensor],
        output_tensor: torch.Tensor,
    ) -> None:
        store_segment_output_torch(output_map, state_buffer, output_tensor)

    def _gather_final_output(
        self,
        state_buffer: Dict[int, torch.Tensor],
        original_input: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        return gather_final_output_torch(
            self.hybrid_mapping.output_sources,
            state_buffer,
            original_input,
            batch_size,
            device,
            _COMPUTE_DTYPE,
        )

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
        recorder_seg: SegmentSpikeRecord | None = None,
    ) -> torch.Tensor:
        """Rate-coded neural segment execution.  Returns spike counts (B, out_dim).

        When ``recorder_seg`` is provided, per-core integer spike counts
        are accumulated during the cycle loop and appended to
        ``recorder_seg.cores`` after execution.  Counts are summed only
        over each core's *active* window ``[core.latency, core.latency + T)``
        so a deep core's "warmup" cycles (when its membrane isn't yet
        being updated) don't inflate its input total.
        """
        mapping = stage.hard_core_mapping
        assert mapping is not None

        T = self.simulation_length
        assert input_spike_train.shape[0] == T

        batch_size = input_spike_train.shape[1]
        device = input_spike_train.device
        input_size = input_spike_train.shape[2]
        # Per-core counts only support B=1; the parity harness asserts
        # this in ``forward_with_recording``.
        recording = recorder_seg is not None
        if recording:
            assert batch_size == 1, "Spike recording requires batch_size == 1"

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

        # Per-core spike-count accumulators.  Allocated only when
        # recording so the non-recording forward path stays untouched.
        record_in_t: list[torch.Tensor] | None = None
        record_out_t: list[torch.Tensor] | None = None
        if recording:
            record_in_t = [
                torch.zeros(max(int(c.axons_per_core - c.available_axons), 1),
                            device=device, dtype=torch.int64)
                for c in cores
            ]
            record_out_t = [
                torch.zeros(max(int(c.neurons_per_core - c.available_neurons), 1),
                            device=device, dtype=torch.int64)
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

                if recording:
                    # Accumulate counts only on cycles when the core was
                    # actually integrating — matches the cycle window
                    # Loihi's reset_offset reproduces.
                    record_in_t[core_idx] += input_signals[core_idx][0].to(torch.int64)
                    record_out_t[core_idx] += buffers[core_idx][0].to(torch.int64)

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

        if recording:
            for core_idx, core in enumerate(cores):
                axon_span_list = axon_spans[core_idx]
                n_always_on = sum(
                    int(sp.length) for sp in axon_span_list if sp.kind == "on"
                )
                recorder_seg.cores.append(
                    CoreSpikeCounts(
                        core_index=core_idx,
                        n_in_used=max(int(core.axons_per_core - core.available_axons), 1),
                        n_out_used=max(int(core.neurons_per_core - core.available_neurons), 1),
                        core_latency=int(core.latency) if core.latency is not None else -1,
                        has_hardware_bias=getattr(core, "hardware_bias", None) is not None,
                        n_always_on_axons=n_always_on,
                        input_spike_count=record_in_t[core_idx].cpu().numpy().astype(np.int64),
                        output_spike_count=record_out_t[core_idx].cpu().numpy().astype(np.int64),
                    )
                )

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
                state_buffer[op.id] = execute_compute_op_torch(
                    op,
                    x,
                    state_buffer,
                    in_scale=in_scales.get(op.id, 1.0),
                    out_scale=out_scales.get(op.id, 1.0),
                    output_dtype=_COMPUTE_DTYPE,
                )
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
    # Encoding-layer spike-train extraction
    # ---------------------------------------------------------------------
    @staticmethod
    def _resolve_lif_perceptron(module):
        """Return the inner ``Perceptron`` with LIFActivation, or None.

        Encoding-layer ComputeOps wrap either a Perceptron directly or a
        Mapper that holds one (e.g. ``PerceptronMapper``,
        ``Conv2DPerceptronMapper``). In LIF mode, when the wrapped
        Perceptron's activation is a ``LIFActivation``, we can ask it to
        emit its actual ``(T, B, ...)`` spike train instead of a mean
        rate — preserving the cycle-accurate spike timing through the
        compute boundary.
        """
        from mimarsinan.models.activations import LIFActivation

        if module is None:
            return None
        candidate = getattr(module, "perceptron", module)
        activation = getattr(candidate, "activation", None)
        if isinstance(activation, LIFActivation) and hasattr(candidate, "forward_spiking"):
            return candidate
        return None

    def _build_segment_input_spike_train(
        self,
        stage,
        seg_input_rates_clamped: torch.Tensor,
        state_buffer_spikes: Dict[int, torch.Tensor],
        *,
        T: int,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Assemble a ``(T, B, in_size)`` spike train for a neural segment.

        For each ``SegmentIOSlice`` in ``stage.input_map``: when the
        producing node already has a spike train cached
        (``state_buffer_spikes`` — populated by encoding-layer compute
        ops with LIFActivation), we slice that train directly so the
        downstream segment integrates the cycle-accurate LIF firing
        pattern. Otherwise we fall back to ``to_uniform_spikes`` on the
        rate, which is the long-standing behaviour for sources that
        don't carry a spike train (raw input -2, NeuralCore outputs that
        already went through a rate boundary, generic ComputeOp results).
        """
        in_size = seg_input_rates_clamped.shape[1]
        spike_train = torch.zeros(
            T, batch_size, in_size, device=device, dtype=_COMPUTE_DTYPE,
        )
        # First pass: write spike-train slices from any source that has one.
        # We track which destination ranges were filled this way so we can
        # skip the uniform-encoding fill below.
        filled_ranges: list[tuple[int, int]] = []
        for s in stage.input_map:
            train = state_buffer_spikes.get(s.node_id)
            if train is None:
                continue
            spike_train[:, :, s.offset : s.offset + s.size] = (
                train[:, :, : s.size].to(_COMPUTE_DTYPE)
            )
            filled_ranges.append((s.offset, s.offset + s.size))

        if not filled_ranges:
            for cycle in range(T):
                spike_train[cycle] = self.to_spikes(
                    seg_input_rates_clamped, cycle,
                ).to(_COMPUTE_DTYPE)
            return spike_train

        # Mixed sources: uniform-encode the rate, then overwrite with
        # any LIF spike-train ranges captured above. (Cheaper than the
        # alternative of generating uniform spikes column-by-column.)
        encoded = torch.zeros_like(spike_train)
        for cycle in range(T):
            encoded[cycle] = self.to_spikes(
                seg_input_rates_clamped, cycle,
            ).to(_COMPUTE_DTYPE)
        for lo, hi in filled_ranges:
            encoded[:, :, lo:hi] = spike_train[:, :, lo:hi]
        return encoded

    def _try_emit_encoding_spike_train(self, op, x: torch.Tensor) -> torch.Tensor | None:
        """When ``op`` is an encoding-layer Perceptron with LIFActivation,
        run its ``forward_spiking`` on ``op.gather_inputs(x, ...)`` (raw
        input only — encoding layers source from -2 by definition) and
        return the ``(T, B, D)`` spike train. Returns ``None`` otherwise.
        """
        if self.spiking_mode != "lif":
            return None
        module = (op.params or {}).get("module") if hasattr(op, "params") else None
        perceptron = self._resolve_lif_perceptron(module)
        if perceptron is None:
            return None
        # Encoding layers have all-raw-input sources (compute-op invariant
        # from ``compute_per_source_scales``). Gather + reshape to the
        # module's expected input layout the same way ``execute`` does.
        gathered = op.gather_inputs(x, {})
        if op.input_shape is not None:
            gathered = gathered.view(gathered.shape[0], *op.input_shape)
        spikes = perceptron.forward_spiking(gathered)
        # Spikes are produced under the LIF effective-weight formulation
        # (``Linear / activation_scale``). They're already in {0, 1}; no
        # extra division by activation_scale is needed for the spike
        # train. Flatten any spatial dims so downstream segment input
        # gather can slice into a 1-D feature axis.
        T_ax = spikes.shape[0]
        B = spikes.shape[1]
        spikes = spikes.reshape(T_ax, B, -1)
        return spikes

    # ---------------------------------------------------------------------
    # Rate-coded forward (state-buffer driven)
    # ---------------------------------------------------------------------
    def _forward_rate(self, x: torch.Tensor) -> torch.Tensor:
        """Rate-coded forward pass using the global state buffer.

        Encoding-layer ComputeOps wrapping a Perceptron with
        ``LIFActivation`` produce a real ``(T, B, D)`` spike train via
        :meth:`_try_emit_encoding_spike_train`. The next neural segment
        consumes that spike train verbatim through ``state_buffer_spikes``
        instead of re-encoding the rate uniformly — preserving the LIF
        spike-timing phase that ``to_uniform_spikes`` would otherwise
        overwrite.
        """
        batch_size = x.shape[0]
        device = x.device
        T = self.simulation_length

        x_compute = x.to(_COMPUTE_DTYPE)
        state_buffer: Dict[int, torch.Tensor] = {-2: x_compute}
        # Parallel buffer of spike trains for entries produced by encoding
        # layers; consumed by the next neural-segment input assembly.
        state_buffer_spikes: Dict[int, torch.Tensor] = {}
        out_scales = getattr(self.hybrid_mapping, "node_activation_scales", {})
        in_scales = getattr(self.hybrid_mapping, "node_input_activation_scales", out_scales)

        # See _forward_ttfs for the rationale; same refcount-prune
        # strategy prevents state_buffer growth within a forward.
        remaining = dict(self._build_consumer_counts())

        for stage_index, stage in enumerate(self.hybrid_mapping.stages):
            if stage.kind == "neural":
                seg_input_rates = self._assemble_segment_input(
                    stage.input_map, state_buffer, batch_size, device
                )
                seg_input_rates_clamped = seg_input_rates.clamp(0.0, 1.0)
                spike_train = self._build_segment_input_spike_train(
                    stage,
                    seg_input_rates_clamped,
                    state_buffer_spikes,
                    T=T,
                    batch_size=batch_size,
                    device=device,
                )

                # Build a recording slot for this neural stage when the
                # parity harness has installed a recorder.  ``cores`` is
                # populated inside ``_run_neural_segment_rate``.
                recorder_seg: SegmentSpikeRecord | None = None
                if self._recorder is not None:
                    recorder_seg = SegmentSpikeRecord(
                        stage_index=stage_index,
                        stage_name=stage.name,
                        schedule_segment_index=stage.schedule_segment_index,
                        schedule_pass_index=stage.schedule_pass_index,
                        seg_input_rates=seg_input_rates_clamped[0]
                            .detach().to(torch.float32).cpu().numpy().reshape(1, -1),
                        seg_input_spike_count=spike_train.sum(dim=0)[0]
                            .to(torch.int64).cpu().numpy(),
                        seg_output_spike_count=np.zeros(0, dtype=np.int64),
                    )

                counts = self._run_neural_segment_rate(
                    stage, input_spike_train=spike_train, recorder_seg=recorder_seg,
                )
                seg_output_rates = counts / float(T)
                self._store_segment_output(stage.output_map, state_buffer, seg_output_rates)
                self._decref_consumers(
                    state_buffer, remaining,
                    (int(s.node_id) for s in stage.input_map),
                )

                if recorder_seg is not None:
                    recorder_seg.seg_output_spike_count = (
                        counts[0].to(torch.int64).cpu().numpy()
                    )
                    self._recorder.segments[stage_index] = recorder_seg

            elif stage.kind == "compute":
                op = stage.compute_op
                assert op is not None
                # Use the shared compute-op helper so SCM/HCM/Sim all run
                # the host op through identical arithmetic. We need the
                # pre-cast float32 result to record into the run-record,
                # so we run the helper without the dtype cast and cast
                # afterwards.
                result = execute_compute_op_torch(
                    op,
                    x,
                    state_buffer,
                    in_scale=in_scales.get(op.id, 1.0),
                    out_scale=out_scales.get(op.id, 1.0),
                )
                state_buffer[op.id] = result.to(_COMPUTE_DTYPE)

                # If this compute stage is an encoding-layer Perceptron
                # with LIFActivation, re-run it with ``forward_spiking``
                # to capture the actual ``(T, B, D)`` spike train. The
                # next neural segment will read this train verbatim
                # instead of re-encoding the rate uniformly.
                spike_train = self._try_emit_encoding_spike_train(op, x)
                if spike_train is not None:
                    state_buffer_spikes[op.id] = spike_train

                if self._recorder is not None:
                    # Snapshot the float32 rate output the next neural
                    # stage would consume. Loihi harness mode reuses
                    # this verbatim instead of re-running the host op.
                    self._recorder.compute_outputs[int(op.id)] = (
                        result.detach().to(torch.float32).cpu().numpy()
                    )
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
    # Spike-count recording entry point (used by Loihi parity harness)
    # ---------------------------------------------------------------------
    def forward_with_recording(
        self, x: torch.Tensor, *, sample_index: int = 0,
    ) -> tuple[torch.Tensor, RunRecord]:
        """Forward a SINGLE sample and return (output, run_record).

        Asserts batch_size == 1 and ``spiking_mode == 'lif'`` because:
          * Per-core counts are aggregated assuming one sample per
            forward, which keeps the recorder allocation small and the
            cycle alignment unambiguous.
          * Loihi's runner is LIF-only (TTFS does not map to its
            SubtractiveLIFReset model), so parity is only meaningful in
            LIF mode — the same constraint LavaLoihiRunner enforces in
            its constructor.
        """
        assert x.shape[0] == 1, "forward_with_recording requires batch_size == 1"
        assert self.spiking_mode == "lif", (
            f"forward_with_recording requires spiking_mode='lif'; got "
            f"{self.spiking_mode!r}"
        )

        record = RunRecord(sample_index=int(sample_index), T=int(self.simulation_length))
        self._recorder = record
        try:
            out = self.forward(x)
        finally:
            self._recorder = None
        return out, record
