"""Spiking simulation for HybridHardCoreMapping (rate-coded and TTFS)."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mimarsinan.chip_simulation import spike_modes
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
from mimarsinan.mapping.core_geometry import used_axons, used_neurons
from mimarsinan.mapping.hybrid_hardcore_mapping import (
    HybridHardCoreMapping,
    HybridStage,
    SegmentIOSlice,
)
from mimarsinan.mapping.ir import ComputeOp, IRSource

from mimarsinan.mapping.spike_source_spans import SpikeSourceSpan, compress_spike_sources
from mimarsinan.models.lif_kernels import lif_fire_and_reset


# float64 matches nevresim TTFS double precision and avoids threshold flip-flops
_COMPUTE_DTYPE = torch.float64


class SpikingHybridCoreFlow(nn.Module):
    """
    Execute a HybridHardCoreMapping via a global state buffer keyed by IR node_id.
    Neural segments use SegmentIOSlice I/O; ComputeOps gather from the buffer.
    Supports rate-coded (LIF) and TTFS (continuous or quantized analytical) modes.
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
        thresholding_mode: str = "<=",
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

        # Single-segment LRU: one segment's weights on GPU at a time (ViT-scale OOM otherwise).
        self._segment_tensor_cache: Dict[int, dict] = {}
        self._segment_tensor_cache_key: int | None = None

        self._recorder: RunRecord | None = None

    def _build_consumer_counts(self) -> Dict[int, int]:
        """Return ``{node_id: downstream_read_count}`` for state-buffer refcount pruning."""
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
        """Drop the cached segment's GPU tensors (no ``empty_cache`` here)."""
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

    def to_spikes(self, tensor: torch.Tensor, cycle: int) -> torch.Tensor:
        return spike_modes.to_spikes(
            tensor,
            cycle,
            simulation_length=self.simulation_length,
            spike_mode=self.spike_mode,
        )

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
        """Return cached segment tensors (axon/output spans, weights, thresholds); upload on miss."""
        mapping = stage.hard_core_mapping
        assert mapping is not None
        key = id(stage)
        cached = self._segment_tensor_cache.get(key)
        if cached is not None and cached.get("device") == device:
            return cached

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
            t = torch.tensor(bank_mat.T, dtype=_COMPUTE_DTYPE, device=device)
            bank_tensors[int(bid)] = t
            return t

        core_params: list[torch.Tensor] = []
        hw_biases = []
        for core_idx, core in enumerate(cores):
            used_ax = used_axons(core, min_one=True)
            used_neu = used_neurons(core, min_one=True)

            placement_dicts = (
                placements_per_core[core_idx]
                if core_idx < len(placements_per_core)
                else []
            )

            core_weight: torch.Tensor | None = None

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
                    core_weight = bank_t[int(bn0):int(bn1), int(ba0):int(ba1)]

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
        """Rate-coded segment: cycle loop over cores; returns spike counts ``(B, out_dim)``."""
        mapping = stage.hard_core_mapping
        assert mapping is not None

        T = self.simulation_length
        assert input_spike_train.shape[0] == T

        batch_size = input_spike_train.shape[1]
        device = input_spike_train.device
        input_size = input_spike_train.shape[2]
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
                if hw_biases[core_idx] is not None:
                    memb_i += hw_biases[core_idx]

                buffers[core_idx] = lif_fire_and_reset(
                    memb_i,
                    thresholds[core_idx],
                    thresholding_mode=self.thresholding_mode,
                    firing_mode=self.firing_mode,
                    output_dtype=_COMPUTE_DTYPE,
                )

                if recording:
                    record_in_t[core_idx] += input_signals[core_idx][0].to(torch.int64)
                    record_out_t[core_idx] += buffers[core_idx][0].to(torch.int64)

            for sp in output_spans:
                d0 = int(sp.dst_start)
                d1 = int(sp.dst_end)
                if sp.kind == "off":
                    continue
                if sp.kind == "on":
                    # Always-on axon: one spike per input cycle only (cycles [0, T)).
                    if cycle < T:
                        output_counts[:, d0:d1] += 1.0
                    continue
                if sp.kind == "input":
                    output_counts[:, d0:d1] += input_spikes[:, int(sp.src_start):int(sp.src_end)]
                    continue
                src_lat = cores[int(sp.src_core)].latency
                if src_lat is None:
                    continue
                # Neuron output: accumulate only inside source core's active window [lat, lat+T).
                if cycle < int(src_lat) or cycle >= int(src_lat) + T:
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
        """TTFS segment: analytical per-core matmul; continuous or quantized closed-form."""
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

        input_activations = input_activations.to(_COMPUTE_DTYPE)

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
            if hw_biases[ci] is not None:
                V = V + hw_biases[ci]

            if quantized:
                from mimarsinan.models.ttfs_kernels import ttfs_quantized_activation

                buffers[ci] = ttfs_quantized_activation(V, thresholds[ci], S)
            else:
                out = F.relu(V)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            x = self.preprocessor(x)
            x = x.view(x.shape[0], -1)

            if self.spiking_mode in self._TTFS_SPIKING_MODES:
                return self._forward_ttfs(x)

            return self._forward_rate(x)
        finally:
            self._evict_segment_cache()
            if isinstance(x, torch.Tensor) and x.is_cuda:
                torch.cuda.empty_cache()

    def _forward_ttfs(self, x: torch.Tensor) -> torch.Tensor:
        """TTFS forward via state buffer; rescales ComputeOp bias via ``node_activation_scales``."""
        T = self.simulation_length
        batch_size = x.shape[0]
        device = x.device
        quantized = self.spiking_mode == "ttfs_quantized"

        x_compute = x.to(_COMPUTE_DTYPE)
        state_buffer: Dict[int, torch.Tensor] = {-2: x_compute}
        from mimarsinan.chip_simulation.hybrid_execution import resolve_stage_compute_scales

        remaining = dict(self._build_consumer_counts())
        from mimarsinan.chip_simulation.hybrid_stage_runner import (
            HybridStageContext,
            run_hybrid_stages,
        )

        def _ctx_factory(stage_index, stage, buf):
            return HybridStageContext(
                stage_index=stage_index,
                stage=stage,
                state_buffer=buf,
                remaining=remaining,
            )

        def _on_neural_ttfs(ctx: HybridStageContext) -> None:
            stage = ctx.stage
            seg_input = self._assemble_segment_input(
                stage.input_map, ctx.state_buffer, batch_size, device
            )
            seg_output = self._run_neural_segment_ttfs(
                stage, input_activations=seg_input, quantized=quantized
            )
            self._store_segment_output(stage.output_map, ctx.state_buffer, seg_output)

        def _after_neural_ttfs(ctx: HybridStageContext) -> None:
            self._decref_consumers(
                ctx.state_buffer,
                ctx.remaining,
                (int(s.node_id) for s in ctx.stage.input_map),
            )

        def _on_compute_ttfs(ctx: HybridStageContext) -> None:
            op = ctx.stage.compute_op
            assert op is not None
            in_scale, out_scale = resolve_stage_compute_scales(
                self.hybrid_mapping, op.id, apply_ttfs=True
            )
            ctx.state_buffer[op.id] = execute_compute_op_torch(
                op,
                x,
                ctx.state_buffer,
                in_scale=in_scale,
                out_scale=out_scale,
                output_dtype=_COMPUTE_DTYPE,
            )

        def _after_compute_ttfs(ctx: HybridStageContext) -> None:
            op = ctx.stage.compute_op
            assert op is not None
            self._decref_consumers(
                ctx.state_buffer,
                ctx.remaining,
                (int(src.node_id) for src in op.input_sources.flatten()
                 if isinstance(src, IRSource) and src.node_id >= 0),
            )

        run_hybrid_stages(
            self.hybrid_mapping,
            state_buffer,
            on_neural=_on_neural_ttfs,
            on_compute=_on_compute_ttfs,
            after_neural=_after_neural_ttfs,
            after_compute=_after_compute_ttfs,
            context_factory=_ctx_factory,
        )

        final = self._gather_final_output(state_buffer, x_compute, batch_size, device)
        return final.to(torch.float32) * float(T)

    @staticmethod
    def _resolve_lif_perceptron(module):
        """Return inner ``Perceptron`` with ``LIFActivation`` for encoding spike trains, else None."""
        from mimarsinan.models.activations import LIFActivation

        if module is None:
            return None
        if hasattr(module, "perceptron") and module is not getattr(module, "perceptron"):
            return None
        candidate = module
        if not hasattr(candidate, "forward_spiking"):
            return None
        activation = getattr(candidate, "activation", None)
        if SpikingHybridCoreFlow._unwrap_lif_activation(activation) is not None:
            return candidate
        return None

    @staticmethod
    def _unwrap_lif_activation(activation):
        """Walk activation wrappers and return inner ``LIFActivation``, or None."""
        from mimarsinan.models.activations import LIFActivation
        from mimarsinan.models.layers import TransformedActivation
        from mimarsinan.tuning.tuners.lif_adaptation_tuner import LIFBlendActivation

        for _ in range(8):  # depth cap — chain is at most 2 wrappers in practice
            if activation is None:
                return None
            if isinstance(activation, LIFActivation):
                return activation
            if isinstance(activation, TransformedActivation):
                activation = getattr(activation, "base_activation", None)
                continue
            if isinstance(activation, LIFBlendActivation):
                activation = activation.lif_activation
                continue
            return None
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
        """Build ``(T, B, in_size)`` spike train; prefer cached LIF trains over uniform encoding."""
        in_size = seg_input_rates_clamped.shape[1]
        spike_train = torch.zeros(
            T, batch_size, in_size, device=device, dtype=_COMPUTE_DTYPE,
        )
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

        encoded = torch.zeros_like(spike_train)
        for cycle in range(T):
            encoded[cycle] = self.to_spikes(
                seg_input_rates_clamped, cycle,
            ).to(_COMPUTE_DTYPE)
        for lo, hi in filled_ranges:
            encoded[:, :, lo:hi] = spike_train[:, :, lo:hi]
        return encoded

    def _try_emit_encoding_spike_train(self, op, x: torch.Tensor) -> torch.Tensor | None:
        """Run encoding Perceptron ``forward_spiking`` when wrapped with LIF; else None."""
        if self.spiking_mode != "lif":
            return None
        module = (op.params or {}).get("module") if hasattr(op, "params") else None
        perceptron = self._resolve_lif_perceptron(module)
        if perceptron is None:
            return None
        gathered = op.gather_inputs(x, {})
        if op.input_shape is not None:
            gathered = gathered.view(gathered.shape[0], *op.input_shape)
        spikes = perceptron.forward_spiking(gathered)
        T_ax = spikes.shape[0]
        B = spikes.shape[1]
        spikes = spikes.reshape(T_ax, B, -1)
        return spikes

    def _forward_rate(self, x: torch.Tensor) -> torch.Tensor:
        """Rate-coded forward; encoding LIF ComputeOps pass spike trains to the next neural segment."""
        batch_size = x.shape[0]
        device = x.device
        T = self.simulation_length

        x_compute = x.to(_COMPUTE_DTYPE)
        state_buffer: Dict[int, torch.Tensor] = {-2: x_compute}
        state_buffer_spikes: Dict[int, torch.Tensor] = {}
        from mimarsinan.chip_simulation.hybrid_execution import resolve_stage_compute_scales

        remaining = dict(self._build_consumer_counts())
        from mimarsinan.chip_simulation.hybrid_stage_runner import (
            HybridStageContext,
            run_hybrid_stages,
        )

        def _ctx_factory(stage_index, stage, buf):
            return HybridStageContext(
                stage_index=stage_index,
                stage=stage,
                state_buffer=buf,
                remaining=remaining,
                state_buffer_spikes=state_buffer_spikes,
                recorder=self._recorder,
            )

        def _on_neural_rate(ctx: HybridStageContext) -> None:
            stage = ctx.stage
            seg_input_rates = self._assemble_segment_input(
                stage.input_map, ctx.state_buffer, batch_size, device
            )
            seg_input_rates_clamped = seg_input_rates.clamp(0.0, 1.0)
            spike_train = self._build_segment_input_spike_train(
                stage,
                seg_input_rates_clamped,
                ctx.state_buffer_spikes,
                T=T,
                batch_size=batch_size,
                device=device,
            )

            recorder_seg: SegmentSpikeRecord | None = None
            if ctx.recorder is not None:
                recorder_seg = SegmentSpikeRecord(
                    stage_index=ctx.stage_index,
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
            self._store_segment_output(
                stage.output_map, ctx.state_buffer, seg_output_rates,
            )

            if recorder_seg is not None and ctx.recorder is not None:
                recorder_seg.seg_output_spike_count = (
                    counts[0].to(torch.int64).cpu().numpy()
                )
                ctx.recorder.segments[ctx.stage_index] = recorder_seg

        def _after_neural_rate(ctx: HybridStageContext) -> None:
            self._decref_consumers(
                ctx.state_buffer,
                ctx.remaining,
                (int(s.node_id) for s in ctx.stage.input_map),
            )

        def _on_compute_rate(ctx: HybridStageContext) -> None:
            op = ctx.stage.compute_op
            assert op is not None
            # Rate/LIF path: state buffer holds spike rates in [0, 1]; do not apply TTFS scales.
            in_scale, out_scale = resolve_stage_compute_scales(
                self.hybrid_mapping, op.id, apply_ttfs=False
            )
            result = execute_compute_op_torch(
                op,
                x,
                ctx.state_buffer,
                in_scale=in_scale,
                out_scale=out_scale,
            )
            ctx.state_buffer[op.id] = result.to(_COMPUTE_DTYPE)

            spike_train = self._try_emit_encoding_spike_train(op, x)
            if spike_train is not None:
                ctx.state_buffer_spikes[op.id] = spike_train

            if ctx.recorder is not None:
                ctx.recorder.compute_outputs[int(op.id)] = (
                    result.detach().to(torch.float32).cpu().numpy()
                )

        def _after_compute_rate(ctx: HybridStageContext) -> None:
            op = ctx.stage.compute_op
            assert op is not None
            self._decref_consumers(
                ctx.state_buffer,
                ctx.remaining,
                (int(src.node_id) for src in op.input_sources.flatten()
                 if isinstance(src, IRSource) and src.node_id >= 0),
            )

        run_hybrid_stages(
            self.hybrid_mapping,
            state_buffer,
            on_neural=_on_neural_rate,
            on_compute=_on_compute_rate,
            after_neural=_after_neural_rate,
            after_compute=_after_compute_rate,
            context_factory=_ctx_factory,
        )

        final = self._gather_final_output(state_buffer, x_compute, batch_size, device)
        return final.to(torch.float32) * float(T)

    def forward_with_recording(
        self, x: torch.Tensor, *, sample_index: int = 0,
    ) -> tuple[torch.Tensor, RunRecord]:
        """Forward one sample (B=1, ``spiking_mode='lif'``) and return output plus spike record."""
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
