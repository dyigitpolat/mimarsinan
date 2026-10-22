"""Hybrid mapping segment I/O and tensor cache."""

from __future__ import annotations

from typing import Dict

import torch

from mimarsinan.chip_simulation.hybrid_run.hybrid_execution import (
    assemble_segment_input_torch,
    decref_consumers,
    gather_final_output_torch,
    store_segment_output_torch,
)
from mimarsinan.chip_simulation import spike_modes
from mimarsinan.mapping.ir import IRSource
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import HybridStage, SegmentIOSlice
from mimarsinan.models.spiking.hybrid.carry import (
    carried_output_ids,
    publish_carried_trains,
    replay_carried_slices,
)
from mimarsinan.models.spiking.hybrid.host import HybridFlowHost
from mimarsinan.spiking.segment_boundary import encode_segment_input
from mimarsinan.models.spiking.hybrid.segment_cache import (
    SEGMENT_CACHE_MAX_BYTES as _SEGMENT_CACHE_MAX_BYTES,
    build_segment_tensors,
)
from mimarsinan.models.spiking.signal_spans import SpanFillPlan
from mimarsinan.models.spiking.spiking_config import COMPUTE_DTYPE


class HybridStageIOMixin(HybridFlowHost):
    """Segment tensor cache and state-buffer I/O."""

    def _encode_segment_input(
        self,
        stage,
        seg_input_rates_clamped: torch.Tensor,
        state_buffer_spikes: Dict[int, torch.Tensor],
        *,
        T: int,
        batch_size: int,
        device: torch.device,
        carried: Dict[int, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Build ``(T, B, in_size)`` spike train slice by slice: cached LIF trains, uniform encodes, and the
        ``carried`` rasters of the stage's own segment replayed at their delivery delays."""
        train = encode_segment_input(
            stage, seg_input_rates_clamped, state_buffer_spikes,
            config=self._boundary_config, hybrid_mapping=self.hybrid_mapping,
            T=T, batch_size=batch_size, device=device,
        )
        replay_carried_slices(train, stage, carried or {})
        return train

    def _carried_output_ids(self) -> Dict[int, tuple]:
        return carried_output_ids(self.hybrid_mapping)

    _publish_carried_trains = staticmethod(publish_carried_trains)

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
        state_buffer_spikes=None,
    ) -> None:
        decref_consumers(
            state_buffer, remaining, src_ids,
            state_buffer_spikes=state_buffer_spikes,
        )

    def _evict_segment_cache(self) -> None:
        """Trim the segment tensor cache to the byte budget.

        Under budget the uploaded weights and memoized latencies are RETAINED
        across stages and forwards (re-upload was the rate path's second wall
        cost); over budget the whole cache drops, restoring the one-live-segment
        VRAM profile for large vehicles.
        """
        total = sum(
            int(entry.get("nbytes", 0))
            for entry in self._segment_tensor_cache.values()
        )
        if total <= _SEGMENT_CACHE_MAX_BYTES:
            return
        self._segment_tensor_cache.clear()
        self._segment_tensor_cache_key = None

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
            input_map, state_buffer, batch_size, device, COMPUTE_DTYPE,
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
            COMPUTE_DTYPE,
        )

    def _fill_signal_tensor_from_spans(
        self,
        out: torch.Tensor,
        *,
        input_spikes: torch.Tensor,
        buffers: list[torch.Tensor],
        plan: SpanFillPlan,
        cycle: int = -1,
        single_spike: bool = False,
        latency: int = 0,
    ) -> None:
        """One core-cycle axon fill via the precomputed gather plan (W3b)."""
        on_value = (1.0 if cycle == latency else 0.0) if single_spike else 1.0
        plan.apply(out, input_spikes=input_spikes, buffers=buffers, on_value=on_value)

    def _get_segment_tensors(self, stage: HybridStage, device: torch.device) -> dict:
        """Return cached segment tensors (axon/output spans, weights, thresholds); upload on miss."""
        assert stage.hard_core_mapping is not None
        key = id(stage)
        cached = self._segment_tensor_cache.get(key)
        if cached is not None and cached.get("device") == device:
            return cached
        cached = build_segment_tensors(stage.hard_core_mapping, device)
        self._segment_tensor_cache[key] = cached
        self._segment_tensor_cache_key = key
        self._evict_segment_cache()
        return cached
