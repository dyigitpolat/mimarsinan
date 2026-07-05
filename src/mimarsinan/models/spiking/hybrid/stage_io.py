"""Hybrid mapping segment I/O and tensor cache."""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np
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
from mimarsinan.mapping.support.core_geometry import used_axons, used_neurons
from mimarsinan.mapping.support.spike_source_spans import (
    SpikeSourceSpan,
    compress_spike_sources,
)
from mimarsinan.models.spiking.hybrid.host import HybridFlowHost
from mimarsinan.models.spiking.hybrid.segment_cache import (
    SEGMENT_CACHE_MAX_BYTES as _SEGMENT_CACHE_MAX_BYTES,
    segment_entry_nbytes as _segment_entry_nbytes,
)
from mimarsinan.models.spiking.signal_spans import fill_signal_from_spans
from mimarsinan.models.spiking.spiking_config import COMPUTE_DTYPE


class HybridStageIOMixin(HybridFlowHost):
    """Segment tensor cache and state-buffer I/O."""

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
        spans: list[SpikeSourceSpan],
        cycle: int = -1,
        single_spike: bool = False,
        latency: int = 0,
    ) -> None:
        def _single_spike_always_on(d0: int, d1: int) -> None:
            out[:, d0:d1].fill_(1.0 if cycle == latency else 0.0)

        on_always_on: Callable[[int, int], None] | None = (
            _single_spike_always_on if single_spike else None
        )

        fill_signal_from_spans(
            out,
            spans,
            read_input=lambda sp: out.__setitem__(
                (slice(None), slice(int(sp.dst_start), int(sp.dst_end))),
                input_spikes[:, int(sp.src_start) : int(sp.src_end)],
            ),
            read_upstream=lambda sp: out.__setitem__(
                (slice(None), slice(int(sp.dst_start), int(sp.dst_end))),
                buffers[int(sp.src_core)][:, int(sp.src_start) : int(sp.src_end)],
            ),
            on_always_on=on_always_on,
            cycle=cycle,
        )

    def _get_segment_tensors(self, stage: HybridStage, device: torch.device) -> dict:
        """Return cached segment tensors (axon/output spans, weights, thresholds); upload on miss."""
        mapping = stage.hard_core_mapping
        assert mapping is not None
        key = id(stage)
        cached = self._segment_tensor_cache.get(key)
        if cached is not None and cached.get("device") == device:
            return cached

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
            output_spans = compress_spike_sources(
                list(np.asarray(output_sources, dtype=object).flatten())
            )

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
            t = torch.tensor(bank_mat.T, dtype=COMPUTE_DTYPE, device=device)
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
                    tile.T, dtype=COMPUTE_DTYPE, device=device,
                )

            core_params.append(core_weight)

            bias = getattr(core, "hardware_bias", None)
            if bias is None:
                hw_biases.append(None)
            else:
                hw_biases.append(
                    torch.tensor(
                        bias[:used_neu], dtype=COMPUTE_DTYPE, device=device,
                    )
                )

        thresholds = [
            torch.tensor(float(core.threshold), dtype=COMPUTE_DTYPE, device=device)
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
            nbytes=0,
        )
        cached["nbytes"] = _segment_entry_nbytes(cached)
        self._segment_tensor_cache[key] = cached
        self._segment_tensor_cache_key = key
        self._evict_segment_cache()
        return cached
