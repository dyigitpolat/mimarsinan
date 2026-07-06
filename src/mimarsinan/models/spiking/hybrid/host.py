"""Static typing contract shared by the SpikingHybridCoreFlow mixins."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Iterable

import torch

from mimarsinan.chip_simulation.recording.spike_recorder import (
    RunRecord,
    SegmentSpikeRecord,
)
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import (
    HybridHardCoreMapping,
    HybridStage,
    SegmentIOSlice,
)
from mimarsinan.models.spiking.signal_spans import SpanFillPlan
from mimarsinan.spiking.segment_boundary import BoundaryConfig

if TYPE_CHECKING:

    class HybridFlowHost:
        """Attributes and cross-mixin methods of the composed SpikingHybridCoreFlow.

        Each mixin subclasses this (``object`` at runtime) so it can rely on the
        surface the sibling mixins and ``SpikingHybridCoreFlow.__init__`` provide.
        """

        hybrid_mapping: HybridHardCoreMapping
        simulation_length: int
        firing_mode: str
        spike_mode: str
        thresholding_mode: str
        spiking_mode: str
        ttfs_cycle_schedule: str
        _boundary_config: BoundaryConfig
        _segment_tensor_cache: Dict[int, dict]
        _segment_tensor_cache_key: int | None
        _recorder: RunRecord | None

        def _build_consumer_counts(self) -> Dict[int, int]: ...

        @staticmethod
        def _decref_consumers(
            state_buffer: Dict[int, torch.Tensor],
            remaining: Dict[int, int],
            src_ids: Iterable[int],
        ) -> None: ...

        def _evict_segment_cache(self) -> None: ...

        def _get_segment_tensors(
            self, stage: HybridStage, device: torch.device
        ) -> dict: ...

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
        ) -> None: ...

        @staticmethod
        def _assemble_segment_input(
            input_map: list[SegmentIOSlice],
            state_buffer: Dict[int, torch.Tensor],
            batch_size: int,
            device: torch.device,
        ) -> torch.Tensor: ...

        @staticmethod
        def _store_segment_output(
            output_map: list[SegmentIOSlice],
            state_buffer: Dict[int, torch.Tensor],
            output_tensor: torch.Tensor,
        ) -> None: ...

        def _gather_final_output(
            self,
            state_buffer: Dict[int, torch.Tensor],
            original_input: torch.Tensor,
            batch_size: int,
            device: torch.device,
        ) -> torch.Tensor: ...

        def _run_neural_segment_rate(
            self,
            stage: HybridStage,
            *,
            input_spike_train: torch.Tensor,
            recorder_seg: SegmentSpikeRecord | None = None,
        ) -> torch.Tensor: ...

        def _encode_segment_input(
            self,
            stage,
            seg_input_rates_clamped: torch.Tensor,
            state_buffer_spikes: Dict[int, torch.Tensor],
            *,
            T: int,
            batch_size: int,
            device: torch.device,
        ) -> torch.Tensor: ...

        def _apply_input_shifts(
            self, input_map, seg_input_rates: torch.Tensor
        ) -> torch.Tensor: ...

else:
    HybridFlowHost = object
