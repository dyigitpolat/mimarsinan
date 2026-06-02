"""Spiking simulation for HybridHardCoreMapping (rate-coded and TTFS)."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from mimarsinan.chip_simulation import spike_modes
from mimarsinan.chip_simulation.hybrid_run.hybrid_execution import decref_consumers
from mimarsinan.chip_simulation.recording.spike_recorder import RunRecord
from mimarsinan.mapping.ir import IRSource
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import HybridHardCoreMapping
from mimarsinan.models.spiking.spiking_config import COMPUTE_DTYPE, validate_spiking_init
from mimarsinan.models.spiking.hybrid.lif_step import HybridLifStepMixin
from mimarsinan.models.spiking.hybrid.rate_forward import HybridRateForwardMixin
from mimarsinan.models.spiking.hybrid.stage_io import HybridStageIOMixin
from mimarsinan.models.spiking.hybrid.ttfs_step import HybridTtfsStepMixin

# Backward compatibility for integration tests.
_COMPUTE_DTYPE = COMPUTE_DTYPE


class SpikingHybridCoreFlow(
    HybridStageIOMixin,
    HybridLifStepMixin,
    HybridRateForwardMixin,
    HybridTtfsStepMixin,
    nn.Module,
):
    """
    Execute a HybridHardCoreMapping via a global state buffer keyed by IR node_id.
    Neural segments use SegmentIOSlice I/O; ComputeOps gather from the buffer.
    Supports rate-coded (LIF) and TTFS (continuous or quantized analytical) modes.
    """

    _TTFS_FIRING_MODES = HybridTtfsStepMixin._TTFS_FIRING_MODES
    _TTFS_SPIKING_MODES = HybridTtfsStepMixin._TTFS_SPIKING_MODES

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
        cycle_accurate_lif_forward: bool = False,
        ttfs_cycle_schedule: str = "cascaded",
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
        self.ttfs_cycle_schedule = ttfs_cycle_schedule
        self.cycle_accurate_lif_forward = bool(cycle_accurate_lif_forward)
        self._use_cycle_accurate_trains = (
            spiking_mode == "lif" and self.cycle_accurate_lif_forward
        )

        validate_spiking_init(
            firing_mode=firing_mode,
            spike_mode=spike_mode,
            thresholding_mode=thresholding_mode,
        )

        from mimarsinan.spiking.segment_encoding import SegmentEncodingConfig
        self._segment_encoding = SegmentEncodingConfig(
            simulation_length=self.simulation_length,
            spiking_mode=self.spiking_mode,
            cycle_accurate=self.cycle_accurate_lif_forward,
            spike_mode=self.spike_mode,
            thresholding_mode=self.thresholding_mode,
            firing_mode=self.firing_mode,
            compute_dtype=COMPUTE_DTYPE,
        )

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from mimarsinan.chip_simulation.spiking_semantics import (
            is_cascaded_ttfs,
            requires_ttfs_firing,
        )

        try:
            x = self.preprocessor(x)
            x = x.view(x.shape[0], -1)

            if is_cascaded_ttfs(self.spiking_mode, self.ttfs_cycle_schedule):
                return self._forward_rate(x)

            if requires_ttfs_firing(self.spiking_mode):
                return self._forward_ttfs(x)

            return self._forward_rate(x)
        finally:
            self._evict_segment_cache()
            if isinstance(x, torch.Tensor) and x.is_cuda:
                torch.cuda.empty_cache()

    def forward_with_recording(
        self, x: torch.Tensor, *, sample_index: int = 0,
    ) -> tuple[torch.Tensor, RunRecord]:
        """Forward one sample (B=1) and return output plus spike record.

        Supported for the per-cycle cascade path: ``lif`` and cascaded
        ``ttfs_cycle_based``. Analytical TTFS modes do not run the recording
        cascade, so they are rejected.
        """
        from mimarsinan.chip_simulation.spiking_semantics import is_cascaded_ttfs

        assert x.shape[0] == 1, "forward_with_recording requires batch_size == 1"
        assert (
            self.spiking_mode == "lif"
            or is_cascaded_ttfs(self.spiking_mode, self.ttfs_cycle_schedule)
        ), (
            f"forward_with_recording requires spiking_mode='lif' or cascaded "
            f"ttfs_cycle_based; got {self.spiking_mode!r}"
        )

        record = RunRecord(sample_index=int(sample_index), T=int(self.simulation_length))
        self._recorder = record
        try:
            out = self.forward(x)
        finally:
            self._recorder = None
        return out, record
