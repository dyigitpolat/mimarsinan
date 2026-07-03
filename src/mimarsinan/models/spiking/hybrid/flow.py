"""Spiking simulation for HybridHardCoreMapping (rate-coded and TTFS)."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from mimarsinan.chip_simulation.recording.spike_recorder import RunRecord
from mimarsinan.chip_simulation.spiking_mode_policy import policy_for_spiking_mode
from mimarsinan.chip_simulation.spiking_semantics import is_cascaded_ttfs
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import HybridHardCoreMapping
from mimarsinan.spiking.segment_boundary import BoundaryConfig
from mimarsinan.models.spiking.spiking_config import COMPUTE_DTYPE, validate_spiking_init
from mimarsinan.models.spiking.hybrid.lif_step import HybridLifStepMixin
from mimarsinan.models.spiking.hybrid.rate_forward import HybridRateForwardMixin
from mimarsinan.models.spiking.hybrid.stage_io import HybridStageIOMixin
from mimarsinan.models.spiking.hybrid.ttfs_step import HybridTtfsStepMixin

# Re-exported by hybrid/__init__ for backward compatibility; keep the alias.
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

        self._boundary_config = BoundaryConfig(
            simulation_length=self.simulation_length,
            spiking_mode=self.spiking_mode,
            cycle_accurate=self.cycle_accurate_lif_forward,
            spike_mode=self.spike_mode,
            thresholding_mode=self.thresholding_mode,
            firing_mode=self.firing_mode,
            compute_dtype=COMPUTE_DTYPE,
        )

        self._segment_tensor_cache: Dict[int, dict] = {}
        self._segment_tensor_cache_key: int | None = None

        self._recorder: RunRecord | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            x = self.preprocessor(x)
            x = x.view(x.shape[0], -1)

            policy = policy_for_spiking_mode(self.spiking_mode, self.ttfs_cycle_schedule)
            if policy.decode_mode() == "timing":
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

        Supported only on the per-cycle cascade path (``lif`` and cascaded
        ``ttfs_cycle_based``); analytical TTFS modes are rejected.
        """
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
