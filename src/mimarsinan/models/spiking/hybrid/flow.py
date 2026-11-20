"""Spiking simulation for HybridHardCoreMapping (rate-coded and TTFS)."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from mimarsinan.chip_simulation.recording.spike_recorder import RunRecord
from mimarsinan.chip_simulation.soma_law import DEFAULT_SOMA_LAW, SomaLaw
from mimarsinan.chip_simulation.spiking_mode_policy import policy_for_spiking_mode
from mimarsinan.chip_simulation.spiking_point import HISTORICAL_WINDOW_FIRING, SpikingPoint, stated_point
from mimarsinan.chip_simulation.window_firing import INTEGRATE_THEN_EMIT
from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.models.spiking.serial import (
    refuse_saturating_membrane,
    require_serial_deployment_admissible,
)
from mimarsinan.chip_simulation.spiking_semantics import is_cascaded_ttfs
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import HybridHardCoreMapping
from mimarsinan.mapping.support.schedule.pass_cut import COLLAPSE, VERBATIM
from mimarsinan.spiking.segment_boundary import BoundaryConfig
from mimarsinan.models.spiking.spiking_config import COMPUTE_DTYPE, validate_spiking_init
from mimarsinan.models.spiking.hybrid.carry import PassReplayRefusal, carried_output_ids, refuse_stochastic_replay
from mimarsinan.models.spiking.hybrid.lif_step import HybridLifStepMixin
from mimarsinan.models.spiking.hybrid.rate_forward import HybridRateForwardMixin
from mimarsinan.models.spiking.hybrid.stage_io import HybridStageIOMixin
from mimarsinan.models.spiking.hybrid.ttfs_step import HybridTtfsStepMixin

# Re-exported by hybrid/__init__ for backward compatibility; keep the alias.
_COMPUTE_DTYPE = COMPUTE_DTYPE


class FlowPointRefusal(ToolchainRefusal, ValueError):
    """[U-LIF2 r2] A builder's point the executor's own fields do not execute, refused by name."""


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
        *,
        # Contract-owned semantics carry NO defaults: a silently defaulted
        # comparator is another family's physics ("<=" is the ttfs family's,
        # and on integer-lattice chips "<" vs "<=" is load-bearing — V9).
        # The constructor-shaped version of the banned get(key, default).
        thresholding_mode: str,
        spiking_mode: str = "lif",
        cycle_accurate_lif_forward: bool = False,
        ttfs_cycle_schedule: str = "cascaded",
        membrane_readout: bool = False,
        membrane_readout_half_step: bool = True,
        phase_dither: bool = False,
        lif_membrane_init: float = 0.0,
        lif_execution_synchronized: bool = False,
        membrane_integer_lattice: bool = False,
        pass_transfer: str = VERBATIM,
        soma_law: SomaLaw = DEFAULT_SOMA_LAW,
        spiking_point: SpikingPoint | None = None,
    ):
        super().__init__()
        # The resolved soma point this executor runs. Constructed from the
        # contract by the builders; the default IS today's law, so every
        # historical construction resolves byte-identically.
        self.soma_law = soma_law
        # ONE discipline per run: a run whose backends disagreed would report
        # numbers from two different computations (see run_pass_transfer).
        self.pass_transfer = str(pass_transfer)
        # [nevresim parity] integer-chip cells snap membranes to the exact
        # chip lattice each cycle (float noise must never decide a tie).
        self.membrane_integer_lattice = bool(membrane_integer_lattice)
        # [§17] certification capture seam: called as (stage, raw_counts)
        # after every neural segment, both disciplines; None disables.
        self.stage_count_recorder: "object | None" = None
        # [ODIN P2] per-cycle raster capture: called as (stage, raster) with
        # the segment output raster (T, B, out) in producer-local time.
        self.stage_raster_recorder: "object | None" = None

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
        # [C2] membrane-augmented readout: final-only LIF output cores decode
        # counts + m_T/theta (half-step charge removed when it was baked).
        self.membrane_readout = bool(membrane_readout)
        self.membrane_readout_half_step = bool(membrane_readout_half_step)

        validate_spiking_init(
            firing_mode=firing_mode,
            spike_mode=spike_mode,
            thresholding_mode=thresholding_mode,
        )

        # The guard is LIF-family semantics; other modes never pre-charge.
        self.lif_membrane_init = (
            float(lif_membrane_init) if spiking_mode == "lif" else 0.0
        )
        # [calculus §16] integrate then emit (LIF-family only): [U-LIF2 W9] executed by the gated cycle-level executor.
        self.lif_execution_synchronized = bool(
            lif_execution_synchronized and spiking_mode == "lif"
        )
        # [U-LIF2 W9] Its closed form (the count executor) instead, asked for by name: the certification's analytic
        # gauge only (count_alignment), never a builder, so the closed form is never the executor of record.
        self.window_count_closed_form = False
        self._boundary_config = BoundaryConfig(
            simulation_length=self.simulation_length,
            spiking_mode=self.spiking_mode,
            cycle_accurate=self.cycle_accurate_lif_forward,
            spike_mode=self.spike_mode,
            thresholding_mode=self.thresholding_mode,
            firing_mode=self.firing_mode,
            compute_dtype=COMPUTE_DTYPE,
            phase_dither=bool(phase_dither),
        )

        refuse_stochastic_replay(hybrid_mapping, spike_mode=spike_mode, pass_transfer=self.pass_transfer)
        refuse_lossy_count_carry(hybrid_mapping, spiking_mode=spiking_mode, ttfs_cycle_schedule=ttfs_cycle_schedule,
                                 spike_mode=spike_mode, pass_transfer=self.pass_transfer)
        # The point-keyed seam: refuse a program the declared soma law cannot
        # execute faithfully BEFORE a single cycle folds (retimed level hops,
        # neuron splitting, coalescing, a non-integral window-start membrane).
        require_serial_deployment_admissible(
            hybrid_mapping, soma_law, membrane_init=self.lif_membrane_init,
        )
        refuse_saturating_membrane(
            soma_law if self.membrane_readout else None,
            mechanism="the membrane-readout logits decode",
            identity=(
                "Q_T = theta*c_T + m_T holds only for a LOSSLESS accumulator, "
                "where the residual membrane IS the unemitted charge."
            ),
        )

        # [U-LIF2 r2] The run's point the builder holds (the contract's): checked against what the fields execute.
        self._run_point = spiking_point
        if spiking_point is not None:
            self._require_fields_execute(spiking_point)

        self._segment_tensor_cache: Dict[int, dict] = {}
        self._segment_tensor_cache_key: int | None = None

        self._recorder: RunRecord | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            x = self.preprocessor(x)
            x = x.view(x.shape[0], -1)

            policy = policy_for_spiking_mode(self.spiking_point())
            if policy.runs_per_cycle_cascade:  # [U-TGAUGE5] the one predicate the relays and the window rule read
                return self._forward_rate(x)
            return self._forward_ttfs(x)
        finally:
            self._evict_segment_cache()
            if isinstance(x, torch.Tensor) and x.is_cuda:
                torch.cuda.empty_cache()

    def spiking_point(self) -> SpikingPoint:
        """[U-LIF2 r2] The point this executor runs: its builder's record (the contract's), else the point its own fields
        state (a flow built directly, or stored before the point existed), whose window firing is the one its two-window
        flag executes, never the declared default."""
        point = getattr(self, "_run_point", None)
        return point if point is not None else self._stated_point()

    def _stated_point(self) -> SpikingPoint:
        boundary = getattr(self, "_boundary_config", None)
        record = {"spiking_mode": self.spiking_mode, "ttfs_cycle_schedule": self.ttfs_cycle_schedule,
                  "spike_phase_dither": bool(getattr(boundary, "phase_dither", False)),
                  "lif_membrane_init": float(getattr(self, "lif_membrane_init", 0.0))}
        firing = INTEGRATE_THEN_EMIT if getattr(self, "lif_execution_synchronized", False) else HISTORICAL_WINDOW_FIRING
        return stated_point(record, window_firing=firing, soma_law=getattr(self, "soma_law", DEFAULT_SOMA_LAW))

    def _require_fields_execute(self, point: SpikingPoint) -> None:
        """Refuse a builder's point the fields would run another function than (the mode, the two-window discipline, the
        declared execution extensions), naming the value."""
        fields = self._stated_point()
        for what, run, executed in (("spiking_mode", point.spiking_mode, fields.spiking_mode),
                                    ("lif_window_firing", point.window_firing, fields.window_firing),
                                    ("execution extensions", point.execution_extensions, fields.execution_extensions)):
            if what == "lif_window_firing" and point.integrates_then_emits == fields.integrates_then_emits:
                continue
            if run != executed:
                raise FlowPointRefusal(f"SpikingHybridCoreFlow: the run's point declares {what}={run!r}, and the "
                                       f"executor's own fields execute {what}={executed!r}: build it from the same "
                                       f"contract")

    def forward_with_recording(
        self, x: torch.Tensor, *, sample_index: int = 0,
    ) -> tuple[torch.Tensor, RunRecord]:
        """Forward one sample (B=1) and return output plus spike record.

        Supported only on the per-cycle cascade path (``lif`` and cascaded
        ``ttfs_cycle_based``); analytical TTFS modes are rejected.
        """
        assert x.shape[0] == 1, "forward_with_recording requires batch_size == 1"
        assert policy_for_spiking_mode(self.spiking_point()).runs_per_cycle_cascade, (
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

    def forward_with_recordings(
        self, x: torch.Tensor, *, first_sample_index: int = 0,
    ) -> tuple[torch.Tensor, list[RunRecord]]:
        """[U-S] Forward a batch one sample at a time (the recorder runs at batch 1): the outputs stacked in batch
        order and one record per sample, ``sample_index`` counting from ``first_sample_index``."""
        outs: list[torch.Tensor] = []
        records: list[RunRecord] = []
        for i in range(int(x.shape[0])):
            out, record = self.forward_with_recording(x[i:i + 1], sample_index=int(first_sample_index) + i)
            outs.append(out)
            records.append(record)
        return torch.cat(outs, dim=0), records


def refuse_lossy_count_carry(hybrid_mapping, *, spiking_mode: str, ttfs_cycle_schedule: str, spike_mode: str,
                             pass_transfer: str) -> None:
    """A cascaded TTFS run carries a count across a pass inside a segment through its single-spike re-encode, the time
    code's lossless codec under the family's one legal input encoding (``spike_generation_mode`` TTFS). Any other
    encoding re-emits an even train and changes the function, so a program that carries across a pass under it is
    refused by name; a program that carries nothing runs."""
    if pass_transfer != COLLAPSE or spike_mode == "TTFS" or not is_cascaded_ttfs(spiking_mode, ttfs_cycle_schedule):
        return
    carried = sorted(i for i, ids in carried_output_ids(hybrid_mapping).items() if ids)
    if carried:
        raise PassReplayRefusal(
            f"cascaded TTFS carries counts across the passes of stages {carried[:4]} through a single-spike re-encode, "
            f"exact only under spike_generation_mode 'TTFS' (the family's one legal value), not {spike_mode!r}")
