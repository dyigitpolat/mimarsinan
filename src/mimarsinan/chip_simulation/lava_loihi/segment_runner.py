"""Wave-parallel host-scheduled Lava neural-segment execution."""

from __future__ import annotations

import time as _time
from typing import TYPE_CHECKING, Dict, Tuple

import numpy as np

from mimarsinan.common.env import loihi_quiet, loihi_wave_workers

from mimarsinan.chip_simulation.execution_bounds import (
    ReusableBoundedPool,
    run_tasks_in_pool_bounded,
)
from mimarsinan.chip_simulation.lava_loihi.core_worker import run_lava_core_task
from mimarsinan.chip_simulation.lava_loihi.segment_assembly import (
    assemble_core_active_input,
)
from mimarsinan.chip_simulation.lava_loihi.wave_schedule import (
    core_dependency_graph,
    wave_levels,
)
from mimarsinan.chip_simulation.recording.spike_recorder import CoreSpikeCounts, SegmentSpikeRecord
from mimarsinan.mapping.packing.softcore import HardCoreMapping
from mimarsinan.mapping.support.core_geometry import used_axons, used_neurons
from mimarsinan.mapping.support.spike_source_spans import compress_spike_sources
from mimarsinan.chip_simulation.behavior_config import NeuralBehaviorConfig
from mimarsinan.chip_simulation.lava_loihi.timing import _LAVA_DTYPE, _SegmentTiming

# (weights, threshold, hardware_bias, input_spikes) — the `_run_core_lava` kwargs.
_CoreTask = Tuple[np.ndarray, float, "np.ndarray | None", np.ndarray]


class LavaSegmentMixin:
    """Host contract: ``T``/``_behavior``/``_simulation_step_timeout_s`` set by ``LavaLoihiRunner.__init__``; ``_run_core_lava`` from ``LavaCoreMixin``."""

    T: int
    _behavior: NeuralBehaviorConfig
    _simulation_step_timeout_s: float

    if TYPE_CHECKING:
        def _run_core_lava(
            self,
            *,
            weights: np.ndarray,
            threshold: float,
            hardware_bias: np.ndarray | None,
            input_spikes: np.ndarray,
        ) -> np.ndarray: ...

    def _dispatch_wave(
        self,
        wave_tasks: Dict[int, _CoreTask],
        *,
        pool: ReusableBoundedPool | None = None,
    ) -> Dict[int, np.ndarray]:
        """Run one wave of dependency-free cores; multi-core waves with funded workers go through the bounded spawn pool (reused across waves via ``pool``)."""
        workers = min(loihi_wave_workers(), len(wave_tasks))
        if workers <= 1:
            return {
                idx: self._run_core_lava(
                    weights=weights,
                    threshold=threshold,
                    hardware_bias=hardware_bias,
                    input_spikes=input_spikes,
                )
                for idx, (weights, threshold, hardware_bias, input_spikes)
                in wave_tasks.items()
            }
        return run_tasks_in_pool_bounded(
            run_lava_core_task,
            {
                idx: (self.T, self._behavior, *task)
                for idx, task in wave_tasks.items()
            },
            max_workers=workers,
            timeout_s=self._simulation_step_timeout_s,
            description=f"lava wave pool ({len(wave_tasks)} cores)",
            pool=pool,
        )

    def _run_neural_segment_scheduled(
        self,
        seg: HardCoreMapping,
        seg_input_rates: np.ndarray,
        *,
        recorder_seg: SegmentSpikeRecord | None = None,
    ) -> np.ndarray:
        """Execute a segment with host-scheduled routing and Lava per-core LIF."""
        T = self.T
        N = seg_input_rates.shape[0]
        seg_in_size = seg_input_rates.shape[1]
        timing = _SegmentTiming.from_mapping(seg, T)

        seg_input_spikes = self._behavior.encode_segment_input(seg_input_rates, T)
        seg_input_logical = np.zeros(
            (seg_in_size, N, timing.sample_stride), dtype=_LAVA_DTYPE,
        )
        seg_input_logical[:, :, :T] = seg_input_spikes.transpose(1, 0, 2)

        core_output_spikes: Dict[int, np.ndarray] = {}
        core_buffer_spikes: Dict[int, np.ndarray] = {}

        waves = wave_levels(core_dependency_graph(seg.cores))

        t0 = _time.time()
        verbose = not loihi_quiet()
        n_cores = len(seg.cores)
        if verbose:
            print(
                f"  [LavaLoihiRunner] segment with {n_cores} cores in "
                f"{len(waves)} waves; T={T}, N={N} — per-wave timing follows",
                flush=True,
            )
        # One reusable spawn pool per segment run: multi-core waves share its
        # workers (spawn paid once), and the `with` tears it down before the
        # segment returns — closed on success, killed on error.
        with ReusableBoundedPool(loihi_wave_workers()) as wave_pool:
            for wave_idx, wave in enumerate(waves):
                t_wave = _time.time()
                wave_tasks: Dict[int, _CoreTask] = {}
                for core_idx in wave:
                    core = seg.cores[core_idx]
                    latency = timing.core_latency(core)
                    used_ax = used_axons(core, min_one=True)
                    used_neu = used_neurons(core, min_one=True)
                    active_input = assemble_core_active_input(
                        core,
                        core_idx=core_idx,
                        N=N,
                        T=T,
                        latency=latency,
                        used_ax=used_ax,
                        seg_input_logical=seg_input_logical,
                        core_buffer_spikes=core_buffer_spikes,
                    )
                    weights = np.asarray(
                        core.core_matrix[:used_ax, :used_neu], dtype=_LAVA_DTYPE,
                    ).T
                    hardware_bias = (
                        np.asarray(core.hardware_bias[:used_neu], dtype=_LAVA_DTYPE)
                        if getattr(core, "hardware_bias", None) is not None
                        else None
                    )
                    wave_tasks[core_idx] = (
                        weights,
                        float(core.threshold),
                        hardware_bias,
                        active_input.reshape(used_ax, N * T),
                    )

                wave_outputs = self._dispatch_wave(wave_tasks, pool=wave_pool)

                for core_idx in wave:
                    core = seg.cores[core_idx]
                    latency = timing.core_latency(core)
                    used_neu = used_neurons(core, min_one=True)
                    active_output = wave_outputs[core_idx].reshape(used_neu, N, T)

                    full_output = np.zeros(
                        (int(core.neurons_per_core), N, timing.sample_stride),
                        dtype=_LAVA_DTYPE,
                    )
                    full_output[:used_neu, :, latency:latency + T] = active_output
                    core_output_spikes[core_idx] = full_output

                    buffered = full_output.copy()
                    hold_start = latency + T
                    if hold_start < timing.sample_stride:
                        buffered[:, :, hold_start:] = buffered[:, :, hold_start - 1:hold_start]
                    core_buffer_spikes[core_idx] = buffered
                if verbose:
                    print(
                        f"    wave {wave_idx + 1:>3}/{len(waves)}: "
                        f"{len(wave)} core(s) (idx={wave}) "
                        f"{_time.time() - t_wave:7.1f}s",
                        flush=True,
                    )

        print(
            f"  [LavaLoihiRunner] scheduled segment run: {len(seg.cores)} cores, "
            f"{timing.total_steps(N)} logical cycles, {_time.time() - t0:.1f}s",
            flush=True,
        )

        out_size = len(seg.output_sources)
        seg_out_spikes = np.zeros(
            (out_size, N, timing.sample_stride), dtype=_LAVA_DTYPE,
        )
        out_spans = compress_spike_sources(seg.output_sources)
        for sp in out_spans:
            d0 = int(sp.dst_start)
            d1 = int(sp.dst_end)
            if sp.kind == "off":
                continue
            if sp.kind == "on":
                # Always-on: gate to cycle < T to match HCM spike counts.
                seg_out_spikes[d0:d1, :, :T] = 1.0
                continue
            if sp.kind == "input":
                seg_out_spikes[d0:d1, :, :] = seg_input_logical[
                    int(sp.src_start) : int(sp.src_end), :, :
                ]
                continue
            seg_out_spikes[d0:d1, :, :] = core_output_spikes[int(sp.src_core)][
                int(sp.src_start) : int(sp.src_end), :, :
            ]

        seg_out_counts = seg_out_spikes.sum(axis=2).T
        seg_out_rates = seg_out_counts / float(T)

        if recorder_seg is not None:
            assert N == 1, "Spike recording requires a single sample (N == 1)"
            recorder_seg.seg_output_spike_count = seg_out_counts[0].astype(np.int64)
            for core_idx, core in enumerate(seg.cores):
                used_ax = used_axons(core, min_one=True)
                used_neu = used_neurons(core, min_one=True)
                latency = timing.core_latency(core)
                active_slice = slice(latency, latency + T)

                out_count = core_output_spikes[core_idx][
                    :used_neu, 0, active_slice
                ].sum(axis=1).astype(np.int64)

                in_count = np.zeros(used_ax, dtype=np.int64)
                n_always_on = 0
                for sp in core.get_axon_source_spans():
                    d0 = int(sp.dst_start)
                    if d0 >= used_ax:
                        continue
                    end = min(int(sp.dst_end), used_ax)
                    take = end - d0
                    if sp.kind == "off":
                        continue
                    if sp.kind == "on":
                        n_always_on += take
                        in_count[d0:end] += T
                        continue
                    if sp.kind == "input":
                        s0 = int(sp.src_start)
                        in_count[d0:end] += seg_input_logical[
                            s0:s0 + take, 0, active_slice
                        ].sum(axis=1).astype(np.int64)
                        continue
                    s0 = int(sp.src_start)
                    src_start = max(latency - 1, 0)
                    src_end = max(latency + T - 1, 0)
                    in_count[d0:end] += core_buffer_spikes[int(sp.src_core)][
                        s0:s0 + take, 0, src_start:src_end
                    ].sum(axis=1).astype(np.int64)

                recorder_seg.cores.append(
                    CoreSpikeCounts(
                        core_index=core_idx,
                        n_in_used=used_ax,
                        n_out_used=used_neu,
                        core_latency=int(core.latency) if core.latency is not None else -1,
                        has_hardware_bias=getattr(core, "hardware_bias", None) is not None,
                        n_always_on_axons=n_always_on,
                        input_spike_count=in_count,
                        output_spike_count=out_count,
                    )
                )

        return seg_out_rates.astype(np.float32)

    def _run_neural_segment(
        self,
        seg: HardCoreMapping,
        seg_input_rates: np.ndarray,
        *,
        recorder_seg: SegmentSpikeRecord | None = None,
    ) -> np.ndarray:
        """Execute one neural segment with HCM-equivalent Lava core dynamics."""
        return self._run_neural_segment_scheduled(
            seg, seg_input_rates, recorder_seg=recorder_seg,
        )
