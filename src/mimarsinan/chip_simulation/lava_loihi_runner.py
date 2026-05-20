"""Host-scheduled Lava Loihi simulation of a HybridHardCoreMapping."""

from __future__ import annotations

import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from mimarsinan.chip_simulation.hybrid_execution import (
    assemble_segment_input_numpy,
    execute_compute_op_numpy,
    gather_final_output_numpy,
    store_segment_output_numpy,
)
from mimarsinan.chip_simulation.simulation_runner import SimulationRunner
from mimarsinan.chip_simulation.spike_recorder import (
    CoreSpikeCounts,
    RunRecord,
    SegmentSpikeRecord,
)
from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory, shutdown_data_loader
from mimarsinan.mapping.hybrid_hardcore_mapping import (
    HybridHardCoreMapping,
    HybridStage,
)
from mimarsinan.mapping.chip_latency import ChipLatency
from mimarsinan.mapping.softcore_mapping import HardCore, HardCoreMapping


# SubtractiveLIFReset is lazy-imported so missing Lava does not break import.

_SUBTRACTIVE_LIF_CLS = None
_LAVA_DTYPE = np.float64


def _make_set_start_method_idempotent() -> None:
    """No-op ``set_start_method`` when a multiprocessing context is already set."""
    import multiprocessing as mp
    import torch.multiprocessing as torch_mp

    def _patch(module) -> None:
        real_set = module.set_start_method
        if getattr(real_set, "_mimarsinan_lava_safe", False):
            return

        def _safe_set(method, force=False):
            current = module.get_start_method(allow_none=True)
            if force or current is None:
                real_set(method, force=force)
            pass  # context already set; skip redundant call

        _safe_set._mimarsinan_lava_safe = True
        module.set_start_method = _safe_set

    _patch(mp)
    _patch(torch_mp)


def _subtractive_lif_cls():
    """Lazy-import and cache SubtractiveLIFReset."""
    global _SUBTRACTIVE_LIF_CLS
    if _SUBTRACTIVE_LIF_CLS is None:
        _make_set_start_method_idempotent()
        from mimarsinan.chip_simulation.subtractive_lif import SubtractiveLIFReset
        _SUBTRACTIVE_LIF_CLS = SubtractiveLIFReset
    return _SUBTRACTIVE_LIF_CLS


from mimarsinan.chip_simulation._spike_encoding import (
    uniform_rate_encode as _uniform_rate_encode,
)


@dataclass
class _StageTrace:
    name: str
    kind: str
    seconds: float
    cores: int = 0
    samples: int = 0


@dataclass
class _RunProfile:
    stages: List[_StageTrace] = field(default_factory=list)
    total_seconds: float = 0.0

    def log(self) -> None:
        print("=== LavaLoihiRunner profile ===")
        by_kind: Dict[str, float] = defaultdict(float)
        for st in self.stages:
            per = f", {st.cores} cores" if st.kind == "neural" else ""
            print(f"  [{st.kind:>7}] {st.seconds:7.2f}s  {st.name}{per}")
            by_kind[st.kind] += st.seconds
        print(f"  ---")
        for k, s in by_kind.items():
            print(f"  Σ {k:>7}: {s:7.2f}s")
        print(f"  Σ total : {self.total_seconds:7.2f}s")


@dataclass(frozen=True)
class _SegmentTiming:
    """Logical sample layout for one Lava neural-segment run."""

    T: int
    segment_latency: int
    sample_stride: int
    pad_head: int = 2
    pipeline_delay: int = 1
    tail: int = 3

    @classmethod
    def from_mapping(cls, mapping: HardCoreMapping, T: int) -> "_SegmentTiming":
        latency = int(ChipLatency(mapping).calculate())
        return cls(T=int(T), segment_latency=latency, sample_stride=int(T) + latency)

    @property
    def warmup_cycles(self) -> int:
        return self.sample_stride

    @property
    def logical_start(self) -> int:
        return self.pad_head + self.warmup_cycles + self.pipeline_delay

    def total_steps(self, n_samples: int) -> int:
        return self.pad_head + self.warmup_cycles + n_samples * self.sample_stride + self.tail

    def core_latency(self, core: HardCore) -> int:
        return max(int(core.latency) if core.latency is not None else 0, 0)

    def active_start(self, core: HardCore) -> int:
        # Lava time_step is 1-based; logical_start is 0-based sink index.
        return (self.logical_start + self.core_latency(core) + 1) % self.sample_stride

    def sample_start(self) -> int:
        return (self.logical_start + 1) % self.sample_stride

    def extract_logical(self, raw: np.ndarray, n_samples: int) -> np.ndarray:
        """Return ``raw`` as (channels, samples, logical_cycles_per_sample)."""
        start = self.logical_start
        end = start + n_samples * self.sample_stride
        return np.asarray(raw[:, start:end], dtype=_LAVA_DTYPE).reshape(
            raw.shape[0], n_samples, self.sample_stride,
        )


class LavaLoihiRunner:
    """Evaluate a HybridHardCoreMapping under a Lava process graph."""

    def __init__(
        self,
        pipeline,
        mapping: HybridHardCoreMapping,
        simulation_length: int,
        preprocessor: nn.Module | None = None,
        *,
        thresholding_mode: str = "<=",
    ):
        self.pipeline = pipeline
        self.mapping = mapping
        self.T = int(simulation_length)
        self.preprocessor = preprocessor if preprocessor is not None else nn.Identity()
        if pipeline is not None:
            thresholding_mode = pipeline.config.get(
                "thresholding_mode", thresholding_mode,
            )
        if thresholding_mode not in ("<", "<="):
            raise ValueError(
                f"thresholding_mode must be '<' or '<='; got {thresholding_mode!r}"
            )
        self.thresholding_mode = str(thresholding_mode)
        firing_mode = "Default"
        if pipeline is not None:
            firing_mode = str(pipeline.config.get("firing_mode", "Default"))
        from mimarsinan.chip_simulation.firing_strategy import FiringStrategyFactory

        self._firing_strategy = FiringStrategyFactory.from_config(
            {
                "firing_mode": firing_mode,
                "thresholding_mode": self.thresholding_mode,
                "spiking_mode": "lif",
            }
        )
        self._firing_strategy.require_backend("lava")

        # pipeline=None: harness mode (spike-parity test; no data loaders).
        if pipeline is None:
            self.device = None
            self.max_samples = 0
            self._data_loader_factory = None
        else:
            self.device = pipeline.config["device"]
            self.max_samples = int(
                pipeline.config.get(
                    "max_loihi_samples",
                    1,
                )
            )

            spiking = pipeline.config.get("spiking_mode", "lif")
            if spiking != "lif":
                raise ValueError(
                    f"LavaLoihiRunner only supports spiking_mode='lif'; got {spiking!r}. "
                    "TTFS modes do not map onto Loihi LIF dynamics."
                )

            self._data_loader_factory = DataLoaderFactory(pipeline.data_provider_factory)

        _subtractive_lif_cls()  # fail fast on missing Lava

        self._profile = _RunProfile()
        self._accuracy: float | None = None

        self._recorder: RunRecord | None = None


    def _load_test_samples(self) -> Tuple[np.ndarray, np.ndarray]:
        provider = self._data_loader_factory.create_data_provider()
        loader = self._data_loader_factory.create_test_loader(
            provider.get_test_batch_size(), provider
        )
        xs: List[np.ndarray] = []
        ys: List[np.ndarray] = []
        total = 0
        for x, y in loader:
            if total >= self.max_samples:
                break
            x_np = x.detach().cpu().numpy()
            y_np = y.detach().cpu().numpy()
            if total + len(x_np) > self.max_samples:
                take = self.max_samples - total
                x_np = x_np[:take]
                y_np = y_np[:take]
            xs.append(x_np)
            ys.append(y_np)
            total += len(x_np)
        shutdown_data_loader(loader)
        return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


    def _preprocess(self, x_np: np.ndarray) -> np.ndarray:
        """Apply the shared preprocessor then flatten to (N, input_size)."""
        x = torch.tensor(x_np, dtype=torch.float32)
        with torch.no_grad():
            x = self.preprocessor(x)
        return x.detach().cpu().numpy().reshape(x.shape[0], -1)


    def _run_core_lava(
        self,
        *,
        weights: np.ndarray,
        threshold: float,
        hardware_bias: np.ndarray | None,
        input_spikes: np.ndarray,
    ) -> np.ndarray:
        """Run one hard core's Dense + SubtractiveLIFReset on Lava."""
        from lava.magma.core.run_conditions import RunSteps
        from lava.magma.core.run_configs import Loihi2SimCfg
        from lava.proc.dense.process import Dense
        from lava.proc.io.sink import RingBuffer as Sink
        from lava.proc.io.source import RingBuffer as Source

        n_out, n_in = weights.shape
        total_cycles = input_spikes.shape[1]
        assert total_cycles % self.T == 0
        N = total_cycles // self.T
        T = self.T

        PAD_HEAD = 2  # absorbs the first 2 cycles when LIF has no useful input
        TAIL = 3      # covers Source→Sink delivery delay for the last sample
        PIPELINE_DELAY = 1  # empirically: data[k] → sink index k + 1

        pad_head_block = np.zeros((n_in, PAD_HEAD), dtype=np.float32)
        warmup = input_spikes[:, :T]
        tail_block = np.zeros((n_in, TAIL), dtype=np.float32)
        data = np.concatenate([pad_head_block, warmup, input_spikes, tail_block], axis=1).astype(_LAVA_DTYPE)
        total_steps = data.shape[1]

        reset_offset = (PAD_HEAD + T + PIPELINE_DELAY + 1) % T

        if hardware_bias is None:
            bias_mant = np.zeros((n_out,), dtype=_LAVA_DTYPE)
        else:
            bias_mant = np.asarray(hardware_bias, dtype=_LAVA_DTYPE).reshape(-1)

        SubLIF = _subtractive_lif_cls()
        src = Source(data=data)
        dense = Dense(weights=weights.astype(_LAVA_DTYPE))
        lif = SubLIF(
            shape=(n_out,),
            du=1,
            dv=0,
            vth=float(threshold),
            bias_mant=bias_mant,
            reset_interval=T,
            reset_offset=reset_offset,
            thresholding_mode=self.thresholding_mode,
            zero_reset=(self._firing_strategy.mode.value == "Novena"),
        )
        sink = Sink(shape=(n_out,), buffer=total_steps)

        src.s_out.connect(dense.s_in)
        dense.a_out.connect(lif.a_in)
        lif.s_out.connect(sink.a_in)

        try:
            lif.run(
                condition=RunSteps(num_steps=total_steps),
                run_cfg=Loihi2SimCfg(select_tag="floating_pt"),
            )
            raw = sink.data.get()  # (n_out, total_steps)
        finally:
            lif.stop()

        start = PAD_HEAD + T + PIPELINE_DELAY
        return np.asarray(raw[:, start : start + N * T], dtype=_LAVA_DTYPE)


    def _run_neural_segment_scheduled(
        self,
        seg: HardCoreMapping,
        seg_input_rates: np.ndarray,
        *,
        recorder_seg: SegmentSpikeRecord | None = None,
    ) -> np.ndarray:
        """Execute a segment with host-scheduled routing and Lava per-core LIF."""
        from mimarsinan.mapping.spike_source_spans import compress_spike_sources

        T = self.T
        N = seg_input_rates.shape[0]
        seg_in_size = seg_input_rates.shape[1]
        timing = _SegmentTiming.from_mapping(seg, T)

        seg_input_spikes = _uniform_rate_encode(seg_input_rates, T)
        seg_input_logical = np.zeros(
            (seg_in_size, N, timing.sample_stride), dtype=_LAVA_DTYPE,
        )
        seg_input_logical[:, :, :T] = seg_input_spikes.transpose(1, 0, 2)

        core_output_spikes: Dict[int, np.ndarray] = {}
        core_buffer_spikes: Dict[int, np.ndarray] = {}

        from mimarsinan.mapping.core_geometry import used_axons, used_neurons

        deps = {
            idx: sorted(
                {
                    int(sp.src_core)
                    for sp in core.get_axon_source_spans()
                    if sp.kind == "core"
                }
            )
            for idx, core in enumerate(seg.cores)
        }
        topo_order: list[int] = []
        visiting: set[int] = set()
        visited: set[int] = set()

        def visit(idx: int) -> None:
            if idx in visited:
                return
            if idx in visiting:
                raise RuntimeError(f"Cycle detected in neural segment at core {idx}")
            visiting.add(idx)
            for dep in deps[idx]:
                visit(dep)
            visiting.remove(idx)
            visited.add(idx)
            topo_order.append(idx)

        for idx in sorted(
            range(len(seg.cores)),
            key=lambda i: (timing.core_latency(seg.cores[i]), i),
        ):
            visit(idx)

        import time as _time
        t0 = _time.time()
        verbose = os.environ.get("MIMARSINAN_LOIHI_QUIET") != "1"
        n_cores = len(topo_order)
        if verbose:
            print(
                f"  [LavaLoihiRunner] segment with {n_cores} cores; T={T}, N={N} "
                f"— per-core timing follows",
                flush=True,
            )
        for step_idx, core_idx in enumerate(topo_order):
            t_core = _time.time()
            core = seg.cores[core_idx]
            latency = timing.core_latency(core)
            used_ax = used_axons(core, min_one=True)
            used_neu = used_neurons(core, min_one=True)
            active_input = np.zeros((used_ax, N, T), dtype=_LAVA_DTYPE)

            for sp in core.get_axon_source_spans():
                d0 = int(sp.dst_start)
                if d0 >= used_ax:
                    continue
                end = min(int(sp.dst_end), used_ax)
                take = end - d0
                if sp.kind == "off":
                    continue
                if sp.kind == "on":
                    active_input[d0:end, :, :] = 1.0
                    continue
                if sp.kind == "input":
                    s0 = int(sp.src_start)
                    active_input[d0:end, :, :] = seg_input_logical[
                        s0:s0 + take, :, latency:latency + T,
                    ]
                    continue

                src_core_id = int(sp.src_core)
                if src_core_id not in core_buffer_spikes:
                    raise RuntimeError(
                        f"Core {core_idx} depends on core {src_core_id}, "
                        "but the source has not been scheduled yet."
                    )
                s0 = int(sp.src_start)
                for local_cycle in range(T):
                    src_cycle = latency + local_cycle - 1
                    if src_cycle < 0:
                        continue
                    active_input[d0:end, :, local_cycle] = core_buffer_spikes[
                        src_core_id
                    ][s0:s0 + take, :, src_cycle]

            weights = np.asarray(
                core.core_matrix[:used_ax, :used_neu], dtype=_LAVA_DTYPE,
            ).T
            hardware_bias = (
                np.asarray(core.hardware_bias[:used_neu], dtype=_LAVA_DTYPE)
                if getattr(core, "hardware_bias", None) is not None
                else None
            )
            t_lava = _time.time()
            active_output = self._run_core_lava(
                weights=weights,
                threshold=float(core.threshold),
                hardware_bias=hardware_bias,
                input_spikes=active_input.reshape(used_ax, N * T),
            ).reshape(used_neu, N, T)
            lava_dt = _time.time() - t_lava

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
                    f"    core {step_idx + 1:>4}/{n_cores} "
                    f"(idx={core_idx}, ax={used_ax}, neu={used_neu}, "
                    f"lat={latency}): lava={lava_dt*1000:7.1f}ms  "
                    f"core_total={(_time.time() - t_core)*1000:7.1f}ms",
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
            # Neuron sources: use core_output_spikes (active window only), not held buffers.
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


    def run(self) -> float:
        t_total = time.time()

        x_np, y_np = self._load_test_samples()
        N = int(x_np.shape[0])
        print(f"[LavaLoihiRunner] Loaded {N} test samples; T={self.T}")

        x_flat = self._preprocess(x_np)
        state_buffer: Dict[int, np.ndarray] = {-2: x_flat}

        from mimarsinan.chip_simulation.hybrid_execution import resolve_stage_compute_scales
        from mimarsinan.chip_simulation.hybrid_stage_runner import run_hybrid_stages

        def _on_neural(_stage_index, stage, state_buffer):
            t0 = time.time()
            seg = stage.hard_core_mapping
            assert seg is not None
            seg_input = assemble_segment_input_numpy(stage.input_map, state_buffer, N)
            seg_output = self._run_neural_segment(seg, seg_input)
            store_segment_output_numpy(stage.output_map, state_buffer, seg_output)
            self._profile.stages.append(
                _StageTrace(
                    name=stage.name,
                    kind="neural",
                    seconds=time.time() - t0,
                    cores=len(seg.cores),
                    samples=N,
                )
            )

        def _on_compute(_stage_index, stage, state_buffer):
            t0 = time.time()
            assert stage.compute_op is not None
            op_id = stage.compute_op.id
            ttfs_in_scale, ttfs_out_scale = resolve_stage_compute_scales(
                self.mapping, op_id, apply_ttfs=False
            )
            result = execute_compute_op_numpy(
                stage.compute_op,
                x_flat,
                state_buffer,
                in_scale=ttfs_in_scale,
                out_scale=ttfs_out_scale,
            )
            state_buffer[op_id] = result
            self._profile.stages.append(
                _StageTrace(
                    name=stage.name,
                    kind="compute",
                    seconds=time.time() - t0,
                    samples=N,
                )
            )

        run_hybrid_stages(
            self.mapping,
            state_buffer,
            on_neural=_on_neural,
            on_compute=_on_compute,
        )

        final = gather_final_output_numpy(
            self.mapping.output_sources, state_buffer, x_flat, N
        )
        preds = np.argmax(final, axis=1)
        correct = int((preds == y_np).sum())
        self._accuracy = correct / max(1, N)

        self._profile.total_seconds = time.time() - t_total
        self._profile.log()
        self.pipeline.reporter.report("Loihi Simulation", self._accuracy)
        return self._accuracy

    @property
    def accuracy(self) -> float | None:
        return self._accuracy

    def run_segments_from_reference(self, ref: RunRecord) -> RunRecord:
        """Run neural stages on Loihi using inputs from a reference RunRecord."""
        assert ref.T == self.T, (
            f"Reference T={ref.T} does not match runner T={self.T}; "
            "check simulation_length consistency."
        )

        out = RunRecord(
            sample_index=ref.sample_index,
            T=ref.T,
            segments={},
            compute_outputs=dict(ref.compute_outputs),
        )
        self._recorder = out
        try:
            for stage_index, stage in enumerate(self.mapping.stages):
                if stage.kind != "neural":
                    continue
                if stage_index not in ref.segments:
                    raise KeyError(
                        f"Reference RunRecord is missing segment for stage_index={stage_index} "
                        f"({stage.name!r}); HCM may have skipped it"
                    )
                ref_seg = ref.segments[stage_index]

                seg = stage.hard_core_mapping
                assert seg is not None
                seg_input_rates = ref_seg.seg_input_rates

                actual_seg = SegmentSpikeRecord(
                    stage_index=stage_index,
                    stage_name=stage.name,
                    schedule_segment_index=stage.schedule_segment_index,
                    schedule_pass_index=stage.schedule_pass_index,
                    seg_input_rates=seg_input_rates,
                    seg_input_spike_count=_uniform_rate_encode(seg_input_rates, self.T)[0]
                        .sum(axis=1)
                        .astype(np.int64),
                    seg_output_spike_count=np.zeros(0, dtype=np.int64),
                )

                seg_out_rates = self._run_neural_segment(
                    seg, seg_input_rates, recorder_seg=actual_seg,
                )

                if actual_seg.seg_output_spike_count.size == 0:
                    actual_seg.seg_output_spike_count = (
                        np.rint(seg_out_rates[0] * self.T).astype(np.int64)
                    )

                out.segments[stage_index] = actual_seg
        finally:
            self._recorder = None
        return out
