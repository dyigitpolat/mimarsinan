"""Host-scheduled Lava Loihi simulation runner."""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from mimarsinan.chip_simulation.behavior_config import NeuralBehaviorConfig
from mimarsinan.chip_simulation.execution_bounds import resolve_simulation_step_timeout_s
from mimarsinan.chip_simulation.hybrid_run.hybrid_execution import (
    apply_input_shifts_numpy,
    assemble_segment_input_numpy,
    compute_input_state_with_shifts,
    execute_compute_op_numpy,
    gather_final_output_numpy,
    resolve_stage_compute_scales,
    store_segment_output_numpy,
)
from mimarsinan.chip_simulation.hybrid_run.hybrid_stage_runner import run_hybrid_stages
from mimarsinan.chip_simulation.lava_loihi.core_lava import LavaCoreMixin, _subtractive_lif_cls
from mimarsinan.chip_simulation.lava_loihi.segment_runner import LavaSegmentMixin
from mimarsinan.chip_simulation.lava_loihi.timing import _RunProfile, _StageTrace
from mimarsinan.chip_simulation.recording.spike_recorder import RunRecord, SegmentSpikeRecord
from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory, shutdown_data_loader
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import HybridHardCoreMapping
from mimarsinan.spiking.segment_boundary import (
    boundary_normalization_scales,
    normalize_boundary_slices_numpy,
)


class LavaLoihiRunner(LavaCoreMixin, LavaSegmentMixin):
    """Evaluate a HybridHardCoreMapping under a Lava process graph."""

    def __init__(
        self,
        mapping: HybridHardCoreMapping,
        simulation_length: int,
        behavior: NeuralBehaviorConfig,
        pipeline=None,
        preprocessor: nn.Module | None = None,
    ):
        self.pipeline = pipeline
        self.mapping = mapping
        self.T = int(simulation_length)
        self._behavior = behavior
        behavior.require_backend("lava")
        self._firing_strategy = behavior.firing_strategy()
        self.thresholding_mode = behavior.thresholding_mode
        self.preprocessor = preprocessor if preprocessor is not None else nn.Identity()

        if pipeline is None:
            self.device = None
            self.max_samples = 0
            self._data_loader_factory = None
            self._simulation_step_timeout_s = resolve_simulation_step_timeout_s(None)
        else:
            self.device = pipeline.config["device"]
            self.max_samples = int(pipeline.config.get("max_loihi_samples", 1))
            self._data_loader_factory = DataLoaderFactory.for_pipeline(pipeline)
            self._simulation_step_timeout_s = resolve_simulation_step_timeout_s(
                pipeline.config.get("simulation_step_timeout_s")
            )

        _subtractive_lif_cls()
        self._profile = _RunProfile()
        self._accuracy: float | None = None
        self._recorder: RunRecord | None = None

    def _load_test_samples(self) -> Tuple[np.ndarray, np.ndarray]:
        factory = self._data_loader_factory
        if factory is None:
            raise RuntimeError("LavaLoihiRunner requires a pipeline to load test samples")
        provider = factory.create_data_provider()
        loader = factory.create_test_loader(
            provider.get_test_batch_size(), provider,
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
        x = torch.tensor(x_np, dtype=torch.float32)
        with torch.no_grad():
            x = self.preprocessor(x)
        return x.detach().cpu().numpy().reshape(x.shape[0], -1)

    def run(self) -> float:
        if self.pipeline is None:
            raise RuntimeError("LavaLoihiRunner.run() requires a pipeline for sample loading")
        t_total = time.time()
        x_np, y_np = self._load_test_samples()
        N = int(x_np.shape[0])
        print(f"[LavaLoihiRunner] Loaded {N} test samples; T={self.T}")

        x_flat = self._preprocess(x_np)
        state_buffer: Dict[int, np.ndarray] = {-2: x_flat}
        # Lava runs host ComputeOps with (1, 1) scales (rate/LIF semantics), so
        # boundary slices carry value-domain results until normalized here.
        wire_divisors = boundary_normalization_scales(self.mapping)
        node_output_shifts = getattr(self.mapping, "node_output_shifts", None)

        def _on_neural(_stage_index, stage, state_buffer):
            t0 = time.time()
            seg = stage.hard_core_mapping
            assert seg is not None
            seg_input = assemble_segment_input_numpy(stage.input_map, state_buffer, N)
            seg_input = normalize_boundary_slices_numpy(
                stage.input_map, seg_input, wire_divisors,
            )
            seg_input = apply_input_shifts_numpy(
                stage.input_map, seg_input, node_output_shifts,
            )
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
                self.mapping, op_id, apply_ttfs=False, op=stage.compute_op,
            )
            result = execute_compute_op_numpy(
                stage.compute_op, x_flat,
                compute_input_state_with_shifts(
                    stage.compute_op, state_buffer, node_output_shifts,
                ),
                in_scale=ttfs_in_scale, out_scale=ttfs_out_scale,
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
            self.mapping, state_buffer, on_neural=_on_neural, on_compute=_on_compute,
        )

        final = gather_final_output_numpy(self.mapping.output_sources, state_buffer, x_flat, N)
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

                encoded = self._behavior.encode_segment_input(seg_input_rates, self.T)
                actual_seg = SegmentSpikeRecord(
                    stage_index=stage_index,
                    stage_name=stage.name,
                    schedule_segment_index=stage.schedule_segment_index,
                    schedule_pass_index=stage.schedule_pass_index,
                    seg_input_rates=seg_input_rates,
                    seg_input_spike_count=encoded[0].sum(axis=1).astype(np.int64),
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
