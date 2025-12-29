from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Iterable

import torch.nn as nn

from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory
from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.models.unified_core_flow import SpikingUnifiedCoreFlow


@dataclass
class UnifiedIRTuningResult:
    tuned_ir_graph: object
    scaled_simulation_length: int
    best_validation_accuracy: float
    best_threshold_scale: float


class UnifiedIRCoreFlowTuner:
    """
    Minimal-but-real tuner for unified IR graphs (NeuralCore + ComputeOp).

    Unlike the legacy CoreFlowTuner (which operates on SoftCoreMapping only),
    this tuner evaluates SpikingUnifiedCoreFlow end-to-end, including ComputeOp
    sync barriers (rate -> op -> respike).

    Currently implemented:
    - Global threshold scaling search over all NeuralCore nodes.
    - Optional simulation-length candidate search (kept simple).
    """

    def __init__(self, pipeline, ir_graph, preprocessor: nn.Module):
        self.pipeline = pipeline
        self.device = pipeline.config["device"]
        self.data_loader_factory = DataLoaderFactory(pipeline.data_provider_factory)

        self.input_shape = pipeline.config["input_shape"]
        self.base_simulation_steps = int(round(pipeline.config["simulation_steps"]))

        self.firing_mode = pipeline.config["firing_mode"]
        self.spike_mode = pipeline.config["spike_generation_mode"]
        self.thresholding_mode = pipeline.config["thresholding_mode"]

        self.preprocessor = preprocessor
        self.ir_graph = ir_graph

        self.report_function = pipeline.reporter.report

        self.accuracy = None

    def _make_flow(self, ir_graph, simulation_steps: int):
        return SpikingUnifiedCoreFlow(
            self.input_shape,
            ir_graph,
            int(simulation_steps),
            self.preprocessor,
            self.firing_mode,
            self.spike_mode,
            self.thresholding_mode,
        )

    def _validate(self, ir_graph, simulation_steps: int) -> float:
        trainer = BasicTrainer(self._make_flow(ir_graph, simulation_steps), self.device, self.data_loader_factory, None)
        trainer.report_function = self.report_function
        return trainer.validate()

    def run(
        self,
        *,
        threshold_scales: Iterable[float] | None = None,
        simulation_lengths: Iterable[int] | None = None,
    ) -> UnifiedIRTuningResult:
        if threshold_scales is None:
            # Cheap default search; enough to avoid skipping tuning.
            threshold_scales = [0.5, 0.75, 1.0, 1.25, 1.5]

        if simulation_lengths is None:
            # Keep it small to avoid long runs; prefer the configured length.
            simulation_lengths = [self.base_simulation_steps]

        best_acc = -1.0
        best_graph = None
        best_scale = 1.0
        best_T = self.base_simulation_steps

        # Baseline print for debugging.
        base_acc = self._validate(self.ir_graph, self.base_simulation_steps)
        print(f"  UnifiedIR baseline SpikingUnifiedCoreFlow val acc (T={self.base_simulation_steps}): {base_acc}")

        for T in simulation_lengths:
            for s in threshold_scales:
                g = copy.deepcopy(self.ir_graph)
                for core in g.get_neural_cores():
                    # Keep thresholds positive/integer-ish.
                    thr = float(core.threshold) if core.threshold is not None else 1.0
                    thr = max(1.0, thr * float(s))
                    core.threshold = thr

                acc = self._validate(g, int(T))
                print(f"  UnifiedIR tune: T={int(T)} threshold_scale={s} -> val acc={acc}")

                if acc > best_acc:
                    best_acc = acc
                    best_graph = g
                    best_scale = float(s)
                    best_T = int(T)

        assert best_graph is not None

        self.accuracy = best_acc
        self.ir_graph = best_graph
        return UnifiedIRTuningResult(
            tuned_ir_graph=best_graph,
            scaled_simulation_length=best_T,
            best_validation_accuracy=best_acc,
            best_threshold_scale=best_scale,
        )

    def validate(self):
        return self.accuracy



