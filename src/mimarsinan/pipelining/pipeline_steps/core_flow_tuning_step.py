from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.tuning.tuners.core_flow_tuner import CoreFlowTuner
from mimarsinan.tuning.tuners.unified_ir_core_flow_tuner import UnifiedIRCoreFlowTuner
from mimarsinan.models.layers import TransformedActivation, ClampDecorator, QuantizeDecorator, ScaleDecorator

import torch.nn as nn
import torch

from math import ceil

class CoreFlowTuningStep(PipelineStep):
    def __init__(self, pipeline):
        requires = ["soft_core_mapping", "activation_scales", "model", "ir_graph"]
        promises = ["tuned_soft_core_mapping", "scaled_simulation_length"]
        updates = ["ir_graph"]
        clears = ["soft_core_mapping"]
        super().__init__(requires, promises, updates, clears, pipeline)
        
        self.tuner = None
        self.preprocessor = None

    def validate(self):
        if self.tuner is None:
            # For unified IR path, just return target metric
            return self.pipeline.get_target_metric()
        return self.tuner.validate()

    def process(self):
        model = self.get_entry('model')
        soft_core_mapping = self.get_entry('soft_core_mapping')
        ir_graph = self.get_entry('ir_graph')
        activation_scales = self.get_entry('activation_scales')  # Access required entry
        
        # Check if we have ComputeOps (unified IR path)
        has_compute_ops = len(ir_graph.get_compute_ops()) > 0 if ir_graph else False
        
        if has_compute_ops or soft_core_mapping is None:
            # Unified IR path - DO NOT skip: tune thresholds (and optionally T) end-to-end,
            # including ComputeOp sync barriers.
            print("[CoreFlowTuningStep] Model contains ComputeOps - running UnifiedIRCoreFlowTuner")

            self.preprocessor = nn.Sequential(model.get_preprocessor(), model.in_act)

            # Keep tuning lightweight by default. You can widen this list later.
            tuner = UnifiedIRCoreFlowTuner(self.pipeline, ir_graph, self.preprocessor)
            result = tuner.run(
                threshold_scales=[0.5, 0.75, 1.0, 1.25, 1.5],
                simulation_lengths=[int(self.pipeline.config["simulation_steps"])],
            )

            self.tuner = tuner

            print(
                f"[CoreFlowTuningStep] Unified IR tuned val acc={result.best_validation_accuracy} "
                f"(threshold_scale={result.best_threshold_scale}, T={result.scaled_simulation_length})"
            )

            # Update ir_graph in cache for downstream steps (HardCoreMapping/Simulation).
            self.update_entry("ir_graph", result.tuned_ir_graph, "pickle")

            self.add_entry("scaled_simulation_length", result.scaled_simulation_length)
            self.add_entry("tuned_soft_core_mapping", None, 'pickle')
            return
        
        scale = model.get_perceptrons()[0].scale_factor
        scale = max(activation_scales)
        print(model.get_perceptrons()[0].scale_factor)
        print(max(activation_scales))
        
        self.preprocessor = nn.Sequential(
            model.get_preprocessor(),
            model.in_act)
        
        self.tuner = CoreFlowTuner(
            self.pipeline, soft_core_mapping, self.preprocessor)
        scaled_simulation_length = self.tuner.run()

        # Keep ir_graph updated even for neural-only models (no-op update).
        self.update_entry("ir_graph", ir_graph, "pickle")

        self.add_entry("scaled_simulation_length", scaled_simulation_length)
        self.add_entry("tuned_soft_core_mapping", self.tuner.mapping, 'pickle')