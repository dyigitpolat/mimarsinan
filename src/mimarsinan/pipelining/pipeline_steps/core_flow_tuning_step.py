from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.tuning.tuners.core_flow_tuner import CoreFlowTuner

import torch.nn as nn

class CoreFlowTuningStep(PipelineStep):
    def __init__(self, pipeline):
        # Unified-only: always tune the IRGraph end-to-end (NeuralCore + ComputeOp),
        # regardless of whether the model contains ComputeOps.
        requires = ["model", "ir_graph"]
        promises = ["scaled_simulation_length"]
        updates = ["ir_graph"]
        clears = []
        super().__init__(requires, promises, updates, clears, pipeline)
        
        self.tuner = None
        self.preprocessor = None

    def validate(self):
        return self.tuner.validate() if self.tuner is not None else self.pipeline.get_target_metric()

    def process(self):
        model = self.get_entry("model")
        ir_graph = self.get_entry("ir_graph")

        has_compute_ops = len(ir_graph.get_compute_ops()) > 0 if ir_graph else False
        if has_compute_ops:
            print("[CoreFlowTuningStep] Unified IR tuning (ComputeOps present): CoreFlowTuner")
        else:
            print("[CoreFlowTuningStep] Unified IR tuning (neural-only): CoreFlowTuner")

        self.preprocessor = nn.Sequential(model.get_preprocessor(), model.in_act)

        # Run threshold tuning using spike rate matching algorithm
        tuner = CoreFlowTuner(self.pipeline, ir_graph, self.preprocessor)
        result = tuner.run()

        self.tuner = tuner

        print(
            f"[CoreFlowTuningStep] Unified IR tuned val acc={result.best_validation_accuracy} "
            f"(T={result.scaled_simulation_length})"
        )

        # Update ir_graph in cache for downstream steps (HardCoreMapping/Simulation).
        self.update_entry("ir_graph", result.tuned_ir_graph, "pickle")
        self.add_entry("scaled_simulation_length", result.scaled_simulation_length)