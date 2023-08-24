from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.tuning.tuners.core_flow_tuner import CoreFlowTuner

class CoreFlowTuningStep(PipelineStep):
    def __init__(self, pipeline):
        requires = ["soft_core_mapping"]
        promises = ["tuned_soft_core_mapping", "scaled_simulation_length"]
        clears = ["soft_core_mapping"]
        super().__init__(requires, promises, clears, pipeline)
        
        self.tuner = None

    def validate(self):
        return self.tuner.validate()

    def process(self):
        self.tuner = CoreFlowTuner(
            self.pipeline, self.get_entry('soft_core_mapping'))
        scaled_simulation_length = self.tuner.run()

        self.add_entry("scaled_simulation_length", scaled_simulation_length)
        self.add_entry("tuned_soft_core_mapping", self.tuner.mapping, 'pickle')