from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.tuning.tuners.core_flow_tuner import CoreFlowTuner

class CoreFlowTuningStep(PipelineStep):
    def __init__(self, pipeline):
        requires = ["soft_core_mapping", "model"]
        promises = ["tuned_soft_core_mapping", "scaled_simulation_length"]
        updates = []
        clears = ["soft_core_mapping"]
        super().__init__(requires, promises, updates, clears, pipeline)
        
        self.tuner = None
        self.preprocessor = None

    def validate(self):
        return self.tuner.validate()

    def process(self):
        self.preprocessor = self.get_entry("model").get_preprocessor()
        
        self.tuner = CoreFlowTuner(
            self.pipeline, self.get_entry('soft_core_mapping'), self.preprocessor)
        scaled_simulation_length = self.tuner.run()

        self.add_entry("scaled_simulation_length", scaled_simulation_length)
        self.add_entry("tuned_soft_core_mapping", self.tuner.mapping, 'pickle')