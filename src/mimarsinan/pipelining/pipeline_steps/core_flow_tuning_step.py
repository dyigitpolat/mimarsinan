from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.tuning.tuners.core_flow_tuner import CoreFlowTuner

class CoreFlowTuningStep(PipelineStep):
    def __init__(self, pipeline):
        requires = ["hard_core_mapping"]
        promises = ["tuned_hard_core_mapping"]
        clears = ["hard_core_mapping"]
        super().__init__(requires, promises, clears, pipeline)
        
        self.tuner = None

    def validate(self):
        return self.tuner.validate()

    def process(self):
        self.tuner = CoreFlowTuner(
            self.pipeline, self.pipeline.cache['hard_core_mapping'])
        self.tuner.run()

        self.pipeline.cache.add("tuned_hard_core_mapping", self.tuner.mapping, 'pickle')
        self.pipeline.cache.remove("hard_core_mapping")