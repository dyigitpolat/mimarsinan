from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.tuning.tuners.core_flow_tuner import CoreFlowTuner

class CoreFlowTuningStep(PipelineStep):
    def __init__(self, pipeline):
        requires = ["hard_core_mapping"]
        promises = ["tuned_hard_core_mapping", "tuned_cf_accuracy", "cf_threshold_scale"]
        clears = ["hard_core_mapping"]
        super().__init__(requires, promises, clears, pipeline)

    def process(self):
        tuner = CoreFlowTuner(self.pipeline, 
                              self.pipeline.cache['hard_core_mapping'])
        
        validation_accuracy, threshold_scale = tuner.run()

        self.pipeline.cache.add("tuned_hard_core_mapping", tuner.mapping, 'pickle')
        self.pipeline.cache.add("tuned_cf_accuracy", validation_accuracy)
        self.pipeline.cache.add("cf_threshold_scale", threshold_scale)

        self.pipeline.cache.remove("hard_core_mapping")