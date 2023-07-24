from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.tuning.tuners.scale_tuner import ScaleTuner 

class ScaleAdaptationStep(PipelineStep):
    def __init__(self, pipeline):
        requires = ["ca_model", "ca_accuracy"]
        promises = ["sa_model", "sa_accuracy"]
        clears = ["ca_model"]
        super().__init__(requires, promises, clears, pipeline)

    def process(self):
        tuner = ScaleTuner(
            self.pipeline,
            model = self.pipeline.cache['ca_model'],
            target_accuracy = self.pipeline.cache['ca_accuracy'] * 0.99,
            lr = self.pipeline.config['lr'])
        
        accuracy = tuner.run()

        assert accuracy > self.pipeline.cache['ca_accuracy'] * 0.9, \
            "Scale adaptation step failed to retain validation accuracy."

        self.pipeline.cache.add("sa_model", tuner.model, 'torch_model')
        self.pipeline.cache.add("sa_accuracy", accuracy)

        self.pipeline.cache.remove("ca_model")

        
