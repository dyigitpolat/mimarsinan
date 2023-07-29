from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.tuning.tuners.weight_quantization_tuner import WeightQuantizationTuner

class WeightQuantizationStep(PipelineStep):
    def __init__(self, pipeline):
        requires = ["pf_model"]
        promises = ["wq_model"]
        clears = ["pf_model"]
        super().__init__(requires, promises, clears, pipeline)

        self.tuner = None
    
    def validate(self):
        return self.tuner.validate()

    def process(self):
        self.tuner = WeightQuantizationTuner(
            self.pipeline,
            model = self.pipeline.cache["pf_model"],
            quantization_bits = self.pipeline.config['weight_bits'],
            target_tq = self.pipeline.config['target_tq'],
            target_accuracy = self.pipeline.get_target_metric(),
            lr = self.pipeline.config['lr'])
        self.tuner.run()

        self.pipeline.cache.add("wq_model", self.tuner.model, 'torch_model')
        self.pipeline.cache.remove("pf_model")