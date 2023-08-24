from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.tuning.tuners.normalization_aware_perceptron_quantization_tuner import NormalizationAwarePerceptronQuantizationTuner

class WeightQuantizationStep(PipelineStep):
    def __init__(self, pipeline):
        requires = ["aq_model"]
        promises = ["wq_model"]
        clears = ["aq_model"]
        super().__init__(requires, promises, clears, pipeline)

        self.tuner = None
    
    def validate(self):
        return self.tuner.validate()

    def process(self):
        self.tuner = NormalizationAwarePerceptronQuantizationTuner(
            self.pipeline,
            model = self.get_entry("aq_model"),
            quantization_bits = self.pipeline.config['weight_bits'],
            target_tq = self.pipeline.config['target_tq'],
            target_accuracy = self.pipeline.get_target_metric(),
            lr = self.pipeline.config['lr'])
        self.tuner.run()

        self.add_entry("wq_model", self.get_entry("aq_model"), 'torch_model')