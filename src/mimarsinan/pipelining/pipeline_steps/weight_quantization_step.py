from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.tuning.tuners.normalization_aware_perceptron_quantization_tuner import NormalizationAwarePerceptronQuantizationTuner

class WeightQuantizationStep(PipelineStep):
    def __init__(self, pipeline):
        requires = []
        promises = []
        updates = ["model"]
        clears = []
        super().__init__(requires, promises, updates, clears, pipeline)

        self.tuner = None
    
    def validate(self):
        return self.tuner.validate()

    def process(self):
        self.tuner = NormalizationAwarePerceptronQuantizationTuner(
            self.pipeline,
            model = self.get_entry("model"),
            quantization_bits = self.pipeline.config['weight_bits'],
            target_tq = self.pipeline.config['target_tq'],
            target_accuracy = self.pipeline.get_target_metric(),
            lr = self.pipeline.config['lr'])
        self.tuner.run()

        self.update_entry("model", self.tuner.model, 'torch_model')