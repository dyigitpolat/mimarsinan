from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.pipelining.pipeline_steps.tuners.weight_quantization_tuner import WeightQuantizationTuner 

class WeightQuantizationStep(PipelineStep):
    def __init__(self, pipeline):
        requires = ["nf_model", "nf_accuracy"]
        promises = ["wq_model", "wq_accuracy"]
        clears = ["nf_model"]
        super().__init__(requires, promises, clears, pipeline)

    def process(self):
        tuner = WeightQuantizationTuner(
            self.pipeline,
            max_epochs = self.pipeline.config['aq_epochs'],
            model = self.pipeline.cache['nf_model'],
            quantization_bits = self.pipeline.config['weight_bits'],
            target_tq = self.pipeline.config['target_tq'],
            target_accuracy = self.pipeline.cache['nf_accuracy'],
            lr = self.pipeline.config['lr'])
        
        validation_accuracy = tuner.run()

        assert validation_accuracy > self.pipeline.cache['nf_accuracy'] * 0.9, \
            "Weight quantization step failed to retain validation accuracy."

        self.pipeline.cache.add("wq_model", tuner.model, 'torch_model')
        self.pipeline.cache.add("wq_accuracy", validation_accuracy)

        self.pipeline.cache.remove("nf_model")