from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory
from mimarsinan.tuning.tuners.normalization_aware_perceptron_quantization_tuner import NormalizationAwarePerceptronQuantizationTuner

class WeightQuantizationStep(PipelineStep):
    def __init__(self, pipeline):
        requires = ["model"]
        promises = []
        updates = ["model"]
        clears = []
        super().__init__(requires, promises, updates, clears, pipeline)

        self.tuner = None
        self.trainer = None
    
    def validate(self):
        return self.trainer.validate()

    def process(self):
        # self.tuner = NormalizationAwarePerceptronQuantizationTuner(
        #     self.pipeline,
        #     model = self.get_entry("model"),
        #     quantization_bits = self.pipeline.config['weight_bits'],
        #     target_tq = self.pipeline.config['target_tq'],
        #     target_accuracy = self.pipeline.get_target_metric(),
        #     lr = self.pipeline.config['lr'])
        # self.tuner.run()

        self.trainer = BasicTrainer(
            self.get_entry("model"),
            self.pipeline.config['device'], 
            DataLoaderFactory(self.pipeline.data_provider_factory),
            self.pipeline.loss)
        self.trainer.report_function = self.pipeline.reporter.report

        for _ in range(10):
            self.trainer.train_until_target_accuracy(
                self.pipeline.config['lr'] / 20, 
                max_epochs=2, 
                target_accuracy=self.pipeline.get_target_metric())

            from mimarsinan.transformations.normalization_aware_perceptron_quantization import NormalizationAwarePerceptronQuantization
            for perceptron in self.get_entry("model").get_perceptrons():
                perceptron.layer = NormalizationAwarePerceptronQuantization(
                    self.pipeline.config['weight_bits'], self.pipeline.config['device']).transform(perceptron).layer

        self.update_entry("model", self.get_entry("model"), 'torch_model')