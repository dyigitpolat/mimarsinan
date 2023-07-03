from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.models.layers import ClampedReLU, NoisyDropout

class PretrainingStep(PipelineStep):
    def __init__(self, pipeline):
        requires = ["init_model"]
        promises = ["pretrained_model", "val_accuracy"]
        clears = ["init_model"]
        super().__init__(requires, promises, clears, pipeline)


    def process(self):
        model = self.pipeline.cache["init_model"]

        trainer = BasicTrainer(
            model, 
            self.pipeline.config['device'], 
            self.pipeline.data_provider, 
            self.pipeline.loss)
        trainer.report_function = self.pipeline.reporter.report
        
        model.set_activation(ClampedReLU())
        model.set_regularization(NoisyDropout(0.5, 0.5, 0.2))

        validation_accuracy = trainer.train_n_epochs(
            self.pipeline.config['lr'], 
            self.pipeline.config['pt_epochs'])
        
        
        self.pipeline.cache.add("pretrained_model", model, 'torch_model')
        self.pipeline.cache.add("val_accuracy", validation_accuracy)
        
        self.pipeline.cache.remove("init_model")
