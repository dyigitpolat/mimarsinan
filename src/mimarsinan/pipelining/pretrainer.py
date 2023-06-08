from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.models.layers import ClampedReLU, NoisyDropout

class Pretrainer:
    def __init__(self, pipeline, epochs):
        # Dependencies
        self.model = pipeline.model

        # Trainer
        self.trainer = BasicTrainer(
            self.model, 
            pipeline.device, 
            pipeline.data_provider, 
            pipeline.loss)
        self.trainer.report_function = pipeline.reporter.report
        
        self.lr = pipeline.lr
        
        # Epochs
        self.epochs = epochs
        
    def run(self):
        self.model.set_activation(ClampedReLU())
        self.model.set_regularization(NoisyDropout(0.5, 0.5, 0.2))

        return self.trainer.train_n_epochs(self.lr, self.epochs)