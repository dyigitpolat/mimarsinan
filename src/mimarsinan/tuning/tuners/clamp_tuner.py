from mimarsinan.tuning.tuners.basic_tuner import BasicTuner

from mimarsinan.models.layers import ClampedReLU_Parametric, ClampedReLU

class ClampTuner(BasicTuner):
    def __init__(self, 
                 pipeline, 
                 model, 
                 target_accuracy, 
                 lr):
        
        super().__init__(
            pipeline, 
            model, 
            target_accuracy, 
            lr)

        self.lr = lr
        self.base_activation = model.activation

    def _get_target_decay(self):
        return 0.999
    
    def _get_previous_parameter_transform(self):
        return lambda x: x
    
    def _get_new_parameter_transform(self):
        return lambda x: x

    def _update_and_evaluate(self, rate):
        self.model.set_activation(ClampedReLU_Parametric(rate, self.base_activation))
        self.trainer.train_one_step(self.lr / 2)
        return self.trainer.validate()

    def run(self):
        super().run()
        self.model.set_activation(ClampedReLU())
        self.trainer.train_until_target_accuracy(self._find_lr() / 2, self.epochs, self._get_target())

        return self.trainer.validate()
