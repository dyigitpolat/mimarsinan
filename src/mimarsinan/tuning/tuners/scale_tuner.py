from mimarsinan.tuning.tuners.basic_tuner import BasicTuner

class ScaleTuner(BasicTuner):
    def __init__(self, 
                 pipeline, 
                 model, 
                 target_accuracy, 
                 lr,
                 adaptation_manager):
        
        super().__init__(
            pipeline, 
            model, 
            target_accuracy, 
            lr)

        self.lr = lr
        self.adaptation_manager = adaptation_manager

    def _get_target_decay(self):
        return 0.999
    
    def _get_previous_parameter_transform(self):
        return lambda x: x
    
    def _get_new_parameter_transform(self):
        return lambda x: x

    def _update_and_evaluate(self, rate):
        self.adaptation_manager.scale_rate = rate
        for perceptron in self.model.get_perceptrons():
            self.adaptation_manager.update_activation(perceptron)

        self.trainer.train_one_step(self._find_lr())
        return self.trainer.validate()

    def run(self):
        super().run()
        return self.trainer.validate()
