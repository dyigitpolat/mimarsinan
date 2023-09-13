from mimarsinan.tuning.tuners.basic_tuner import BasicTuner

class ClampTuner(BasicTuner):
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

        for perceptron in self.model.get_perceptrons():
            self.adaptation_manager.update_activation(perceptron)

    def _get_target_decay(self):
        return 0.999
    
    def _get_previous_parameter_transform(self):
        return lambda x: x
    
    def _get_new_parameter_transform(self):
        return lambda x: x
    
    def _calculate_base_threshold(self, stats, rate):
        if stats.in_max is None:
            return 1.0
        
        clamp_limit = 1.0
        return clamp_limit * rate + (1.0 - rate) * stats.in_max

    def _update_and_evaluate(self, rate):
        self.adaptation_manager.clamp_rate = rate
        for perceptron in self.model.get_perceptrons():
            perceptron.base_threshold = \
                self._calculate_base_threshold(perceptron.activation.get_stats(), rate)
            self.adaptation_manager.update_activation(perceptron)

        self.trainer.train_one_step(self._find_lr())
        return self.trainer.validate()

    def run(self):
        super().run()
        
        return self.trainer.validate()
