from mimarsinan.tuning.tuners.perceptron_tuner import PerceptronTuner
class ClampTuner(PerceptronTuner):
    def __init__(self, 
                 pipeline, 
                 model, 
                 target_accuracy, 
                 lr,
                 adaptation_manager,
                 activation_scales):
        
        super().__init__(
            pipeline, 
            model, 
            target_accuracy, 
            lr)

        self.lr = lr
        self.adaptation_manager = adaptation_manager
        self.activation_scales = activation_scales

        for perceptron in self.model.get_perceptrons():
            self.adaptation_manager.update_activation(self.pipeline.config, perceptron)

    def _get_target_decay(self):
        return 0.999
    
    def _get_previous_perceptron_transform(self, _):
        return lambda p: None
    
    def _get_new_perceptron_transform(self, _):
        return lambda p: None
    
    def _calculate_activation_scales(self, in_max):
        for perceptron in self.model.get_perceptrons():
            perceptron.set_scale_factor(in_max)
            perceptron.set_activation_scale(in_max)

    def _update_and_evaluate(self, rate):
        self._calculate_activation_scales(max(self.activation_scales))

        self.adaptation_manager.clamp_rate = rate
        for perceptron in self.model.get_perceptrons():
            self.adaptation_manager.update_activation(self.pipeline.config, perceptron)

        self.trainer.train_one_step(0)
        return self.trainer.validate()

    def run(self):
        self.trainer.validate()
        super().run()
        return self.trainer.validate()
