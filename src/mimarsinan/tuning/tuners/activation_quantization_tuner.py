from mimarsinan.tuning.tuners.perceptron_tuner import PerceptronTuner

class ActivationQuantizationTuner(PerceptronTuner):
    def __init__(self, 
                 pipeline, 
                 model, 
                 target_tq, 
                 target_accuracy, 
                 lr,
                 adaptation_manager):
        
        super().__init__(
            pipeline, 
            model, 
            target_accuracy, 
            lr)

        self.target_tq = target_tq
        self.adaptation_manager = adaptation_manager
        self.base_input_activation = model.get_input_activation()

    def _get_target_decay(self):
        return 0.99

    def _update_and_evaluate(self, rate):
        self.adaptation_manager.quantization_rate = rate
        for perceptron in self.model.get_perceptrons():
            self.adaptation_manager.update_activation(self.pipeline.config, perceptron)
        
        self.trainer.train_one_step(0)
        return self.trainer.validate()

    def run(self):
        super().run()

        return self.trainer.validate()
