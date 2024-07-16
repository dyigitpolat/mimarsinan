from mimarsinan.tuning.tuners.perceptron_tuner import PerceptronTuner

from mimarsinan.transformations.perceptron_transformer import PerceptronTransformer

import torch

class ScaleTuner(PerceptronTuner):
    def __init__(self, 
                 pipeline, 
                 model, 
                 target_accuracy, 
                 lr,
                 adaptation_manager,
                 in_scales,
                 out_scales):
        
        super().__init__(
            pipeline, 
            model, 
            target_accuracy, 
            lr)

        self.lr = lr
        self.in_scales = in_scales
        self.out_scales = out_scales
        self.adaptation_manager = adaptation_manager

    def _get_target_decay(self):
        return 0.99

    def _update_and_evaluate(self, rate):
        for g_idx, perceptron_group in enumerate(self.model.perceptron_flow.get_perceptron_groups()):
            for perceptron in perceptron_group:
                scale = self.out_scales[g_idx]
                in_scale = self.in_scales[g_idx]

                t = scale * (1-rate) + rate
                perceptron.set_scale_factor(t)
                perceptron.set_activation_scale(t)

                self.adaptation_manager.update_activation(self.pipeline.config, perceptron)

                PerceptronTransformer().apply_effective_bias_transform(perceptron, lambda p: (p / scale) * rate + p * (1-rate))
                PerceptronTransformer().apply_effective_weight_transform(perceptron, lambda p: (p * in_scale / scale) * rate + p * (1-rate))

        self.trainer.train_one_step(self.lr)
        return self.trainer.validate()

    def run(self):
        self.trainer.validate()
        super().run()
        return self.trainer.validate()
