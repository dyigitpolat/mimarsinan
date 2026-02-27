from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory
from mimarsinan.tuning.shift_calculation import calculate_activation_shift

from mimarsinan.transformations.perceptron_transformer import PerceptronTransformer

import torch.nn as nn
import torch

class ActivationShiftStep(PipelineStep):
    def __init__(self, pipeline):
        requires = ["model", "adaptation_manager"]
        promises = []
        updates = ["model", "adaptation_manager"]
        clears = []
        super().__init__(requires, promises, updates, clears, pipeline)
        self.trainer = None

    def validate(self):
        return self.trainer.validate()

    def process(self):
        model = self.get_entry("model")

        self.trainer = BasicTrainer(
            model, 
            self.pipeline.config['device'], 
            DataLoaderFactory(self.pipeline.data_provider_factory),
            self.pipeline.loss)
        self.trainer.report_function = self.pipeline.reporter.report
        
        adaptation_manager = self.get_entry('adaptation_manager')
        config = self.pipeline.config
        use_ttfs = config.get("spiking_mode", "rate") in ("ttfs", "ttfs_quantized")

        for perceptron in model.get_perceptrons():
            print(perceptron.activation_scale)
            print(perceptron.scale_factor)
            if not use_ttfs:
                shift_amount = calculate_activation_shift(config["target_tq"], perceptron.activation_scale)
                # Add to effective bias so that pre-activation (BN/layer output) increases by shift_amount,
                # so the ShiftDecorator's subtraction cancels. Effective bias is in normalized space
                # (pre_activation / activation_scale), so we add shift_amount / activation_scale.
                act_scale = perceptron.activation_scale
                if torch.is_tensor(act_scale):
                    act_scale = act_scale.to(shift_amount.device) if torch.is_tensor(shift_amount) else act_scale
                effective_bias_shift = shift_amount / act_scale

                adaptation_manager.shift_rate = 1.0
                adaptation_manager.update_activation(config, perceptron)
                PerceptronTransformer().apply_effective_bias_transform(perceptron, lambda b: b + effective_bias_shift)
            else:
                # TTFS: the quantization decorator internally applies +shift
                # before ReLU (via nested ShiftDecorator).  Bias compensation
                # is NOT done here because quantization_rate is still 0 at
                # this point.  After ActivationQuantizationStep trains the
                # model with the shift active, SoftCoreMappingStep adds the
                # shift to the effective bias right before IR mapping.
                adaptation_manager.update_activation(config, perceptron)
        
        self.trainer.train_until_target_accuracy(
            self.pipeline.config['lr'] / 20, 
            max_epochs=2, 
            target_accuracy=self.pipeline.get_target_metric(),
            warmup_epochs=0)
        
        print(self.validate())
        
        self.update_entry("adaptation_manager", adaptation_manager, "pickle")
        self.update_entry("model", model, 'torch_model')
