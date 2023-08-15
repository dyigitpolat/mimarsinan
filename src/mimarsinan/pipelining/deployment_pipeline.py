from mimarsinan.pipelining.pipeline import Pipeline
from mimarsinan.model_training.training_utilities import BasicClassificationLoss

from mimarsinan.pipelining.pipeline_steps import *

import numpy as np
import torch

class DeploymentPipeline(Pipeline):
    default_deployment_parameters = {
        'lr': 0.001,
        'training_epochs': 10,
        'tuner_epochs': 10,
        'nas_cycles': 5,
        'nas_batch_size': 50
    }

    default_platform_constraints = {
        'max_axons': 256,
        'max_neurons': 256,
        'target_tq': 32,
        'simulation_steps': 32,
        'weight_bits': 8
    }

    def __init__(
        self,
        data_provider,
        deployment_parameters,
        platform_constraints,
        reporter,
        working_directory):

        super().__init__(working_directory)

        self.data_provider = data_provider
        self.reporter = reporter

        self.config = {}
        self._initialize_config(deployment_parameters, platform_constraints)
        self._display_config()

        self.loss = BasicClassificationLoss()
        
        self.add_pipeline_step("Architecture Search", ArchitectureSearchStep(self))
        self.add_pipeline_step("Model Building", ModelBuildingStep(self))
        self.add_pipeline_step("Pretraining", PretrainingStep(self))
        self.add_pipeline_step("Clamp Adaptation", ClampAdaptationStep(self))
        self.add_pipeline_step("Scale Adaptation", ScaleAdaptationStep(self))
        self.add_pipeline_step("Noise Adaptation", NoiseAdaptationStep(self))
        self.add_pipeline_step("Activation Shifting", ActivationShiftStep(self))
        self.add_pipeline_step("Activation Quantization", ActivationQuantizationStep(self))
        self.add_pipeline_step("Normalization Fusion", NormalizationFusionStep(self))
        self.add_pipeline_step("Perceptron Fusion", PerceptronFusionStep(self))
        self.add_pipeline_step("Weight Quantization", WeightQuantizationStep(self))
        self.add_pipeline_step("Soft Core Mapping", SoftCoreMappingStep(self))
        self.add_pipeline_step("Hard Core Mapping", HardCoreMappingStep(self))
        self.add_pipeline_step("CoreFlow Tuning", CoreFlowTuningStep(self))
        self.add_pipeline_step("Simulation", SimulationStep(self))
        
    def _initialize_config(self, deployment_parameters, platform_constraints):
        self.config.update(self.default_deployment_parameters)
        self.config.update(deployment_parameters)
        
        self.config.update(self.default_platform_constraints)
        self.config.update(platform_constraints)

        self.config['input_shape'] = self.data_provider.get_input_shape()
        self.config['output_shape'] = self.data_provider.get_output_shape()
        if len(self.config['output_shape']) == 0: self.config['output_shape'] = (1,)

        self.config['input_size'] = np.prod(self.config['input_shape'])
        self.config['output_size'] = np.prod(self.config['output_shape'])
        self.config['num_classes'] = self.data_provider.get_prediction_mode().num_classes

        self.config['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _display_config(self):
        print("Deployment configuration:")
        for key, value in self.config.items():
            print(f"{key}: {value}")









