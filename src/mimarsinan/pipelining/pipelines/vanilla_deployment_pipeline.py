from mimarsinan.pipelining.pipeline import Pipeline
from mimarsinan.data_handling.data_provider_factory import DataProviderFactory

from mimarsinan.pipelining.pipeline_steps import *

import numpy as np
import torch

class VanillaDeploymentPipeline(Pipeline):
    default_deployment_parameters = {
        'lr': 0.001,
        'training_epochs': 10,
        'degradation_tolerance': 0.95,
        'model_config': {
            "patch_n_1": 2,
            "patch_m_1": 2,
            "patch_c_1": 16,
            "fc_k_1": 2,
            "fc_w_1": 16,
            "patch_n_2": 2,
            "patch_c_2": 16,
            "fc_k_2": 2,
            "fc_w_2": 16
        }
    }

    default_platform_constraints = {
        'max_axons': 256,
        'max_neurons': 256,
        'target_tq': 32,
        'simulation_steps': 32,
        'weight_bits': 8,
        'allow_axon_tiling': False,
    }

    def __init__(
        self,
        data_provider_factory: DataProviderFactory,
        deployment_parameters,
        platform_constraints,
        reporter,
        working_directory):

        super().__init__(working_directory)

        self.data_provider_factory = data_provider_factory
        self.reporter = reporter

        self.config = {}
        data_provider = self.data_provider_factory.create()
        self.loss = data_provider.create_loss()
        self._initialize_config(deployment_parameters, platform_constraints, data_provider=data_provider)
        self._display_config()
        
        self.add_pipeline_step("Model Configuration", ModelConfigurationStep(self))
        self.add_pipeline_step("Model Building", ModelBuildingStep(self))
        self.add_pipeline_step("Pretraining", PretrainingStep(self))
        self.add_pipeline_step("Normalization Fusion", NormalizationFusionStep(self))
        self.add_pipeline_step("Soft Core Mapping", SoftCoreMappingStep(self))
        self.add_pipeline_step("Core Quantization Verification", CoreQuantizationVerificationStep(self))
        self.add_pipeline_step("CoreFlow Tuning", CoreFlowTuningStep(self))
        self.add_pipeline_step("Hard Core Mapping", HardCoreMappingStep(self))
        self.add_pipeline_step("Simulation", SimulationStep(self))
        
    def _initialize_config(self, deployment_parameters, platform_constraints, *, data_provider=None):
        self.config.update(self.default_deployment_parameters)
        self.config.update(deployment_parameters)
        
        self.config.update(self.default_platform_constraints)
        self.config.update(platform_constraints)

        if data_provider is None:
            data_provider = self.data_provider_factory.create()
        self.config['input_shape'] = data_provider.get_input_shape()

        self.config['input_size'] = np.prod(self.config['input_shape'])
        self.config['num_classes'] = data_provider.get_prediction_mode().num_classes

        self.config['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tolerance = self.config['degradation_tolerance']

    def _display_config(self):
        print("Deployment configuration:")
        for key, value in self.config.items():
            print(f"{key}: {value}")









