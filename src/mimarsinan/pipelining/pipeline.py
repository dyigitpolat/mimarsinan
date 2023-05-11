from mimarsinan.pipelining.model_building.patched_perceptron_flow_builder import PatchedPerceptronFlowBuilder
from mimarsinan.pipelining.pretrainer import Pretrainer
from mimarsinan.pipelining.activation_quantization_tuner import ActivationQuantizationTuner
from mimarsinan.pipelining.normalization_fuser import NormalizationFuser
from mimarsinan.pipelining.soft_core_mapper import SoftCoreMapper
from mimarsinan.pipelining.hard_core_mapper import HardCoreMapper
from mimarsinan.pipelining.core_flow_tuner import CoreFlowTuner
from mimarsinan.pipelining.simulation_runner import SimulationRunner

from mimarsinan.models.perceptron_flow import ppf_loss

import numpy as np
import torch

class Pipeline:
    def __init__(self, 
        training_dataloader, 
        validation_dataloader, 
        test_dataloader,
        num_classes,
        platform_constraints: dict,
        reporter,
        working_directory):

        # Platform constraints
        self.max_axons = platform_constraints['max_axons']
        self.max_neurons = platform_constraints['max_neurons']
        self.target_tq = platform_constraints['target_tq']

        # Data
        self.training_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader
        self.test_dataloader = test_dataloader

        # Metadata
        self.input_shape = next(iter(training_dataloader))[0].shape[1:]
        self.output_shape = next(iter(training_dataloader))[1].shape[1:]
        if len(self.output_shape) == 0: self.output_shape = (1,)

        self.input_size = np.prod(self.input_shape)
        self.output_size = np.prod(self.output_shape)
        self.num_classes = num_classes

        # Reporting
        self.reporter = reporter

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model
        self.model = PatchedPerceptronFlowBuilder(
            self.max_axons, self.max_neurons, 
            self.input_shape, self.num_classes).build()
        
        # Loss definitions
        self.pt_loss = ppf_loss
        self.aq_loss = ppf_loss
        self.wq_loss = ppf_loss
        
        # File
        self.working_directory = working_directory
        
    def run(self):
        # Pretraining
        pretraining_accuracy = Pretrainer(self, 10).run()

        # Activation quantization
        ActivationQuantizationTuner(
            self, 10, self.target_tq, pretraining_accuracy).run()

        # Batchnorm fusion
        NormalizationFuser(self).run()

        # Weight quantization
        ActivationQuantizationTuner(
            self, 10, self.target_tq, pretraining_accuracy).run()

        # Soft core mapping
        soft_core_mapping = SoftCoreMapper(self).run()
        
        # Hard core mapping
        hard_core_mapping = HardCoreMapper(self, soft_core_mapping).run()

        # CoreFlow tuning
        core_flow_accuracy, threshold_scale = CoreFlowTuner(
            self, hard_core_mapping, self.target_tq).run()

        # Chip simulation
        chip_accuracy = SimulationRunner(
            self, hard_core_mapping, threshold_scale, self.target_tq).run()



    







    


