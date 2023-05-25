from mimarsinan.pipelining.model_building.patched_perceptron_flow_builder import PatchedPerceptronFlowBuilder
from mimarsinan.pipelining.pretrainer import Pretrainer
from mimarsinan.pipelining.activation_shifter import ActivationShifter
from mimarsinan.pipelining.activation_quantization_tuner import ActivationQuantizationTuner
from mimarsinan.pipelining.normalization_fuser import NormalizationFuser
from mimarsinan.pipelining.weight_quantization_tuner import WeightQuantizationTuner
from mimarsinan.pipelining.soft_core_mapper import SoftCoreMapper
from mimarsinan.pipelining.hard_core_mapper import HardCoreMapper
from mimarsinan.pipelining.core_flow_tuner import CoreFlowTuner
from mimarsinan.pipelining.simulation_runner import SimulationRunner

from mimarsinan.models.perceptron_flow import ppf_loss
from mimarsinan.common.file_utils import prepare_containing_directory

import numpy as np
import torch

class BasicClassificationPipeline:
    def __init__(self, 
        training_dataloader, 
        validation_dataloader, 
        test_dataloader,
        num_classes,
        platform_constraints: dict,
        training_parameters: dict,
        reporter,
        working_directory):

        # Platform constraints
        self.max_axons = platform_constraints['max_axons']
        self.max_neurons = platform_constraints['max_neurons']
        self.target_tq = platform_constraints['target_tq']
        self.simulation_steps = platform_constraints['simulation_steps']

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
        self.model_complexity = 2
        self.model = PatchedPerceptronFlowBuilder(
            self.max_axons, self.max_neurons, 
            self.input_shape, self.num_classes,
            self.model_complexity).build()
        
        # Training hyper parameters
        self.lr = training_parameters['lr']
        self.pretraining_epochs = training_parameters['pretraining_epochs']
        self.aq_cycle_epochs = training_parameters['aq_cycle_epochs']
        self.wq_cycle_epochs = training_parameters['wq_cycle_epochs']
        self.aq_cycles = training_parameters['aq_cycles']
        self.wq_cycles = training_parameters['wq_cycles']
        
        # Loss definitions
        self.pt_loss = ppf_loss
        self.aq_loss = ppf_loss
        self.wq_loss = ppf_loss
        
        # File
        self.working_directory = working_directory
        prepare_containing_directory(self.working_directory)
        
    def run(self):
        print("Pretraining...")
        pretraining_accuracy = Pretrainer(self, self.pretraining_epochs).run()

        print("Shifting activation...")
        shift_accuracy = ActivationShifter(
            self, self.aq_cycle_epochs, self.target_tq, pretraining_accuracy).run()
        print(f"Accuracy after activation shift: {shift_accuracy}")
        assert shift_accuracy > pretraining_accuracy * 0.95

        print("Activation quantization...")
        aq_accuracy = ActivationQuantizationTuner(
            self, self.aq_cycle_epochs, self.target_tq, shift_accuracy).run()
        print(f"AQ final accuracy: {aq_accuracy}")
        assert aq_accuracy > shift_accuracy * 0.9

        print("Normalization fusion...")
        fn_accuracy = NormalizationFuser(self).run()
        print(f"Fused normalization accuracy: {fn_accuracy}")
        assert fn_accuracy > aq_accuracy * 0.9

        print("Weight quantization...")
        wq_accuracy = WeightQuantizationTuner(
            self, self.wq_cycle_epochs, self.target_tq, aq_accuracy).run()
        print(f"WQ final accuracy: {fn_accuracy}")
        assert wq_accuracy > fn_accuracy * 0.9

        print("Soft core mapping...")
        soft_core_mapping = SoftCoreMapper(self).run()
        
        print("Hard core mapping...")
        hard_core_mapping = HardCoreMapper(self, soft_core_mapping).run()

        print("CoreFlow tuning...")
        core_flow_accuracy, threshold_scale = CoreFlowTuner(
            self, hard_core_mapping, self.target_tq).run()
        print(f"Scale: {threshold_scale}")
        print(f"CoreFlow final accuracy: {core_flow_accuracy}")

        print("Simulation...")
        chip_accuracy = SimulationRunner(
            self, hard_core_mapping, threshold_scale, self.simulation_steps).run()
        print(f"Simulation accuracy: {chip_accuracy}")
        