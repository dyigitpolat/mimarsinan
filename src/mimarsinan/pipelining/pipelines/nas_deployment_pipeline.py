from mimarsinan.pipelining.pipeline import Pipeline
from mimarsinan.model_training.training_utilities import *
from mimarsinan.data_handling.data_provider_factory import DataProviderFactory

from mimarsinan.models.layers import TransformedActivation

from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory

from mimarsinan.visualization.activation_function_visualization import ActivationFunctionVisualizer
from mimarsinan.visualization.histogram_visualization import HistogramVisualizer

from mimarsinan.pipelining.pipeline_steps import *

import numpy as np
import torch

import os

class NASDeploymentPipeline(Pipeline):
    default_deployment_parameters = {
        'lr': 0.001,
        'training_epochs': 10,
        'tuner_epochs': 10,
        'nas_cycles': 5,
        'nas_batch_size': 50,
        'nas_workers': 1,
        'degradation_tolerance': 0.95
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
        data_provider_factory: DataProviderFactory,
        deployment_parameters,
        platform_constraints,
        reporter,
        working_directory):

        super().__init__(working_directory)

        self.data_provider_factory = data_provider_factory
        self.reporter = reporter

        self.config = {}
        self._initialize_config(deployment_parameters, platform_constraints)
        self._display_config()

        self.loss = CustomClassificationLoss()
        
        self.add_pipeline_step("Architecture Search", ArchitectureSearchStep(self))
        self.add_pipeline_step("Model Building", ModelBuildingStep(self))
        self.add_pipeline_step("Pretraining", PretrainingStep(self))
        self.add_pipeline_step("Clamp Adaptation", ClampAdaptationStep(self))
        #self.add_pipeline_step("Noise Adaptation", NoiseAdaptationStep(self))
        self.add_pipeline_step("Activation Shifting", ActivationShiftStep(self))
        self.add_pipeline_step("Activation Quantization", ActivationQuantizationStep(self))
        #self.add_pipeline_step("Scale Adaptation", ScaleAdaptationStep(self))
        #self.add_pipeline_step("Parameter Scale Adaptation", ParameterScaleAdaptationStep(self))
        self.add_pipeline_step("Weight Quantization", WeightQuantizationStep(self))
        self.add_pipeline_step("Quantization Verification", QuantizationVerificationStep(self))
        self.add_pipeline_step("Normalization Fusion", NormalizationFusionStep(self))
        self.add_pipeline_step("Soft Core Mapping", SoftCoreMappingStep(self))
        self.add_pipeline_step("CoreFlow Tuning", CoreFlowTuningStep(self))
        self.add_pipeline_step("Hard Core Mapping", HardCoreMappingStep(self))
        self.add_pipeline_step("Simulation", SimulationStep(self))

        def post_step_hook(step):
            self._visualize_activations(step)
            self._visualize_activation_histograms(step)

        self.register_post_step_hook(post_step_hook)
        
    def _initialize_config(self, deployment_parameters, platform_constraints):
        self.config.update(self.default_deployment_parameters)
        self.config.update(deployment_parameters)
        
        self.config.update(self.default_platform_constraints)
        self.config.update(platform_constraints)

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

    def _visualize_activations(self, step):
        if 'model' in step.promises or 'model' in step.updates:
            path = self.working_directory + f"/{step.name}_act/"
            os.makedirs(path, exist_ok=True)

            model = self.cache.get(self._create_real_key(step.name, 'model'))
            for idx, perceptron in enumerate(model.get_perceptrons()):
                ActivationFunctionVisualizer(perceptron.activation, -3, 3, 0.001).plot(f"{path}/p_{idx}.png")

    def _visualize_activation_histograms(self, step):
        if 'model' in step.promises or 'model' in step.updates:
            path = self.working_directory + f"/{step.name}_act_hist/"
            os.makedirs(path, exist_ok=True)

            model = self.cache.get(self._create_real_key(step.name, 'model'))

            for perceptron in model.get_perceptrons():
                perceptron.activation.decorate(StatsDecorator())

            BasicTrainer(
                    model, 
                    self.config['device'], 
                    DataLoaderFactory(self.data_provider_factory),
                    self.loss).validate()

            for idx, perceptron in enumerate(model.get_perceptrons()):
                stats = perceptron.activation.pop_decorator()

                hist = stats.in_hist.tolist()
                bin_edges = stats.in_hist_bin_edges.tolist()

                trimmed_hist, trimmed_edges = self._trim_histogram(hist, bin_edges)

                rob_max = self._find_robust_max(trimmed_hist, trimmed_edges)
                HistogramVisualizer(hist[-(len(trimmed_hist)):], trimmed_edges, -10, 10, v_line = rob_max).plot(f"{path}/h_{idx}.png")


    def _trim_histogram(self, hist, bin_edges):
        index_cross_zero = -1

        print(len(hist), len(bin_edges))
        for idx, edge in enumerate(bin_edges):
            if(edge > 0):
                index_cross_zero = idx
                break
        
        clamped_histogram = hist[index_cross_zero:]
        clamped_edges = bin_edges[index_cross_zero:]

        print(len(clamped_histogram), len(clamped_edges))

        for idx, _ in enumerate(clamped_histogram):
            clamped_histogram[idx] *= clamped_edges[idx]
        
        return clamped_histogram, clamped_edges

    def _find_robust_max(self, hist, bin_edges):
        hist_sum = 0
        for value in hist:
            hist_sum += value

        rate = 0.8
        current_sum = 0

        for idx, value in enumerate(reversed(hist)):
            current_sum += value
            print(idx, current_sum / hist_sum)
            if(current_sum / hist_sum > (1.0 - rate)):
                print(idx, bin_edges[-idx+1])
                return bin_edges[-idx+1]
            
        return bin_edges[-1]











