from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.transformations.chip_quantization import ChipQuantization
from mimarsinan.models.core_flow import CoreFlow
from mimarsinan.models.spiking_core_flow import SpikingCoreFlow
from mimarsinan.models.stable_spiking_core_flow import StableSpikingCoreFlow

from mimarsinan.mapping.chip_latency import *

from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory

import copy
import math
import random

class CoreFlowTuner:
    def __init__(self, pipeline, mapping, preprocessor):
        self.device = pipeline.config["device"]
        self.data_loader_factory = DataLoaderFactory(pipeline.data_provider_factory)
        self.input_shape = pipeline.config["input_shape"]

        self.preprocessor = preprocessor

        self.mapping = mapping
        self.target_tq = pipeline.config["target_tq"]
        self.simulation_steps = round(pipeline.config["simulation_steps"])
        self.report_function = pipeline.reporter.report
        self.quantization_bits = pipeline.config["weight_bits"]

        self.core_flow_trainer = BasicTrainer(
            CoreFlow(self.input_shape, self.mapping, self.target_tq, self.preprocessor), 
            self.device, self.data_loader_factory,
            None)
        # self.core_flow_trainer.set_validation_batch_size(100)
        self.core_flow_trainer.report_function = self.report_function

        self.firing_mode = pipeline.config["firing_mode"]
        self.spike_mode = pipeline.config["spike_generation_mode"]

        self.accuracy = None

    def run(self):
        non_quantized_mapping = copy.deepcopy(self.mapping)
        core_flow = CoreFlow(self.input_shape, non_quantized_mapping, self.target_tq, self.preprocessor)
        print(f"  Non-Quantized CoreFlow Accuracy: {self._validate_core_flow(core_flow)}")

        quantized_mapping = copy.deepcopy(self.mapping)
        ChipQuantization(self.quantization_bits).quantize(quantized_mapping.cores)
        core_flow = SpikingCoreFlow(self.input_shape, quantized_mapping, int(self.simulation_steps), self.preprocessor, self.firing_mode, self.spike_mode)
        print(f"  Original SpikingCoreFlow Accuracy: {self._validate_core_flow(core_flow)}")

        stable_core_flow = StableSpikingCoreFlow(self.input_shape, quantized_mapping, int(self.simulation_steps), self.preprocessor, self.firing_mode)
        print(f"  Original StableSpikingCoreFlow Accuracy: {self._validate_core_flow(stable_core_flow)}")


        max_lr = max([core.threshold for core in stable_core_flow.cores]) / 10
        self._tune_thresholds(
            stable_core_flow.get_core_spike_rates(), 
            tuning_cycles=40, lr=1.0, core_flow_model=core_flow)
        
        final_mapping = copy.deepcopy(core_flow.core_mapping)
        core_flow = SpikingCoreFlow(self.input_shape, final_mapping, int(self.simulation_steps), self.preprocessor, self.firing_mode, self.spike_mode)
        print(f"  Original 2 SpikingCoreFlow Accuracy: {self._validate_core_flow(core_flow)}")
        
        self._quantize_thresholds(final_mapping, 1.0)
        scaled_simulation_steps = self.simulation_steps
        core_flow = SpikingCoreFlow(self.input_shape, final_mapping, scaled_simulation_steps, self.preprocessor, self.firing_mode, self.spike_mode)
        self.accuracy = self._test_core_flow(core_flow)
        print(f"  Final SpikingCoreFlow Accuracy: {self.accuracy}")

        self.mapping = final_mapping
        return scaled_simulation_steps

    
    def _tune_thresholds(self, stable_spike_rates, tuning_cycles, lr, core_flow_model):
        print("  Tuning thresholds...")
        print(stable_spike_rates)
        print(core_flow_model.get_core_spike_rates())

        latency_groups = {}
        for idx, core in enumerate(core_flow_model.cores):
            if core.latency not in latency_groups:
                latency_groups[core.latency] = []
            
            latency_groups[core.latency].append(idx)

        thresholds = [round(0.99 * core.threshold) for core in core_flow_model.cores]
        best_thresholds = [t for t in thresholds]

        acc = 0
        max_acc = 0
        perturbations = [None for _ in core_flow_model.cores]
        lr_ = lr
        for i in range(tuning_cycles):
            acc = self._validate_core_flow(core_flow_model)
            print("lr: ", lr_)

            print("acc: ", acc)

            print(f"  Tuning cycle {i+1}/{tuning_cycles}...")
            self._update_core_thresholds(
                stable_spike_rates, lr_, core_flow_model, thresholds, perturbations, latency_groups)
            
            lr_ *= math.pow(0.8, 1 / tuning_cycles)

            if acc > max_acc:
                max_acc = acc
                best_thresholds = [core.threshold for core in core_flow_model.cores]

        for core_id, core in enumerate(core_flow_model.cores):
            core.threshold = best_thresholds[core_id]
            core_flow_model.refresh_thresholds()

    def _update_core_thresholds(self, stable_spike_rates, lr, core_flow_model, thresholds, perturbations, latency_groups):
        current_perturbations = []
        for core_id, core in enumerate(core_flow_model.cores):
            core_spike_rate = core_flow_model.get_core_spike_rates()[core_id] + 0.01
            target_spike_rate = random.uniform(0.995, 1.01) * stable_spike_rates[core_id] + 0.01
            perturbation = target_spike_rate - core_spike_rate
            perturbations[core_id] = perturbation
            current_perturbations.append(perturbation)

        for core_id, core in enumerate(core_flow_model.cores):
            if core.latency - 1 in latency_groups:
                for other_id in latency_groups[core.latency - 1]:
                    perturbations[other_id] += perturbations[core_id] * 0.1

        scale = max([abs(p) for p in current_perturbations])
        for core_id, core in enumerate(core_flow_model.cores):
            perturbation = perturbations[core_id]
            perturbation = max(-0.99, min(0.99, perturbation))

            new_thresh = thresholds[core_id] * (1 - perturbation * lr)
            thresholds[core_id] = new_thresh

            core.threshold = round(thresholds[core_id])
            if core.threshold <= 0:
                core.threshold = 1
        
        for pert, core in zip(perturbations, core_flow_model.cores):
            if pert is None:
                self.print_colorful("gray", f"{core.threshold} ")
                continue

            if abs(pert) > 0.03:
                self.print_colorful("gray", f"{core.threshold} ")
            elif pert < -0.02:
                self.print_colorful("blue", f"{core.threshold} ")
            elif pert > 0.02:
                self.print_colorful("red", f"{core.threshold} ")
            else:
                self.print_colorful("green", f"{core.threshold} ")
        print()

        core_flow_model.refresh_thresholds()

    def _quantize_thresholds(self, mapping, quantization_scale):
        for core in mapping.cores:
            core.threshold = round(core.threshold * quantization_scale)

    def _validate_core_flow(self, core_flow):
        self.core_flow_trainer.model = core_flow.to(self.device)
        return self.core_flow_trainer.validate()

    def _test_core_flow(self, core_flow):
        self.core_flow_trainer.model = core_flow.to(self.device)
        return self.core_flow_trainer.test()
    
    def validate(self):
        return self.accuracy
    
    def print_colorful(self, color, text):
        colors = {
            "red": "\033[91m",
            "green": "\033[92m",
            "yellow": "\033[93m",
            "blue": "\033[94m",
            "purple": "\033[95m",
            "cyan": "\033[96m",
            "gray": "\033[90m"
        }
        if color not in colors:
            raise ValueError(f"Invalid color: {color}. Choose from: {', '.join(colors.keys())}")
        print(f"{colors[color]}{text}\033[0m", end="")
