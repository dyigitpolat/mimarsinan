from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.transformations.chip_quantization import ChipQuantization
from mimarsinan.models.core_flow import CoreFlow
from mimarsinan.models.spiking_core_flow import SpikingCoreFlow

from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory

import copy
import math

class CoreFlowTuner:
    def __init__(self, pipeline, mapping):
        self.device = pipeline.config["device"]
        self.data_loader_factory = DataLoaderFactory(pipeline.data_provider_factory)
        self.input_shape = pipeline.config["input_shape"]

        self.mapping = mapping
        self.target_tq = pipeline.config["target_tq"]
        self.simulation_steps = round(pipeline.config["simulation_steps"])
        self.report_function = pipeline.reporter.report
        self.quantization_bits = pipeline.config["weight_bits"]

        self.core_flow_trainer = BasicTrainer(
            CoreFlow(self.input_shape, self.mapping, self.target_tq), 
            self.device, self.data_loader_factory,
            None)
        self.core_flow_trainer.set_validation_batch_size(100)
        self.core_flow_trainer.report_function = self.report_function

        self.accuracy = None

    def run(self):
        non_quantized_mapping = copy.deepcopy(self.mapping)
        core_flow = CoreFlow(self.input_shape, non_quantized_mapping, self.target_tq)
        print(f"  Non-Quantized CoreFlow Accuracy: {self._validate_core_flow(core_flow)}")

        unscaled_quantized_mapping = copy.deepcopy(self.mapping)
        ChipQuantization(self.quantization_bits).unscaled_quantize(unscaled_quantized_mapping.cores)
        core_flow = CoreFlow(self.input_shape, unscaled_quantized_mapping, self.target_tq)
        print(f"  Quantized CoreFlow Accuracy: {self._validate_core_flow(core_flow)}")

        quantized_mapping = copy.deepcopy(self.mapping)
        ChipQuantization(self.quantization_bits).quantize(quantized_mapping.cores)
        core_flow = SpikingCoreFlow(self.input_shape, quantized_mapping, int(self.simulation_steps * self._get_step_scale(quantized_mapping)))
        print(f"  Original SpikingCoreFlow Accuracy: {self._validate_core_flow(core_flow)}")

        self._tune_thresholds(
            self._get_core_sums(unscaled_quantized_mapping), cycles=10, lr=0.3, mapping=quantized_mapping)
        
        self._quantize_thresholds(quantized_mapping, 1.0)
        scaled_simulation_steps = math.ceil(self.simulation_steps)
        core_flow = SpikingCoreFlow(self.input_shape, quantized_mapping, scaled_simulation_steps)
        self.accuracy = self._validate_core_flow(core_flow)
        print(f"  Final SpikingCoreFlow Accuracy: {self.accuracy}")

        self.mapping = quantized_mapping
        return scaled_simulation_steps

    def _get_core_sums(self, mapping):
        core_flow = CoreFlow(self.input_shape, mapping, self.target_tq)
        core_flow_trainer = BasicTrainer(
            core_flow, 
            self.device, self.data_loader_factory,
            None)
        core_flow_trainer.report_function = self.report_function 
        
        core_flow_trainer.validate()

        for core, core_sum in zip(mapping.cores, core_flow.core_sums):
            core_sum /= core.threshold
        
        return core_flow.core_sums

    
    def _tune_thresholds(self, core_sums, cycles, lr, mapping):
        print("  Tuning thresholds...")
        best_acc = 0

        core_sums_mean = 0
        for core_sum in core_sums:
            core_sums_mean += core_sum
        core_sums_mean /= len(core_sums)
        
        import numpy as np
        for _ in range(cycles):
            print(f"    Tuning Cycle {_ + 1}/{cycles}")
            
            spiking_core_flow = SpikingCoreFlow(self.input_shape, mapping, math.ceil(self.simulation_steps))
            spiking_core_flow_trainer = BasicTrainer(
            spiking_core_flow, 
                self.device, self.data_loader_factory,
                None)
            spiking_core_flow_trainer.report_function = self.report_function 

            acc = spiking_core_flow_trainer.validate()

            spike_sum_mean = 0
            for core_sum in spiking_core_flow.core_sums:
                spike_sum_mean += core_sum / math.ceil(self.simulation_steps)
            spike_sum_mean /= len(spiking_core_flow.core_sums)

            print(f"    acc: {acc}")

            if acc > best_acc:
                best_acc = acc
                best_thresholds = [core.threshold for core in mapping.cores]

            for idx, core in enumerate(mapping.cores):
                rate_numerator = core_sums[idx] / core_sums_mean
                rate_denominator = (spiking_core_flow.core_sums[idx] / self.simulation_steps) / spike_sum_mean
                #print(f"    core {idx}... rate_numerator: {rate_numerator}, rate_denominator: {rate_denominator}")
                
                if rate_denominator == 0: 
                    print(f"    WARNING: rate_denominator for core {idx} is 0, setting to 0.5")
                    rate_denominator = 0.5

                rate = rate_numerator / rate_denominator
                target_threshold = core.threshold / rate
                updated_threshold = (1 - lr) * core.threshold + (lr) * target_threshold
                core.threshold = updated_threshold

                print(f"    core {idx}... core_sum        : {rate_numerator}")
                print(f"    core {idx}... spiking_core_sum: {rate_denominator}")
                print(f"    core {idx}... rate: {rate}")
                print(f"    core {idx}... threshold: {core.threshold}")
                
        
        for idx, core in enumerate(mapping.cores):
            core.threshold = best_thresholds[idx]

    def _quantize_thresholds(self, mapping, quantization_scale):
        for core in mapping.cores:
            core.threshold = round(core.threshold * quantization_scale)

    def _validate_core_flow(self, core_flow):
        self.core_flow_trainer.model = core_flow.to(self.device)
        return self.core_flow_trainer.validate()
    
    def validate(self):
        return self.accuracy
