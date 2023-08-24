from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.transformations.chip_quantization import ChipQuantization
from mimarsinan.models.layers import CQ_Activation
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

        unscaled_quantized_mapping = copy.deepcopy(self.mapping)
        ChipQuantization(self.quantization_bits).unscaled_quantize(unscaled_quantized_mapping.cores)

        core_flow = CoreFlow(self.input_shape, unscaled_quantized_mapping, self.target_tq)
        print(f"  CoreFlow Accuracy: {self._validate_core_flow(core_flow)}")

        quantized_mapping = copy.deepcopy(self.mapping)
        ChipQuantization(self.quantization_bits).quantize(quantized_mapping.cores)

        core_flow = SpikingCoreFlow(self.input_shape, quantized_mapping, self.simulation_steps)
        print(f"  Original SpikingCoreFlow Accuracy: {self._validate_core_flow(core_flow)}")

        self._tune_thresholds(
            self._get_core_sums(non_quantized_mapping), cycles=20, lr=0.1, mapping=quantized_mapping)
        
        quantization_scale = self._calculate_quantization_scale(quantized_mapping)
        self._quantize_thresholds(quantized_mapping, quantization_scale)

        scaled_simulation_steps = math.ceil(self.simulation_steps * quantization_scale)
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
        return core_flow.core_sums
    
    def _tune_thresholds(self, core_sums, cycles, lr, mapping):
        print("  Tuning thresholds...")
        best_acc = 0
        
        base_thresholds = [core.threshold for core in mapping.cores]
        for _ in range(cycles):
            print(f"    Tuning Cycle {_ + 1}/{cycles}")
            
            spiking_core_flow = SpikingCoreFlow(self.input_shape, mapping, self.simulation_steps)
            spiking_core_flow_trainer = BasicTrainer(
                spiking_core_flow, 
                self.device, self.data_loader_factory,
                None)
            spiking_core_flow_trainer.report_function = self.report_function 

            acc = spiking_core_flow_trainer.validate()
            print(f"    acc: {acc}")

            if acc > best_acc:
                best_acc = acc
                best_thresholds = [core.threshold for core in mapping.cores]

            for idx, core in enumerate(mapping.cores):
                rate_numerator = core_sums[idx] / base_thresholds[idx]
                rate_denominator = spiking_core_flow.core_sums[idx] / (spiking_core_flow.thresholds[idx].item() * self.simulation_steps)
                #print(f"    core {idx}... rate_numerator: {rate_numerator}, rate_denominator: {rate_denominator}")
                
                if rate_denominator == 0: 
                    print(f"    WARNING: rate_denominator for core {idx} is 0, setting to 0.5")
                    rate_denominator = 0.5

                rate = rate_numerator / rate_denominator
                threshold_delta = core.threshold - (core.threshold / rate) 
                core.threshold = core.threshold - (threshold_delta * lr) 

                print(f"    core {idx}... core_sum        : {core_sums[idx] / base_thresholds[idx]}")
                print(f"    core {idx}... spiking_core_sum: {spiking_core_flow.core_sums[idx] / (spiking_core_flow.thresholds[idx].item() * self.simulation_steps)}")
                print(f"    core {idx}... rate: {rate}")
                print(f"    core {idx}... threshold: {core.threshold}")
                
        
        for idx, core in enumerate(mapping.cores):
            core.threshold = best_thresholds[idx]
    
    def _calculate_quantization_scale(self, mapping):
        found = False
        tolerance = 1 / (2 * self.simulation_steps)
        scale = 1 + tolerance
        search_precision = tolerance / 200

        print("    Calculating quantization scale...")
        while not found:
            thresholds = [core.threshold for core in mapping.cores]
            scaled_thresholds = [threshold * scale for threshold in thresholds]

            found = True
            for scaled_threshold in scaled_thresholds:
                error = abs(1 - (round(scaled_threshold) / (scaled_threshold)))
                if error > tolerance:
                    found = False

            scale += search_precision
        
        print("    Scale = ", scale)
        
        return scale

    def _quantize_thresholds(self, mapping, quantization_scale):
        for core in mapping.cores:
            core.threshold = round(core.threshold * quantization_scale)

    def _validate_core_flow(self, core_flow):
        self.core_flow_trainer.model = core_flow.to(self.device)
        return self.core_flow_trainer.validate()
    
    def validate(self):
        return self.accuracy
