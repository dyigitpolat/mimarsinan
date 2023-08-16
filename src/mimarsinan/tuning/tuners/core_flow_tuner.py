from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.transformations.chip_quantization import ChipQuantization
from mimarsinan.models.layers import CQ_Activation
from mimarsinan.models.core_flow import CoreFlow
from mimarsinan.models.spiking_core_flow import SpikingCoreFlow

import copy

class CoreFlowTuner:
    def __init__(self, pipeline, mapping):
        self.device = pipeline.config["device"]
        self.data_provider = pipeline.data_provider
        self.input_shape = pipeline.config["input_shape"]

        self.mapping = mapping
        self.target_tq = pipeline.config["target_tq"]
        self.simulation_steps = round(pipeline.config["simulation_steps"])
        self.report_function = pipeline.reporter.report
        self.quantization_bits = pipeline.config["weight_bits"]

        self.accuracy = None

    def run(self):
        unscaled_quantized_mapping = copy.deepcopy(self.mapping)
        ChipQuantization(self.quantization_bits).unscaled_quantize(unscaled_quantized_mapping.cores)

        core_flow = CoreFlow(self.input_shape, unscaled_quantized_mapping, self.target_tq)
        print(f"  CoreFlow Accuracy: {self._validate_core_flow(core_flow)}")

        quantized_mapping = copy.deepcopy(self.mapping)
        ChipQuantization(self.quantization_bits).quantize(quantized_mapping.cores)

        core_flow = SpikingCoreFlow(self.input_shape, quantized_mapping, self.simulation_steps)
        print(f"  Original SpikingCoreFlow Accuracy: {self._validate_core_flow(core_flow)}")

        self._tune_thresholds(
            self._get_core_sums(), cycles=20, lr=0.1, mapping=quantized_mapping)

        core_flow = SpikingCoreFlow(self.input_shape, quantized_mapping, self.simulation_steps)
        self.accuracy = self._validate_core_flow(core_flow)
        print(f"  Final SpikingCoreFlow Accuracy: {self.accuracy}")

        self.mapping = quantized_mapping
        return self.accuracy

    def _get_core_sums(self):
        core_flow = CoreFlow(self.input_shape, self.mapping, self.target_tq)
        core_flow_trainer = BasicTrainer(
            core_flow, 
            self.device, self.data_provider,
            None)
        core_flow_trainer.report_function = self.report_function 
        
        core_flow_trainer.validate()
        return core_flow.core_sums
    
    def _tune_thresholds(self, core_sums, cycles, lr, mapping):
        print("  Tuning thresholds...")
        best_acc = 0
        for _ in range(cycles):
            print(f"    Tuning Cycle {_ + 1}/{cycles}")
            
            spiking_core_flow = SpikingCoreFlow(self.input_shape, mapping, self.simulation_steps)
            spiking_core_flow_trainer = BasicTrainer(
                spiking_core_flow, 
                self.device, self.data_provider,
                None)
            spiking_core_flow_trainer.report_function = self.report_function 

            acc = spiking_core_flow_trainer.validate()
            print(f"    acc: {acc}")

            if acc > best_acc:
                best_acc = acc
                best_thresholds = [core.threshold for core in mapping.cores]

            for idx, core in enumerate(mapping.cores):
                rate_numerator = core_sums[idx]
                rate_denominator = spiking_core_flow.core_sums[idx] / self.simulation_steps
                #print(f"    core {idx}... rate_numerator: {rate_numerator}, rate_denominator: {rate_denominator}")
                
                if rate_denominator == 0: 
                    print(f"    WARNING: rate_denominator for core {idx} is 0, setting to 0.5")
                    rate_denominator = 0.5

                rate = rate_numerator / rate_denominator
                threshold_delta = core.threshold - (core.threshold / rate) 
                core.threshold = core.threshold - (threshold_delta * lr) 

                print(f"    core {idx}... rate: {rate}")
                print(f"    core {idx}... threshold: {core.threshold}")
                
        
        for idx, core in enumerate(mapping.cores):
            core.threshold = best_thresholds[idx]
            core.threshold = round(core.threshold)

    def _validate_core_flow(self, core_flow):
        core_flow_trainer = BasicTrainer(
            core_flow, 
            self.device, self.data_provider,
            None)
        core_flow_trainer.report_function = self.report_function
        
        return core_flow_trainer.validate()
    
    def validate(self):
        return self.accuracy
