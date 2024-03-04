from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.transformations.chip_quantization import ChipQuantization
from mimarsinan.models.core_flow import CoreFlow
from mimarsinan.models.spiking_core_flow import SpikingCoreFlow
from mimarsinan.models.stable_spiking_core_flow import StableSpikingCoreFlow

from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory

import copy
import math

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

        self.accuracy = None

    def run(self):
        non_quantized_mapping = copy.deepcopy(self.mapping)
        core_flow = CoreFlow(self.input_shape, non_quantized_mapping, self.target_tq, self.preprocessor)
        print(f"  Non-Quantized CoreFlow Accuracy: {self._validate_core_flow(core_flow)}")

        quantized_mapping = copy.deepcopy(self.mapping)
        ChipQuantization(self.quantization_bits).quantize(quantized_mapping.cores)
        core_flow = SpikingCoreFlow(self.input_shape, quantized_mapping, int(self.simulation_steps), self.preprocessor)
        print(f"  Original SpikingCoreFlow Accuracy: {self._validate_core_flow(core_flow)}")

        stable_core_flow = StableSpikingCoreFlow(self.input_shape, quantized_mapping, int(self.simulation_steps), self.preprocessor)
        print(f"  Original StableSpikingCoreFlow Accuracy: {self._validate_core_flow(stable_core_flow)}")

        self._tune_thresholds(
            stable_core_flow.get_core_spike_rates(), tuning_cycles=20, lr=0.5, core_flow_model=core_flow)
        
        final_mapping = copy.deepcopy(core_flow.core_mapping)
        core_flow = SpikingCoreFlow(self.input_shape, final_mapping, int(self.simulation_steps), self.preprocessor)
        print(f"  Original 2 SpikingCoreFlow Accuracy: {self._validate_core_flow(core_flow)}")
        
        self._quantize_thresholds(final_mapping, 1.0)
        scaled_simulation_steps = math.ceil(self.simulation_steps)
        core_flow = SpikingCoreFlow(self.input_shape, final_mapping, scaled_simulation_steps, self.preprocessor)
        self.accuracy = self._test_core_flow(core_flow)
        print(f"  Final SpikingCoreFlow Accuracy: {self.accuracy}")

        self.mapping = quantized_mapping
        return scaled_simulation_steps

    
    def _tune_thresholds(self, stable_spike_rates, tuning_cycles, lr, core_flow_model):
        print("  Tuning thresholds...")
        print(stable_spike_rates)
        print(core_flow_model.get_core_spike_rates())

        thresholds = [core.threshold for core in core_flow_model.cores]
        for i in range(tuning_cycles):
            print("acc: ", self._validate_core_flow(core_flow_model))

            print(f"  Tuning cycle {i+1}/{tuning_cycles}...")
            self._update_core_thresholds(
                stable_spike_rates, lr, core_flow_model, thresholds)


    def _update_core_thresholds(self, stable_spike_rates, lr, core_flow_model, thresholds):
        for core_id, core in enumerate(core_flow_model.cores):
            core_spike_rate = core_flow_model.get_core_spike_rates()[core_id]
            target_spike_rate = stable_spike_rates[core_id]
            thresholds[core_id] *= (1 + lr * (core_spike_rate - target_spike_rate))

            core.threshold = round(thresholds[core_id])
        
        print("  Updated thresholds: ", [core.threshold for core in core_flow_model.cores])
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
