from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.transformations.chip_quantization import ChipQuantization
from mimarsinan.models.layers import CQ_Activation
from mimarsinan.models.core_flow import CoreFlow
from mimarsinan.models.spiking_core_flow import SpikingCoreFlow

class CoreFlowTuner:
    def __init__(self, pipeline, mapping):
        self.device = pipeline.config["device"]
        self.data_provider = pipeline.data_provider
        self.input_shape = pipeline.config["input_shape"]

        self.mapping = mapping
        self.target_tq = pipeline.config["target_tq"]
        self.simulation_steps = pipeline.config["simulation_steps"]
        self.report_function = pipeline.reporter.report
        self.quantization_bits = pipeline.config["weight_bits"]

    def _tune_thresholds(self):
        print("  Tuning thresholds...")

        core_flow = CoreFlow(self.input_shape, self.mapping)
        core_flow.set_activation(CQ_Activation(self.target_tq))
        core_flow_trainer = BasicTrainer(
            core_flow, 
            self.device, self.data_provider,
            None)
        core_flow_trainer.report_function = self.report_function 
        
        core_flow_trainer.validate_train()
        core_sums = core_flow.core_sums
        
        cycles = 20
        lr = 0.1
        best_acc = 0
        for _ in range(cycles):
            print(f"    Tuning Cycle {_ + 1}/{cycles}")
            
            spiking_core_flow = SpikingCoreFlow(self.input_shape, self.mapping, self.simulation_steps)
            spiking_core_flow_trainer = BasicTrainer(
                spiking_core_flow, 
                self.device, self.data_provider,
                None)
            spiking_core_flow_trainer.report_function = self.report_function 

            acc = spiking_core_flow_trainer.validate_train()
            print(f"    acc: {acc}")

            if acc > best_acc:
                best_acc = acc
                best_thresholds = [core.threshold for core in self.mapping.cores]

            for idx, core in enumerate(self.mapping.cores):
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
                
        
        for idx, core in enumerate(self.mapping.cores):
            core.threshold = best_thresholds[idx]
            core.threshold = round(core.threshold)

    def _validate_core_flow(self, core_flow):
        core_flow_trainer = BasicTrainer(
            core_flow, 
            self.device, self.data_provider,
            None)
        core_flow_trainer.report_function = self.report_function
        
        return core_flow_trainer.validate_train()

    def run(self):
        core_flow = CoreFlow(self.input_shape, self.mapping)
        core_flow.set_activation(CQ_Activation(self.target_tq))
        print(f"  CQ CoreFlow Accuracy: {self._validate_core_flow(core_flow)}")

        scale = ChipQuantization(bits = self.quantization_bits).quantize(
            self.mapping.cores)
        
        print(f"  scale: {scale}")
        self.simulation_steps = round(self.simulation_steps)
        
        spiking_core_flow = SpikingCoreFlow(self.input_shape, self.mapping, self.simulation_steps)
        print(f"  Original Spiking CoreFlow Accuracy: {self._validate_core_flow(spiking_core_flow)}")
        
        self._tune_thresholds()
        
        spiking_core_flow = SpikingCoreFlow(self.input_shape, self.mapping, self.simulation_steps)
        accuracy = self._validate_core_flow(spiking_core_flow)
        print(f"  Tuned Spiking CoreFlow Accuracy: {accuracy}")

        return accuracy, scale