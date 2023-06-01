from mimarsinan.model_training.weight_transform_trainer import WeightTransformTrainer
from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.transformations.chip_quantization import ChipQuantization
from mimarsinan.models.layers import CQ_Activation
from mimarsinan.models.core_flow import CoreFlow
from mimarsinan.models.spiking_core_flow import SpikingCoreFlow

class CoreFlowTuner:
    def __init__(self, pipeline, mapping, target_tq):
        self.model = pipeline.model
        self.device = pipeline.device
        self.data_provider = pipeline.data_provider
        self.input_shape = pipeline.input_shape

        self.mapping = mapping
        self.target_tq = target_tq
        self.simulation_steps = pipeline.simulation_steps
        self.report_function = pipeline.reporter.report

    def _tune_thresholds(self):
        print("  Tuning thresholds...")

        core_flow = CoreFlow(self.input_shape, self.mapping)
        core_flow.set_activation(CQ_Activation(self.target_tq))
        core_flow_trainer = BasicTrainer(
            core_flow, 
            self.device, self.data_provider,
            None)
        core_flow_trainer.report_function = self.report_function 
        
        core_flow_trainer.validate()
        core_avgs, chip_avg = core_flow.core_avgs, core_flow.chip_avg

        for core in self.mapping.cores:
            core.threshold = core.threshold * 0.9

        spiking_core_flow = SpikingCoreFlow(self.input_shape, self.mapping, self.simulation_steps)
        spiking_core_flow_trainer = BasicTrainer(
            spiking_core_flow, 
            self.device, self.data_provider,
            None)
        spiking_core_flow_trainer.report_function = self.report_function 
        
        spiking_core_flow_trainer.validate()
        for idx, core in enumerate(self.mapping.cores):
            rate_numerator = core_avgs[idx] / chip_avg
            rate_denominator = spiking_core_flow.core_avgs[idx] / spiking_core_flow.chip_avg
            
            if rate_denominator == 0: 
                print(f"    WARNING: rate_denominator for core {idx} is 0, setting to 0.5")
                rate_denominator = 0.5
            rate = rate_numerator / rate_denominator

            factor = 5
            rate = ((factor - 1) + rate) / factor
            core.threshold = core.threshold / rate
        
        for core in self.mapping.cores:
            core.threshold = round(core.threshold)

    def _validate_core_flow(self, core_flow):
        core_flow_trainer = BasicTrainer(
            core_flow, 
            self.device, self.data_provider,
            None)
        core_flow_trainer.report_function = self.report_function
        
        return core_flow_trainer.validate()

    def run(self):
        core_flow = CoreFlow(self.input_shape, self.mapping)
        core_flow.set_activation(CQ_Activation(self.target_tq))
        print(f"  CQ CoreFlow Accuracy: {self._validate_core_flow(core_flow)}")

        scale = ChipQuantization(bits = 4).quantize(
            self.mapping.cores)
        
        spiking_core_flow = SpikingCoreFlow(self.input_shape, self.mapping, self.simulation_steps)
        print(f"  Original Spiking CoreFlow Accuracy: {self._validate_core_flow(spiking_core_flow)}")
        
        self._tune_thresholds()
        
        spiking_core_flow = SpikingCoreFlow(self.input_shape, self.mapping, self.simulation_steps)
        accuracy = self._validate_core_flow(spiking_core_flow)
        print(f"  Tuned Spiking CoreFlow Accuracy: {accuracy}")

        return accuracy, scale