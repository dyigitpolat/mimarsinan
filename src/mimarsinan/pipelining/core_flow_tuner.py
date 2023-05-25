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
        self.train_loader = pipeline.training_dataloader
        self.test_loader = pipeline.test_dataloader
        self.input_shape = pipeline.input_shape

        self.mapping = mapping
        self.target_tq = target_tq
        self.simulation_steps = pipeline.simulation_steps

    def _tune_thresholds(self):
        print("  Tuning thresholds...")

        core_flow = CoreFlow(self.input_shape, self.mapping)
        core_flow.set_activation(CQ_Activation(self.target_tq))
        core_flow_trainer = BasicTrainer(
            core_flow, 
            self.device, self.train_loader, self.test_loader, 
            None)
        
        core_flow_trainer.validate_train()
        core_avgs, chip_avg = core_flow.core_avgs, core_flow.chip_avg

        for core in self.mapping.cores:
            core.threshold = core.threshold * 0.9

        cycles = 10
        for _ in range(cycles):
            spiking_core_flow = SpikingCoreFlow(self.input_shape, self.mapping, self.simulation_steps)
            spiking_core_flow_trainer = BasicTrainer(
                spiking_core_flow, 
                self.device, self.train_loader, self.test_loader,
                None)
            
            spiking_accuracy = spiking_core_flow_trainer.validate_train()
            print(f"    Current Spiking CoreFlow Accuracy: {spiking_accuracy}")
            for idx, core in enumerate(self.mapping.cores):
                rate_numerator = core_avgs[idx] / chip_avg
                rate_denominator = spiking_core_flow.core_avgs[idx] / spiking_core_flow.chip_avg
                rate = rate_numerator / rate_denominator

                factor = cycles // 2
                rate = ((factor - 1) + rate) / factor
                core.threshold = core.threshold / rate
        
        for core in self.mapping.cores:
            core.threshold = round(core.threshold)

    def _validate_core_flow(self, core_flow):
        core_flow_trainer = BasicTrainer(
            core_flow, 
            self.device, self.train_loader, self.test_loader, 
            None)
        
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