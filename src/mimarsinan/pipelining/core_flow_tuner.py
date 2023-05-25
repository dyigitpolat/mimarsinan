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

    def run(self):
        
        core_flow = CoreFlow(self.input_shape, self.mapping)
        core_flow.set_activation(CQ_Activation(self.target_tq))
        core_flow_trainer = BasicTrainer(
            core_flow, 
            self.device, self.train_loader, self.test_loader, 
            None)
        
        accuracy = core_flow_trainer.validate()
        scale = ChipQuantization(bits = 4).quantize(
            self.mapping.cores)
        
        spiking_core_flow = SpikingCoreFlow(self.input_shape, self.mapping, self.simulation_steps)
        spiking_core_flow_trainer = BasicTrainer(
            spiking_core_flow, 
            self.device, self.train_loader, self.test_loader,
            None)
        
        spiking_accuracy = spiking_core_flow_trainer.validate_train()
        print(f"  Accuracy: {accuracy}, Spiking Accuracy: {spiking_accuracy}")
        
        for core in self.mapping.cores:
            core.threshold = round(core.threshold * 0.5)

        for cycle in range(10):
            spiking_core_flow = SpikingCoreFlow(self.input_shape, self.mapping, self.simulation_steps)
            spiking_core_flow_trainer = BasicTrainer(
                spiking_core_flow, 
                self.device, self.train_loader, self.test_loader,
                None)
            
            spiking_accuracy = spiking_core_flow_trainer.validate_train()
            for idx, core in enumerate(self.mapping.cores):
                rate_numerator = core_flow.core_avgs[idx] / core_flow.chip_avg
                rate_denominator = spiking_core_flow.core_avgs[idx] / spiking_core_flow.chip_avg
                rate = rate_numerator / rate_denominator
                rate = (9 + rate) / 10

                print("core threshold before: ", core.threshold)
                core.threshold = core.threshold / rate
                print("core threshold after: ", core.threshold)
        
        for core in self.mapping.cores:
            core.threshold = round(core.threshold)

        spiking_core_flow = SpikingCoreFlow(self.input_shape, self.mapping, self.simulation_steps)
        spiking_core_flow_trainer = BasicTrainer(
            spiking_core_flow, 
            self.device, self.train_loader, self.test_loader,
            None)
        
        spiking_accuracy = spiking_core_flow_trainer.validate_train()

        print(f"  Accuracy: {accuracy}, Spiking Accuracy: {spiking_accuracy}")

        return accuracy, scale