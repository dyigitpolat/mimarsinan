from mimarsinan.model_training.weight_transform_trainer import WeightTransformTrainer
from mimarsinan.transformations.chip_quantization import ChipQuantization
from mimarsinan.models.layers import CQ_Activation
from mimarsinan.models.core_flow import CoreFlow

class CoreFlowTuner:
    def __init__(self, pipeline, mapping, target_tq):
        self.model = pipeline.model
        self.device = pipeline.device
        self.train_loader = pipeline.training_dataloader
        self.test_loader = pipeline.test_dataloader
        self.input_shape = pipeline.input_shape

        self.mapping = mapping
        self.target_tq = target_tq

    def run(self):
        core_flow = CoreFlow(self.input_shape, self.mapping)
        core_flow.set_activation(CQ_Activation(self.target_tq))
        core_flow_trainer = WeightTransformTrainer(
            core_flow, 
            self.device, self.train_loader, self.test_loader, 
            None, None)
        
        accuracy = core_flow_trainer.validate()
        scale = ChipQuantization(bits = 4).quantize(
            self.hard_core_mapping.cores)

        return accuracy, scale