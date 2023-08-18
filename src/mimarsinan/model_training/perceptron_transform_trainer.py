from mimarsinan.model_training.weight_transform_trainer import WeightTransformTrainer
from mimarsinan.models.perceptron_mixer.perceptron_flow import PerceptronFlow


class PerceptronTransformTrainer(WeightTransformTrainer):
    def __init__(
            self, model, device, data_provider_factory, loss_function, perceptron_transformation):
        super().__init__(model, device, data_provider_factory, loss_function, None)
        assert isinstance(model, PerceptronFlow)

        self.perceptron_transformation = perceptron_transformation

    def _update_and_transform_model(self):
        for perceptron, aux_perceptron in zip(self.model.get_perceptrons(), self.aux_model.get_perceptrons()):
            transformed_perceptron = self.perceptron_transformation(aux_perceptron).to(self.device)
            for param, aux_param in zip(perceptron.parameters(), transformed_perceptron.parameters()):
                param.data[:] = aux_param.data[:]
