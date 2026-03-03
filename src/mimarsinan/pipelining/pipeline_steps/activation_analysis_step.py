from mimarsinan.pipelining.pipeline_step import PipelineStep
from mimarsinan.model_training.basic_trainer import BasicTrainer

from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory
from mimarsinan.models.layers import SavedTensorDecorator

import torch

# Epsilon below which activations are treated as pruned (zero) and excluded.
PRUNED_THRESHOLD = 1e-9


def scale_from_activations(flat_acts, pruned_threshold=PRUNED_THRESHOLD):
    """Compute 99th-percentile (by cumulative sum) scale from activations.

    Only non-pruned activations (above pruned_threshold) are included so that
    post-pruning statistics are not skewed and clamping does not over-degrade.
    """
    active_mask = flat_acts > pruned_threshold
    active_acts = flat_acts[active_mask]

    if active_acts.numel() == 0:
        return max(flat_acts.max().item(), 1.0) if flat_acts.numel() > 0 else 1.0

    sorted_acts, _ = torch.sort(active_acts)
    cumsum_acts = torch.cumsum(sorted_acts, dim=0)
    norm_cumsum = cumsum_acts / cumsum_acts[-1]
    threshold_idx = torch.searchsorted(norm_cumsum, 0.99)
    return sorted_acts[threshold_idx].item()


class ActivationAnalysisStep(PipelineStep):
    def __init__(self, pipeline):
        requires = ["model"]
        promises = ["activation_scales"]
        updates = []
        clears = []
        super().__init__(requires, promises, updates, clears, pipeline)

        self.trainer = None

    def validate(self):
        return self.trainer.validate()

    def process(self):
        model = self.get_entry("model")

        self.trainer = BasicTrainer(
            model, 
            self.pipeline.config['device'], 
            DataLoaderFactory(self.pipeline.data_provider_factory),
            self.pipeline.loss)

        for perceptron in model.get_perceptrons():
            perceptron.activation.decorate(SavedTensorDecorator())

        self.trainer.validate()

        activation_scales = []
        for perceptron in model.get_perceptrons():
            saved_tensor = perceptron.activation.pop_decorator()
            flat_acts = saved_tensor.latest_output.view(-1)
            activation_scales.append(scale_from_activations(flat_acts))

        print(activation_scales)

        self.add_entry("activation_scales", activation_scales)


        