from mimarsinan.pipelining.pipeline_step import PipelineStep
from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.tuning.tuning_budget import tuning_budget_from_pipeline

from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory
from mimarsinan.models.layers import SavedTensorDecorator, TransformedActivation

import torch

# Epsilon below which activations are treated as pruned (zero) and excluded.
PRUNED_THRESHOLD = 1e-9
DEFAULT_SCALE_QUANTILE = 0.99
MIN_SCALE = 1e-6
MIN_ANALYSIS_BATCHES = 2
MAX_ANALYSIS_BATCHES = 4
MAX_SAMPLES_PER_BATCH = 8192


def _analysis_batch_count(pipeline) -> int:
    """Bound activation-stat collection cost while avoiding single-batch calibration."""
    budget = tuning_budget_from_pipeline(pipeline)
    return max(MIN_ANALYSIS_BATCHES, min(MAX_ANALYSIS_BATCHES, budget.validation_steps))


def _sample_activation_values(flat_acts, max_samples=MAX_SAMPLES_PER_BATCH):
    """Deterministically subsample a large activation vector for cheap aggregation."""
    flat_acts = flat_acts.detach().reshape(-1).to(torch.float32)
    if flat_acts.numel() <= max_samples:
        return flat_acts.cpu()

    indices = torch.linspace(
        0,
        flat_acts.numel() - 1,
        steps=int(max_samples),
        device=flat_acts.device,
    ).round().long()
    return flat_acts.index_select(0, indices).cpu()


def scale_from_activations(
    flat_acts,
    pruned_threshold=PRUNED_THRESHOLD,
    *,
    quantile=DEFAULT_SCALE_QUANTILE,
    min_scale=MIN_SCALE,
):
    """Compute a count-based activation quantile over positive activations.

    Only non-pruned activations (above pruned_threshold) are included so that
    post-pruning statistics are not skewed and clamping does not over-degrade.
    """
    active_mask = flat_acts > pruned_threshold
    active_acts = flat_acts[active_mask]

    if active_acts.numel() == 0:
        return max(flat_acts.max().item(), 1.0) if flat_acts.numel() > 0 else 1.0

    q = torch.quantile(
        active_acts.to(torch.float32),
        float(quantile),
        interpolation="higher",
    ).item()
    return max(float(q), float(min_scale))


def _activation_scale_stats(
    perceptrons,
    sampled_activations,
    activation_scales,
    *,
    num_batches,
    quantile,
    max_samples_per_batch,
):
    layer_stats = []
    for idx, (perceptron, samples, scale) in enumerate(
        zip(perceptrons, sampled_activations, activation_scales)
    ):
        sample_count = int(samples.numel())
        active_samples = samples[samples > PRUNED_THRESHOLD]
        if sample_count > 0:
            sample_min = float(samples.min().item())
            sample_median = float(torch.quantile(samples, 0.5).item())
            sample_max = float(samples.max().item())
        else:
            sample_min = 0.0
            sample_median = 0.0
            sample_max = 0.0

        layer_stats.append(
            {
                "index": idx,
                "name": perceptron.name,
                "scale": float(scale),
                "sample_count": sample_count,
                "active_sample_count": int(active_samples.numel()),
                "sample_min": sample_min,
                "sample_median": sample_median,
                "sample_max": sample_max,
            }
        )

    sorted_scales = sorted(float(s) for s in activation_scales) or [1.0]
    return {
        "num_batches": int(num_batches),
        "quantile": float(quantile),
        "pruned_threshold": float(PRUNED_THRESHOLD),
        "max_samples_per_batch": int(max_samples_per_batch),
        "summary": {
            "min_scale": sorted_scales[0],
            "median_scale": sorted_scales[len(sorted_scales) // 2],
            "max_scale": sorted_scales[-1],
        },
        "layers": layer_stats,
    }


def _attach_saved_tensor_decorator(perceptron):
    """Attach a SavedTensorDecorator and return a cleanup callback."""
    decorator = SavedTensorDecorator()
    activation = perceptron.activation
    if hasattr(activation, "decorate") and hasattr(activation, "pop_decorator"):
        activation.decorate(decorator)

        def cleanup():
            activation.pop_decorator()

        return decorator, cleanup

    wrapped_activation = TransformedActivation(activation, [decorator])
    perceptron.set_activation(wrapped_activation)

    def cleanup():
        perceptron.set_activation(activation)

    return decorator, cleanup


class ActivationAnalysisStep(PipelineStep):
    def __init__(self, pipeline):
        requires = ["model"]
        promises = ["activation_scales", "activation_scale_stats"]
        updates = []
        clears = []
        super().__init__(requires, promises, updates, clears, pipeline)

        self.trainer = None

    def validate(self):
        if self.trainer is not None:
            return self.trainer.validate()
        return self.pipeline.get_target_metric()

    def cleanup(self):
        if self.trainer is not None:
            self.trainer.close()

    def process(self):
        model = self.get_entry("model")

        self.trainer = BasicTrainer(
            model,
            self.pipeline.config['device'],
            DataLoaderFactory(self.pipeline.data_provider_factory),
            self.pipeline.loss)

        perceptrons = list(model.get_perceptrons())
        decorators = []
        cleanup_callbacks = []
        for perceptron in perceptrons:
            decorator, cleanup = _attach_saved_tensor_decorator(perceptron)
            decorators.append(decorator)
            cleanup_callbacks.append(cleanup)

        n_batches = _analysis_batch_count(self.pipeline)
        sampled_activations = [[] for _ in perceptrons]

        try:
            self.trainer.model.eval()
            with torch.no_grad():
                for x, _ in self.trainer.iter_validation_batches(n_batches):
                    x = x.to(self.pipeline.config["device"])
                    _ = self.trainer.model(x)

                    for idx, decorator in enumerate(decorators):
                        latest = decorator.latest_output
                        if latest is None:
                            continue
                        sampled = _sample_activation_values(
                            latest, max_samples=MAX_SAMPLES_PER_BATCH
                        )
                        if sampled.numel() > 0:
                            sampled_activations[idx].append(sampled)
                        decorator.latest_output = None
        finally:
            for cleanup in reversed(cleanup_callbacks):
                cleanup()

        merged_samples = []
        activation_scales = []
        for layer_samples in sampled_activations:
            if layer_samples:
                merged = torch.cat(layer_samples)
            else:
                merged = torch.empty(0, dtype=torch.float32)
            merged_samples.append(merged)
            activation_scales.append(
                scale_from_activations(
                    merged,
                    quantile=DEFAULT_SCALE_QUANTILE,
                    min_scale=MIN_SCALE,
                )
            )

        activation_scale_stats = _activation_scale_stats(
            perceptrons,
            merged_samples,
            activation_scales,
            num_batches=n_batches,
            quantile=DEFAULT_SCALE_QUANTILE,
            max_samples_per_batch=MAX_SAMPLES_PER_BATCH,
        )
        scale_summary = activation_scale_stats["summary"]
        print(
            "[ActivationAnalysisStep] "
            f"batches={n_batches}, q={DEFAULT_SCALE_QUANTILE}, "
            f"scale_range=[{scale_summary['min_scale']:.4f}, {scale_summary['max_scale']:.4f}]"
        )

        self.trainer.val_iter = iter(self.trainer.validation_loader)

        self.add_entry("activation_scales", activation_scales)
        self.add_entry("activation_scale_stats", activation_scale_stats)