from typing import Iterable, cast

from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
from mimarsinan.pipelining.core.registry.trainer_factory import make_basic_trainer
from mimarsinan.pipelining.core.steps.trainer_pipeline_step import TrainerPipelineStep
from mimarsinan.spiking.gain_correction import per_perceptron_cascade_depth
from mimarsinan.tuning.orchestration.install_resolution import (
    ChannelStatsAccumulator,
    attach_activation_decorator,
    build_value_install_gauge,
    emit_value_gauge,
    gauge_summary,
    needs_quantile_deflation,
    value_gauge_thresholds,
    value_grid_levels,
)
from mimarsinan.models.nn.layers import SavedTensorDecorator
from mimarsinan.pipelining.pipeline_steps.activation_utils import (
    activation_scale_stats,
    analysis_batch_count,
    calibration_policy,
)

import torch

PRUNED_THRESHOLD = 1e-9
DEFAULT_SCALE_QUANTILE = 0.99
MIN_SCALE = 1e-6
MAX_SAMPLES_PER_BATCH = 8192
# Cap the analysis batch: per-perceptron saved-tensor decorators pin every full
# output tensor in VRAM, so large-input backbones OOM without a cap. Explicit
# activation_analysis_batch_size wins, then the calibration profile, then this.
ANALYSIS_BATCH_SIZE_CAP = 16


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
    """Count-based activation quantile over non-pruned positives, so post-pruning stats stay unskewed."""
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


def _attach_saved_tensor_decorator(perceptron):
    """Attach a sampling ``SavedTensorDecorator`` and return a cleanup callback.

    ``sample_to_cpu`` subsamples + moves to CPU inside the decorator so the GPU
    activation frees immediately (avoids OOM on large-input backbones).
    """
    decorator = SavedTensorDecorator(sample_to_cpu=True, max_samples=MAX_SAMPLES_PER_BATCH)
    cleanup = attach_activation_decorator(perceptron, decorator)
    return decorator, cleanup


class ActivationAnalysisStep(TrainerPipelineStep):
    REQUIRES = ("model",)
    PROMISES = (
        "activation_scales",
        "activation_scale_stats",
        "install_resolution_gauge",
    )

    def __init__(self, pipeline):
        super().__init__(self.REQUIRES, self.PROMISES, self.UPDATES, self.CLEARS, pipeline)

    def process(self):
        model = self.get_entry("model")
        self.trainer = make_basic_trainer(self.pipeline, model)

        override = self.pipeline.config.get("activation_analysis_batch_size")
        profile_cap = calibration_policy(self.pipeline).analysis_batch_size_cap
        cap = ANALYSIS_BATCH_SIZE_CAP if profile_cap is None else int(profile_cap)
        current_val_bs = self.trainer.validation_batch_size
        target_bs = int(override) if override else min(current_val_bs, cap)
        if target_bs > 0 and target_bs < current_val_bs:
            self.trainer.set_validation_batch_size(target_bs)

        perceptrons = list(model.get_perceptrons())
        decorators = []
        channel_accumulators = []
        cleanup_callbacks = []
        for perceptron in perceptrons:
            decorator, cleanup = _attach_saved_tensor_decorator(perceptron)
            decorators.append(decorator)
            cleanup_callbacks.append(cleanup)
            # [MBH-A6] channel-resolved capture rides the same forward pass.
            accumulator = ChannelStatsAccumulator()
            channel_accumulators.append(accumulator)
            cleanup_callbacks.append(
                attach_activation_decorator(perceptron, accumulator)
            )

        n_batches = analysis_batch_count(self.pipeline)
        sampled_activations = [[] for _ in perceptrons]

        try:
            self.trainer.model.eval()
            with torch.no_grad():
                # cast: the validation cache yields (input, target) tensor pairs; _gpu_val_cache is untyped upstream.
                val_batches = cast(
                    "Iterable[tuple[torch.Tensor, torch.Tensor]]",
                    self.trainer.iter_validation_batches(n_batches),
                )
                for x, _ in val_batches:
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

        # activation_scale is BOTH the LIF/TTFS decode scale and the clamp ceiling, so raising the quantile toward 1.0 trades rate-resolution for less saturation clipping.
        quantile = float(
            self.pipeline.config.get("activation_scale_quantile", DEFAULT_SCALE_QUANTILE)
        )
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
                    quantile=quantile,
                    min_scale=MIN_SCALE,
                )
            )

        deflated_layers = self._deflate_starved_scales(
            perceptrons, channel_accumulators, merged_samples, activation_scales,
            quantile=quantile,
        )

        scale_stats = activation_scale_stats(
            perceptrons,
            merged_samples,
            activation_scales,
            num_batches=n_batches,
            quantile=quantile,
            max_samples_per_batch=MAX_SAMPLES_PER_BATCH,
            pruned_threshold=PRUNED_THRESHOLD,
        )
        if deflated_layers is not None:
            scale_stats["deflated_layers"] = deflated_layers
        scale_summary = scale_stats["summary"]
        print(
            "[ActivationAnalysisStep] "
            f"batches={n_batches}, q={quantile}, "
            f"scale_range=[{scale_summary['min_scale']:.4f}, {scale_summary['max_scale']:.4f}]"
        )

        validation_loader = self.trainer.validation_loader
        assert validation_loader is not None, "trainer was closed during analysis"
        self.trainer.val_iter = iter(validation_loader)

        gauge = self._a6_value_gauge(
            model, perceptrons, channel_accumulators, activation_scales,
        )

        self.add_entry("activation_scales", activation_scales)
        self.add_entry("activation_scale_stats", scale_stats)
        self.add_entry("install_resolution_gauge", gauge_summary(gauge))

    def _deflate_starved_scales(
        self, perceptrons, accumulators, merged_samples, scales, *, quantile,
    ):
        """[5v B1(i)] sync-recipe-gated starvation-aware quantile: a hop whose
        ``quantile`` theta starves the grid recomputes its scale at 0.99."""
        plan = DeploymentPlan.of(self.pipeline)
        armed = bool(
            self.pipeline.config.get("starvation_aware_scale_quantile", False)
        ) and plan.is_synchronized_ttfs
        if not armed:
            return None
        levels = value_grid_levels(plan.spiking_mode, self.pipeline.config)
        if levels is None or quantile <= DEFAULT_SCALE_QUANTILE:
            return []
        deflated = []
        for idx, (perceptron, acc) in enumerate(zip(perceptrons, accumulators)):
            if not needs_quantile_deflation(acc.per_channel_q99(), scales[idx], levels):
                continue
            scales[idx] = scale_from_activations(
                merged_samples[idx], quantile=DEFAULT_SCALE_QUANTILE, min_scale=MIN_SCALE,
            )
            deflated.append(perceptron.name)
            print(
                f"[MBH-A6] quantile-deflate hop={perceptron.name} {quantile} -> "
                f"{DEFAULT_SCALE_QUANTILE} (starved at the {levels}-step grid)",
                flush=True,
            )
        return deflated

    def _a6_value_gauge(self, model, perceptrons, accumulators, scales):
        """[MBH-A6] the pre-flight value-starvation gauge at the install grid
        (warn-only; thresholds conditioned on the install kernel per the
        tier-0.1 corpus calibration)."""
        spiking_mode = DeploymentPlan.of(self.pipeline).spiking_mode
        levels = value_grid_levels(spiking_mode, self.pipeline.config)
        if levels is None:
            return None
        min_levels, mass_warn = value_gauge_thresholds(spiking_mode)
        depths = per_perceptron_cascade_depth(model.get_mapper_repr())
        gauge = build_value_install_gauge(
            list(zip(perceptrons, accumulators)), scales, depths, levels,
            min_levels=min_levels, mass_warn=mass_warn,
        )
        emit_value_gauge(type(self).__name__, gauge, reporter=self.pipeline.reporter)
        return gauge