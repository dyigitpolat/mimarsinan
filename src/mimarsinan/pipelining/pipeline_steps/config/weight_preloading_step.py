"""WeightPreloadingStep -- load pretrained weights into the model and optionally fine-tune."""

from mimarsinan.common.measurement import (
    MetricTolerance,
    assert_matches_recorded_baseline,
)
from mimarsinan.common.pretrained import unusable_baseline_reason
from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
from mimarsinan.pipelining.core.registry.trainer_factory import make_basic_trainer
from mimarsinan.pipelining.core.steps.trainer_pipeline_step import TrainerPipelineStep
from mimarsinan.model_training.training_recipe import build_recipe
from mimarsinan.model_training.weight_loading import resolve_weight_strategy


class WeightPreloadingStep(TrainerPipelineStep):
    """Load pretrained weights into the model and optionally fine-tune."""

    REQUIRES = ("model", "model_builder")
    UPDATES = ("model",)

    @classmethod
    def applies_to(cls, plan):
        return bool(plan.weight_source)

    def __init__(self, pipeline):
        super().__init__(self.REQUIRES, self.PROMISES, self.UPDATES, self.CLEARS, pipeline)

    def _assert_recorded_baseline(self, trainer, weight_set) -> None:
        """The loaded weights must reproduce the accuracy the weight set records.

        Run before any fine-tuning, so the number asserted is the load's own.
        """
        if weight_set is None:
            return
        reason = unusable_baseline_reason(weight_set, self.pipeline.config)
        if reason is not None:
            print(f"[WeightPreloadingStep] Recorded baseline not asserted: {reason}")
            return
        expected = float(weight_set["expected_accuracy"])
        measured, samples = trainer.validate_measured()
        print(
            f"[WeightPreloadingStep] Recorded baseline {expected:.4f}; measured "
            f"{measured:.4f} over {samples} validation samples"
        )
        assert_matches_recorded_baseline(
            measured,
            expected=expected,
            tolerance=MetricTolerance.for_sampled_proportion(expected, samples),
            observable="the preloaded model's validation accuracy",
            recorded_as=f"weight set {weight_set['id']!r} on {weight_set['dataset']}",
            causes=(
                f"this run's data pipeline does not reproduce the preprocessing "
                f"the weight set records ({weight_set.get('preprocessing')})",
            ),
        )

    def process(self):
        model = self.get_entry("model")
        builder = self.get_entry("model_builder")

        plan = DeploymentPlan.of(self.pipeline)
        weight_source = plan.weight_source
        weight_set = plan.pretrained_weight_set
        strategy = resolve_weight_strategy(
            weight_source,
            model_builder=builder,
            weight_set_id=None if weight_set is None else str(weight_set["id"]),
        )

        if strategy is None:
            print("[WeightPreloadingStep] No weight_source configured, skipping.")
            self.update_entry("model", model, "torch_model")
            return

        if weight_set is not None:
            print(
                f"[WeightPreloadingStep] Weight set: {weight_set['id']} "
                f"({weight_set['label']}; {weight_set['task']} on "
                f"{weight_set['dataset']}, {weight_set['num_classes']} classes)"
            )

        device = self.pipeline.config["device"]
        model, info = strategy.load(model)
        model = model.to(device)

        matched = info.get("matched", "?")
        missing = info.get("missing_keys", [])
        unexpected = info.get("unexpected_keys", [])
        source = info.get("source", weight_source)

        print(f"[WeightPreloadingStep] Loaded weights from: {source}")
        print(f"  Matched parameters: {matched}")
        if missing:
            print(f"  Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

        finetune_epochs = int(self.pipeline.config.get("finetune_epochs", 0))
        recipe = build_recipe(self.pipeline.config)
        trainer = make_basic_trainer(self.pipeline, model, recipe=recipe)
        self.trainer = trainer
        self._assert_recorded_baseline(trainer, weight_set)

        if finetune_epochs > 0:
            lr = self.pipeline.config.get("finetune_lr", self.pipeline.config["lr"])
            recipe_tag = f" recipe={recipe.optimizer}" if recipe is not None else ""
            print(
                f"[WeightPreloadingStep] Fine-tuning for {finetune_epochs} epochs "
                f"(lr={lr}{recipe_tag})"
            )
            warmup_epochs = 0 if recipe is not None else 5
            self.trainer.train_n_epochs(lr, finetune_epochs, warmup_epochs=warmup_epochs)
        else:
            print("[WeightPreloadingStep] No fine-tuning (finetune_epochs=0)")

        val_acc = self.validate()
        print(f"[WeightPreloadingStep] Validation accuracy: {val_acc:.4f}")
        self.update_entry("model", model, "torch_model")
