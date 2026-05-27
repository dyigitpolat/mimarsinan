"""
WeightPreloadingStep -- load pretrained weights into the model.

Replaces PretrainingStep when pretrained weights are available.
Supports torchvision pretrained weights, local checkpoints, and URLs.
Optionally fine-tunes for a configurable number of epochs after loading.
"""

from mimarsinan.pipelining.core.registry.trainer_factory import make_basic_trainer
from mimarsinan.pipelining.core.steps.trainer_pipeline_step import TrainerPipelineStep
from mimarsinan.model_training.training_recipe import build_recipe
from mimarsinan.model_training.weight_loading import resolve_weight_strategy

import torch


class WeightPreloadingStep(TrainerPipelineStep):
    """Load pretrained weights into the model and optionally fine-tune."""

    def __init__(self, pipeline):
        requires = ["model", "model_builder"]
        promises = []
        updates = ["model"]
        clears = []
        super().__init__(requires, promises, updates, clears, pipeline)

    def process(self):
        model = self.get_entry("model")
        builder = self.get_entry("model_builder")

        weight_source = self.pipeline.config.get("weight_source", "")
        strategy = resolve_weight_strategy(weight_source, model_builder=builder)

        if strategy is None:
            print("[WeightPreloadingStep] No weight_source configured, skipping.")
            self.update_entry("model", model, "torch_model")
            return

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
        self.trainer = make_basic_trainer(
            self.pipeline, model, recipe=recipe
        )

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
