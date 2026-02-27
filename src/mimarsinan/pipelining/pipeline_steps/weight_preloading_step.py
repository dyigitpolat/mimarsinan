"""
WeightPreloadingStep -- load pretrained weights into the model.

Replaces PretrainingStep when pretrained weights are available.
Supports torchvision pretrained weights, local checkpoints, and URLs.
Optionally fine-tunes for a configurable number of epochs after loading.
"""

from mimarsinan.pipelining.pipeline_step import PipelineStep
from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.model_training.weight_loading import resolve_weight_strategy
from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory

import torch


class WeightPreloadingStep(PipelineStep):
    """Load pretrained weights into the model and optionally fine-tune.

    This step replaces ``PretrainingStep`` when the pipeline is configured
    with ``weight_source``.  It resolves a weight-loading strategy from the
    config, loads weights with ``strict=False`` to handle classifier-head
    mismatches, and optionally fine-tunes for ``finetune_epochs``.

    Config keys consumed:
        ``weight_source`` (str):
            ``"torchvision"`` -- pretrained weights from the builder's factory.
            A file path (.pt / .pth / .ckpt) -- local checkpoint.
            A URL (http/https) -- download and load.

        ``finetune_epochs`` (int, default 0):
            Number of fine-tuning epochs after weight loading.
            Set to 0 to skip fine-tuning entirely.
    """

    def __init__(self, pipeline):
        requires = ["model", "model_builder"]
        promises = []
        updates = ["model"]
        clears = []
        super().__init__(requires, promises, updates, clears, pipeline)

        self.trainer = None

    def validate(self):
        if self.trainer is not None:
            return self.trainer.validate()
        return self.pipeline.get_target_metric()

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

        self.trainer = BasicTrainer(
            model,
            device,
            DataLoaderFactory(self.pipeline.data_provider_factory),
            self.pipeline.loss,
        )
        self.trainer.report_function = self.pipeline.reporter.report

        if finetune_epochs > 0:
            lr = self.pipeline.config.get("finetune_lr", self.pipeline.config["lr"])
            print(f"[WeightPreloadingStep] Fine-tuning for {finetune_epochs} epochs (lr={lr})")
            self.trainer.train_n_epochs(lr, finetune_epochs, warmup_epochs=1)
        else:
            print("[WeightPreloadingStep] No fine-tuning (finetune_epochs=0)")

        val_acc = self.validate()
        print(f"[WeightPreloadingStep] Validation accuracy: {val_acc:.4f}")

        self.update_entry("model", model, "torch_model")
