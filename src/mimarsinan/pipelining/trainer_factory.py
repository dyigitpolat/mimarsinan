"""Shared BasicTrainer construction for pipeline steps."""

from __future__ import annotations

from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory
from mimarsinan.model_training.basic_trainer import BasicTrainer


def make_basic_trainer(pipeline, model, loss=None) -> BasicTrainer:
    return BasicTrainer(
        model,
        pipeline.config["device"],
        DataLoaderFactory(pipeline.data_provider_factory),
        loss if loss is not None else pipeline.loss,
    )
