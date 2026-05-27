"""Shared model_config / model_builder cache emission for config steps."""

from __future__ import annotations

from mimarsinan.pipelining.core.registry.model_registry import ModelRegistry


def emit_model_config_entries(step, pipeline_config: dict) -> tuple[object, dict]:
    """Create builder and return ``(builder, model_config)`` for cache promises."""
    model_type = pipeline_config["model_type"]
    builder_cls = ModelRegistry.get_builder_cls(model_type)
    model_config = pipeline_config["model_config"]
    builder = builder_cls(
        pipeline_config["device"],
        pipeline_config["input_shape"],
        pipeline_config["num_classes"],
        pipeline_config,
    )
    step.add_entry("model_builder", builder, "pickle")
    step.add_entry("model_config", model_config)
    return builder, model_config
