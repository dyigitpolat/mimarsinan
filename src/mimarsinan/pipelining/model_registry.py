"""Model type registry: builders self-register and expose config schema for GUI.

Builders register via @ModelRegistry.register(id, label, category) and implement
get_config_schema() so the GUI can build dynamic forms. Pipeline steps continue
to use their builder dicts and do not depend on this module for instantiation.
"""

from __future__ import annotations

from typing import Any


class ModelRegistry:
    """Registry of model types; builders register and expose metadata + config schema."""

    _registry: dict[str, dict[str, Any]] = {}

    @classmethod
    def register(cls, model_id: str, *, label: str, category: str) -> Any:
        """Decorator to register a builder class. Builder must implement get_config_schema()."""

        def decorator(builder_cls: type) -> type:
            cls._registry[model_id] = {
                "label": label,
                "category": category,
                "builder_cls": builder_cls,
            }
            return builder_cls

        return decorator

    @classmethod
    def _ensure_builders_loaded(cls) -> None:
        """Import builders package so all builders run their @ModelRegistry.register."""
        import mimarsinan.models.builders  # noqa: F401

    @classmethod
    def get_model_types(cls) -> list[dict[str, Any]]:
        """Return list of model types for the GUI (id, label, category)."""
        cls._ensure_builders_loaded()
        return [
            {"id": mid, "label": info["label"], "category": info["category"]}
            for mid, info in sorted(cls._registry.items())
        ]

    @classmethod
    def get_model_config_schema(cls, model_type: str) -> list[dict[str, Any]]:
        """Return config schema fields for the given model type from its builder.

        Returns a list of field descriptors with key, type, label, default,
        and optional options, min, max, step. Returns empty list if model_type
        is unknown or the builder has no get_config_schema.
        """
        cls._ensure_builders_loaded()
        entry = cls._registry.get(model_type)
        if entry is None:
            return []
        builder_cls = entry["builder_cls"]
        get_schema = getattr(builder_cls, "get_config_schema", None)
        if get_schema is None:
            return []
        schema = get_schema()
        return list(schema) if schema else []


# Public API used by GUI server (kept for minimal call-site changes)
def get_model_types() -> list[dict[str, Any]]:
    """Return list of model types for the GUI (id, label, category)."""
    return ModelRegistry.get_model_types()


def get_model_config_schema(model_type: str) -> list[dict[str, Any]]:
    """Return config schema for the given model type from its registered builder."""
    return ModelRegistry.get_model_config_schema(model_type)
