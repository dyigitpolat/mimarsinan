"""Model type registry: builders self-register and expose config schema + NAS hooks."""

from __future__ import annotations

from typing import Any


class ModelRegistry:
    """Registry of model types; builders register and expose metadata + config schema."""

    _registry: dict[str, dict[str, Any]] = {}

    @classmethod
    def register(cls, model_id: str, *, label: str, category: str) -> Any:
        """Decorator to register a builder class.

        The builder must implement get_config_schema(); it may also implement
        get_nas_search_options() and validate_config() to join the generic NAS flow.
        """

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
        import mimarsinan.models.builders  # noqa: F401  # pyright: ignore[reportUnusedImport] — side-effect registration

    @classmethod
    def get_category(cls, model_id: str) -> str | None:
        """Return the category for the given model_id, or None if unknown."""
        cls._ensure_builders_loaded()
        entry = cls._registry.get(model_id)
        if entry is None:
            return None
        return entry.get("category")

    @classmethod
    def get_builder_cls(cls, model_id: str) -> type:
        """Return the builder class registered under model_id.

        Raises ValueError for unknown model types with a helpful message.
        """
        cls._ensure_builders_loaded()
        entry = cls._registry.get(model_id)
        if entry is None:
            raise ValueError(
                f"Unknown model_type: {model_id!r}. "
                f"Available types: {sorted(cls._registry.keys())}"
            )
        return entry["builder_cls"]

    @classmethod
    def get_workload_profile(cls, model_id: str):
        """The builder-declared ``ModelWorkloadProfile``; ``None`` only when the
        model type is UNKNOWN.

        A known builder that declares no hook still has a registration — the EMPTY
        one — so "this builder registers no pretrained weights" is a claim the
        framework can read, distinct from "no builder was consulted".
        """
        from mimarsinan.common.workload_profile import ModelWorkloadProfile

        cls._ensure_builders_loaded()
        entry = cls._registry.get(model_id)
        if entry is None:
            return None
        hook = getattr(entry["builder_cls"], "workload_profile", None)
        return ModelWorkloadProfile() if hook is None else hook()

    @classmethod
    def builder_classes(cls) -> dict[str, type]:
        """model_type id -> registered builder class (the one builder SSOT view)."""
        cls._ensure_builders_loaded()
        return {mid: info["builder_cls"] for mid, info in cls._registry.items()}

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
        """Return config schema field descriptors for the model type, or [] if unknown."""
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


def get_model_types() -> list[dict[str, Any]]:
    """Return list of model types for the GUI (id, label, category)."""
    return ModelRegistry.get_model_types()


def get_model_config_schema(model_type: str) -> list[dict[str, Any]]:
    """Return config schema for the given model type from its registered builder."""
    return ModelRegistry.get_model_config_schema(model_type)
