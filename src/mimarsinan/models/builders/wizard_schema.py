"""
Collect wizard config schema from model builders.

Each builder class exposes WIZARD_SCHEMA (label, description, config_schema).
This module only aggregates from BUILDERS_REGISTRY; it does not define schemas.
"""

from __future__ import annotations

from typing import Any, Dict, List

from . import BUILDERS_REGISTRY


def get_all_model_type_schemas() -> List[Dict[str, Any]]:
    """Return list of {id, label, description, config_schema} from each builder class."""
    out: List[Dict[str, Any]] = []
    for model_id, builder_cls in BUILDERS_REGISTRY.items():
        schema = getattr(builder_cls, "WIZARD_SCHEMA", None)
        
        if schema is None:
            # Shim for newer builders that don't have WIZARD_SCHEMA attribute
            label = getattr(builder_cls, "label", model_id)
            config_schema = []
            if hasattr(builder_cls, "get_config_schema"):
                config_schema = builder_cls.get_config_schema()
            
            schema = {
                "label": label,
                "description": "",
                "config_schema": config_schema
            }
            
        out.append({"id": model_id, **schema})
    return out


def get_config_schema_for_model_type(model_type: str) -> Dict[str, Any]:
    """Return config_schema dict for a given model type, or empty dict if unknown."""
    builder_cls = BUILDERS_REGISTRY.get(model_type)
    if builder_cls is None:
        return {}
    schema = getattr(builder_cls, "WIZARD_SCHEMA", None)
    if schema is None:
        return {}
    return schema.get("config_schema", {})
