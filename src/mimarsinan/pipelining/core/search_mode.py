"""Search mode resolution from pipeline config."""

from __future__ import annotations


def derive_search_mode(config: dict) -> str:
    """Derive the internal search mode from config keys."""
    model_search = config.get("model_config_mode", "user") == "search"
    hw_search = config.get("hw_config_mode", "fixed") == "search"
    if model_search and hw_search:
        return "joint"
    if model_search:
        return "model"
    if hw_search:
        return "hardware"
    return "fixed"
