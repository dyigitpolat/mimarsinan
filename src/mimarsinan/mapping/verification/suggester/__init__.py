"""Hardware configuration suggestion from layout verification stats."""

from mimarsinan.mapping.verification.suggester.hw_config_suggester import suggest_hardware_config
from mimarsinan.mapping.verification.suggester.hw_config_suggester_scheduled import (
    suggest_hardware_config_for_model,
    suggest_hardware_config_scheduled,
)
from mimarsinan.mapping.verification.suggester.hw_suggestion_types import HardwareSuggestion

__all__ = [
    "HardwareSuggestion",
    "suggest_hardware_config",
    "suggest_hardware_config_scheduled",
    "suggest_hardware_config_for_model",
]
