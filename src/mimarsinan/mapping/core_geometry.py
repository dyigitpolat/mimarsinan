"""Occupied axon/neuron counts for HardCore-style objects."""

from __future__ import annotations

from typing import Any


def used_axons(core: Any, *, min_one: bool = False) -> int:
    """Live axon count: ``axons_per_core - available_axons``."""
    n = int(core.axons_per_core) - int(getattr(core, "available_axons", 0))
    return max(n, 1) if min_one else n


def used_neurons(core: Any, *, min_one: bool = False) -> int:
    """Live neuron count: ``neurons_per_core - available_neurons``."""
    n = int(core.neurons_per_core) - int(getattr(core, "available_neurons", 0))
    return max(n, 1) if min_one else n
