"""Spawn-pool worker executing one Lava core run outside the scheduling process."""

from __future__ import annotations

import multiprocessing

import numpy as np

from mimarsinan.chip_simulation.behavior_config import NeuralBehaviorConfig
from mimarsinan.chip_simulation.lava_loihi.core_lava import (
    LavaCoreMixin,
    _subtractive_lif_cls,
)


class _WaveCoreHost(LavaCoreMixin):
    """Minimal LavaCoreMixin host rebuilt from the picklable per-task state."""

    def __init__(self, T: int, behavior: NeuralBehaviorConfig) -> None:
        self.T = int(T)
        self._behavior = behavior
        self.thresholding_mode = behavior.thresholding_mode


def run_lava_core_task(
    T: int,
    behavior: NeuralBehaviorConfig,
    weights: np.ndarray,
    threshold: float,
    hardware_bias: np.ndarray | None,
    input_spikes: np.ndarray,
) -> np.ndarray:
    """Run one core's Dense+LIF Lava graph; identical semantics to the inline ``_run_core_lava`` unit."""
    # Lava's channel runtime deadlocks under spawn-context actors, so the fork
    # method it pins must be restored before any lava import; forking is safe
    # here because a fresh spawn worker owns no CUDA/OpenMP state.
    multiprocessing.set_start_method("fork", force=True)
    _subtractive_lif_cls()
    return _WaveCoreHost(T, behavior)._run_core_lava(
        weights=weights,
        threshold=threshold,
        hardware_bias=hardware_bias,
        input_spikes=input_spikes,
    )
