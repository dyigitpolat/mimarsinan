"""Small law-carrying value programs for the instance tests: a CIFAR-style residual net (two convolutions, a
projection shortcut, a host add, ReLUs and a pool on the host) built the pipeline's way, packed onto few cores so it
runs in several passes. Every constant here is a test constant."""

from __future__ import annotations

import functools
from typing import Any, Dict

from mimarsinan.deployment_record.platform_physics.registry import get_platform_physics
from mimarsinan.pipelining.core.simulation_factory import build_hybrid_mapping_for_pipeline
from unit.chip_simulation.value_run.integer_programs import build_integer_program, integer_inputs, residual_block_net

AXONS, NEURONS = 256, 32
#: Test platform: few cores (several passes, rewritten per period), cell write granularity, one write per inference
#: rate and the five-year floor of the exploration, K declared.
RATE_HZ, FLOOR_H = 30.0, 43_800.0


@functools.lru_cache(maxsize=1)
def residual_program() -> Any:
    return build_integer_program(residual_block_net(), (3, 16, 16), num_classes=10, max_axons=AXONS,
                                 max_neurons=NEURONS)


def perceptron_names(program: Any) -> Dict[int, str]:
    return {i: p.name for i, p in enumerate(program.model.get_perceptrons())}


def platform(count: int, *, k: int = 1, granularity: str = "cell") -> Dict[str, Any]:
    return {"cores": [{"max_axons": AXONS, "max_neurons": NEURONS, "count": count, "write_granularity": granularity}],
            "weight_bits": 8, "core_semantics": "mvm", "activation_bits": 8, "allow_scheduling": True,
            "max_schedule_passes": 256, "allow_coalescing": False, "allow_neuron_splitting": True,
            "inference_batch_K": k, "inference_rate_hz": RATE_HZ, "min_lifetime_hours": FLOOR_H}


def built(count: int, **kwargs: Any) -> Any:
    program = residual_program()
    return build_hybrid_mapping_for_pipeline(program.ir, platform(count, **kwargs))


def inputs(batch: int = 16) -> Any:
    return integer_inputs(residual_program(), batch)


def neurram() -> Any:
    return get_platform_physics("neurram")
