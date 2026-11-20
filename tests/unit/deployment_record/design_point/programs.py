"""Small law-carrying value programs for the instance tests: a CIFAR-style residual net (two convolutions, a
projection shortcut, a host add, ReLUs and a pool on the host) built the pipeline's way, packed onto few cores so it
runs in several passes. Every constant here is a test constant."""

from __future__ import annotations

import functools
from typing import Any, Dict

from mimarsinan.deployment_record.platform_physics.registry import get_platform_physics
from mimarsinan.pipelining.core.simulation_factory import build_hybrid_mapping_for_pipeline
from unit.chip_simulation.value_run.integer_programs import build_integer_program, integer_inputs, residual_block_net
from mimarsinan.deployment_record.platform_physics.hosts.description import host_description_block, host_descriptions

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


#: [E-6, targeted pass] The hybrid declaration of Figure 1's mixed row (NeuRRAM static tiles beside SRAM-CIM operand
#: cores, ``tests/fixtures/design_point/hybrid_neurram_sram_cim.json``) at test counts, with a small softmax attention
#: block (``operand_programs.attention_block``, depth 1) built the pipeline's way onto its operand core type.
HYBRID = "tests/fixtures/design_point/hybrid_neurram_sram_cim.json"
#: A host block: the host's MAC rate, dispatch time and power. [U-HOST2] The first read host description's own rows,
#: as its block declares them (the admission rule admits a host row only as its description's row or a declared
#: derivation of its rows, never another value under its name).
HOST_BLOCK = {key: row for key, row in host_description_block(host_descriptions()[0]).items()
              if key in ("host_macs_per_s", "t_host_op_overhead", "p_host")}


def hybrid_declaration(static: int, operand: int, *, scheduling: bool, k: int = 1,
                       overrides: Any = None, static_write_physics: Any = None) -> Dict[str, Any]:
    """The fixture's platform constraints at ``static`` + ``operand`` cores, the run keys of the instance. [U-C C-5]
    The 8-bit value grid departs from the target's input width, so it is a declared departure (``target_overrides``
    with its note), as the instance script declares it. [U-C3R] ``static_write_physics`` names another write physics
    for the static core type (its cells are written under that profile's write rows; every other constant stays the
    fixture's), so a test can judge the same row under a technology with other write constants."""
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[4]
    declared = json.loads((root / HYBRID).read_text())["platform_constraints"]
    cores = [dict(c) for c in declared["target_overrides"]["cores"]["value"]]
    cores[0]["count"], cores[1]["count"] = int(static), int(operand)
    if static_write_physics is not None:
        cores[0]["write_physics"] = str(static_write_physics)
    width = {"value": 8, "note": "the vehicle's 8-bit value grid (the integer readout law's), a declared departure"}
    # [U-CHIPKEYS] the fixture's other declared departures (the lanes' off ablation beside the operand core type) stay
    return {**declared, "target_overrides": {**declared["target_overrides"],
                                             "cores": {**declared["target_overrides"]["cores"], "value": cores},
                                             "activation_bits": width},
            "inference_batch_K": int(k), "inference_rate_hz": RATE_HZ,
            "min_lifetime_hours": FLOOR_H,
            **({"allow_scheduling": True, "max_schedule_passes": 4096} if scheduling else {}),
            **({"platform_physics_overrides": overrides} if overrides else {})}


def resolved_hybrid(static: int, operand: int, *, scheduling: bool, k: int = 1, overrides: Any = None,
                    static_write_physics: Any = None) -> Dict[str, Any]:
    from mimarsinan.config_schema.runtime import run_config
    from mimarsinan.pipelining.core.platform_constraints_resolver import build_platform_constraints_resolved
    return build_platform_constraints_resolved(run_config({}, hybrid_declaration(
        static, operand, scheduling=scheduling, k=k, overrides=overrides, static_write_physics=static_write_physics)))


@functools.lru_cache(maxsize=4)
def attention_on_hybrid(static: int, operand: int, scheduling: bool, static_write_physics: Any = None) -> Any:
    """(program, hybrid mapping, resolved platform) of the small softmax block on the hybrid declaration, the IR's
    latencies tagged as Soft Core Mapping tags them (``IRLatency``) before the build ([U-C3R] ``static_write_physics``
    as ``hybrid_declaration`` takes it)."""
    import contextlib
    import io
    from mimarsinan.mapping.latency.ir import IRLatency
    from unit.chip_simulation.value_run.operand_programs import LAW, attention_block, build_operand_program
    platform = resolved_hybrid(static, operand, scheduling=scheduling, static_write_physics=static_write_physics)
    with contextlib.redirect_stdout(io.StringIO()):
        program = build_operand_program(attention_block(depth=1), (3, 8, 8), num_classes=5,
                                        platform={**platform, "readout_law": dict(LAW),
                                                  "readout_ranges": "per_consumer_group"})
        IRLatency(program.ir).calculate()
        hybrid = build_hybrid_mapping_for_pipeline(program.ir, {**platform, "weight_bits": 8})
    return program, hybrid, platform
