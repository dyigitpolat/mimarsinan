from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from mimarsinan.mapping.platform.coalescing import coalescing_fragment_count
from mimarsinan.mapping.platform.mapping_structure import (
    compute_core_input_count,
    compute_fc_tiling_mode,
    compute_psum_params,
)
import operator

from mimarsinan.mapping.support.compute_modules import ComputeAdapter
from mimarsinan.mapping.mapping_utils import (
    ComputeOpMapper,
    Conv1DPerceptronMapper,
    Conv2DPerceptronMapper,
    Mapper,
    ModelRepresentation,
    PerceptronMapper,
    StackMapper,
)


@dataclass
class HWEstimate:
    mappable: bool
    reason: str | None
    cores_total: int
    # Human-readable summary of core shapes, tiling, etc.
    details: str


def _ceil_div(a: int, b: int) -> int:
    return int((int(a) + int(b) - 1) // int(b))


def estimate_fc_cores(
    *,
    in_features: int,
    out_features: int,
    instances: int = 1,
    has_bias: bool,
    max_axons: int,
    max_neurons: int,
    hardware_bias: bool = False,
    allow_coalescing: bool = True,
) -> int:
    """Return estimated hardware core count for one FC layer (matches layout tiling modes)."""
    est = _estimate_map_fc(
        in_features=in_features,
        out_features=out_features,
        instances=instances,
        has_bias=has_bias,
        max_axons=max_axons,
        max_neurons=max_neurons,
        hardware_bias=hardware_bias,
        allow_coalescing=allow_coalescing,
    )
    return est.cores_total if est.mappable else 0


def _estimate_map_fc(
    *,
    in_features: int,
    out_features: int,
    instances: int,
    has_bias: bool,
    max_axons: int,
    max_neurons: int,
    hardware_bias: bool = False,
    allow_coalescing: bool = True,
) -> HWEstimate:
    """
    Estimate hardware cores for an FC layer using ``mapping_structure`` helpers
    (same tiling mode / psum / coalescing semantics as ``LayoutIRMapping``).
    """
    required_axons = compute_core_input_count(
        int(in_features), has_bias=has_bias, hardware_bias=hardware_bias
    )
    bias_ax = required_axons - int(in_features)

    mode = compute_fc_tiling_mode(
        int(in_features),
        int(out_features),
        int(max_axons),
        int(max_neurons),
        has_bias,
        hardware_bias,
        allow_coalescing,
    )

    if mode == "single":
        groups = _ceil_div(out_features, max_neurons)
        cores_total = int(instances) * int(groups)
        details = (
            f"mode=single\n"
            f"required_axons={required_axons} (features={in_features}, bias={bias_ax}) <= max_axons={max_axons}\n"
            f"output_groups=ceil({out_features}/{max_neurons})={groups}\n"
            f"cores_total={cores_total}"
        )
        return HWEstimate(True, None, cores_total, details)

    if mode == "coalescing":
        k = coalescing_fragment_count(required_axons, int(max_axons))
        out_groups = _ceil_div(out_features, max_neurons)
        cores_total = int(instances) * k * out_groups
        details = (
            f"mode=coalescing\n"
            f"required_axons={required_axons} > max_axons={max_axons}\n"
            f"fragments={k}\n"
            f"output_groups=ceil({out_features}/{max_neurons})={out_groups}\n"
            f"cores_total={cores_total}"
        )
        return HWEstimate(True, None, cores_total, details)

    if mode == "output_tiled":
        out_groups = _ceil_div(out_features, max_neurons)
        cores_total = int(instances) * out_groups
        details = (
            f"mode=output_tiled\n"
            f"output_groups=ceil({out_features}/{max_neurons})={out_groups}\n"
            f"cores_total={cores_total}"
        )
        return HWEstimate(True, None, cores_total, details)

    # psum
    try:
        pp = compute_psum_params(
            int(in_features),
            int(out_features),
            int(max_axons),
            int(max_neurons),
            has_bias,
            hardware_bias,
        )
    except ValueError as exc:
        return HWEstimate(False, str(exc), 0, f"UNMAPPABLE: {exc}")

    out_blocks = _ceil_div(out_features, pp.out_block_size)
    cores_per_instance_per_block = 2 * pp.tile_count + 1
    cores_total = int(instances) * int(out_blocks) * int(cores_per_instance_per_block)

    details = (
        f"mode=psum\n"
        f"required_axons={required_axons} (features={in_features}, bias={bias_ax}) > max_axons={max_axons}\n"
        f"tile_count={pp.tile_count}\n"
        f"out_block_size={pp.out_block_size}\n"
        f"out_blocks={out_blocks}\n"
        f"cores_per_instance_per_block=2*tile_count+1={cores_per_instance_per_block}\n"
        f"cores_total={cores_total}"
    )
    return HWEstimate(True, None, cores_total, details)

