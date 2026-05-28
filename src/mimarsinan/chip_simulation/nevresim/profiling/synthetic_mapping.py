"""Synthetic HardCoreMapping factory for nevresim compile profiling."""

from __future__ import annotations

from typing import Literal

import numpy as np

from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.mapping.packing.softcore.hard_core import HardCore
from mimarsinan.mapping.packing.softcore.hard_core_mapping import HardCoreMapping

TopologyPreset = Literal[
    "contiguous_input",
    "contiguous_crosscore",
    "fragmented_crosscore",
    "patch_embed_like",
    "mixed_vit_like",
]

PRESET_ALIASES: dict[str, TopologyPreset] = {
    "contiguous_input": "contiguous_input",
    "contiguous_crosscore": "contiguous_crosscore",
    "fragmented_crosscore": "fragmented_crosscore",
    "patch_embed_like": "patch_embed_like",
    "mixed_vit_like": "mixed_vit_like",
}


def _input_src(idx: int) -> SpikeSource:
    return SpikeSource(-2, idx, is_input=True, is_off=False)


def _core_src(core: int, neuron: int) -> SpikeSource:
    return SpikeSource(core, neuron, is_input=False, is_off=False)


def _off_src() -> SpikeSource:
    return SpikeSource(-1, 0, is_input=False, is_off=True)


def _weight_matrix(axons_per_core: int, neurons_per_core: int) -> np.ndarray:
    rng = np.random.RandomState(0)
    mat = rng.randn(axons_per_core, neurons_per_core).astype(np.float64) * 0.05
    limit = min(axons_per_core, neurons_per_core)
    mat[:limit, :limit] = np.eye(limit, dtype=np.float64)
    return mat


def _make_core(
    axons_per_core: int,
    neurons_per_core: int,
    sources: list[SpikeSource],
    *,
    threshold: float = 1.0,
    latency: int = 0,
) -> HardCore:
    core = HardCore(axons_per_core, neurons_per_core, has_bias_capability=False)
    padded = list(sources)
    while len(padded) < axons_per_core:
        padded.append(_off_src())
    core.axon_sources = padded[:axons_per_core]
    core.core_matrix = _weight_matrix(axons_per_core, neurons_per_core)
    core.threshold = threshold
    core.latency = latency
    core.available_axons = 0
    core.available_neurons = 0
    return core


def _sources_contiguous_input(core_id: int, axons_per_core: int) -> list[SpikeSource]:
    base = core_id * axons_per_core
    return [_input_src(base + i) for i in range(axons_per_core)]


def _sources_contiguous_crosscore(
    core_id: int,
    axons_per_core: int,
    neurons_per_core: int,
) -> list[SpikeSource]:
    if core_id == 0:
        return [_input_src(i) for i in range(axons_per_core)]
    prev = core_id - 1
    return [_core_src(prev, i % neurons_per_core) for i in range(axons_per_core)]


def _sources_fragmented_crosscore(
    core_id: int,
    axons_per_core: int,
    neurons_per_core: int,
    *,
    input_size: int,
) -> list[SpikeSource]:
    sources: list[SpikeSource] = []
    for axon_idx in range(axons_per_core):
        if core_id == 0:
            sources.append(_input_src((axon_idx * 5 + 3) % max(1, input_size)))
        else:
            src_core = (core_id + axon_idx * 3 + 1) % core_id
            src_neuron = (axon_idx * 7 + core_id) % max(1, neurons_per_core)
            sources.append(_core_src(src_core, src_neuron))
    return sources


def _finalize_mapping(
    cores: list[HardCore],
    *,
    output_count: int | None = None,
) -> HardCoreMapping:
    hcm = HardCoreMapping(chip_cores=[])
    hcm.cores = cores
    last = len(cores) - 1
    n_out = output_count if output_count is not None else min(cores[-1].neurons_per_core, 8)
    n_out = max(1, min(n_out, cores[-1].neurons_per_core))
    hcm.output_sources = np.array(
        [_core_src(last, i) for i in range(n_out)],
        dtype=object,
    )
    return hcm


def compute_input_size(
    preset: TopologyPreset,
    core_count: int,
    axons_per_core: int,
) -> int:
    if preset in ("contiguous_input", "patch_embed_like", "mixed_vit_like"):
        return core_count * axons_per_core
    return axons_per_core


def build_synthetic_mapping(
    preset: TopologyPreset,
    core_count: int,
    axons_per_core: int,
    neurons_per_core: int,
) -> tuple[HardCoreMapping, int]:
    """Build a synthetic HardCoreMapping and its input buffer size."""
    if core_count < 1:
        raise ValueError("core_count must be >= 1")

    input_size = compute_input_size(preset, core_count, axons_per_core)
    cores: list[HardCore] = []

    if preset == "contiguous_input":
        for core_id in range(core_count):
            sources = _sources_contiguous_input(core_id, axons_per_core)
            cores.append(_make_core(axons_per_core, neurons_per_core, sources))

    elif preset == "contiguous_crosscore":
        for core_id in range(core_count):
            sources = _sources_contiguous_crosscore(core_id, axons_per_core, neurons_per_core)
            cores.append(_make_core(axons_per_core, neurons_per_core, sources))

    elif preset == "fragmented_crosscore":
        for core_id in range(core_count):
            sources = _sources_fragmented_crosscore(
                core_id, axons_per_core, neurons_per_core, input_size=input_size,
            )
            cores.append(_make_core(axons_per_core, neurons_per_core, sources))

    elif preset == "patch_embed_like":
        for core_id in range(core_count):
            sources = _sources_contiguous_input(core_id, axons_per_core)
            cores.append(_make_core(axons_per_core, neurons_per_core, sources))

    elif preset == "mixed_vit_like":
        embed_count = max(1, core_count - 1)
        for core_id in range(embed_count):
            sources = _sources_contiguous_input(core_id, axons_per_core)
            cores.append(_make_core(axons_per_core, neurons_per_core, sources))
        if core_count > 1:
            mlp_core_id = core_count - 1
            sources = _sources_fragmented_crosscore(
                mlp_core_id, axons_per_core, neurons_per_core, input_size=input_size,
            )
            for axon_idx in range(axons_per_core):
                if axon_idx % 4 == 0:
                    src_core = axon_idx % embed_count
                    sources[axon_idx] = _core_src(src_core, axon_idx % neurons_per_core)
            cores.append(_make_core(axons_per_core, neurons_per_core, sources))
        else:
            sources = _sources_fragmented_crosscore(0, axons_per_core, neurons_per_core, input_size=input_size)
            cores.append(_make_core(axons_per_core, neurons_per_core, sources))

    else:
        raise ValueError(f"Unknown preset: {preset!r}")

    return _finalize_mapping(cores), input_size


def build_multi_segment_fanout(
    segment_count: int,
    cores_per_segment: int,
    axons_per_core: int,
    neurons_per_core: int,
    *,
    preset: TopologyPreset = "patch_embed_like",
) -> list[tuple[HardCoreMapping, int]]:
    """Build N independent segments for E5 multi-segment fan-out experiments."""
    segments: list[tuple[HardCoreMapping, int]] = []
    for _ in range(segment_count):
        segments.append(
            build_synthetic_mapping(preset, cores_per_segment, axons_per_core, neurons_per_core)
        )
    return segments
