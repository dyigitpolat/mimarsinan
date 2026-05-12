"""Build a SANA-FE Network from a single ``HardCoreMapping`` neural segment.

Per segment we create:

* one neuron group per ``HardCore`` (sized to the core's used neuron count),
* one ``input`` neuron group of size ``seg_in_size`` carrying segment-input
  spike trains injected by the runner,
* one ``always_on`` neuron group of size 1, **only** when the segment
  references the always-on bias source.

Synapses are emitted by walking each core's ``axon_sources`` once.  An
axon at position ``a`` of core ``c`` is connected to *every* destination
neuron in the same core, with weights ``core.core_matrix[a, n]`` — zero
entries are skipped to avoid wasteful synapses.  Duplicate source-spec
axons are aggregated: two axons that read the same ``SpikeSource`` and
target the same destination collapse into one synapse whose weight is
their sum (matches the documented Lava+SANA-FE behaviour and is verified
by ``test_loihi_duplicate_source_axons_accumulate_weights``).

Returns
-------
network
    The freshly built ``sanafe.Network`` (mockable via ``_sanafe()``).
core_to_group
    Dict ``{HardCore index → sanafe NeuronGroup}`` so the runner can
    look up per-core neurons for spike-count extraction.
input_group
    The ``input`` neuron group, or ``None`` when ``seg_in_size == 0``.
always_on_group
    The ``always_on`` group, or ``None`` when no axon references it.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from .arch_synth import _sanafe


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _used_axons(core: Any) -> int:
    """Number of axons that carry signal.

    ``len(core.axon_sources)`` is the canonical source of truth: each
    softcore added by ``HardCore.add_softcore`` extends the list, so the
    list length always equals ``axons_per_core - available_axons`` for a
    real HardCore.  The fallback for old fakes that only set
    ``available_axons`` is the subtraction path.
    """
    if hasattr(core, "axon_sources"):
        return len(core.axon_sources)
    return int(core.axons_per_core) - int(getattr(core, "available_axons", 0))


def _used_neurons(core: Any) -> int:
    """Number of neurons used by this core (excludes trailing free slots)."""
    return int(core.neurons_per_core) - int(getattr(core, "available_neurons", 0))


def _pack_tile_index(core_global_idx: int, cores_per_tile: int) -> Tuple[int, int]:
    """Map a flat core index to (tile_index, core_in_tile_index)."""
    if cores_per_tile <= 0:
        return 0, core_global_idx
    return core_global_idx // cores_per_tile, core_global_idx % cores_per_tile


def _segment_needs_always_on(hcm: Any) -> bool:
    for core in hcm.cores:
        for s in core.axon_sources:
            if getattr(s, "is_always_on_", False):
                return True
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_network_for_segment(
    arch: Any,
    hcm: Any,
    *,
    tile_offset: int,
    core_offset: int,
    seg_in_size: int,
    cores_per_tile: int = 0,
) -> Tuple[Any, Dict[int, Any], Optional[Any], Optional[Any]]:
    """Construct a ``sanafe.Network`` for a single neural segment.

    Parameters
    ----------
    arch
        A ``sanafe.Architecture`` (or test fake) with ``.tiles[t].cores[c]``.
    hcm
        A ``HardCoreMapping``-shaped object with ``.cores``.
    tile_offset
        First tile this segment may place cores into.  Lets multi-segment
        runs share one ``Architecture``.
    core_offset
        First core within ``tile_offset`` to occupy.
    seg_in_size
        Width of the segment input (axons of kind ``input``).
    cores_per_tile
        Tile packing density; ``0`` means "all in one tile".  Must match
        the ``ArchSpec`` used to build ``arch``.
    """
    sanafe = _sanafe()
    net = sanafe.Network()

    # 1. Input + always-on groups (created up front; always-on only when needed).
    input_group = (
        net.create_neuron_group("input", seg_in_size, model_attributes={})
        if seg_in_size > 0 else None
    )
    always_on_group = None
    if _segment_needs_always_on(hcm):
        always_on_group = net.create_neuron_group(
            "always_on", 1, model_attributes={"force_spike_every_cycle": True},
        )

    # 2. One neuron group per HardCore, mapped to the spec's tile/core slot.
    core_to_group: Dict[int, Any] = {}
    for core_idx, core in enumerate(hcm.cores):
        used_neurons = _used_neurons(core)
        if used_neurons <= 0:
            # Degenerate: an entirely-unused core has no neurons to simulate;
            # skip group creation but keep the index reserved so downstream
            # core_to_group lookups remain consistent.
            continue
        group = net.create_neuron_group(
            f"core{core_idx}", used_neurons,
            model_attributes={"threshold": float(core.threshold)},
        )

        for n_idx in range(used_neurons):
            neuron = group[n_idx]
            attrs: dict = {"threshold": float(core.threshold)}
            if core.hardware_bias is not None:
                attrs["bias"] = float(np.asarray(core.hardware_bias)[n_idx])
            neuron.set_attributes(model_attributes=attrs)

            tile_local = core_offset + core_idx
            tile_idx, core_in_tile = _pack_tile_index(tile_local, cores_per_tile)
            tile_idx += tile_offset
            neuron.map_to_core(arch.tiles[tile_idx].cores[core_in_tile])

        core_to_group[core_idx] = group

    # 3. Wire axons.  For each (src_neuron, dst_neuron) we collapse all
    #    contributing axons into one synapse with the summed weight.
    for core_idx, core in enumerate(hcm.cores):
        if core_idx not in core_to_group:
            continue
        used_axons = _used_axons(core)
        used_neurons = _used_neurons(core)
        if used_neurons <= 0:
            continue
        accum: Dict[Tuple[int, int], float] = {}  # (src_key, dst_idx) -> weight
        sources_by_key: Dict[int, Any] = {}

        # First pass: collapse axon weights per (source neuron object, destination index).
        for a in range(used_axons):
            src = core.axon_sources[a]
            if getattr(src, "is_off_", False):
                continue

            # Resolve source neuron (input group, always-on group, or another core's group).
            if getattr(src, "is_input_", False):
                src_group = input_group
                src_neuron_idx = int(src.neuron_)
            elif getattr(src, "is_always_on_", False):
                src_group = always_on_group
                src_neuron_idx = 0
            else:
                src_core = int(src.core_)
                src_group = core_to_group.get(src_core)
                src_neuron_idx = int(src.neuron_)
                if src_group is None:
                    continue   # source core was degenerate

            src_neuron = src_group[src_neuron_idx]
            src_key = id(src_neuron)
            sources_by_key[src_key] = src_neuron

            for n_idx in range(used_neurons):
                w = float(core.core_matrix[a, n_idx])
                if w == 0.0:
                    continue
                key = (src_key, n_idx)
                accum[key] = accum.get(key, 0.0) + w

        # Second pass: emit one synapse per (source neuron, destination) pair.
        dst_group = core_to_group[core_idx]
        for (src_key, n_idx), w in accum.items():
            sources_by_key[src_key].connect_to_neuron(
                dst_group[n_idx], {"weight": w},
            )

    return net, core_to_group, input_group, always_on_group
