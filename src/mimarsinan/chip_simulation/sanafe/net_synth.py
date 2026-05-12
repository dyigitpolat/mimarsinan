"""Build a SANA-FE Network from a single ``HardCoreMapping`` neural segment.

Hardware-faithful mapping (see ``sanafe_per_core_input_neurons`` memory):

* Each mimarsinan ``HardCore`` maps 1:1 to one SANA-FE core.
* For each axon ``a`` on HardCore ``c``:
    - if ``is_off_``: skip
    - if ``is_input_``: emit an input neuron on the **same** SANA-FE core
      (using that core's local ``inputs[a]`` soma slot).  Multiple
      HardCores reading the same external input each get their own copy
      of the input neuron, fed the same spike train.
    - if ``is_always_on_``: emit an always-on neuron on the same core.
    - otherwise (cross-core source): connect the upstream HardCore's LIF
      neuron directly to this HardCore's LIF neurons via NoC.
* LIF neurons use the mimarsinan ``soma`` plugin (no Loihi compartment
  cap).  Dendrite uses the mimarsinan ``dendrite`` plugin (no
  ``accumulator``-style 1024 cap).  See ``arch_synth._render_arch_yaml``.

Returns
-------
network
    Freshly built ``sanafe.Network``.
core_to_group
    ``{HardCore index → sanafe NeuronGroup (LIF neurons)}``.
core_input_neurons
    ``{(HardCore index, axon index within that core) → sanafe Neuron}``
    map covering every input axon.  The runner uses this to inject the
    correct rate-encoded spike train per (core, axon) pair.
core_always_on_neurons
    ``{HardCore index → sanafe Neuron}`` for cores that reference the
    always-on bias source.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from .arch_synth import _sanafe
from .neuron_model import input_neuron_attributes, lif_model_attributes
from .presets import (
    SOMA_INPUT_RANGE_NAME, SOMA_LIF_NAME, SYNAPSE_NAME,
)


def _used_axons(core: Any) -> int:
    if hasattr(core, "axon_sources"):
        return len(core.axon_sources)
    return int(core.axons_per_core) - int(getattr(core, "available_axons", 0))


def _used_neurons(core: Any) -> int:
    return int(core.neurons_per_core) - int(getattr(core, "available_neurons", 0))


def _pack_tile_index(core_global_idx: int, cores_per_tile: int) -> Tuple[int, int]:
    if cores_per_tile <= 0:
        return 0, core_global_idx
    return core_global_idx // cores_per_tile, core_global_idx % cores_per_tile


def build_network_for_segment(
    arch: Any,
    hcm: Any,
    *,
    tile_offset: int,
    core_offset: int,
    cores_per_tile: int = 0,
) -> Tuple[
    Any,
    Dict[int, Any],
    Dict[Tuple[int, int], Any],
    Dict[int, Any],
]:
    """Construct a ``sanafe.Network`` for one neural segment.

    Returns ``(network, core_to_group, core_input_neurons,
    core_always_on_neurons)``.
    """
    sanafe = _sanafe()
    net = sanafe.Network()

    # 1. One LIF group per HardCore, mapped to its corresponding SANA-FE core.
    core_to_group: Dict[int, Any] = {}
    core_input_neurons: Dict[Tuple[int, int], Any] = {}
    core_always_on_neurons: Dict[int, Any] = {}

    def _resolve_core(hardcore_idx: int) -> Any:
        tile_local = core_offset + hardcore_idx
        tile_idx, core_in_tile = _pack_tile_index(tile_local, cores_per_tile)
        tile_idx += tile_offset
        return arch.tiles[tile_idx].cores[core_in_tile]

    for core_idx, core in enumerate(hcm.cores):
        sanafe_core = _resolve_core(core_idx)
        used_neurons = _used_neurons(core)
        used_axons = _used_axons(core)

        # ----- LIF neurons (one group per HardCore) -----
        if used_neurons > 0:
            group = net.create_neuron_group(
                f"core{core_idx}", used_neurons, model_attributes={},
            )
            for n_idx in range(used_neurons):
                neuron = group[n_idx]
                bias = None
                if core.hardware_bias is not None:
                    bias = float(np.asarray(core.hardware_bias)[n_idx])
                neuron.set_attributes(
                    soma_hw_name=SOMA_LIF_NAME,
                    default_synapse_hw_name=SYNAPSE_NAME,
                    log_spikes=True,
                    model_attributes=lif_model_attributes(
                        threshold=float(core.threshold),
                        hardware_bias=bias,
                    ),
                )
                neuron.map_to_core(sanafe_core)
            core_to_group[core_idx] = group

        # ----- Per-axon input / always-on neurons (on the same core) -----
        # We only materialise neurons for axons that are actually wired to
        # an input or always-on source.  Cross-core axons are wired in
        # step 2 below directly from the upstream LIF neuron.
        input_axon_indices = []
        always_on_axon_indices = []
        for a in range(used_axons):
            src = core.axon_sources[a]
            if getattr(src, "is_off_", False):
                continue
            if getattr(src, "is_input_", False):
                input_axon_indices.append(a)
            elif getattr(src, "is_always_on_", False):
                always_on_axon_indices.append(a)

        if input_axon_indices:
            in_group = net.create_neuron_group(
                f"core{core_idx}_in", len(input_axon_indices),
                model_attributes={},
            )
            for offset, a in enumerate(input_axon_indices):
                n = in_group[offset]
                n.set_attributes(
                    soma_hw_name=f"{SOMA_INPUT_RANGE_NAME}[{a}]",
                    default_synapse_hw_name=SYNAPSE_NAME,
                    model_attributes=input_neuron_attributes(),
                )
                n.map_to_core(sanafe_core)
                core_input_neurons[(core_idx, a)] = n

        if always_on_axon_indices:
            # One always-on source per HardCore is enough; the multiple
            # axons that read from "always-on" can all fan out from a single
            # neuron via separate synapses (collapsed below).  We pick the
            # first always-on axon's index as the soma slot.
            ao_slot = always_on_axon_indices[0]
            ao_group = net.create_neuron_group(
                f"core{core_idx}_on", 1, model_attributes={},
            )
            ao_group[0].set_attributes(
                soma_hw_name=f"{SOMA_INPUT_RANGE_NAME}[{ao_slot}]",
                default_synapse_hw_name=SYNAPSE_NAME,
                model_attributes=input_neuron_attributes(),
            )
            ao_group[0].map_to_core(sanafe_core)
            core_always_on_neurons[core_idx] = ao_group[0]

    # 2. Wire axons.  Per-axon source resolution; collapse duplicate
    # (src_neuron, dst_neuron) pairs into one summed-weight synapse.
    for core_idx, core in enumerate(hcm.cores):
        dst_group = core_to_group.get(core_idx)
        if dst_group is None:
            continue
        used_axons = _used_axons(core)
        used_neurons = _used_neurons(core)
        if used_neurons <= 0:
            continue

        accum: Dict[Tuple[int, int], float] = {}  # (src_key, dst_idx) → weight
        sources_by_key: Dict[int, Any] = {}

        for a in range(used_axons):
            src = core.axon_sources[a]
            if getattr(src, "is_off_", False):
                continue
            src_neuron: Optional[Any] = None
            if getattr(src, "is_input_", False):
                src_neuron = core_input_neurons.get((core_idx, a))
            elif getattr(src, "is_always_on_", False):
                src_neuron = core_always_on_neurons.get(core_idx)
            else:
                src_core_idx = int(src.core_)
                src_group = core_to_group.get(src_core_idx)
                if src_group is None:
                    continue
                src_neuron = src_group[int(src.neuron_)]

            if src_neuron is None:
                continue
            src_key = id(src_neuron)
            sources_by_key[src_key] = src_neuron

            for n_idx in range(used_neurons):
                w = float(core.core_matrix[a, n_idx])
                if w == 0.0:
                    continue
                key = (src_key, n_idx)
                accum[key] = accum.get(key, 0.0) + w

        for (src_key, n_idx), w in accum.items():
            sources_by_key[src_key].connect_to_neuron(
                dst_group[n_idx],
                {"weight": w, "synapse_hw_name": SYNAPSE_NAME},
            )

    return net, core_to_group, core_input_neurons, core_always_on_neurons


# ---------------------------------------------------------------------------
# Per-sample spike-train injection used by the runner
# ---------------------------------------------------------------------------


def set_input_spike_trains(
    core_input_neurons: Dict[Tuple[int, int], Any],
    hcm: Any,
    encoded: np.ndarray,
) -> None:
    """Inject spike trains into every per-core input neuron.

    For each ``(core_idx, axon_idx) → neuron`` entry in
    ``core_input_neurons``, looks up the axon's logical input index
    ``k = hcm.cores[core_idx].axon_sources[axon_idx].neuron_`` and feeds
    the row ``encoded[0, k, :]`` as the neuron's ``spikes`` attribute.

    ``encoded`` is the ``(1, seg_in_size, T)`` binary tensor produced by
    ``uniform_rate_encode``.  Multiple per-core input neurons sharing the
    same logical index ``k`` get the same spike train.
    """
    for (core_idx, axon_idx), neuron in core_input_neurons.items():
        src = hcm.cores[core_idx].axon_sources[axon_idx]
        k = int(src.neuron_)
        if k >= encoded.shape[1]:
            train: list[int] = []
        else:
            train = [int(b) for b in encoded[0, k, :].tolist()]
        neuron.set_attributes(
            model_attributes=input_neuron_attributes(train),
        )


def set_always_on_spike_trains(
    core_always_on_neurons: Dict[int, Any],
    T: int,
) -> None:
    """Always-on neurons fire every cycle: positional bit array of all 1s."""
    train = [1] * T
    for neuron in core_always_on_neurons.values():
        neuron.set_attributes(
            model_attributes=input_neuron_attributes(train),
        )
