"""Liveness analysis for unified IR graphs.

A NeuralCore is one of:

- :attr:`NodeLiveness.LIVE` - has at least one live axon weight, and at least
  one neuron is reachable from a model output.
- :attr:`NodeLiveness.BIAS_ONLY` - every axon is dead (off-source or zero
  weight), but ``hardware_bias`` can still drive at least one neuron above
  threshold within ``simulation_steps`` LIF cycles, and at least one neuron
  has a downstream live consumer.
- :attr:`NodeLiveness.DEAD` - either the core cannot emit any spike, or no
  downstream consumer needs any of its neurons. ``DEAD`` wins over
  ``BIAS_ONLY``: a bias emitter with nothing to drive is dead.

The reachability half of the analysis reuses
:func:`mimarsinan.mapping.pruning.graph.compute_global_pruned_sets`,
which already encodes the "ComputeOps are barriers" rule via persistent
consumers. A neuron is *reachable* iff the global propagation does **not**
prune its column.

Bias-can-fire is a sound under-approximation by default
(``bias_only_emission_check="conservative"``):
``max(|bias|) * simulation_steps >= threshold``. This matches the simplest
LIF integration ``v[t+1] = v[t] + bias`` and ignores leak / reset. An exact
mode is reserved as a future extension behind the same flag.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Mapping

import numpy as np

from mimarsinan.mapping.ir import (
    ComputeOp,
    IRGraph,
    IRSource,
    NeuralCore,
)
from mimarsinan.mapping.pruning.graph import (
    GlobalPruningResult,
    compute_global_pruned_sets,
)


class NodeLiveness(Enum):
    """Three-valued liveness classification for NeuralCore nodes."""

    LIVE = "live"
    BIAS_ONLY = "bias_only"
    DEAD = "dead"


@dataclass(frozen=True)
class LivenessResult:
    """Per-node liveness classifications and human-readable reasons.

    ``per_node`` and ``reasons`` are both keyed by ``NeuralCore.id``.
    ComputeOp ids are deliberately absent: the analysis does not classify
    them (they are barriers, never sources of spikes).
    """

    per_node: Mapping[int, NodeLiveness]
    reasons: Mapping[int, str]


def compute_liveness(
    graph: IRGraph,
    *,
    simulation_steps: int,
    spiking_mode: str = "lif",
    thresholding_mode: str = "<",
    bias_only_emission_check: str = "conservative",
    pruning_result: GlobalPruningResult | None = None,
    zero_threshold: float = 1e-8,
) -> LivenessResult:
    """Classify every NeuralCore in ``graph`` as LIVE / BIAS_ONLY / DEAD.

    Args:
        graph: The IR graph to analyze (mutated by neither this function
            nor the underlying propagation pass).
        simulation_steps: Integration window for LIF-style bias checks;
            quantized TTFS uses this as ``S``.
        spiking_mode: ``lif``, ``rate``, ``ttfs``, or ``ttfs_quantized``; selects
            bias-only activation predicate (see :mod:`liveness_semantics`).
        thresholding_mode: Reserved for future use ("<" / "<=" semantics);
            currently has no effect on the conservative check.
        bias_only_emission_check: ``"conservative"`` (default) uses the
            sum bound above. ``"exact"`` is reserved for a precise LIF
            integration check; not yet implemented.
        pruning_result: Optional pre-computed
            :class:`GlobalPruningResult`. When ``None`` we run
            :func:`compute_global_pruned_sets` ourselves with
            ``zero_threshold``.
        zero_threshold: Threshold passed through to the propagation pass
            when we run it ourselves.

    Returns:
        :class:`LivenessResult` keyed by NeuralCore id.

    Raises:
        ValueError: When ``simulation_steps <= 0``.
    """
    if simulation_steps <= 0:
        raise ValueError(
            f"compute_liveness: simulation_steps must be positive, "
            f"got {simulation_steps!r}"
        )
    if bias_only_emission_check not in ("conservative", "exact"):
        raise ValueError(
            f"compute_liveness: bias_only_emission_check must be "
            f"'conservative' or 'exact', got {bias_only_emission_check!r}"
        )

    neural_cores = [n for n in graph.nodes if isinstance(n, NeuralCore)]
    if not neural_cores:
        return LivenessResult(per_node={}, reasons={})

    if pruning_result is None:
        pruning_result = compute_global_pruned_sets(
            graph, zero_threshold=zero_threshold
        )

    output_protected_ids = _node_ids_referenced_by_outputs(graph)

    per_node: Dict[int, NodeLiveness] = {}
    reasons: Dict[int, str] = {}

    for node in neural_cores:
        nid = node.id
        try:
            mat = node.get_core_matrix(graph)
        except (ValueError, KeyError):
            mat = None

        if mat is None:
            per_node[nid] = NodeLiveness.DEAD
            reasons[nid] = "core matrix could not be resolved"
            continue

        n_axons, n_neurons = mat.shape
        pruned_rows = pruning_result.pruned_rows_per_node.get(nid, set())
        pruned_cols = pruning_result.pruned_cols_per_node.get(nid, set())

        is_output_protected = nid in output_protected_ids
        any_neuron_reachable = (
            is_output_protected or len(pruned_cols) < n_neurons
        )
        if not any_neuron_reachable:
            per_node[nid] = NodeLiveness.DEAD
            reasons[nid] = "no downstream consumer reads any neuron"
            continue

        all_axons_dead = len(pruned_rows) == n_axons
        live_axon_has_weight = _has_live_axon_with_weight(
            node=node,
            mat=np.asarray(mat),
            pruned_rows=pruned_rows,
            zero_threshold=zero_threshold,
        )

        if live_axon_has_weight:
            per_node[nid] = NodeLiveness.LIVE
            reasons[nid] = "live"
            continue

        bias_can_fire = _bias_can_activate_for_mode(
            spiking_mode=spiking_mode,
            node=node,
            n_neurons=n_neurons,
            simulation_steps=simulation_steps,
            zero_threshold=zero_threshold,
            mode=bias_only_emission_check,
        )

        if not bias_can_fire:
            if is_output_protected:
                # Output-referenced core that cannot emit is still kept
                # alive: removing it would break the output contract
                # enforced by ``_validate_outputs_remain``. Surface this
                # as BIAS_ONLY with a clear reason so the UI can warn.
                per_node[nid] = NodeLiveness.BIAS_ONLY
                reasons[nid] = (
                    "output-referenced core has no live axon weight and "
                    "no firing bias; preserved for output contract "
                    "(produces zero output)"
                )
                continue
            per_node[nid] = NodeLiveness.DEAD
            if all_axons_dead:
                reasons[nid] = (
                    f"all axons dead and hardware_bias cannot activate "
                    f"({spiking_mode!r}, T={simulation_steps})"
                )
            else:
                reasons[nid] = (
                    f"no live axon weight and hardware_bias cannot activate "
                    f"({spiking_mode!r}, T={simulation_steps})"
                )
            continue

        per_node[nid] = NodeLiveness.BIAS_ONLY
        n_alive_cols = n_neurons - len(pruned_cols)
        reasons[nid] = (
            f"all axon weights dead; {n_alive_cols} of {n_neurons} "
            f"neurons active from hardware_bias only ({spiking_mode!r})"
        )

    # Sanity: ComputeOps are not in the result. Reserved for symmetry.
    _ = ComputeOp
    _ = IRSource
    return LivenessResult(per_node=per_node, reasons=reasons)


def _node_ids_referenced_by_outputs(graph: IRGraph) -> "frozenset[int]":
    """Return the set of node ids referenced by ``graph.output_sources``."""
    if graph.output_sources is None or not graph.output_sources.size:
        return frozenset()
    out: set[int] = set()
    for src in graph.output_sources.flatten():
        if isinstance(src, IRSource) and src.node_id >= 0:
            out.add(int(src.node_id))
    return frozenset(out)


def _has_live_axon_with_weight(
    *,
    node: NeuralCore,
    mat: np.ndarray,
    pruned_rows,
    zero_threshold: float,
) -> bool:
    """True iff some surviving axon row has a non-negligible weight.

    "Surviving" means: not in ``pruned_rows`` AND its ``IRSource`` is not
    ``is_off()``. We treat input-data and always-on sources as live (they
    do supply a non-zero signal at runtime).
    """
    if mat.size == 0:
        return False
    abs_max_per_row = np.abs(mat).max(axis=1)
    flat_sources = node.input_sources.flatten()
    n_axons = mat.shape[0]
    for i in range(n_axons):
        if i in pruned_rows:
            continue
        if i < len(flat_sources):
            src = flat_sources[i]
            if isinstance(src, IRSource) and src.is_off():
                continue
        if i < abs_max_per_row.size and abs_max_per_row[i] >= zero_threshold:
            return True
    return False


def _bias_can_activate_for_mode(
    *,
    spiking_mode: str,
    node: NeuralCore,
    n_neurons: int,
    simulation_steps: int,
    zero_threshold: float,
    mode: str,
) -> bool:
    """Mode-aware bias-only activation check via :mod:`liveness_semantics`."""
    from mimarsinan.mapping.pruning.liveness_semantics import bias_can_activate

    bias = getattr(node, "hardware_bias", None)
    if bias is None:
        return False
    arr = np.asarray(bias)
    if arr.size == 0 or arr.size != n_neurons:
        return False
    threshold = float(getattr(node, "threshold", 1.0))
    return bias_can_activate(
        spiking_mode=spiking_mode,
        bias=bias,
        threshold=threshold,
        simulation_steps=simulation_steps,
        zero_threshold=zero_threshold,
        bias_only_emission_check=mode,
    )
