"""[U-PCH] Per-neuron threshold registers: the declared capability, the row's per-neuron weight scales and every
consumer's one reading of a core's threshold.

A chip description declares a threshold register per neuron with ``core_value_granularity: {"threshold":
"per_neuron"}``; absent, a spiking core holds one threshold register (``core_residency.default_residency_policy``).
A row declares ``weight_scale_granularity: per_neuron`` to project each output neuron on its own scale ``s_c`` (Weight
Quantization's ``max(1, floor(q_max / max_j |w_cj|))``) with ``theta_c = s_c``; absent, one scale per perceptron, byte
for byte. Its core thresholds are then one integer per neuron column, carried from the IR through the packed program
to every executor. A consumer that holds one register per core reads a threshold through ``core_threshold``, which
refuses a per-neuron one by name. [U-PCH3] Every reader of a per-core or per-node value reads it by its holder's
declared layout (``mapping.support.value_layouts``), never by its size; a hard core assembles over a soft core's
neuron range every singleton it declares per neuron (``place_neuron_value``)."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

import numpy as np
import torch

from mimarsinan.mapping.platform.core_residency import Granularity, resolve_residency_policy
from mimarsinan.mapping.support.grid_declarations import (
    WEIGHT_SCALE_GRANULARITIES,
    WEIGHT_SCALE_GRANULARITY_KEY as KEY,
    WEIGHT_SCALE_PER_NEURON as PER_NEURON,
    WEIGHT_SCALE_PER_PERCEPTRON as PER_PERCEPTRON,
)
from mimarsinan.mapping.support.value_layouts import ValueLayoutError, neuron_entries
from mimarsinan.models.nn.lif_kernels import PerNeuronThresholdRefusal

__all__ = ["KEY", "PER_NEURON", "PER_PERCEPTRON", "PerNeuronThresholdRefusal", "WEIGHT_SCALE_GRANULARITIES"]


def per_neuron_weight_scales(config: Mapping[str, Any], *, value_domain: bool) -> bool:
    """Whether the row projects each output neuron on its own scale; refused by name on a chip whose cores hold one
    threshold register (its ``core_value_granularity``) and on a value-domain core (no threshold register)."""
    declared = config.get(KEY) or PER_PERCEPTRON
    if declared not in WEIGHT_SCALE_GRANULARITIES:
        raise PerNeuronThresholdRefusal(f"{KEY} {declared!r} is not one of {list(WEIGHT_SCALE_GRANULARITIES)}")
    if declared == PER_PERCEPTRON:
        return False
    if value_domain:
        raise PerNeuronThresholdRefusal(
            f"{KEY} 'per_neuron' sets every output neuron's threshold register (theta_c = s_c), but a value-domain "
            f"core has no threshold register: declare {KEY} 'per_perceptron'")
    grain = resolve_residency_policy(config, value_domain=False)["threshold"]
    if grain is not Granularity.PER_NEURON:
        raise PerNeuronThresholdRefusal(
            f"{KEY} 'per_neuron' gives every output neuron its own threshold (theta_c = s_c), but the chip "
            f"description declares one threshold register per core (core_value_granularity threshold "
            f"{grain.value!r}): declare a chip whose cores hold a threshold register per neuron "
            f"(core_value_granularity {{'threshold': 'per_neuron'}}), or {KEY} 'per_perceptron'")
    return True


def is_per_neuron(value: Any) -> bool:
    """Whether ``value`` (a scale or a threshold) holds one entry per neuron rather than one per core."""
    return value is not None and np.ndim(value) >= 1


def _array(value: Any) -> np.ndarray:
    return np.asarray(value.detach().cpu() if hasattr(value, "detach") else value, dtype=np.float64)


def neuron_thresholds(threshold: Any, count: int, width: Optional[int] = None) -> np.ndarray:
    """The thresholds of a core's first ``count`` neurons (its used neurons) of the ``width`` neurons its registers
    are held for (``count`` by default; a hard core's register column covers every neuron of the core): its one
    register repeated, or its per-neuron registers. [U-LIVE r2] A per-neuron array of any other size is read for
    another number of neurons and refused by name (``ValueLayoutError``), never taken at its first entries: a site
    that dropped a neuron and kept its register is a refusal, never a misaligned threshold."""
    values = _array(threshold)
    if values.ndim == 0:
        return np.full(int(count), float(values))
    held_for = int(count) if width is None else int(width)
    if values.size != held_for or int(count) > held_for:
        raise ValueLayoutError(
            f"a per-neuron threshold of {values.size} registers read for {int(count)} of {held_for} neurons: a "
            f"register per neuron of another neuron set, refused by name")
    return values.reshape(-1)[:int(count)].copy()


def threshold_tensor(threshold: Any, count: int, *, dtype: torch.dtype, device: Any,
                     width: Optional[int] = None) -> torch.Tensor:
    """A core's threshold as its executor reads it: a 0-d tensor for one register (byte for byte as before), one entry
    per used neuron for per-neuron registers (``width``: the neurons they are held for, ``neuron_thresholds``)."""
    if not is_per_neuron(threshold):
        return torch.tensor(float(threshold), dtype=dtype, device=device)
    return torch.tensor(neuron_thresholds(threshold, count, width), dtype=dtype, device=device)


def one_register(core: Any, consumer: str) -> Any:
    """[U-PCH3] ``core``'s threshold as it is held, for a consumer that holds one register per core (which reads and
    refuses the held value by its own rule); a per-neuron threshold is refused by name."""
    threshold: Any = getattr(core, "threshold", None)
    if is_per_neuron(threshold):
        raise PerNeuronThresholdRefusal(
            f"{consumer} holds one threshold register per core, but core "
            f"{getattr(core, 'name', None) or getattr(core, 'id', '?')!r} carries one threshold per neuron "
            f"({KEY} 'per_neuron'): declare {KEY} 'per_perceptron' for this consumer")
    return threshold


def core_threshold(core: Any, consumer: str) -> float:
    """The one threshold register of ``core`` for a consumer that holds one per core; a per-neuron threshold is
    refused by name."""
    return float(one_register(core, consumer))


def refuse_per_neuron_cores(ir_graph: Any, consumer: str) -> None:
    """Refuse by name a program whose cores hold per-neuron thresholds, for a consumer that reads one per core."""
    for node in ir_graph.get_neural_cores():
        core_threshold(node, consumer)


def threshold_of_neuron(threshold: Any, index: int) -> Any:
    """Neuron ``index``'s threshold: the core's one register as it is (byte for byte), or that neuron's register (the
    first one beyond the registers: a padded neuron, which carries no weights)."""
    if not is_per_neuron(threshold):
        return threshold
    values = _array(threshold).reshape(-1)
    return float(values[int(index)] if int(index) < values.size else values[0])


def threshold_columns(threshold: Any, lo: int, hi: Optional[int]) -> Any:
    """The threshold of neuron columns ``lo:hi`` of a core (``hi`` None: to its last): its one register, or those
    columns' registers."""
    if not is_per_neuron(threshold):
        return threshold
    return _array(threshold).reshape(-1)[int(lo):None if hi is None else int(hi)].copy()


def threshold_bound(threshold: Any) -> float:
    """A core's threshold as a bound's term reads it: its one register, or the largest magnitude of its per-neuron
    registers (``None`` reads 0)."""
    if threshold is None:
        return 0.0
    if not is_per_neuron(threshold):
        return float(threshold or 0.0)
    return float(np.max(np.abs(_array(threshold)), initial=0.0))


def thresholds_integral(threshold: Any, tol: float = 0.0) -> bool:
    """Whether every threshold register of a core is an integer (within ``tol``)."""
    values = _array(threshold if threshold is not None else 0.0)
    return bool(np.all(np.abs(values - np.round(values)) <= float(tol)))


def threshold_key(threshold: Any, count: int) -> Tuple[float, ...]:
    """A core's used neurons' thresholds as one comparable value (``None`` reads 0)."""
    return tuple(neuron_thresholds(0.0 if threshold is None else threshold, count).tolist())


def threshold_report(threshold: Any) -> Any:
    """A core's threshold for a report: its one register, or its per-neuron registers as a list."""
    if threshold is None:
        return None
    return _array(threshold).reshape(-1).tolist() if is_per_neuron(threshold) else float(threshold)


def report_registers(reported: Any) -> list:
    """[U-PCH2] Every register of a reported threshold (``threshold_report``'s float or list) as floats, for a reader
    that summarizes cores (a distribution, a range); none for an absent one."""
    return [] if reported is None else [float(v) for v in np.asarray(reported, dtype=np.float64).reshape(-1)]


def register_and_neurons(threshold: Any) -> Tuple[float, Optional[np.ndarray]]:
    """A threshold split for an executor with one register and an optional per-neuron override: ``(value, None)``
    for one register, ``(largest, per-neuron values)`` for per-neuron registers."""
    if not is_per_neuron(threshold):
        return float(threshold), None
    values = _array(threshold).reshape(-1)
    return float(values.max(initial=0.0)), values.copy()


def place_neuron_value(hard_core: Any, softcore: Any, name: str, neuron_offset: int) -> bool:
    """[U-PCH3] Write a soft core's value ``name`` (a singleton the hard core declares per neuron) over its neuron
    range of ``hard_core`` when either holds one entry per neuron (as its hardware bias is placed) and answer True;
    False when both hold one value, which the residency check adopts and compares."""
    incoming = getattr(softcore, name, None)
    current = getattr(hard_core, name, None)
    if not (is_per_neuron(incoming) or is_per_neuron(current)):
        return False
    width = int(softcore.get_output_count())
    if is_per_neuron(current):
        column = _array(current).reshape(-1).copy()
    else:
        fill = _array(incoming).reshape(-1)[0] if current is None else float(current)
        column = np.full(int(hard_core.neurons_per_core), float(fill))
    column[int(neuron_offset):int(neuron_offset) + width] = neuron_thresholds(incoming, width)
    setattr(hard_core, name, column)
    return True


def node_neuron_channels(node: Any, width: int) -> np.ndarray:
    """The perceptron output channel of each of an IR node's ``width`` columns (its ``perceptron_output_slice`` start
    plus its column before IR pruning); refused by name for a node with no perceptron slice."""
    out_slice = getattr(node, "perceptron_output_slice", None)
    if getattr(node, "perceptron_index", None) is None or out_slice is None:
        raise PerNeuronThresholdRefusal(
            f"IR node {getattr(node, 'name', '?')!r} carries a per-neuron weight scale but no perceptron output "
            f"slice, so no column has a neuron's scale")
    columns = np.arange(int(width), dtype=np.int64)
    mask = getattr(node, "pre_pruning_col_mask", None)
    if mask is not None:
        columns = np.flatnonzero(~np.asarray(mask, dtype=bool)).astype(np.int64)[:int(width)]
    return int(out_slice[0]) + columns


def trained_neuron_scales(parameter_scale: Any, *, fractional: bool = False) -> np.ndarray:
    """A per-neuron trained scale as the chip's per-neuron threshold registers: each entry already on the integer
    lattice (``max(1, round(s_c))``), refused loud off it as a per-perceptron scale is (``chip_quantize``); a declared
    fractional rule keeps it as is."""
    values = _array(parameter_scale).reshape(-1)
    if fractional:
        return values.copy()
    snapped = np.maximum(1.0, np.round(values))
    if np.any(np.abs(values - snapped) > 1e-6 * snapped):
        raise ValueError(
            f"quantize_ir_graph: a trained per-neuron parameter_scale is not on the integer lattice (chip "
            f"threshold_t=int): {values[np.abs(values - snapped) > 1e-6 * snapped][:4].tolist()}; refusing to "
            f"truncate at deploy.")
    return snapped


def localized_neuron_scales(owner: Any, channels: np.ndarray) -> Tuple[Any, Any]:
    """The ``parameter_scale`` and ``bias_scale`` of an IR node or weight bank at its columns, whose perceptron channels
    are ``channels``: a per-neuron one read by its declared layout (its perceptron's, one per output channel), one held
    value as it is. [U-PCH3] Read for the caller, never written back: the holder keeps its declared layout."""
    out = []
    for name in ("parameter_scale", "bias_scale"):
        value = getattr(owner, name, None)
        if is_per_neuron(value):
            width = len(channels)
            value = torch.as_tensor(neuron_entries(owner, name, np.arange(width), width=width, channels=channels))
        out.append(value)
    return out[0], out[1]


def bank_neuron_channels(bank_id: int, width: int, nodes: Any) -> np.ndarray:
    """The perceptron output channel of each of a weight bank's ``width`` columns: one base read off every node that
    references the bank (its ``perceptron_output_slice`` start less its ``weight_row_slice`` start), refused by name
    when the nodes disagree or none carries a slice."""
    bases = set()
    for node in nodes:
        if getattr(node, "weight_bank_id", None) is None or int(node.weight_bank_id) != int(bank_id):
            continue
        out_slice, rows = getattr(node, "perceptron_output_slice", None), getattr(node, "weight_row_slice", None)
        if out_slice is not None:
            bases.add(int(out_slice[0]) - (int(rows[0]) if rows is not None else 0))
    if len(bases) != 1:
        raise PerNeuronThresholdRefusal(
            f"weight bank {int(bank_id)} carries a per-neuron weight scale, but its nodes place its columns at "
            f"{sorted(bases) or 'no'} perceptron channel base(s), so no column has one neuron's scale")
    return bases.pop() + np.arange(int(width), dtype=np.int64)
