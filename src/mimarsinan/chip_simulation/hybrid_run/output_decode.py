"""The one decode of an output into the value domain (U-TGAUGE2): every comparison of two outputs is made in one unit.

A twin holds its output at a gauge, the one its producer encoded it at, and every reader that reads the output as a
value (a class, a loss, a gate comparing two twins or two executors) decodes it by the inverse of that encode through
:func:`decode_held`, the held value times its gauge per element:

- the walk decodes its output node at the gauge its own tables hold it at (``LifSegmentPolicy.decode_output``);
- a deployed program holds each output element in its producer's state buffer, ``state = value / gauge``
  (:func:`program_output_gauge`, read from the program alone: a deployment carries no model), and a spiking executor's
  final read is that state times its window (:func:`executor_output_values` divides the window out first).

[U-TGAUGE3] The placement rule: every element at its producer's gauge read at its source index, through the one
placement (``gauge_placement.place_gauges``) from the program's IR source table (``output_sources``); the walks read the
same rule from the identity program's table. A bare packed mapping holds no such table: its counts rank as its values
only where its output cores hold them at one gauge, and the read refuses by name where they differ
(:func:`bare_mapping_output_gauge`).

The executors' own outputs keep their count currency: records, the per-neuron parity gathers and executor-to-executor
comparisons read counts against counts, which is one unit too. A gauge is read at the record's boundary dtype
(``SPIKING_ARITHMETIC.boundary_dtype``, the dtype of the state buffers it scales); the decode runs at the held read's.
"""

from __future__ import annotations

from typing import Any, TypeVar, cast

import numpy as np
import torch
import torch.nn as nn

from mimarsinan.chip_simulation.hybrid_run.hybrid_execution import (
    compute_op_owns_scale_domain,
    resolve_stage_compute_scales,
)
from mimarsinan.chip_simulation.spiking_semantics import requires_ttfs_firing
from mimarsinan.mapping.ir.deployment_dtype import SPIKING_ARITHMETIC
from mimarsinan.mapping.support.gauge_placement import place_gauges, source_index
from mimarsinan.mapping.support.scale_broadcast import broadcast_scale_along

__all__ = ["DeployedValueReadout", "OutputGaugeError", "bare_mapping_output_gauge", "decode_held",
           "executor_output_values", "program_output_gauge"]


class OutputGaugeError(ValueError):
    """A program output whose gauge cannot be placed element by element (a per-channel emitted gauge on an op that
    declares no output shape)."""


Held = TypeVar("Held", torch.Tensor, np.ndarray)


def decode_held(held: Held, gauge) -> Held:
    """``held`` (batch first) in the value domain: times ``gauge``, the gauge its producer encoded it at. ``gauge`` is
    ``None`` (an absolute value: the identity), a scalar, or a tensor or array that broadcasts against ``held`` (one
    gauge per output element, the batch axis first)."""
    if gauge is None:
        return held
    if isinstance(held, np.ndarray):
        return cast(Held, held * np.asarray(gauge, dtype=SPIKING_ARITHMETIC.boundary_np))
    if isinstance(gauge, np.ndarray):
        gauge = torch.from_numpy(gauge).to(device=held.device, dtype=held.dtype)
    return cast(Held, held * gauge)


def _emitted_gauge(op, module) -> "float | np.ndarray":
    """A domain-owning host op's emitted gauge per element of its flattened output: the ``output_scale`` it divides
    its value by, broadcast along its channel axis as its own forward broadcasts it."""
    scale = torch.as_tensor(module.output_scale).detach().to(SPIKING_ARITHMETIC.boundary_dtype).reshape(-1)
    if scale.numel() == 1 or bool(torch.all(scale == scale[0])):
        return float(scale[0])
    shape = getattr(op, "output_shape", None)
    if shape is None:
        raise OutputGaugeError(
            f"host op {op.id} emits a per-channel gauge ({scale.numel()} values) but declares no output shape, so its "
            f"output elements cannot be decoded one by one")
    like = torch.zeros(1, *tuple(int(d) for d in shape), dtype=SPIKING_ARITHMETIC.boundary_dtype)
    return broadcast_scale_along(scale, like, getattr(module, "channel_axis", None)).expand_as(like).reshape(-1).numpy()


def _held_gauge(program, node_id: int, op, timing: bool) -> "float | np.ndarray":
    """The gauge ``node_id``'s state buffer holds its value at (``state = value / gauge``): a core at its threshold
    (``node_buffer_scales``, per neuron where it carries a vector); a domain-owning host op at its emitted gauge; any
    other host op at the family's stage convention: a timing family re-normalizes it by the op's out-scale
    (``resolve_stage_compute_scales``), a rate family leaves it at its buffer gauge (a raw value at one)."""
    buffers = getattr(program, "node_buffer_scales", None) or {}
    folds = getattr(program, "node_activation_scales", None) or {}
    if op is None:
        return buffers.get(node_id, folds.get(node_id, 1.0))
    module = (op.params or {}).get("module")
    if compute_op_owns_scale_domain(op):
        return _emitted_gauge(op, module)
    if timing:
        return float(resolve_stage_compute_scales(program, node_id, apply_ttfs=True, op=op)[1])
    return buffers.get(node_id, 1.0)


def program_output_gauge(program, *, timing: bool = False) -> np.ndarray:
    """The deployed program's gauge per output element (at the record's boundary dtype, in ``output_sources`` order):
    each element's source at the gauge its state buffer holds it at; the raw input, an always-on source and an
    unconnected one at one. ``timing``: the program runs under a timing (TTFS) family's host-op convention."""
    ops = {int(stage.compute_op.id): stage.compute_op for stage in program.stages
           if stage.kind == "compute" and stage.compute_op is not None}
    ids, idx = source_index(program.output_sources)
    return place_gauges(ids, idx, lambda nid: _held_gauge(program, nid, ops.get(nid), timing),
                        dtype=SPIKING_ARITHMETIC.boundary_np)


def _core_gauge(core, neuron: int):
    """[U-TGAUGE4] The gauge a packed core holds ``neuron`` at, as its holder declares the threshold register: per core
    (``activation_scale``, the value the static check's H1-theta reads: one value, or a vector over the core's used
    neurons, read at the neuron); ``None`` where the core declares none. A declaration this read does not place refuses
    by name: a register the target stores per neuron range (``activation_scale`` outside the core's
    ``constrained_properties``: each resident's value over its own neurons, which only the program's table places), and
    a vector that spans not the core's used neurons (residents sharing one per-channel vector: which resident's channel a
    neuron holds is not declared). It read the vector by its size at the hard core's neuron index: a per-neuron
    declaration read as no gauge (the counts ranked), and two residents of one vector raised numpy's IndexError."""
    constrained = getattr(core, "constrained_properties", None)
    if constrained is not None and "activation_scale" not in constrained:
        raise OutputGaugeError(
            "a packed core whose target stores the activation scale per neuron range holds each resident's gauge over "
            "its own neurons, which only the program's source table places: run it as a program")
    value = getattr(core, "activation_scale", None)
    if value is None:
        return None
    flat = np.asarray(torch.as_tensor(value).detach().cpu(), dtype=SPIKING_ARITHMETIC.boundary_np).reshape(-1)
    used = int(getattr(core, "neurons_per_core", flat.size)) - int(getattr(core, "available_neurons", 0) or 0)
    if flat.size == 1 or flat.size == used:
        return float(flat[0] if flat.size == 1 else flat[neuron])
    raise OutputGaugeError(
        f"a packed core declares one {flat.size}-value activation scale over {used} used neurons: which resident's "
        f"channel output neuron {neuron} holds is not declared, so its gauge cannot be read")


def bare_mapping_output_gauge(mapping) -> None:
    """[U-TGAUGE3] A bare packed mapping (no program, so no gauge table by source): its output counts rank as its values
    where every output core holds its outputs at one gauge (``None``: the counts' own ranking); where the gauges its
    output cores declare differ per output, the read refuses by name, never a count ranking."""
    sources = getattr(mapping, "output_sources", None)
    gauges = set()
    for src in [] if sources is None else list(sources):
        if getattr(src, "is_off_", False) or getattr(src, "is_input_", False) or getattr(src, "is_always_on_", False):
            continue
        gauges.add(_core_gauge(mapping.cores[int(src.core_)], int(src.neuron_)))
    if len(gauges) > 1:
        raise OutputGaugeError(
            f"a bare packed mapping holds no gauge table, and its output cores hold its outputs at {len(gauges)} "
            f"gauges ({sorted(g for g in gauges if g is not None)[:4]}...): its counts do not rank as its values; "
            f"run it as a program (its HybridHardCoreMapping), which decodes each output at its gauge")
    return None


def executor_output_values(executor, out: torch.Tensor, *, gauge: "np.ndarray | None" = None) -> torch.Tensor:
    """A spiking executor's final read ``out`` (its held state times its window, the count currency) in the value
    domain, by its program's output gauge (``gauge``: one read beforehand). An executor without a window (the value
    flow) already returns values."""
    window = getattr(executor, "simulation_length", None)
    if window is None:
        return out
    if gauge is None:
        gauge = program_output_gauge(
            executor.hybrid_mapping, timing=requires_ttfs_firing(getattr(executor, "spiking_mode", "lif")))
    return decode_held(out / float(window), gauge)


class DeployedValueReadout(nn.Module):
    """The deployed readout of a spiking executor: its final read in the value domain (the classes a reader takes are
    the value domain's), the program's output gauge read once."""

    def __init__(self, executor: Any):
        super().__init__()
        self.executor = executor
        self._gauge = program_output_gauge(
            executor.hybrid_mapping, timing=requires_ttfs_firing(getattr(executor, "spiking_mode", "lif")))

    @property
    def hybrid_mapping(self):
        """The program it reads (its window and family stay the executor's: this readout is no executor to decode)."""
        return self.executor.hybrid_mapping

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return executor_output_values(self.executor, self.executor(x), gauge=self._gauge)
