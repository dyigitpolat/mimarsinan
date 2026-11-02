"""[U-IN F-IN] The raw input is an absolute producer at the boundary gauge in both twins.

The input root (``InputMapper``) is a host buffer of raw values, ABSOLUTE in the value-domain classification, whose
boundary-table gauge is the run's input data scale ``s`` (``propagate_boundary_scale``). The NF encodes a core's read of
it as ``clamp(x / s)`` and carries ``s`` back (``normalize_boundary_value``), the one transcode U-SC2 F-2 made every
input-rooted host chain take (``wire_gauge_seam.arm_absolute_core_entries``). A core that reads the raw input itself (the
encoding layer placed on chip) takes the same transcode here, generic over the layout nodes between the root and the
core: the source walk folds ``s`` into the core (``fold_input_root_gauge``, through each node's own
``propagate_source_scale``), the IR carries ``s`` as the raw input's fold gauge over its unit buffer gauge
(``stamp_ir_input_gauge``; ``activation_scales.compute_node_output_scales`` and ``compute_node_buffer_scales``), so the
divisor table every executor reads divides the raw slice by ``s`` before the clamp
(``segment_boundary.boundary_normalization_scales``). A host op reading the raw buffer decodes it at one, as before: the
root has no wrap slot, and the gauge it emits to a host op is its unit host currency (``InputMapper``).

The gauge belongs to the rate/LIF wire currency the wire-gauge seam establishes (``establish_wire_gauge`` passes it); a
source walk the seam never stamped (a value-domain or TTFS program) folds the raw input at one. The stamp is written only
where a core folds the raw input at a gauge other than one, so every other graph is left as it was.
"""

from __future__ import annotations

from typing import Any

from mimarsinan.common.refusal.toolchain_refusal import ToolchainRefusal
from mimarsinan.mapping.ir import IRGraph, NeuralCore
from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper
from mimarsinan.mapping.mappers.module_mapper import ModuleMapper
from mimarsinan.mapping.mappers.structural import InputMapper
from mimarsinan.mapping.support.activation_scales import INPUT_WIRE_GAUGE, RAW_INPUT

__all__ = ["InputGaugeRefusal", "fold_input_root_gauge", "input_wire_gauge", "stamp_ir_input_gauge"]

_RTOL = 1e-12


class InputGaugeRefusal(ToolchainRefusal, NotImplementedError):
    """A node reads the raw input at a non-unit wire gauge that is neither a core, a host op nor a layout node."""


def input_wire_gauge(holder: Any) -> float:
    """The gauge at which a core reads the raw input, stamped on a mapper graph or an IR; one when none is."""
    return float(getattr(holder, "__dict__", {}).get(INPUT_WIRE_GAUGE, 1.0))


def _carries(node) -> bool:
    """A node whose IR is its sources' (a layout node, a forward-only module): the raw input's currency passes it."""
    return bool(getattr(node, "moves_elements_only", False)) or isinstance(node, ModuleMapper)


def fold_input_root_gauge(model_repr, out_scales: dict, gauge: float | None) -> int:
    """Fold the raw input's wire gauge into every core that reads it (directly or through layout nodes), each node
    applying its own source-scale rule to the root's vector at that gauge; host ops keep the unit host currency the
    walk gave them. ``gauge`` is the seam's (``None``: the one stamped on the graph); the stamp is kept only where a
    core folds a gauge other than one. Returns the number of cores folded."""
    stamped = gauge is None
    gauge = input_wire_gauge(model_repr) if stamped else float(gauge)
    folded = 0
    if abs(gauge - 1.0) > _RTOL:
        model_repr._ensure_exec_graph()
        core_read: dict = {}
        for node in model_repr._exec_order:
            deps = model_repr._deps.get(node, [])
            if isinstance(node, InputMapper):
                core_read[node] = out_scales[node] * gauge
                continue
            if not any(d in core_read for d in deps):
                continue
            view = {d: core_read.get(d, out_scales.get(d)) for d in deps if d in core_read or d in out_scales}
            perceptron = getattr(node, "perceptron", None)
            if perceptron is not None:
                if not getattr(perceptron, "is_encoding_layer", False):  # a host-run perceptron reads it raw
                    node.propagate_source_scale(deps, view)
                    folded += 1
            elif _carries(node):
                core_read[node] = node.propagate_source_scale(deps, view)
            elif not isinstance(node, ComputeOpMapper):
                raise InputGaugeRefusal(
                    f"{type(node).__name__} {getattr(node, 'name', None)!r} reads the raw input at the wire gauge "
                    f"{gauge:.6g}: neither a core, a host op nor a layout node, so no fold of the input root is defined "
                    "for it (U-IN). Remedy: declare encoding_layer_placement subsume")
    if not stamped:
        if folded:
            setattr(model_repr, INPUT_WIRE_GAUGE, gauge)
        else:
            model_repr.__dict__.pop(INPUT_WIRE_GAUGE, None)
    return folded


def _reads_raw_input(core: NeuralCore) -> bool:
    return any(int(s.node_id) == RAW_INPUT for s in core.input_sources.flatten())


def stamp_ir_input_gauge(ir_graph: IRGraph, model_repr) -> None:
    """The IR carries the raw input's wire gauge when a core of it reads the raw input (the program's node maps read it
    off the IR, so a pickled program keeps it)."""
    gauge = input_wire_gauge(model_repr)
    if abs(gauge - 1.0) > _RTOL and any(_reads_raw_input(n) for n in ir_graph.get_neural_cores()):
        setattr(ir_graph, INPUT_WIRE_GAUGE, gauge)
