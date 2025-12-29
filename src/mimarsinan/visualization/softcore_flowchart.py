from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from mimarsinan.mapping.mapping_utils import (
    AddMapper,
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


def _estimate_map_fc(
    *,
    in_features: int,
    out_features: int,
    instances: int,
    has_bias: bool,
    max_axons: int,
    max_neurons: int,
    allow_axon_tiling: bool,
) -> HWEstimate:
    """
    Estimate the number of hardware cores produced by SoftCoreMapping.map_fc (without actually mapping),
    including axon tiling (partial sums + accumulator) when enabled.
    """

    bias_ax = 1 if has_bias else 0
    required_axons = int(in_features) + bias_ax

    if required_axons <= max_axons:
        groups = _ceil_div(out_features, max_neurons)
        # Each group creates `instances` cores.
        cores_total = int(instances) * int(groups)
        details = (
            f"no axon-tiling\n"
            f"required_axons={required_axons} (features={in_features}, bias={bias_ax}) <= max_axons={max_axons}\n"
            f"output_groups=ceil({out_features}/{max_neurons})={groups}\n"
            f"core_matrix≈{required_axons}x<= {max_neurons} (axons x neurons), cores_per_group={instances}"
        )
        return HWEstimate(True, None, cores_total, details)

    # Axon overflow
    if not allow_axon_tiling:
        return HWEstimate(
            False,
            f"requires axon tiling: required_axons={required_axons} > max_axons={max_axons}",
            0,
            f"UNMAPPABLE: required_axons={required_axons} > max_axons={max_axons} and allow_axon_tiling=False",
        )

    # Replicate SoftCoreMapping.map_fc tiling logic
    tile_size = max_axons  # bias excluded from partials
    tile_count = _ceil_div(in_features, tile_size)

    max_out_by_accum_axons = (max_axons - bias_ax) // (2 * tile_count)
    if max_out_by_accum_axons <= 0:
        return HWEstimate(
            False,
            "accumulator cannot fit",
            0,
            f"UNMAPPABLE: accumulator needs 2*tile_count*out_block + bias_axons <= max_axons "
            f"(tile_count={tile_count}, max_axons={max_axons}, bias_axons={bias_ax})",
        )

    out_block = min(max_neurons, int(max_out_by_accum_axons))
    if out_block <= 0:
        return HWEstimate(False, "out_block <= 0", 0, f"UNMAPPABLE: out_block={out_block}")

    out_blocks = _ceil_div(out_features, out_block)
    cores_per_instance_per_block = 2 * tile_count + 1  # pos tiles + neg tiles + accum
    cores_total = int(instances) * int(out_blocks) * int(cores_per_instance_per_block)

    details = (
        f"axon-tiling enabled\n"
        f"required_axons={required_axons} (features={in_features}, bias={bias_ax}) > max_axons={max_axons}\n"
        f"tile_count=ceil({in_features}/{tile_size})={tile_count}\n"
        f"accum_axons=2*tile_count*out_block + bias_axons <= max_axons\n"
        f"max_out_by_accum_axons=floor(({max_axons}-{bias_ax})/(2*{tile_count}))={max_out_by_accum_axons}\n"
        f"out_block=min(max_neurons={max_neurons}, max_out_by_accum_axons)={out_block}\n"
        f"out_blocks=ceil({out_features}/{out_block})={out_blocks}\n"
        f"cores_per_instance_per_block=2*tile_count+1={cores_per_instance_per_block}\n"
        f"total_cores=instances({instances})*out_blocks({out_blocks})*cores_per_instance_per_block({cores_per_instance_per_block})={cores_total}\n"
        f"core_shapes:\n"
        f"  partial: <={tile_size}x{out_block} (no bias)\n"
        f"  accum: {2*tile_count*out_block + bias_ax}x{out_block}"
    )
    return HWEstimate(True, None, cores_total, details)


def _node_display_name(node: Any) -> str:
    if hasattr(node, "name"):
        try:
            return str(getattr(node, "name"))
        except Exception:
            pass
    if isinstance(node, PerceptronMapper):
        return f"Perceptron:{getattr(node.perceptron, 'name', '<unnamed>')}"
    return type(node).__name__


def _dot_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n")


def generate_softcore_flowchart_dot(
    mapper_repr: ModelRepresentation,
    *,
    input_shape: tuple[int, ...],
    max_axons: int,
    max_neurons: int,
    allow_axon_tiling: bool = False,
    device: torch.device | None = None,
) -> str:
    """
    Generate a Graphviz DOT flowchart for the mapper graph, annotated with estimated SoftCoreMapping cost.

    This does NOT require successful mapping (pooling ops can exist); it uses forward shape tracing + estimation.
    """

    if device is None:
        device = torch.device("cpu")

    # Trace forward shapes through the mapper graph (single batch)
    mapper_repr._ensure_exec_graph()  # noqa: SLF001
    exec_order: list[Mapper] = list(mapper_repr._exec_order)  # noqa: SLF001
    deps = mapper_repr._deps  # noqa: SLF001

    x0 = torch.randn(1, *input_shape, device=device)
    values: dict[Mapper, torch.Tensor] = {}
    for node in exec_order:
        d = deps.get(node, [])
        if len(d) == 0:
            values[node] = node.forward(x0)
        elif len(d) == 1:
            values[node] = node.forward(values[d[0]])
        else:
            values[node] = node.forward(tuple(values[dep] for dep in d))

    # Build DOT
    lines: list[str] = []
    lines.append("digraph SoftCoreFlowchart {")
    lines.append("  rankdir=LR;")
    lines.append("  node [shape=record, fontname=\"Helvetica\", fontsize=10];")
    lines.append("  edge [fontname=\"Helvetica\", fontsize=9];")

    node_id: dict[Mapper, str] = {}
    for idx, node in enumerate(exec_order):
        nid = f"n{idx}"
        node_id[node] = nid

        out_t = values.get(node)
        out_shape = None
        if out_t is not None and hasattr(out_t, "shape"):
            # drop batch dim if present
            if len(out_t.shape) >= 1:
                out_shape = tuple(out_t.shape[1:])

        hw = None
        hw_text = "HW: n/a"
        sw_text = "SW: n/a"

        # Estimate hardware usage for known mappable ops
        if isinstance(node, PerceptronMapper):
            p = node.perceptron
            in_f = int(p.layer.weight.shape[1])
            out_f = int(p.layer.weight.shape[0])
            inst = 1
            sw_text = f"SW perceptrons=1 (in_features={in_f}, out_features={out_f})"
            hw = _estimate_map_fc(
                in_features=in_f,
                out_features=out_f,
                instances=inst,
                has_bias=(p.layer.bias is not None),
                max_axons=int(max_axons),
                max_neurons=int(max_neurons),
                allow_axon_tiling=bool(allow_axon_tiling),
            )
        elif isinstance(node, Conv2DPerceptronMapper):
            k_h, k_w = node.kernel_size
            in_f = int(node.in_channels * k_h * k_w)
            out_f = int(node.out_channels)
            sw_text = f"SW perceptrons=1 (shared conv weights, patch={in_f}, out_channels={out_f})"
            if out_shape is not None and len(out_shape) == 3:
                _, h, w = out_shape
                inst = int(h * w)
            else:
                inst = 1
            hw = _estimate_map_fc(
                in_features=in_f,
                out_features=out_f,
                instances=inst,
                has_bias=bool(node.bias),
                max_axons=int(max_axons),
                max_neurons=int(max_neurons),
                allow_axon_tiling=bool(allow_axon_tiling),
            )
        elif isinstance(node, Conv1DPerceptronMapper):
            in_f = int(node.in_channels * node.kernel_size)
            out_f = int(node.out_channels)
            sw_text = f"SW perceptrons=1 (shared conv weights, patch={in_f}, out_channels={out_f})"
            if out_shape is not None and len(out_shape) == 2:
                _, l = out_shape
                inst = int(l)
            else:
                inst = 1
            hw = _estimate_map_fc(
                in_features=in_f,
                out_features=out_f,
                instances=inst,
                has_bias=bool(node.bias),
                max_axons=int(max_axons),
                max_neurons=int(max_neurons),
                allow_axon_tiling=bool(allow_axon_tiling),
            )
        elif isinstance(node, AddMapper):
            # Add is mapped as a linear op (concat + identity weights). Estimate roughly.
            # For visualization purposes, treat it as 1 instance, axons=2*features, neurons=features.
            if out_shape is not None and len(out_shape) == 1:
                feat = int(out_shape[0])
                sw_text = f"SW perceptrons=0 (Add op)"
                hw = _estimate_map_fc(
                    in_features=2 * feat,
                    out_features=feat,
                    instances=1,
                    has_bias=False,
                    max_axons=int(max_axons),
                    max_neurons=int(max_neurons),
                    allow_axon_tiling=bool(allow_axon_tiling),
                )

        if hw is not None:
            hw_text = f"HW cores={hw.cores_total}\\n{hw.details}"
            if not hw.mappable and hw.reason is not None:
                hw_text = f"HW UNMAPPABLE\\n{hw.reason}"
        else:
            # Heuristic: explicit non-neural ops are implemented as *OpMapper in this repo.
            if type(node).__name__.endswith("OpMapper"):
                hw_text = "HW UNSUPPORTED (non-spiking op)"

        label_parts = [
            _dot_escape(_node_display_name(node)),
            _dot_escape(f"type={type(node).__name__}"),
        ]
        if out_shape is not None:
            label_parts.append(_dot_escape(f"out_shape={out_shape}"))
        label_parts.append(_dot_escape(sw_text))
        label_parts.append(_dot_escape(hw_text))
        label = "{ " + " | ".join(label_parts) + " }"

        # Color unsupported/non-mappable nodes
        attrs = []
        if hw is not None and not hw.mappable:
            attrs.append("color=red")
        if type(node).__name__.endswith("OpMapper"):
            attrs.append("color=orange")
        # StackMapper isn't directly mappable (it just structures tensors)
        if isinstance(node, StackMapper):
            attrs.append("color=gray")

        attr_str = ""
        if attrs:
            attr_str = " [" + ",".join(attrs) + "]"

        lines.append(f"  {nid} [label=\"{label}\"]{attr_str};")

    # Edges
    for node in exec_order:
        nid = node_id[node]
        for dep in deps.get(node, []):
            if dep is None:
                continue
            lines.append(f"  {node_id[dep]} -> {nid};")

    lines.append("}")
    return "\n".join(lines)


def write_softcore_flowchart_dot(
    mapper_repr: ModelRepresentation,
    out_path: str,
    *,
    input_shape: tuple[int, ...],
    max_axons: int,
    max_neurons: int,
    allow_axon_tiling: bool = False,
    device: torch.device | None = None,
) -> None:
    dot = generate_softcore_flowchart_dot(
        mapper_repr,
        input_shape=input_shape,
        max_axons=max_axons,
        max_neurons=max_neurons,
        allow_axon_tiling=allow_axon_tiling,
        device=device,
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(dot)


