from __future__ import annotations

from typing import Any

import torch

from mimarsinan.mapping.mapping_utils import (
    Mapper,
    ModelRepresentation,
    PerceptronMapper,
)


from mimarsinan.visualization.softcore_flowchart_estimate import _estimate_map_fc

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
    device: torch.device | None = None,
    hardware_bias: bool = False,
    allow_coalescing: bool = True,
) -> str:
    """Graphviz DOT flowchart for the mapper graph annotated with estimated SoftCoreMapping cost.

    Uses forward shape tracing + estimation; does not require successful mapping.
    """

    if device is None:
        device = torch.device("cpu")

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
            if len(out_t.shape) >= 1:
                out_shape = tuple(out_t.shape[1:])

        hw = None
        hw_text = "HW: n/a"

        _est_kw = dict(
            max_axons=int(max_axons),
            max_neurons=int(max_neurons),
            hardware_bias=hardware_bias,
            allow_coalescing=allow_coalescing,
        )

        estimate = node.flowchart_node_estimate(out_shape)
        sw_text = estimate.sw_text
        if estimate.fc_spec is not None:
            spec = estimate.fc_spec
            hw = _estimate_map_fc(
                in_features=spec.in_features,
                out_features=spec.out_features,
                instances=spec.instances,
                has_bias=spec.has_bias,
                **_est_kw,
            )

        if hw is not None:
            if hw.mappable:
                hw_text = f"HW cores≈{hw.cores_total}\n{hw.details}"
            else:
                hw_text = f"HW UNMAPPABLE: {hw.reason}\n{hw.details}"

        label = f"{_node_display_name(node)}\\n{sw_text}\\n{hw_text}"
        if out_shape is not None:
            label += f"\\nout_shape={out_shape}"

        lines.append(f'  {nid} [label="{_dot_escape(label)}"];')

    for node in exec_order:
        for dep in deps.get(node, []):
            lines.append(f"  {node_id[dep]} -> {node_id[node]};")

    lines.append("}")
    return "\n".join(lines)


def write_softcore_flowchart_dot(
    mapper_repr: ModelRepresentation,
    path: str,
    *,
    input_shape: tuple[int, ...],
    max_axons: int,
    max_neurons: int,
    device: torch.device | None = None,
    hardware_bias: bool = False,
    allow_coalescing: bool = True,
) -> None:
    dot = generate_softcore_flowchart_dot(
        mapper_repr,
        input_shape=input_shape,
        max_axons=max_axons,
        max_neurons=max_neurons,
        device=device,
        hardware_bias=hardware_bias,
        allow_coalescing=allow_coalescing,
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(dot)
