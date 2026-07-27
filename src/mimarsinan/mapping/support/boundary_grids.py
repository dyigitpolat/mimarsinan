"""[mvm AQ] Realize the model's boundary grids onto the mapped IR."""

from __future__ import annotations

from mimarsinan.mapping.ir import NeuralCore
from mimarsinan.models.nn.activations.value_quantizer import (
    BoundaryGrid,
    boundary_grid_of,
)


def _perceptron_grids(model_repr) -> dict:
    """{perceptron_index: BoundaryGrid} for every entry that installed one."""
    model_repr._ensure_exec_graph()
    exec_order = model_repr._exec_order
    assert exec_order is not None  # populated by _ensure_exec_graph
    grids: dict = {}
    for node in exec_order:
        perceptron = getattr(node, "perceptron", None)
        index = getattr(node, "perceptron_index", None)
        if perceptron is None or index is None:
            continue
        grid = boundary_grid_of(getattr(perceptron, "input_activation", None))
        if grid is not None:
            grids[int(index)] = grid
    return grids


def install_boundary_grids(ir_graph, model) -> int:
    """Stamp each entry's realized grid onto its cores; return how many.

    The grid is a first-class field, NOT a value smuggled through the event
    domain's ``input_activation_scale`` — so the spiking boundary walk may
    run unconditionally without clobbering it.
    """
    grids = _perceptron_grids(model.get_mapper_repr())
    if not grids:
        return 0
    stamped = 0
    for node in ir_graph.nodes:
        if not isinstance(node, NeuralCore):
            continue
        grid = grids.get(node.perceptron_index)
        if grid is not None:
            node.boundary_grid = grid
            stamped += 1
    return stamped


def widest_boundary_step(ir_graph) -> "float | None":
    """One LSB of the coarsest armed grid — the AQ certificate's unit."""
    steps = [
        node.boundary_grid.step
        for node in ir_graph.nodes
        if isinstance(node, NeuralCore)
        and isinstance(node.boundary_grid, BoundaryGrid)
        and node.boundary_grid.armed
    ]
    return max(steps) if steps else None
