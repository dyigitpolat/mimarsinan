"""Lower a param-free equal-width residual add to an on-chip signed-IF merge core."""

from __future__ import annotations

import operator

import torch
import torch.nn as nn

from mimarsinan.mapping.layout.layout_source_view_ops import concat_source_views
from mimarsinan.mapping.mappers.base import Mapper
from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper
from mimarsinan.mapping.mappers.perceptron_mapper import PerceptronMapper
from mimarsinan.mapping.support.compute_modules import ComputeAdapter
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron

MERGE_NAME_SUFFIX = "_merge"


class _ResidualConcatMapper(Mapper):
    """Concatenate two equal-width branches into one ``2*width`` axon space (flattened to 1-D)."""

    def __init__(self, source_mappers, name: str = "ResidualConcat"):
        super().__init__()
        self._source_mappers_list = list(source_mappers)
        self.name = name

    def get_source_mappers(self):
        return [m for m in self._source_mappers_list if m is not None]

    def _map_to_ir(self, ir_mapping):
        views = []
        for mapper in self.get_source_mappers():
            arr = mapper.map_to_ir(ir_mapping)
            views.append(arr.flatten() if hasattr(arr, "flatten") else arr)
        return concat_source_views(views)

    def _forward_impl(self, x):
        return torch.cat(tuple(x), dim=1)

    def propagate_source_scale(self, deps, out_scales):
        from mimarsinan.mapping.mappers.scale_propagation import present_source_scales

        parts = present_source_scales(deps, out_scales)
        return torch.cat(parts) if parts else None

    def propagate_boundary_scale(self, deps, out_scales, default):
        # Lane-parallel concat: the scalar wire scale must cover the widest
        # branch (§6b contract-1); equal-θ branches reduce to the old mean.
        from mimarsinan.mapping.mappers.scale_propagation import max_source_scale

        return max_source_scale(deps, out_scales, float(default))


def _is_unbound_add(mapper) -> bool:
    """A param-free elementwise ``operator.add`` ComputeOpMapper (no bound tensors)."""
    module = getattr(mapper, "module", None)
    return (
        isinstance(mapper, ComputeOpMapper)
        and isinstance(module, ComputeAdapter)
        and getattr(module, "fn", None) is operator.add
        and getattr(module, "_bound_count", 0) == 0
    )


def _branch_width(mapper) -> int | None:
    """Output width of a branch mapper from its recorded output shape, else ``None``."""
    shape = getattr(mapper, "output_shape", None)
    if shape is None:
        p = getattr(mapper, "perceptron", None)
        if p is not None:
            return int(p.output_channels)
        return None
    flat = 1
    for d in shape:
        flat *= int(d)
    return int(flat)


def _build_merge_perceptron(width: int, name: str) -> Perceptron:
    """Param-free identity-concat merge ``y = [I | I] @ [z ; F]`` (signed, no bias, identity activation).

    Installs the frozen identity-concat weight bank; the merge inherits its scales from the
    normal scale-propagation pass so the same scale governs torch NF and the deployed HCM neuron.
    """
    merge = Perceptron(
        output_channels=width,
        input_features=2 * width,
        bias=False,
        name=name,
    )
    eye = torch.eye(width)
    weight = torch.cat([eye, eye], dim=1)
    merge.layer.weight.data = weight
    merge.layer.weight.requires_grad_(False)
    merge.base_activation = nn.Identity()
    merge.activation = nn.Identity()
    return merge


def _rebind_source(consumer, old, new) -> bool:
    """Re-point every reference to ``old`` in ``consumer``'s source containers to
    ``new``. Returns True if any reference was changed."""
    changed = False
    container = getattr(consumer, "_source_mapper_container", None)
    if container is not None:
        for i, s in enumerate(container):
            if s is old:
                container[i] = new
                changed = True
    for attr in ("_sources_list", "_source_mappers_list"):
        lst = getattr(consumer, attr, None)
        if lst is not None:
            for i, s in enumerate(lst):
                if s is old:
                    lst[i] = new
                    changed = True
    return changed


def lower_residual_adds_to_onchip_merge(model_repr) -> int:
    """Rewrite every param-free equal-width residual add to an on-chip merge Perceptron; return the count.

    Mutates the mapper graph in place and invalidates the cached exec graph. Unequal-width
    adds (projection residuals) are left as host ComputeOps."""
    model_repr._ensure_exec_graph()
    exec_order = list(model_repr._exec_order)
    deps = model_repr._deps

    consumers: dict = {n: [] for n in exec_order}
    for node in exec_order:
        for dep in deps.get(node, []):
            if dep in consumers:
                consumers[dep].append(node)

    output = model_repr.output_layer_mapper
    lowered = 0
    for node in exec_order:
        if not _is_unbound_add(node):
            continue
        branches = node.get_source_mappers()
        if len(branches) != 2:
            continue
        widths = [_branch_width(b) for b in branches]
        if widths[0] is None or widths[0] != widths[1]:
            continue
        width = widths[0]

        base = node.name or "residual"
        concat = _ResidualConcatMapper(branches, name=f"{base}_concat")
        merge = _build_merge_perceptron(width, name=f"{base}{MERGE_NAME_SUFFIX}")
        merge_mapper = PerceptronMapper(concat, merge)

        for consumer in consumers.get(node, []):
            _rebind_source(consumer, node, merge_mapper)
        if node is output:
            model_repr.output_layer_mapper = merge_mapper
        lowered += 1

    if lowered:
        model_repr._exec_order = None
        model_repr._deps = None
        model_repr._consumer_count = None
        model_repr._ensure_exec_graph()
    return lowered
