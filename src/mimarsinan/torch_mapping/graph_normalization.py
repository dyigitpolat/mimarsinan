"""Graph normalization passes that fuse consecutive MM ops to maximize Perceptron packaging."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.fx as fx

from mimarsinan.torch_mapping.converter_handlers.symbolic.operand_writes import RUN_AFTER, observable_write
from mimarsinan.torch_mapping.fx_rewrites.batch_axis_push_forward import push_batch_axis_forward
from mimarsinan.torch_mapping.fx_rewrites.chunk_to_narrow import split_chunks
from mimarsinan.torch_mapping.fx_rewrites.rewrite_support import write_ordered
from mimarsinan.torch_mapping.fx_rewrites.sdpa_decomposition import decompose_sdpa
from mimarsinan.torch_mapping.fx_shape_utils import node_output_shape, node_target_str, shared_state_sites
from mimarsinan.torch_mapping.representability_analyzer import BATCHNORM_TYPES, batchnorm_folds_into


_MM_MODULES = (nn.Linear, nn.Conv1d, nn.Conv2d)


def _get_sole_user_module(
    node: fx.Node,
    modules: dict[str, nn.Module],
) -> tuple[Optional[fx.Node], Optional[nn.Module]]:
    """Return (user_node, user_module) if ``node`` has exactly one user that is a call_module."""
    users = list(node.users)
    if len(users) != 1:
        return None, None
    user = users[0]
    if user.op != "call_module":
        return None, None
    mod = modules.get(node_target_str(user))
    return user, mod


def _find_next_linear_through_foldables(
    node: fx.Node,
    modules: dict[str, nn.Module],
) -> tuple[Optional[fx.Node], list[tuple[fx.Node, nn.Module]]]:
    """Walk forward through Identity and exactly-foldable BatchNorm modules to the next Linear.

    Returns ``(next_linear_node, chain)`` where *chain* lists the intermediate
    ``(node, module)`` pairs; ``(None, [])`` if no fusable Linear is reachable.
    """
    linear = modules[node_target_str(node)]
    shape = node_output_shape(node)
    rank = len(shape) if shape is not None else None
    chain: list[tuple[fx.Node, nn.Module]] = []
    current = node
    while True:
        user_node, user_mod = _get_sole_user_module(current, modules)
        if user_node is None or user_mod is None:
            return None, []
        if isinstance(user_mod, nn.Linear):
            return user_node, chain
        foldable = isinstance(user_mod, nn.Identity) or (
            isinstance(user_mod, BATCHNORM_TYPES) and batchnorm_folds_into(user_mod, linear, rank)
        )
        if not foldable:
            return None, []
        chain.append((user_node, user_mod))
        current = user_node


def _fold_bn_into_affine(
    weight: torch.Tensor, bias: torch.Tensor, bn_mod: nn.BatchNorm1d | nn.BatchNorm2d,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``BN(W@x+b) = (diag(γ/σ)@W)@x + (γ/σ*(b−μ)+β)``, as new tensors (the source Linear is never written)."""
    mean, var = bn_mod.running_mean, bn_mod.running_var
    if mean is None or var is None:
        raise ValueError(f"cannot fold a BatchNorm without running statistics: {bn_mod}")
    gamma = (
        bn_mod.weight.detach()
        if bn_mod.weight is not None
        else torch.ones(weight.shape[0], device=weight.device, dtype=weight.dtype)
    )
    beta = (
        bn_mod.bias.detach()
        if bn_mod.bias is not None
        else torch.zeros(weight.shape[0], device=weight.device, dtype=weight.dtype)
    )
    scale = gamma / torch.sqrt(var + bn_mod.eps)
    return scale.unsqueeze(1) * weight, scale * (bias - mean) + beta


def _weight_and_bias(linear: nn.Linear) -> tuple[torch.Tensor, torch.Tensor]:
    weight = linear.weight.detach()
    if linear.bias is not None:
        return weight, linear.bias.detach()
    return weight, torch.zeros(weight.shape[0], device=weight.device, dtype=weight.dtype)


def _fuse_linear_pair(
    gm: fx.GraphModule,
    node1: fx.Node,
    mod1: nn.Linear,
    chain_bns: list[nn.BatchNorm1d | nn.BatchNorm2d],
    node2: fx.Node,
    mod2: nn.Linear,
) -> None:
    """Fuse ``mod2 ∘ BN* ∘ mod1`` into one new Linear at ``node2``: W = W2 @ W1', b = W2 @ b1' + b2."""
    with torch.no_grad():
        W1, b1 = _weight_and_bias(mod1)
        for bn_mod in chain_bns:
            W1, b1 = _fold_bn_into_affine(W1, b1, bn_mod)
        W2, b2 = _weight_and_bias(mod2)

        W_fused = W2 @ W1
        b_fused = W2 @ b1 + b2

    fused = nn.Linear(W_fused.shape[1], W_fused.shape[0], bias=True, device=W_fused.device, dtype=W_fused.dtype)
    with torch.no_grad():
        fused.weight.copy_(W_fused)
        fused.bias.copy_(b_fused)

    # A fresh target: node2's module may be called at other sites, which must keep their own weights.
    fused_target = _fresh_target(gm, f"{node2.name}_fused")
    gm.add_submodule(fused_target, fused)
    node2.target = fused_target

    node2.args = node1.args


def _fresh_target(gm: fx.GraphModule, base: str) -> str:
    taken = dict(gm.named_modules())
    target, k = base, 0
    while target in taken or hasattr(gm, target):
        k += 1
        target = f"{base}_{k}"
    return target


def normalize_fx_graph(gm: fx.GraphModule, foreign_modules: frozenset[str] = frozenset(),
                       decompose_leaf_modules: bool = False) -> fx.GraphModule:
    """Run in-place normalization passes: the FX rewrites (``fx_rewrites/``; the attention one under
    ``decompose_leaf_modules``, first: its call has no declared batch rule, so what it feeds is provable only once it is
    rewritten), consecutive-Linear fusion, then dead-code elimination, which keeps an observable write in place
    (``operand_writes``). A module of ``foreign_modules`` (outside the declared operator set, ``symbolic.purity``) is
    never fused: fusion computes with its state at conversion."""
    writes = write_ordered(gm)
    if decompose_leaf_modules:
        decompose_sdpa(gm, writes)
    split_chunks(gm, writes)
    push_batch_axis_forward(gm, writes)
    modules: dict[str, nn.Module] = dict(gm.named_modules())
    graph = gm.graph

    # A node ordered against a write in place (``operand_writes.RUN_AFTER``) keeps its own call: a fusion would read the
    # first Linear's operand at the second's place, which may come after the write (review round 16, targeted pass).
    ordered = {name for node in graph.nodes for name in (node.name, *node.meta.get(RUN_AFTER, ()))
               if node.meta.get(RUN_AFTER)}
    fused = True
    while fused:
        fused = False
        modules = dict(gm.named_modules())
        # A fusion would consume one call site of shared state and hide the share from the D13 census.
        shared_sites = {site for _node, sites in shared_state_sites(gm, modules) for site in sites}
        for node in list(graph.nodes):
            if node.op != "call_module":
                continue
            mod = modules.get(node_target_str(node))
            if not isinstance(mod, nn.Linear):
                continue

            next_node, chain = _find_next_linear_through_foldables(node, modules)
            if next_node is None:
                continue
            next_mod = modules.get(node_target_str(next_node))
            if not isinstance(next_mod, nn.Linear):
                continue

            if mod.out_features != next_mod.in_features:
                continue
            if shared_sites.intersection([node.name, next_node.name, *(n.name for n, _m in chain)]):
                continue
            if foreign_modules.intersection(str(n.target) for n in (node, next_node, *(n for n, _m in chain))):
                continue
            if ordered.intersection(n.name for n in (node, next_node, *(n for n, _m in chain))):
                continue

            chain_bns = [m for _chain_node, m in chain if isinstance(m, BATCHNORM_TYPES)]
            _fuse_linear_pair(gm, node, mod, chain_bns, next_node, next_mod)
            fused = True
            break

    # A write in place whose operand a later node reads is kept (``operand_writes``; FX's own rule keeps an aten
    # operator's but drops ``torch.resize_as_(z, y)``); after the tracer's rewrite none is left but through its result.
    order = {node: index for index, node in enumerate(graph.nodes)}
    modules = dict(gm.named_modules())
    graph.eliminate_dead_code(is_impure_node=lambda node: node.is_impure() or observable_write(node, modules, order))
    gm.recompile()

    return gm
