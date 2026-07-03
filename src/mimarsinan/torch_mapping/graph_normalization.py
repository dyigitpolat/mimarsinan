"""Graph normalization passes that fuse consecutive MM ops to maximize Perceptron packaging."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.fx as fx

from mimarsinan.torch_mapping.fx_shape_utils import node_target_str


_MM_MODULES = (nn.Linear, nn.Conv1d, nn.Conv2d)

_FOLDABLE_MODULES = (nn.Identity, nn.BatchNorm1d, nn.BatchNorm2d)


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
    """Walk forward through foldable modules to the next Linear.

    Returns ``(next_linear_node, chain)`` where *chain* lists the intermediate
    ``(node, module)`` pairs; ``(None, [])`` if no fusable Linear is reachable.
    """
    chain: list[tuple[fx.Node, nn.Module]] = []
    current = node
    while True:
        user_node, user_mod = _get_sole_user_module(current, modules)
        if user_node is None or user_mod is None:
            return None, []
        if isinstance(user_mod, nn.Linear):
            return user_node, chain
        if isinstance(user_mod, _FOLDABLE_MODULES):
            chain.append((user_node, user_mod))
            current = user_node
        else:
            return None, []


def _fold_bn_into_linear(linear_mod: nn.Linear, bn_mod: nn.Module) -> None:
    """Fold BatchNorm parameters into a preceding Linear in-place.

    ``BN(W@x+b) = (diag(γ/σ)@W)@x + (γ/σ*(b−μ)+β)``.
    """
    if not isinstance(bn_mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
        return
    if bn_mod.running_mean is None or bn_mod.running_var is None:
        return

    with torch.no_grad():
        W = linear_mod.weight.data
        b = (
            linear_mod.bias.data
            if linear_mod.bias is not None
            else torch.zeros(W.shape[0], device=W.device, dtype=W.dtype)
        )

        gamma = (
            bn_mod.weight.data
            if bn_mod.weight is not None
            else torch.ones(W.shape[0], device=W.device, dtype=W.dtype)
        )
        beta = (
            bn_mod.bias.data
            if bn_mod.bias is not None
            else torch.zeros(W.shape[0], device=W.device, dtype=W.dtype)
        )
        mean = bn_mod.running_mean
        var = bn_mod.running_var
        eps = bn_mod.eps

        scale = gamma / torch.sqrt(var + eps)

        linear_mod.weight.data = scale.unsqueeze(1) * W
        new_bias = scale * (b - mean) + beta
        if linear_mod.bias is not None:
            linear_mod.bias.data = new_bias
        else:
            linear_mod.bias = nn.Parameter(new_bias)


def _fuse_linear_pair(
    gm: fx.GraphModule,
    node1: fx.Node,
    mod1: nn.Linear,
    node2: fx.Node,
    mod2: nn.Linear,
) -> None:
    """Fuse two consecutive Linear layers: W_fused = W2 @ W1, b_fused = W2 @ b1 + b2."""
    with torch.no_grad():
        W1 = mod1.weight.data
        W2 = mod2.weight.data
        b1 = mod1.bias.data if mod1.bias is not None else torch.zeros(W1.shape[0])
        b2 = mod2.bias.data if mod2.bias is not None else torch.zeros(W2.shape[0])

        W_fused = W2 @ W1
        b_fused = W2 @ b1 + b2

    fused = nn.Linear(W_fused.shape[1], W_fused.shape[0], bias=True)
    with torch.no_grad():
        fused.weight.copy_(W_fused)
        fused.bias.copy_(b_fused)

    fused_target = node_target_str(node2)
    gm.add_submodule(fused_target, fused)

    node2.args = node1.args


def normalize_fx_graph(gm: fx.GraphModule) -> fx.GraphModule:
    """Run in-place normalization passes: consecutive-Linear fusion then dead-code elimination."""
    modules: dict[str, nn.Module] = dict(gm.named_modules())
    graph = gm.graph

    fused = True
    while fused:
        fused = False
        modules = dict(gm.named_modules())
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

            for _chain_node, chain_mod in chain:
                if isinstance(chain_mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    _fold_bn_into_linear(mod, chain_mod)

            _fuse_linear_pair(gm, node, mod, next_node, next_mod)
            fused = True
            break

    graph.eliminate_dead_code()
    gm.recompile()

    return gm
