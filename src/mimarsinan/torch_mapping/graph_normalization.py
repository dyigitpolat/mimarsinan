"""Graph normalization passes that fuse consecutive MM ops to maximize Perceptron packaging."""

from __future__ import annotations

import operator
from typing import Optional

import torch
import torch.nn as nn
import torch.fx as fx

from mimarsinan.torch_mapping.fx_shape_utils import node_target_str
from mimarsinan.torch_mapping.representability_analyzer import (
    OpInfo,
    RepresentabilityError,
    RepresentabilityReport,
)


_MM_MODULES = (nn.Linear, nn.Conv1d, nn.Conv2d)

_FOLDABLE_MODULES = (nn.Identity, nn.BatchNorm1d, nn.BatchNorm2d)

# A write through any of these still lands on the tensor it was taken from, so the
# mutation walk follows them back to the module attribute they alias.
_ALIASING_METHODS = frozenset({
    "view", "reshape", "flatten", "permute", "transpose", "t", "squeeze", "unsqueeze",
    "contiguous", "detach", "narrow", "select", "expand", "__getitem__",
})
_INPLACE_DUNDER_METHODS = frozenset({
    "__setitem__", "__iadd__", "__isub__", "__imul__", "__itruediv__", "__ifloordiv__",
    "__imod__", "__ipow__", "__iand__", "__ior__", "__ixor__", "__ilshift__", "__irshift__",
})
_INPLACE_FUNCTIONS = (
    operator.setitem, operator.iadd, operator.isub, operator.imul, operator.itruediv,
    operator.ifloordiv, operator.imod, operator.ipow, operator.iand, operator.ior,
    operator.ixor, operator.ilshift, operator.irshift, setattr,
)


def _module_state_root(arg: object) -> Optional[fx.Node]:
    """The ``get_attr`` node (a buffer, parameter or baked constant) ``arg`` aliases, if any."""
    node = arg
    seen: set[fx.Node] = set()
    while isinstance(node, fx.Node) and node not in seen:
        seen.add(node)
        if node.op == "get_attr":
            return node
        aliases = (
            node.op == "call_function" and node.target in (getattr, operator.getitem)
        ) or (
            node.op == "call_method" and node_target_str(node) in _ALIASING_METHODS
        )
        if not aliases or not node.args:
            return None
        node = node.args[0]
    return None


def _mutated_module_state(node: fx.Node) -> Optional[fx.Node]:
    """The module-state root ``node`` writes into, or ``None`` when it writes none."""
    if node.op == "call_method":
        name = node_target_str(node)
        inplace = (name.endswith("_") and not name.endswith("__")) or name in _INPLACE_DUNDER_METHODS
        if inplace and node.args:
            return _module_state_root(node.args[0])
    elif node.op == "call_function":
        if node.target in _INPLACE_FUNCTIONS and node.args:
            return _module_state_root(node.args[0])
    else:
        return None
    return _module_state_root(node.kwargs.get("out"))


def refuse_module_state_mutations(gm: fx.GraphModule) -> None:
    """Refuse a forward that writes module state (a buffer, parameter or baked constant).

    Such a write is hidden state carried across calls; it has no deployment semantics,
    and dead-code elimination would otherwise drop the unused-result update and admit a
    stateless graph silently. Raised BEFORE any normalization pass runs.
    """
    unsupported = [
        OpInfo(
            node.name, node.op, str(getattr(node.target, "__name__", node.target)),
            reason=(
                f"mutates module state '{root.target}' (get_attr node '{root.name}') inside "
                f"forward; hidden state carried across calls is not admissible, and "
                f"dead-code elimination would silently drop the update. Stage: graph "
                f"normalization, before dead-code elimination. Keep buffers and parameters "
                f"read-only in forward, or move the stateful section to the host caller."
            ),
        )
        for node in gm.graph.nodes
        if (root := _mutated_module_state(node)) is not None
    ]
    if unsupported:
        raise RepresentabilityError(
            RepresentabilityReport(is_representable=False, unsupported_ops=unsupported)
        )


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
    """Run in-place normalization passes: consecutive-Linear fusion then dead-code elimination.

    A forward that writes module state is refused first (``RepresentabilityError``),
    while the write is still in the graph.
    """
    refuse_module_state_mutations(gm)
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
