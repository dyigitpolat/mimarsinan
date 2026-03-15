"""
Graph normalization passes for FX-traced models.

Runs BEFORE representability analysis and conversion to maximize Perceptron
packaging.  Each pass operates on the GraphModule in-place.

Passes:
  1. **Dead code elimination**: Remove unused nodes.
  2. **Consecutive mm fusion**: Fuse `Linear → Linear` (with only Identity/BN
     in between) into a single Linear so the combined op can absorb a
     downstream activation and become one Perceptron.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.fx as fx


# Module types that represent matmul-capable ops.
_MM_MODULES = (nn.Linear, nn.Conv1d, nn.Conv2d)

# Module types that can be skipped through when looking for a consecutive mm.
_PASSTHROUGH_MODULES = (nn.Identity,)


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
    mod = modules.get(user.target)
    return user, mod


def _walk_through_passthrough(
    node: fx.Node,
    modules: dict[str, nn.Module],
) -> Optional[fx.Node]:
    """Walk forward through passthrough modules (Identity) to find the next non-trivial node.

    Returns the first non-passthrough user node, or None if the chain ends.
    Each node in the chain must have exactly one user (no branching).
    """
    current = node
    while True:
        user_node, user_mod = _get_sole_user_module(current, modules)
        if user_node is None or user_mod is None:
            return None
        if not isinstance(user_mod, _PASSTHROUGH_MODULES):
            return user_node
        current = user_node


def _fuse_linear_pair(
    gm: fx.GraphModule,
    node1: fx.Node,
    mod1: nn.Linear,
    node2: fx.Node,
    mod2: nn.Linear,
) -> None:
    """Fuse two consecutive Linear layers: W_fused = W2 @ W1, b_fused = W2 @ b1 + b2."""
    with torch.no_grad():
        W1 = mod1.weight.data  # (mid, in)
        W2 = mod2.weight.data  # (out, mid)
        b1 = mod1.bias.data if mod1.bias is not None else torch.zeros(W1.shape[0])
        b2 = mod2.bias.data if mod2.bias is not None else torch.zeros(W2.shape[0])

        W_fused = W2 @ W1  # (out, in)
        b_fused = W2 @ b1 + b2  # (out,)

    fused = nn.Linear(W_fused.shape[1], W_fused.shape[0], bias=True)
    with torch.no_grad():
        fused.weight.copy_(W_fused)
        fused.bias.copy_(b_fused)

    # Register fused module, replacing mod2's target
    fused_target = node2.target
    gm.add_submodule(fused_target, fused)

    # Rewire: node2 now takes node1's INPUT (skip node1 entirely)
    node2.args = node1.args

    # Remove intermediate nodes (node1 and passthrough nodes between them)
    # by making them unused — DCE will clean them up.


def normalize_fx_graph(gm: fx.GraphModule) -> fx.GraphModule:
    """Run graph normalization passes on a traced GraphModule.

    Currently implements:
    1. Consecutive Linear fusion (through Identity passthrough)
    2. Dead code elimination
    """
    modules = dict(gm.named_modules())
    graph = gm.graph

    # Pass 1: Fuse consecutive Linear layers connected through passthrough ops.
    # Iterate until no more fusions are possible (fixpoint).
    fused = True
    while fused:
        fused = False
        for node in list(graph.nodes):
            if node.op != "call_module":
                continue
            mod = modules.get(node.target)
            if not isinstance(mod, nn.Linear):
                continue

            # Walk forward through passthrough to find next mm
            next_node = _walk_through_passthrough(node, modules)
            if next_node is None:
                continue
            next_mod = modules.get(next_node.target)
            if not isinstance(next_mod, nn.Linear):
                continue

            # Dimensions must be compatible
            if mod.out_features != next_mod.in_features:
                continue

            _fuse_linear_pair(gm, node, mod, next_node, next_mod)
            fused = True
            break  # restart iteration after mutation

    # Pass 2: Dead code elimination
    graph.eliminate_dead_code()
    gm.recompile()

    # Refresh modules dict after mutations
    return gm
