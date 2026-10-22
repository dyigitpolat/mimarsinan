"""E270's declared rounding bound tau(P) = 64 u max(D(P), 16), and stage 2's reassociation bound beta(P)."""

from __future__ import annotations

import operator
from typing import Callable, Dict, List, Mapping, Optional

import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F

UNIT_ROUNDOFF = 2.0 ** -53
"""The float64 unit roundoff u."""

EXACTNESS_ACCEPTANCE = 1e-12
"""The owner's acceptance: a conversion equals its native model within 1e-12 (fp64), or it is refused by name."""

# Contractions that are not module calls; each is one weight layer of the path it lies on.
_CONTRACTION_FUNCTIONS = frozenset({
    torch.matmul, operator.matmul, torch.mm, torch.bmm, torch.einsum, torch.addmm, torch.baddbmm,
    F.linear, F.bilinear, F.conv1d, F.conv2d, F.conv3d, F.conv_transpose1d, F.conv_transpose2d,
    F.conv_transpose3d,
})
_CONTRACTION_METHODS = frozenset({"matmul", "mm", "bmm", "__matmul__"})


def declared_rounding_bound(weight_layer_depth: int) -> float:
    """tau(P) for a program whose longest input-to-output path holds ``weight_layer_depth`` weight layers."""
    return 64.0 * UNIT_ROUNDOFF * max(int(weight_layer_depth), 16)


def conversion_value_bound(weight_layer_depth: int) -> float:
    """Stage 1's refusal line: tau(P), and never above the owner's 1e-12 acceptance.

    The converted model keeps the native operation order (a fold or a fused pair reassociates within one
    weight layer), so the declared rounding bound of E270 applies.
    """
    return min(declared_rounding_bound(weight_layer_depth), EXACTNESS_ACCEPTANCE)


def lowering_value_bound(weight_layer_depth: int, reassociation: Optional[float]) -> float:
    """A lowering stage's refusal line: tau(P) + beta(P), never above the owner's 1e-12 acceptance.

    Cores sum every dot product in their own order; beta(P) (``reassociation_bound``) carries that
    reassociation to the output, a host op after a cancelling sum (a division) amplifying it by the
    data's condition number. ``None`` (not derived) leaves tau(P).
    """
    return min(declared_rounding_bound(weight_layer_depth) + (reassociation or 0.0), EXACTNESS_ACCEPTANCE)


def gamma(n: int) -> float:
    """gamma_n = n u / (1 - n u): the relative error bound of an n-term float64 sum in any order (Higham 3.1)."""
    nu = int(n) * UNIT_ROUNDOFF
    return nu / (1.0 - nu)


_CORE_MODULES = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)
REASSOCIATION_OUTPUT_BUDGET = 4096
"""Per-sample output elements beyond which beta(P) is not derived (one reverse pass per element)."""


def _reassociation_radius(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """2 gamma_n (sum_i |w_i x_i| + |b|) per output element of a weight layer at its input ``x``."""
    if not isinstance(module, _CORE_MODULES):
        raise TypeError(f"{type(module).__name__} is not a weight layer")
    weight = module.weight.abs()
    bias = None if module.bias is None else module.bias.abs()
    radius = 2.0 * gamma(module.weight[0].numel() + int(bias is not None))
    if isinstance(module, nn.Linear):
        return radius * F.linear(x.abs(), weight, bias)
    return radius * module._conv_forward(x.abs(), weight, bias)


def reassociation_bound(
    reference: nn.Module, inputs: torch.Tensor, select_output: Callable[[object], torch.Tensor],
) -> Optional[float]:
    """beta(P): max over output elements of sum_k |d out / d h_k| 2 gamma_n (|W||x| + |b|), over max(1, max|ref|).

    h_k is each weight layer's output (a core of the lowering; n its fan-in plus the bias), measured on the
    witness ``inputs``: two summation orders of one dot product differ by at most 2 gamma_n sum|w_i x_i|, and
    first-order propagation carries that to the output. ``None`` beyond the output budget.
    """
    offsets: List[torch.Tensor] = []
    radii: List[torch.Tensor] = []

    def perturb(module: nn.Module, args, output: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            radii.append(_reassociation_radius(module, args[0]))
        offsets.append(torch.zeros_like(output, requires_grad=True))
        return output + offsets[-1]

    handles = [m.register_forward_hook(perturb) for m in reference.modules() if isinstance(m, _CORE_MODULES)]
    try:
        with torch.enable_grad():
            out = select_output(reference(inputs))
            flat = out.reshape(out.shape[0], -1)
            if not offsets or flat.shape[1] > REASSOCIATION_OUTPUT_BUDGET:
                return None if offsets else 0.0
            bound = torch.zeros_like(flat.detach())
            for j in range(flat.shape[1]):
                grads = torch.autograd.grad(flat[:, j].sum(), offsets, retain_graph=True, allow_unused=True)
                for grad, radius in zip(grads, radii):
                    if grad is not None:
                        bound[:, j] += (grad.abs() * radius).reshape(radius.shape[0], -1).sum(1)
    finally:
        for handle in handles:
            handle.remove()
    values = flat.detach()
    finite = torch.isfinite(values) & torch.isfinite(bound)
    if not bool(finite.any()):
        return None
    reference_max = float(values[torch.isfinite(values)].abs().max())
    return float(bound[finite].max()) / max(1.0, reference_max)


def _weight_layers(node: fx.Node, modules: Mapping[str, nn.Module]) -> int:
    if node.op == "call_module":
        module = modules.get(str(node.target))
        if module is None:
            return 0
        return sum(1 for p in module.parameters() if p.dim() >= 2 and p.is_floating_point())
    if node.op == "call_function":
        return int(node.target in _CONTRACTION_FUNCTIONS)
    if node.op == "call_method":
        return int(node.target in _CONTRACTION_METHODS)
    return 0


def weight_layer_depth(gm: fx.GraphModule) -> int:
    """D(P): the largest count of weight layers on any input-to-output path of the traced graph."""
    modules = dict(gm.named_modules())
    depth: Dict[fx.Node, int] = {}
    for node in gm.graph.nodes:
        upstream = [depth[arg] for arg in node.all_input_nodes if arg in depth]
        depth[node] = _weight_layers(node, modules) + max(upstream, default=0)
    return max(depth.values(), default=0)
