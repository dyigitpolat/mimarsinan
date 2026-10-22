"""E270's declared rounding bound tau(P) = 64 u max(D(P), 16), stage 2's reassociation bound beta(P), and the
reference's own reassociation floor a refusal is settled against."""

from __future__ import annotations

import math
import operator
from dataclasses import replace
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F

from mimarsinan.torch_mapping.value_parity.comparison import relative_error

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
    enough: Optional[float] = None,
) -> Optional[float]:
    """beta(P): max over output elements of sum_k |d out / d h_k| 2 gamma_n (|W||x| + |b|), over max(1, max|ref|).

    h_k is each weight layer's output (a core of the lowering; n its fan-in plus the bias), measured on the
    witness ``inputs``: two summation orders of one dot product differ by at most 2 gamma_n sum|w_i x_i|, and
    first-order propagation carries that to the output. ``None`` beyond the output budget. ``enough``: the first
    output column (one reverse pass each) whose bound reaches it ends the visit and is returned, a lower bound of
    beta(P) that decides a line tau(P) + beta(P) as beta(P) does.
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
            bound, values = torch.zeros_like(flat.detach()), flat.detach()
            known = values[torch.isfinite(values)]
            scale = max(1.0, float(known.abs().max())) if known.numel() else math.inf
            for j in range(flat.shape[1]):
                grads = torch.autograd.grad(flat[:, j].sum(), offsets, retain_graph=True, allow_unused=True)
                for grad, radius in zip(grads, radii):
                    if grad is not None:
                        bound[:, j] += (grad.abs() * radius).reshape(radius.shape[0], -1).sum(1)
                column = bound[:, j][torch.isfinite(values[:, j]) & torch.isfinite(bound[:, j])]
                if enough is not None and column.numel() and float(column.max()) / scale >= enough:
                    return float(column.max()) / scale
    finally:
        for handle in handles:
            handle.remove()
    finite = torch.isfinite(values) & torch.isfinite(bound)
    if not bool(finite.any()):
        return None
    return float(bound[finite].max()) / scale


REFERENCE_FLOOR_MARGIN = 4.0
"""A declared policy constant (beside ``EXACTNESS_ACCEPTANCE``), never a tolerance: an error within this multiple of
the reference's own floor is not shown to be the conversion's and is refused by its own name, nothing accepted (two
legal fp64 evaluations of one program differ by a small multiple of one another, never by orders of magnitude)."""


def _reversed_reduction(module: nn.Module, args: Tuple[Any, ...], output: torch.Tensor) -> torch.Tensor:
    """A weight layer's output summed in the reverse order: input features reversed (within each group) and the weight
    alike, the same function in exact arithmetic and one more fp64 summation order."""
    x = args[0]
    if isinstance(module, nn.Linear):
        return F.linear(x.flip(-1), module.weight.flip(1), module.bias)
    if not isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)) or x.dim() != module.weight.dim():
        return output
    shape, groups = x.shape, module.groups
    flipped = x.reshape(shape[0], groups, shape[1] // groups, *shape[2:]).flip(2).reshape(shape)
    return module._conv_forward(flipped, module.weight.flip(1), module.bias)


def reference_floor(
    program: nn.Module, inputs: torch.Tensor, select_output: Callable[[object], torch.Tensor],
) -> Optional[float]:
    """The native program's own reassociation floor on ``inputs``: the probe's metric between it and its twin whose
    every weight layer sums its reduction in the reverse order (``_reversed_reduction``); ``None`` without one.

    No conversion takes part: a program whose floor exceeds a stage's line is not reproducible within that line on
    these inputs. Weight layers called as functions are not reversed.
    """
    layers = [m for m in program.modules() if isinstance(m, _CORE_MODULES)]
    if not layers:
        return None
    with torch.no_grad():
        want = select_output(program(inputs))
        handles = [m.register_forward_hook(_reversed_reduction) for m in layers]
        try:
            twin = select_output(program(inputs))
        finally:
            for handle in handles:
                handle.remove()
    return relative_error(twin, want).relative_error


def settle_line(
    result: Any, program: nn.Module, inputs: torch.Tensor, select_output: Callable[[object], torch.Tensor],
    derive_line: Optional[Callable[[torch.Tensor, float], Tuple[float, Optional[float]]]] = None,
) -> Any:
    """A comparison above its stage's base line (a refused ``ValueParityResult``), settled on its ``inputs``.

    ``derive_line(inputs, error)`` (stage 2: tau(P) + beta(P), capped at the acceptance) is derived only for an error
    within the acceptance, the most the capped line reaches; a comparison still refused carries the reference's own
    floor (``reference_floor``), which names the refusal the reference's where the floor exceeds the line
    (``ValueParityResult.reference_undetermined``).
    """
    error = result.relative_error
    if derive_line is not None and error is not None and error <= EXACTNESS_ACCEPTANCE:
        line, beta = derive_line(inputs, error)
        result = replace(result, ok=error <= line, bound=line, reassociation_bound=beta)
    if result.ok:
        return result
    return replace(result, reference_floor=reference_floor(program, inputs, select_output))


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
