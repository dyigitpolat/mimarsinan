"""The leaf decomposition registry: which leaf instances ``decompose_leaf_modules`` replaces, and the replaced model.

The leaf program is traced once (``trace_model``) to read how each registered leaf is called. An instance is replaced
when its exact type is registered, it is called once, its call and every submodule's runs its class's ``forward`` alone
(``symbolic.module_call.call_reason``: no forward hook, no forward bound on the instance, no compiled call), its
configuration is one the decomposed module computes (no ``bias_k``/``bias_v``, no zero attention, an encoder layer's
activation one torch's eval fast path computes as the layer does), no mask reaches it and the attention weights it
returns are never read; every other instance stays the leaf, the host op of its own call. A boolean mask
computed from activations is refused by name (``ActivationMaskError``). The replaced model is a structural copy of
the user's: each module on a path to a replaced leaf is copied with its own submodule table, every parameter and
buffer shared by reference, so the user's model is never changed.
"""

from __future__ import annotations

import copy
import inspect
import operator
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, cast

import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.torch_mapping.converter_handlers.shape_value_mixin import depends_on_input
from mimarsinan.torch_mapping.converter_handlers.symbolic.module_call import call_reason
from mimarsinan.torch_mapping.leaf_decomposition.attention import (
    DecomposedAttention,
    DecomposedEncoder,
    DecomposedEncoderLayer,
)
from mimarsinan.torch_mapping.leaf_decomposition.recurrence import lowers_recurrent_leaf
from mimarsinan.torch_mapping.torch_graph_tracer import trace_model

_MASKS = {nn.MultiheadAttention: ("key_padding_mask", "attn_mask"),
          nn.TransformerEncoderLayer: ("src_mask", "src_key_padding_mask"),
          nn.TransformerEncoder: ("mask", "src_key_padding_mask")}

DECOMPOSITIONS: Dict[type, Callable[[Any], nn.Module]] = {
    nn.MultiheadAttention: DecomposedAttention,
    nn.TransformerEncoderLayer: DecomposedEncoderLayer,
    nn.TransformerEncoder: DecomposedEncoder,
}
"""Registered leaf types and the traceable module each is replaced by before tracing (``TransformerDecoderLayer``
stays a leaf)."""

CONVERSION_LOWERINGS: Dict[type, Callable[[nn.Module, bool, Optional[int]], bool]] = {
    kind: lowers_recurrent_leaf for kind in (nn.LSTM, nn.GRU, nn.RNN)
}
"""Registered leaf types lowered at conversion instead (``recurrence.py``: Wave 0's frozen batch domain declares batch
rules for torch's own recurrent modules only, so a replacement traced in their place would have none), each with the
predicate of the calls its lowering takes (the module, one activation operand, the tuple element read). With
``DECOMPOSITIONS`` it is the one rule set of ``decompose_leaf_modules``; the converter's module-call dispatch reads it
through ``lowered_at_conversion``."""


def lowered_at_conversion(module: nn.Module, single_source: bool, carried: Optional[int]) -> bool:
    """``module``'s call (one activation operand when ``single_source``, ``carried`` its tuple element read) is one
    the conversion-time lowering of its registered type takes."""
    takes = CONVERSION_LOWERINGS.get(type(module))
    return takes is not None and takes(module, single_source, carried)


class ActivationMaskError(ToolchainRefusal, ValueError):
    """A boolean mask computed from activations reaches an attention the conversion decomposes (or a scaled dot
    product attention): its host op would carry the mask in the program's value dtype, which changes its meaning."""

    def __init__(self, operator_name: str, node_name: str, mask: str):
        self.operator, self.node_name, self.mask = operator_name, node_name, mask
        super().__init__(f"{operator_name} (FX node {node_name!r}): its {mask} is a boolean mask computed from "
                         "activations; host ops exchange activations in the program's value dtype, so the conversion "
                         "is refused")


@dataclass
class Decomposition:
    """The replaced model, and per replaced leaf parameter (its qualified name) the replaced model's parameters that
    view its rows, stacked along axis 0 (``reference.decomposition_reference`` reads it)."""

    model: nn.Module
    stacked: Dict[str, List[str]] = field(default_factory=dict)
    replaced: Tuple[str, ...] = ()


def _bound(module: nn.Module, node: fx.Node) -> Mapping[str, Any]:
    call = inspect.signature(type(module).forward).bind(module, *node.args, **node.kwargs)
    call.apply_defaults()
    return call.arguments


def refuse_activation_mask(operator_name: str, node: fx.Node, mask: str, value: Any) -> None:
    """``ActivationMaskError`` when ``value`` is a boolean FX value computed from activations."""
    meta = value.meta.get("tensor_meta") if isinstance(value, fx.Node) else None
    if getattr(meta, "dtype", None) is torch.bool and depends_on_input(value):
        raise ActivationMaskError(operator_name, node.name, mask)


def _fast_path_activation(layer: nn.Module) -> bool:
    """The layer's activation is the function torch's eval fast path computes for it: the fast path reads
    ``activation_relu_or_gelu`` (set by ``isinstance``) and computes exact ReLU or GELU, so a tanh GELU or a ReLU or GELU
    subclass is another function there than in the slow path the decomposition follows; with the flag 0 there is no
    fast path."""
    activation = getattr(layer, "activation", None)
    if not getattr(layer, "activation_relu_or_gelu", 0):
        return True
    return activation is F.relu or activation is F.gelu or type(activation) is nn.ReLU or (
        type(activation) is nn.GELU and getattr(activation, "approximate", "none") == "none")


def _decomposable(module: nn.Module) -> bool:
    """A configuration the decomposed module computes: exact registered types all the way down, no ``bias_k`` and
    ``bias_v``, no zero attention, an encoder layer's activation the one its fast path computes."""
    kind = type(module)
    if kind is nn.MultiheadAttention:
        return module.bias_k is None and not module.add_zero_attn
    if kind is nn.TransformerEncoderLayer:
        return _fast_path_activation(module) and _decomposable(cast(nn.Module, module.self_attn))
    return kind is nn.TransformerEncoder and all(type(layer) is nn.TransformerEncoderLayer and _decomposable(layer)
                                                 for layer in cast(nn.ModuleList, module.layers))


def _eligible(module: nn.Module, node: fx.Node, calls: int) -> bool:
    """``module``, called at ``node``, is replaced (the module docstring); a boolean activation mask refuses."""
    arguments = _bound(module, node)
    for mask in _MASKS[type(module)]:
        refuse_activation_mask(type(module).__name__, node, mask, arguments.get(mask))
    weights_read = any(user.target is operator.getitem and user.args[1] == 1 and user.users for user in node.users)
    unmasked = all(arguments.get(mask) is None for mask in _MASKS[type(module)]) and not arguments.get("is_causal")
    forward_alone = all(call_reason(m) is None for m in module.modules())
    return calls == 1 and unmasked and forward_alone and not weights_read and _decomposable(module)


def _rows(part: torch.Tensor, whole: torch.Tensor) -> Optional[int]:
    """The first row of ``whole`` that ``part`` views (its storage, its trailing layout), ``None`` when it views none."""
    if part.untyped_storage().data_ptr() != whole.untyped_storage().data_ptr() or part.dim() != whole.dim() \
            or part.shape[1:] != whole.shape[1:] or part.stride() != whole.stride():
        return None
    step = int(whole.stride(0)) if whole.dim() else 1
    start, rest = divmod(int(part.storage_offset()) - int(whole.storage_offset()), step)
    return start if rest == 0 and 0 <= start and start + part.shape[0] <= whole.shape[0] else None


def provenance(leaf: nn.Module, replacement: nn.Module) -> Dict[str, List[str]]:
    """Per parameter of ``leaf`` the replacement's parameters that view its rows, in row order, read from their
    storage (never from their names): the probe's reference stacks them back into the leaf's own parameter, so a
    decomposition putting a row in the wrong projection computes another function than the reference. A parameter
    reused whole under its own name is omitted; one whose rows the views do not tile exactly refuses."""
    found: Dict[str, List[Tuple[int, str]]] = {}
    natives = dict(leaf.named_parameters())
    for name, part in replacement.named_parameters():
        for native_name, whole in natives.items():
            start = 0 if part is whole else _rows(part, whole)
            if start is not None:
                found.setdefault(native_name, []).append((start, name))
                break
    stacked: Dict[str, List[str]] = {}
    for native_name, parts in found.items():
        parts.sort()
        sizes = [int(dict(replacement.named_parameters())[name].shape[0]) for _start, name in parts]
        tiled = [start for start, _name in parts] == [sum(sizes[:i]) for i in range(len(sizes))] and \
            sum(sizes) == natives[native_name].shape[0]
        if not tiled:
            raise ValueError(f"the decomposition of {type(leaf).__name__} does not tile {native_name!r} exactly")
        if [name for _start, name in parts] != [native_name]:
            stacked[native_name] = [name for _start, name in parts]
    return stacked


def _replace(root: nn.Module, path: str, module: nn.Module) -> nn.Module:
    """``root`` with the submodule at ``path`` replaced by ``module``; each module on the path copied with its own
    submodule table, nothing else copied."""
    head, _, rest = path.partition(".")
    clone = copy.copy(root)
    object.__setattr__(clone, "_modules", dict(vars(root)["_modules"]))
    clone._modules[head] = module if not rest else _replace(cast(nn.Module, root._modules[head]), rest, module)
    return clone


def decompose_leaves(model: nn.Module, input_shape: Tuple[int, ...], device: Any = "cpu") -> Decomposition:
    """``model`` with every eligible registered leaf replaced (the module docstring); a model with no registered leaf
    is returned as it is, untraced."""
    if not any(type(module) in DECOMPOSITIONS for module in model.modules()):
        return Decomposition(model)
    graph = trace_model(model, tuple(input_shape), device=device).graph
    modules = dict(model.named_modules())
    sites: Dict[str, List[fx.Node]] = {}
    for node in graph.nodes:
        if node.op == "call_module" and type(modules.get(str(node.target))) in DECOMPOSITIONS:
            sites.setdefault(str(node.target), []).append(node)
    decomposed = Decomposition(model)
    for path, nodes in sites.items():
        leaf = modules[path]
        if not all(_eligible(leaf, node, len(nodes)) for node in nodes):
            continue
        replacement = DECOMPOSITIONS[type(leaf)](leaf)
        decomposed.model = _replace(decomposed.model, path, replacement)
        for name, parts in provenance(leaf, replacement).items():
            decomposed.stacked[f"{path}.{name}"] = [f"{path}.{part}" for part in parts]
        decomposed.replaced += (path,)
    return decomposed
