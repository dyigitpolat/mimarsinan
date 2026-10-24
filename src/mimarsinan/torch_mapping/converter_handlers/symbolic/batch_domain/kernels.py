"""Declared batch rules of torch's tensor kernels (review round 14, the owner's shape-proof domain).

A kernel's output size that depends on the batch is accepted only when it equals, in the affine batch domain, the
size its declared batch rule states; every other such size is ``Opaque``. ``rule_of`` declares, by kernel (the rules
are in ``kernel_rules.py``, ``layouts.py`` and ``selections.py``):

- ``SAME``: the first tensor operand's shape (activations, dropout, softmax, the norms, casts, ``*_like`` factories,
  cumulative and flipping kernels, ``slice_scatter`` and ``select_scatter`` (their input with a part replaced), an
  encoder or decoder layer and stack, whose first operand is ``src`` or ``tgt``);
  ``OPERAND``: a named operand's shape (``nn.Transformer``: ``tgt``'s); ``GLU``: the input's with axis ``dim`` halved;
- ``BROADCAST``: the broadcast of the tensor operands' shapes (torch's pointwise operators: the ``pointwise`` tag of
  the aten operator of the same name, and Python's arithmetic and comparison operators on tensors);
- ``REDUCE``: the first operand's shape without (or with 1 at) the reduced axes;
- ``LEADING(k)``: the first operand's leading axes pass through and its last ``k`` axes, with every other operand,
  are batch-free (linear and bilinear: 1; convolution over d spatial axes: d + 1; pooling, padding over d axes: d;
  interpolation: every spatial axis; embedding: 0): the trailing output extents are then batch-free, torch's own,
  where every operand but the kernel's tensor operands (input, weight, bias) is a batch-free number: a tensor given
  as a size (a pad amount, an output size, a class count) is data, and the rule states nothing (review round 15);
- ``MATMUL``, ``EINSUM``: the batch axes broadcast, the matrix axes pass through;
- the layout kernels of ``layouts.py`` and ``selections.py``, the factories (their size arguments), the attention and
  recurrent modules (``module_rules.py``: their output, weight and state shapes), and ``nonzero``/``masked_select`` (a
  data-dependent size).

Every rule reads its call bound to the kernel's signature (``binding.py``, review round 16): the tensor operands in the
signature's order, whatever order the call passes them in by keyword, and every keyword under the name torch's
schema gives it: torch's Python argument parser reads numpy's names ``axis``, ``keepdims``, ``x``, ``a``, ``x1`` and
``x2`` as ``dim``, ``keepdim``, ``input`` and ``other`` for every builtin function and tensor method, and so does the
binding (``binding.canonical_keywords``), once for every node of the traced graph, which every handler of the
converter reads too (review round 16, targeted pass).

Soundness lemma (kernels). Let every operand extent and scalar have its domain value's meaning at every batch B >= 1.
For a declared kernel, at every B >= 1 at which the kernel returns, each output extent the rule states equals the
rule's domain value at B: the rule is the kernel's shape function as torch defines it, written in the domain's closed
operations (a pass-through, a permutation, a sum, a product or an exact quotient of operand extents and batch-free
constants), and a batch-free extent the rule takes from torch (``LEADING``) is computed by the kernel from the
batch-free extents of its tensor operands and batch-free numbers only, never from a tensor's data (a tensor where the
kernel reads a number makes the rule state nothing), and is read from torch's FakeTensor run only: where that run
raised, every extent the rule would read from it is ``Opaque`` (the seeded traces are a sample, never a proof; review
round 15). The pass accepts a stated extent only where torch's own run agrees with it (``acceptance.py``, whose lemma
completes this one) and makes every other extent ``Opaque``. [corrected in review round 15: the lemma said "computed by
the kernel from batch-free operands only", and a 0-dim tensor given as a pad amount counted as batch-free; where torch's
run raised, the rule read that extent from the batch-1 trace] [corrected in review round 16: the rules read a call's
tensor operands in the order the call passed them, keywords in their order (``mha(key=kv, query=q, value=kv)`` stated
``kv``'s shape, ``torch.matmul(other=W, input=y)`` ``W @ y``'s, ``lstm(hx=(h0, c0), input=q)`` ``h0``'s); every
``.transformer`` and ``.activation`` module had ``SAME`` (``nn.Transformer`` stated ``src``'s shape for ``tgt``'s,
``nn.GLU`` its input's for the halved one); ``einsum`` took a letter's first extent other than 1 where torch broadcasts
it. Now every call is bound to its signature (``binding.py``), ``nn.Transformer`` has ``OPERAND`` (``tgt``) and
``nn.GLU`` and ``F.glu`` ``GLU``, a letter's extents broadcast; and every stated extent is cross-checked at two batches
at least (``acceptance.py``)] [corrected in review round 16, targeted pass: the binding read torch's numpy names as
names of no parameter, so a rule read ``dim`` as absent (``torch.sum(y, axis=1)`` a full reduction,
``torch.squeeze(y, axis=1)`` every axis squeezed, ``torch.cat([y, y], axis=1)`` along axis 0); only the cross-check kept
those extents opaque]
"""

from __future__ import annotations

import functools
import operator
import re
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

from mimarsinan.torch_mapping.converter_handlers.symbolic.batch_domain import kernel_rules as rules
from mimarsinan.torch_mapping.converter_handlers.symbolic.batch_domain import module_rules as modules
from mimarsinan.torch_mapping.converter_handlers.symbolic.batch_domain.kernel_rules import DATA_DEPENDENT, Rule, same
from mimarsinan.torch_mapping.converter_handlers.symbolic.batch_domain.layouts import LAYOUT_RULES

__all__ = ["DATA_DEPENDENT", "Rule", "rule_of", "same"]


_SAME = frozenset("""dropout dropout1d dropout2d dropout3d alpha_dropout feature_alpha_dropout softmax log_softmax
softmin layer_norm batch_norm group_norm instance_norm local_response_norm normalize rms_norm cumsum cumprod
logcumsumexp flip fliplr flipud roll contiguous clone detach float double half bfloat16 long int short bool byte char
to type type_as cpu cuda requires_grad_ zeros_like ones_like empty_like full_like rand_like randn_like tril triu sort
argsort slice_scatter select_scatter nan_to_num leaky_relu hardswish log_sigmoid logsigmoid _threshold threshold relu6
rrelu softplus softshrink hardshrink tanhshrink softsign silu mish gelu elu selu celu prelu data""".split())
_REDUCE = frozenset("""sum mean nansum nanmean amax amin max min argmax argmin prod std var norm logsumexp all any
count_nonzero median nanmedian""".split())
_FACTORY = frozenset("zeros ones empty full rand randn new_zeros new_ones new_empty new_full".split())
_AFFINE_OPERANDS = ("input", "weight", "bias")
_LEADING = {"linear": (1, _AFFINE_OPERANDS), "embedding": (0, ("input", "weight")), "one_hot": (0, ("tensor",)),
            "unfold": (3, ("input",)), "pixel_shuffle": (3, ("input",)), "pixel_unshuffle": (3, ("input",))}
"""``LEADING`` kernels by name: the trailing axes consumed, and the tensor operands (every other operand is a
number, review round 15)."""
_POINTWISE_OPERATORS = frozenset(id(fn) for fn in (
    operator.add, operator.sub, operator.mul, operator.truediv, operator.floordiv, operator.mod, operator.pow,
    operator.and_, operator.or_, operator.xor, operator.lshift, operator.rshift, operator.neg, operator.pos,
    operator.invert, operator.abs, operator.eq, operator.ne, operator.lt, operator.le, operator.gt, operator.ge))


@functools.lru_cache(maxsize=None)
def _pointwise_name(name: str) -> bool:
    """The aten operator of the name (an in-place form's too) carries torch's ``pointwise`` tag."""
    packet = getattr(torch.ops.aten, name.rstrip("_") or name, None)
    try:
        return packet is not None and any(torch.Tag.pointwise in getattr(packet, o).tags for o in packet.overloads())
    except (AttributeError, RuntimeError):
        return False


def _numbered(name: str) -> Optional[int]:
    found = re.search(r"(\d)d", name)
    return int(found.group(1)) if found else None


def _named_rule(name: str, functional: bool) -> Optional[Rule]:
    if name in LAYOUT_RULES:
        return LAYOUT_RULES[name]
    if name in _SAME:
        return rules.same
    if name in _REDUCE:
        return rules.reduce
    if name in _FACTORY:
        return rules.factory
    if name in _LEADING and (functional or name not in ("embedding", "unfold")):
        return rules.leading(_LEADING[name][0], operands=_LEADING[name][1])
    if name in ("matmul", "mm", "bmm", "addmm", "baddbmm", "__matmul__"):
        return rules.matmul
    if name in ("nonzero", "argwhere", "masked_select"):
        return rules.data
    if name == "einsum":
        return rules.einsum
    if name == "bilinear":
        return rules.bilinear
    if name == "glu" and functional:
        return modules.glu
    if name == "arange":
        return rules.arange
    if name == "tensor":
        return rules.tensor
    if name == "pad" and functional:
        return rules.leading(None, rules.padded)
    if name in ("interpolate", "upsample", "upsample_nearest", "upsample_bilinear"):
        return rules.leading(None, rules.spatial)
    number = _numbered(name)
    if number is not None and re.match(r"(conv|conv_transpose)\dd$", name):
        return rules.leading(number + 1, operands=_AFFINE_OPERANDS)
    if number is not None and re.match(r"(max_pool|avg_pool|lp_pool|adaptive_avg_pool|adaptive_max_pool|"
                                       r"fractional_max_pool)\dd(_with_indices)?$", name):
        return rules.leading(number)
    return rules.pointwise if _pointwise_name(name) else None


def _module_rule(module: nn.Module) -> Optional[Rule]:
    kind, name = type(module), type(module).__name__
    where = kind.__module__
    if isinstance(module, nn.MultiheadAttention):
        return modules.attention
    if isinstance(module, (nn.LSTM, nn.GRU, nn.RNN)):
        return modules.recurrent
    if isinstance(module, nn.Transformer):
        return modules.operand(1, "tgt")  # the output has tgt's shape (review round 16: ``same`` stated src's)
    if isinstance(module, nn.GLU):
        return modules.glu  # halves an axis (review round 16: ``same`` stated the input's)
    if name in ("Flatten", "Unflatten"):
        return LAYOUT_RULES[name]
    if where.endswith((".activation", ".dropout", ".normalization", ".batchnorm", ".instancenorm", ".transformer")) \
            or name in ("Identity", "ChannelShuffle", "ChannelsLastBatchNorm1d"):
        return rules.same
    if name in ("Linear", "LazyLinear"):
        return rules.leading(1)
    if name == "Bilinear":
        return rules.bilinear
    if name in ("Embedding",):
        return rules.leading(0)
    if name in ("PixelShuffle", "PixelUnshuffle", "Unfold"):
        return rules.leading(3)
    if where.endswith(".upsampling"):
        return rules.leading(None, rules.spatial)
    number = _numbered(name)
    if number is not None and where.endswith(".conv"):
        return rules.leading(number + 1)
    if number is not None and where.endswith((".pooling", ".padding")):
        return rules.leading(number)
    return None


def rule_of(op: str, target: Any, module: Optional[nn.Module]) -> Optional[Tuple[str, Rule]]:
    """``(name, rule)`` of a declared kernel call (``op``: the FX node's op), ``None`` for an undeclared one."""
    if op == "call_module":
        rule = _module_rule(module) if module is not None else None
        return None if rule is None else (type(module).__name__, rule)
    if op == "call_method":
        rule = _named_rule(str(target), functional=False)
        return None if rule is None else (str(target), rule)
    if id(target) in _POINTWISE_OPERATORS:
        return target.__name__, rules.pointwise
    if target is operator.getitem:
        return "getitem", LAYOUT_RULES["getitem"]
    if target is operator.matmul:
        return "matmul", rules.matmul
    name = getattr(target, "__name__", None)
    if not isinstance(name, str):
        return None
    functional = getattr(target, "__module__", "") in ("torch.nn.functional", "torch._C._nn")
    rule = _named_rule(name, functional)
    return None if rule is None else (name, rule)
