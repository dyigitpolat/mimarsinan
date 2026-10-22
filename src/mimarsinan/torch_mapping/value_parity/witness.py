"""Probe witnesses: the fp64 witness program, the native fp64 reference, non-trivial BatchNorm statistics and seeded
inputs.

The probe's witness program is the traced graph the returned conversion is converted from (review round 14,
``fp64_program``): FX's own deep copy of the traced ``GraphModule`` (its graph and the modules and tensors it calls,
never the user's root object), with the parameters and buffers cast to fp64, as ``model.double()`` casts them. So the
certificate covers the returned program: the root's ``__deepcopy__`` never runs, a called module's copy is checked
against the traced program node by node and tensor by tensor (``program_binding``), and a dtype the traced program
holds (``self.dtype`` read at trace time) is the same literal in both. The native reference is the witness program
itself where it computes the user's model's own call bit for bit in the model's dtype on every input the probe
compares (``traced_as_native``; an fp32 model reading its dtype, ``F.softmax(..., dtype=self.dtype)``, computes another
function in fp64), else the user's model itself (``NativeFp64``): its own call, with its parameters and buffers
replaced by the witness program's fp64 tensors.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import torch
import torch.fx as fx
import torch.nn as nn
from torch.fx.passes.shape_prop import ShapeProp

from mimarsinan.torch_mapping.conversion_probe import guarded_forward

_BATCHNORMS = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)


_ATOMS = (int, float, complex, bool, str, bytes, type(None), type(Ellipsis), torch.dtype, torch.device, torch.layout,
          torch.memory_format)


def _immutable(value: Any) -> bool:
    """An immutable value: an atom of exact type, a tuple (a subclass, a namedtuple, with no instance attribute) or a
    ``frozenset`` of immutable values."""
    if type(value) in _ATOMS:
        return True
    if isinstance(value, tuple) and not getattr(value, "__dict__", None) or type(value) is frozenset:
        return all(map(_immutable, value))
    return False


def _immutables(value: Any, found: Dict[int, Any], depth: int = 0) -> None:
    """The immutable non-atom values ``value`` holds, through lists, dicts and tuples, by ``id``."""
    if isinstance(value, (tuple, frozenset)) and type(value) is not tuple and _immutable(value):
        found[id(value)] = value
    if depth < 4 and isinstance(value, (list, tuple, dict)):
        for item in (value.values() if isinstance(value, dict) else value):
            _immutables(item, found, depth + 1)


def _computed(value: Any, found: Dict[int, Any]) -> None:
    """A tensor attribute computed with autograd (not a graph leaf: ``spectral_norm``'s ``weight``, set by its hook in
    the traced run), which ``copy.deepcopy`` refuses: its copy is the same values, detached (review round 15)."""
    if isinstance(value, torch.Tensor) and not value.is_leaf:
        found[id(value)] = value.detach().clone()


def faithful_copy(model: nn.Module) -> nn.Module:
    """A deep copy that holds the model's own immutable attribute values (a namedtuple, a tuple subclass): copying one
    rebuilds it through its class (``__reduce_ex__``, ``__new__``), which a class may not round-trip, so the copy would
    be another program (review round 13); a computed tensor attribute is copied detached (``_computed``)."""
    memo: Dict[int, Any] = {}
    for module in model.modules():
        for key, value in vars(module).items():
            if key not in ("_parameters", "_buffers", "_modules"):
                _immutables(value, memo)
                _computed(value, memo)
        for value in vars(module).get("_buffers", {}).values():
            _computed(value, memo)
    return copy.deepcopy(model, memo)


def fp64_snapshot(model: nn.Module) -> nn.Module:
    """A CPU, fp64, eval-mode deep copy (``faithful_copy``); taken before conversion, which may mutate its input. The
    witness of ``probe_witness_conversion``; ``convert_torch_model`` probes ``fp64_program``."""
    return faithful_copy(model).to("cpu", torch.float64).eval()


def fp64_program(traced: fx.GraphModule, input_shape: Tuple[int, ...]) -> fx.GraphModule:
    """The witness program: ``traced`` (taken before any rewrite) deep-copied, on the CPU in fp64 and eval mode, its
    node metadata refreshed by an fp64 shape propagation at batch 1 (``ShapeProp``)."""
    witness = cast(fx.GraphModule, faithful_copy(traced).to("cpu", torch.float64).eval())
    with torch.random.fork_rng(devices=[]):
        ShapeProp(witness).propagate(torch.randn(1, *tuple(input_shape), dtype=torch.float64))
    return witness


def _named_tensors(module: nn.Module) -> Dict[str, torch.Tensor]:
    return {**dict(module.named_parameters(remove_duplicate=False)), **dict(module.named_buffers(remove_duplicate=False))}


class NativeFp64(nn.Module):
    """The native model in fp64 (``dtype``), uncopied: ``model``'s own call (eval mode) on its parameters and buffers
    replaced by ``state`` (by qualified name), every floating one cast to ``dtype``. ``model`` is held, not registered:
    nothing reaches it through this module."""

    def __init__(self, model: nn.Module, state: Dict[str, torch.Tensor], dtype: torch.dtype = torch.float64):
        super().__init__()
        own = {**dict(model.named_parameters()), **dict(model.named_buffers())}
        tensors = {name: (state[name] if name in state else value).detach().to(
                   "cpu", dtype if value.is_floating_point() else value.dtype).clone()
                   for name, value in own.items()}
        object.__setattr__(self, "native", model)
        object.__setattr__(self, "tensors", tensors)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        native, tensors = cast(nn.Module, self.native), cast(Dict[str, torch.Tensor], self.tensors)
        modes = [(module, module.training) for module in native.modules()]
        native.eval()
        try:
            return torch.func.functional_call(native, tensors, args, kwargs, strict=False)
        finally:
            for module, training in modes:
                module.training = training


def _floating_dtype(module: nn.Module) -> Optional[torch.dtype]:
    kinds = {t.dtype for t in _named_tensors(module).values() if t.is_floating_point()}
    return next(iter(kinds)) if len(kinds) == 1 else None


def _bitwise(found: Any, want: Any) -> bool:
    found = found[0] if isinstance(found, (tuple, list)) and found else found
    want = want[0] if isinstance(want, (tuple, list)) and want else want
    if not isinstance(found, torch.Tensor) or not isinstance(want, torch.Tensor) or found.shape != want.shape \
            or found.dtype != want.dtype:
        return False
    if not found.is_floating_point():
        return torch.equal(found, want)
    return bool(((found == want) | (torch.isnan(found) & torch.isnan(want))).all())


class _Called(nn.Module):
    """``program`` called on ``state`` (``torch.func.functional_call``)."""

    def __init__(self, program: nn.Module, state: Dict[str, torch.Tensor]):
        super().__init__()
        object.__setattr__(self, "program", program)
        object.__setattr__(self, "state", state)

    def forward(self, *args: Any) -> Any:
        return torch.func.functional_call(cast(nn.Module, self.program), cast(Dict[str, torch.Tensor], self.state),
                                          args, strict=False)


def traced_as_native(model: nn.Module, program: nn.Module, inputs: Sequence[torch.Tensor]) -> bool:
    """The traced program computes ``model``'s own call bit for bit in the model's floating dtype, on every input the
    probe compares (the first seeded one alone included) and the witness's tensors cast to that dtype (review round
    14). A dtype the model reads at trace time
    (``self.dtype``) is then the same literal in both: the witness program in fp64 is the model's function in fp64,
    where the model's own call in fp64 would read another dtype. An fp64 model is not checked (False): its own call
    in fp64 is that function, and it equals the witness program bit for bit on the probe's inputs exactly where this
    check would hold, so the reference changes no verdict."""
    dtype = _floating_dtype(model)
    if dtype is None or dtype == torch.float64:
        return False
    state = {name: t.detach().to(dtype) if t.is_floating_point() else t for name, t in _named_tensors(program).items()}
    native, traced = NativeFp64(model, state, dtype), _Called(program, state)
    for x in inputs:
        want, unrun = guarded_forward(native, x.to(dtype))
        found, failed = guarded_forward(traced, x.to(dtype))
        if unrun is not None or failed is not None or not _bitwise(found, want):
            return False
    return True


def native_reference(model: nn.Module):
    """The probe's native reference of ``model`` given its randomized witness program and the probe's inputs: the
    witness program itself where it computes the model's own call bit for bit in the model's dtype
    (``traced_as_native``), else the model's own call in fp64 (``NativeFp64``)."""
    def build(program: nn.Module, inputs: Sequence[torch.Tensor]) -> nn.Module:
        if traced_as_native(model, program, inputs):
            return program
        return NativeFp64(model, _named_tensors(program))
    return build


def _uniform(n: int, low: float, high: float, generator: torch.Generator) -> torch.Tensor:
    return torch.rand(n, generator=generator, dtype=torch.float64) * (high - low) + low


def randomize_batchnorm_statistics(model: nn.Module, *, seed: int) -> int:
    """Draw non-trivial statistics and affine parameters for every BatchNorm in place; returns the count.

    Running mean U(-0.2, 0.2), running variance U(0.5, 2), weight U(0.5, 2) and
    bias U(-0.2, 0.2), so a dropped, misplaced or eps-shifted BatchNorm changes
    the output (default statistics make it nearly an identity).
    """
    generator = torch.Generator().manual_seed(seed)
    count = 0
    with torch.no_grad():
        for module in model.modules():
            if not isinstance(module, _BATCHNORMS):
                continue
            mean, var = module.running_mean, module.running_var
            if mean is None or var is None:
                continue
            n = mean.numel()
            mean.copy_(_uniform(n, -0.2, 0.2, generator))
            var.copy_(_uniform(n, 0.5, 2.0, generator))
            if module.weight is not None:
                module.weight.copy_(_uniform(n, 0.5, 2.0, generator))
            if module.bias is not None:
                module.bias.copy_(_uniform(n, -0.2, 0.2, generator))
            count += 1
    return count


def randomize_zero_parameters(model: nn.Module, *, seed: int) -> List[str]:
    """Draw every all-zero floating parameter in place (a zero head or token compares nothing); returns their names.

    Matrices draw U(-b, b) with b = 1/sqrt(fan_in), the default Linear scale; vectors and scalars U(-0.2, 0.2).
    """
    generator = torch.Generator().manual_seed(seed)
    drawn: List[str] = []
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not param.is_floating_point() or param.numel() == 0 or bool(param.any()):
                continue
            bound = 1.0 / math.sqrt(param.numel() // param.shape[0]) if param.dim() >= 2 else 0.2
            param.copy_(_uniform(param.numel(), -bound, bound, generator).view(param.shape))
            drawn.append(name)
    return drawn


PROBE_INPUT_KINDS = ("N(0, 1)", "U(0, 1)", "zero", "-U(0, 1)")
"""The seeded inputs: random halves, then the all-zero input (silence) and a negated uniform one."""


def probe_inputs(
    input_shape: Tuple[int, ...], *, seed: int, batch: int = 6,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """``batch`` seeded inputs: ``batch - 2`` random (half N(0, 1), half U(0, 1)), one zero, one -U(0, 1)."""
    if batch < 4:
        raise ValueError(
            f"the value probe needs batch >= 4 (got {batch}): two random inputs so that "
            "batch-dependent broadcasts are exercised, the zero input and a negated one"
        )
    generator = torch.Generator().manual_seed(seed)
    random = batch - 2
    half = random // 2
    shape = tuple(input_shape)
    normal = torch.randn(half, *shape, generator=generator, dtype=torch.float64)
    uniform = torch.rand(random - half, *shape, generator=generator, dtype=torch.float64)
    zero = torch.zeros(1, *shape, dtype=torch.float64)
    negative = -torch.rand(1, *shape, generator=generator, dtype=torch.float64)
    return torch.cat([normal, uniform, zero, negative]).to(dtype)


ESCALATED_SCALES = (10.0, 100.0)
"""Magnitudes every reference is compared at again: a branch dead on the seeded inputs is exercised if alive here."""

INTERMEDIATE_SCALES = (3.0, 30.0)
"""The magnitudes between: a branch alive only between all of the probe's magnitudes is still never exercised."""


def scaled_probe_inputs(
    input_shape: Tuple[int, ...], *, seed: int, scales: Tuple[float, float], batch: int = 6,
) -> torch.Tensor:
    """``batch`` seeded fp64 inputs: the first half ``scales[0]`` N(0, 1), the rest ``scales[1]`` N(0, 1)."""
    generator = torch.Generator().manual_seed(seed)
    normal = torch.randn(batch, *tuple(input_shape), generator=generator, dtype=torch.float64)
    half = batch // 2
    factors = torch.tensor([scales[0]] * half + [scales[1]] * (batch - half), dtype=torch.float64)
    return normal * factors.view(batch, *([1] * len(tuple(input_shape))))


def escalated_probe_inputs(input_shape: Tuple[int, ...], *, seed: int, batch: int = 6) -> torch.Tensor:
    """``batch`` seeded fp64 inputs: the first half 10·N(0, 1), the rest 100·N(0, 1)."""
    return scaled_probe_inputs(input_shape, seed=seed, scales=ESCALATED_SCALES, batch=batch)


def intermediate_probe_inputs(input_shape: Tuple[int, ...], *, seed: int, batch: int = 6) -> torch.Tensor:
    """``batch`` seeded fp64 inputs: the first half 3·N(0, 1), the rest 30·N(0, 1)."""
    return scaled_probe_inputs(input_shape, seed=seed, scales=INTERMEDIATE_SCALES, batch=batch)


def magnitude_groups(outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """The outputs on a scaled input set, split by magnitude (the halves ``scaled_probe_inputs`` draws)."""
    half = int(outputs.shape[0]) // 2
    return outputs[:half], outputs[half:]
