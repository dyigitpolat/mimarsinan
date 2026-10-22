"""Value-parity conversion probe (E270): the converted model against the native model on seeded fp64 inputs."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Callable, List, Mapping, NamedTuple, Optional, Sequence, Tuple, Union, cast

import torch
import torch.fx as fx
import torch.nn as nn

from mimarsinan.common.refusal import is_toolchain_refusal, refusal_in_chain
from mimarsinan.torch_mapping.conversion_probe import extract_failing_node_name, guarded, guarded_forward
from mimarsinan.torch_mapping.torch_graph_tracer import trace_model
from mimarsinan.torch_mapping.value_parity.comparison import degenerate_kind, relative_error, uninformative_kind
from mimarsinan.torch_mapping.value_parity.verdict import (
    STAGE_CONVERTED,
    STAGE_NATIVE,
    ConversionValueParityError,
    ValueParityResult,
    degenerate_reference,
    describe_failure,
)
from mimarsinan.torch_mapping.value_parity.contract import NATIVE_EVAL_IDENTITY, FunctionContract
from mimarsinan.torch_mapping.value_parity.localization import DivergingNode, first_diverging_node
from mimarsinan.torch_mapping.value_parity.rounding import conversion_value_bound, weight_layer_depth
from mimarsinan.torch_mapping.value_parity.witness import (
    escalated_probe_inputs,
    faithful_copy,
    intermediate_probe_inputs,
    magnitude_groups,
    probe_inputs,
    randomize_batchnorm_statistics,
    randomize_zero_parameters,
)

__all__ = ["ProbeWitness", "WitnessConversion", "describe_failure", "is_toolchain_refusal", "judge", "probe_conversion",
           "probe_conversion_values"]

BATCH_ONE = "batch 1 (the first seeded input alone)"
ESCALATED = "escalated-magnitude inputs (10 N(0, 1), 100 N(0, 1))"
INTERMEDIATE = "intermediate-magnitude inputs (3 N(0, 1), 30 N(0, 1))"


class WitnessConversion(NamedTuple):
    flow: nn.Module
    absorption_plan: Mapping[str, str]
    program: Optional[tuple] = None  # the traced program (``program_binding.program_record``), review round 13


@dataclass(frozen=True)
class ProbeWitness:
    reference: nn.Module
    conversion: WitnessConversion
    inputs: torch.Tensor
    escalated_inputs: torch.Tensor
    intermediate_inputs: torch.Tensor
    input_shape: Tuple[int, ...]
    weight_layer_depth: Optional[int]
    randomized_parameters: Tuple[str, ...]
    program: Optional[nn.Module] = None  # the witness program before conversion (review round 14), when probed

    @property
    def bound(self) -> float:
        return conversion_value_bound(self.weight_layer_depth or 0)

    @property
    def traced(self) -> nn.Module:
        """The traced program localization and the rounding bounds read: the witness program, else the reference."""
        return self.program if self.program is not None else self.reference


def prepare_witness(
    snapshot: nn.Module,
    convert: Callable[[nn.Module], WitnessConversion],
    input_shape: Tuple[int, ...],
    *,
    contract: FunctionContract,
    seed: int,
    batch: int,
    native: Optional[Callable[[nn.Module, Sequence[torch.Tensor]], nn.Module]] = None,
) -> ProbeWitness:
    """Randomize the snapshot's BatchNorms and all-zero parameters, keep a pristine reference, convert the snapshot.

    ``native``: the native reference of the randomized snapshot given the probe's inputs (``witness.native_reference``:
    the snapshot is then the witness program, kept pristine for localization); else the reference is a copy of the
    snapshot. A failing witness conversion propagates as itself: a refusal by name, an anonymous crash loud.
    """
    randomize_batchnorm_statistics(snapshot, seed=seed)
    randomized = tuple(randomize_zero_parameters(snapshot, seed=seed))
    inputs = (probe_inputs(input_shape, seed=seed + 1, batch=batch),
              escalated_probe_inputs(input_shape, seed=seed + 2, batch=batch),
              intermediate_probe_inputs(input_shape, seed=seed + 3, batch=batch))
    program = cast(fx.GraphModule, faithful_copy(snapshot).eval()) if native is not None else None
    reference = (native(cast(nn.Module, program), (inputs[0], inputs[0][:1], *inputs[1:])) if native is not None
                 else contract.reference(faithful_copy(snapshot))).eval()
    conversion = convert(snapshot)
    conversion.flow.eval()
    # A reference that cannot be traced fails its own stage below; its refusal line is then the strictest one.
    depth, _ = guarded(lambda: weight_layer_depth(
        program if program is not None else trace_model(reference, tuple(input_shape), device="cpu")))
    return ProbeWitness(reference, conversion, *inputs, tuple(input_shape), depth, randomized, program)


def locate_in_converted_model(witness: ProbeWitness, candidate: nn.Module, bound: float) -> Optional[DivergingNode]:
    plan = witness.conversion.absorption_plan
    return first_diverging_node(witness.traced, candidate, plan, witness.input_shape, witness.inputs, bound)


Locator = Callable[[ProbeWitness, nn.Module, float], Optional[DivergingNode]]
LineDeriver = Callable[[ProbeWitness, torch.Tensor], Tuple[float, Optional[float]]]
"""``(line, beta)`` for a comparison on the given inputs that measured above the stage's base line."""


def _reference_output(
    witness: ProbeWitness, inputs: torch.Tensor, contract: FunctionContract, base: ValueParityResult, where: str,
) -> Union[torch.Tensor, ValueParityResult]:
    """The contracted native output on ``inputs``, or the native-stage refusal when there is none."""
    raw_want, exc = guarded_forward(witness.reference, inputs)
    if exc is not None:
        return replace(
            base, stage=STAGE_NATIVE, error=exc,
            reason=f"the native model does not run in fp64 on the probe inputs ({where}): {describe_failure(exc)}",
        )
    try:
        return contract.select_output(raw_want)
    except TypeError as type_exc:
        return replace(base, stage=STAGE_NATIVE, error=type_exc, reason=str(type_exc))


@dataclass(frozen=True)
class _Comparison:
    inputs: torch.Tensor
    want: torch.Tensor
    compared_on: Optional[str]


def judge(
    witness: ProbeWitness,
    candidate: nn.Module,
    *,
    stage: str,
    contract: FunctionContract,
    bound: float,
    context: str,
    locate: Optional[Locator] = locate_in_converted_model,
    flat_output: bool = False,
    derive_line: Optional[LineDeriver] = None,
) -> ValueParityResult:
    """Compare ``candidate`` with the contracted reference: seeded inputs, the first alone, two scaled input sets.

    Every reference is compared at 10 and 100 N(0, 1) (escalated) and 3 and 30 N(0, 1) (intermediate): a branch
    dead on the seeded inputs is exercised only where it is alive at one of these magnitudes, never one alive
    only between them. A reference with no information on the seeded inputs (no finite non-zero entry, or constant
    across the samples under the probe's metric) and no finite non-zero entry on the escalated ones is refused by
    name; it is flagged constant when it carries none at any magnitude on its own. ``flat_output``: the candidate
    returns each sample's output flattened (an IR value program). ``derive_line`` widens ``bound`` above it.
    """
    base = ValueParityResult(
        ok=False, context=context, stage=stage, contract=contract.name, bound=bound,
        samples=int(witness.inputs.shape[0]), weight_layer_depth=witness.weight_layer_depth,
        randomized_parameters=witness.randomized_parameters,
    )
    comparisons: List[_Comparison] = []
    for inputs, compared_on in ((witness.inputs, None), (witness.inputs[:1], BATCH_ONE),
                                (witness.escalated_inputs, ESCALATED), (witness.intermediate_inputs, INTERMEDIATE)):
        want = _reference_output(witness, inputs, contract, base, compared_on or "seeded")
        if isinstance(want, ValueParityResult):
            return want
        comparisons.append(_Comparison(inputs, want, compared_on))
    seeded, escalated, intermediate = comparisons[0].want, comparisons[2].want, comparisons[3].want
    uninformative = uninformative_kind(seeded, bound)
    if uninformative and degenerate_kind(escalated):
        return degenerate_reference(base, uninformative, str(degenerate_kind(escalated)))
    magnitudes = (seeded, *magnitude_groups(escalated), *magnitude_groups(intermediate))
    base = replace(base, reference_constant=all(uninformative_kind(want, bound) for want in magnitudes),
                   escalated=True)
    measured: List[ValueParityResult] = []
    for comparison in comparisons:
        result = _compare(witness, candidate, comparison, base, locate, flat_output, derive_line)
        if not result.ok:
            return result
        measured.append(result)
    worst = max(measured, key=lambda r: r.relative_error or 0.0)
    return replace(worst, bound=max(r.bound for r in measured), compared_on=None, reassociation_bound=max(
        (r.reassociation_bound for r in measured if r.reassociation_bound is not None), default=None))


def _compare(
    witness: ProbeWitness, candidate: nn.Module, comparison: _Comparison, base: ValueParityResult,
    locate: Optional[Locator], flat_output: bool, derive_line: Optional[LineDeriver],
) -> ValueParityResult:
    base = replace(base, compared_on=comparison.compared_on)
    prefix = f"on {comparison.compared_on}: " if comparison.compared_on else ""
    samples = int(comparison.inputs.shape[0])
    got, exc = guarded_forward(candidate, comparison.inputs)
    if exc is not None:
        return replace(
            base, error=exc, first_diverging_node=extract_failing_node_name(exc),
            reason=f"{prefix}forward failed on {samples} seeded fp64 inputs: {describe_failure(exc)}",
        )
    if not isinstance(got, torch.Tensor):
        return replace(base, reason=f"{prefix}returned {type(got).__name__}, not a tensor")
    want = comparison.want
    if flat_output and got.numel() == want.numel():
        got = got.reshape(want.shape)
    local = replace(witness, inputs=comparison.inputs)

    def located(result: ValueParityResult) -> ValueParityResult:
        found = None if locate is None else locate(local, candidate, result.bound)
        if found is None:
            return result
        return replace(result, first_diverging_node=found.name, first_diverging_operator=found.operator)

    if tuple(got.shape) != tuple(want.shape):
        return located(replace(
            base, reason=f"{prefix}output shape {tuple(got.shape)} differs from the native {tuple(want.shape)}"))
    delta = relative_error(got, want)
    result = replace(
        base, ok=delta.relative_error <= base.bound, max_abs_error=delta.max_abs_error,
        reference_max=delta.reference_max, relative_error=delta.relative_error,
    )
    if not result.ok and derive_line is not None:
        line, beta = derive_line(witness, comparison.inputs)
        result = replace(result, ok=delta.relative_error <= line, bound=line, reassociation_bound=beta)
    return result if result.ok else located(result)


def conclude(result: ValueParityResult, *, started: float, strict: bool) -> ValueParityResult:
    """Stamp the wall time; strict, raise a refusal by name, or the anonymous crash that left no comparison."""
    result = replace(result, wall_s=time.perf_counter() - started)
    if strict and not result.ok:
        error = result.error if result.stage != STAGE_NATIVE else None
        if error is not None and refusal_in_chain(error) is None:
            raise error
        raise ConversionValueParityError(result)
    return result


WitnessStage = Callable[..., ValueParityResult]
"""A later probe stage: ``stage(witness, contract=, bound=, context=)`` judges a lowering of the same witness
(``bound=None``: the stage derives its own refusal line)."""


def probe_conversion(
    snapshot: nn.Module,
    convert: Callable[[nn.Module], WitnessConversion],
    input_shape: Tuple[int, ...],
    *,
    later_stages: Sequence[WitnessStage] = (),
    contract: FunctionContract = NATIVE_EVAL_IDENTITY,
    bound: Optional[float] = None,
    seed: int = 0,
    batch: int = 6,
    strict: bool = True,
    context: str = "probe_conversion",
    native: Optional[Callable[[nn.Module, Sequence[torch.Tensor]], nn.Module]] = None,
) -> Tuple[ValueParityResult, ...]:
    """Stage 1 (``convert(snapshot)`` against ``contract``), then each later stage on the same witness; RNG-neutral.

    ``snapshot`` is the probe's own fp64 witness (randomized and converted in place: a copy of the native model, or
    with ``native`` the witness program, ``prepare_witness``). The refusal line is tau(P) for stage 1 and tau(P) +
    beta(P) (capped at the 1e-12 acceptance) for a lowering stage, unless ``bound`` overrides both. A stage runs only
    after every earlier one passed.
    """
    started = time.perf_counter()
    results = []
    with torch.random.fork_rng(devices=[]):
        witness = prepare_witness(snapshot, convert, input_shape, contract=contract, seed=seed, batch=batch,
                                  native=native)
        results.append(conclude(judge(
            witness, witness.conversion.flow, stage=STAGE_CONVERTED, contract=contract,
            bound=witness.bound if bound is None else bound, context=context,
        ), started=started, strict=strict))
        for stage in later_stages:
            if not results[-1].ok:
                break
            started = time.perf_counter()
            results.append(conclude(stage(witness, contract=contract, bound=bound, context=context), started=started,
                                    strict=strict))
    return tuple(results)


def probe_conversion_values(
    snapshot: nn.Module,
    convert: Callable[[nn.Module], WitnessConversion],
    input_shape: Tuple[int, ...],
    *,
    contract: FunctionContract = NATIVE_EVAL_IDENTITY,
    bound: Optional[float] = None,
    seed: int = 0,
    batch: int = 6,
    strict: bool = True,
    context: str = "probe_conversion_values",
) -> ValueParityResult:
    """Stage 1 only: ``convert(snapshot)`` against ``contract``; ``strict=True`` raises above the refusal line."""
    (result,) = probe_conversion(snapshot, convert, input_shape, contract=contract, bound=bound, seed=seed,
                                 batch=batch, strict=strict, context=context)
    return result
