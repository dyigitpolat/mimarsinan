"""Member census: the stage-2 verdict at C0, the conversion's IR, both MAC counters, and the ladder rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch.nn as nn

from mimarsinan.common.refusal import ToolchainRefusal, refusal_in_chain
from mimarsinan.mapping.capability.census import LevelCensus, ladder_census
from mimarsinan.mapping.capability.ladder import LADDER, CapabilityLevel
from mimarsinan.mapping.capability.mac_split import MacCounters
from mimarsinan.mapping.ir import ComputeOp, IRGraph
from mimarsinan.mapping.map_model_to_ir import map_model_to_ir
from mimarsinan.mapping.platform.packaging_contract import PackagingContract
from mimarsinan.mapping.verification.onchip_fraction import estimate_onchip_fraction
from mimarsinan.mapping.verification.onchip_majority import count_host_ops
from mimarsinan.torch_mapping.conversion_probe import guarded
from mimarsinan.torch_mapping.converter import probe_stages, probe_witness_conversion, trace_and_convert

AGREEMENT, REFUSED = "fp64 agreement", "refused"
#: A measured comparison above its bound: an exactness failure of the toolchain, never a capability refusal.
VALUE_MISMATCH = "value mismatch"
#: The probe returned without a stage ``probe_stages`` requires: no verdict, never an agreement.
INCOMPLETE = "incomplete verdict"
#: The C0 column's name: the stage-2 fp64 identity comparison; "exact" is kept for atol-0 integer certificates (CL4).
VERDICT_COLUMN = "fp64 agreement (max relative error)"


@dataclass(frozen=True)
class ExactnessVerdict:
    """The C0 verdict: the worst relative error over the probe stages, or the refusal by its type name.

    ``refusal`` is the probe's own refusal class; ``root_refusal`` the innermost named refusal it wraps (a
    lowering refusal inside a stage-2 value-parity refusal), the name a capability count reads.
    """

    status: str
    relative_error: Optional[float] = None
    bound: Optional[float] = None
    stages: Tuple[Tuple[str, float], ...] = ()
    refusal: Optional[str] = None
    refusal_type: Optional[type] = None
    reason: Optional[str] = None
    column: str = VERDICT_COLUMN
    root_refusal: Optional[str] = None
    root_refusal_type: Optional[type] = None
    value_mismatch: bool = False
    missing_stages: Tuple[str, ...] = ()


def root_refusal(refusal: ToolchainRefusal) -> ToolchainRefusal:
    """The named refusal a probe stage wraps (its ``result.error``), followed down to the innermost one."""
    seen = set()
    current = refusal
    while id(current) not in seen:
        seen.add(id(current))
        error = getattr(getattr(current, "result", None), "error", None)
        inner = refusal_in_chain(error) if isinstance(error, BaseException) else None
        if inner is None:
            return current
        current = inner
    return current


def _refused(refusal: ToolchainRefusal) -> ExactnessVerdict:
    """A measured comparison above its bound is a value mismatch; a comparison the reference cannot decide
    (``ValueParityResult.reference_undetermined``: ``ReferenceUndeterminedError``) is refused by that name, its
    measured error kept beside it, because no mismatch is shown."""
    root = root_refusal(refusal)
    result = getattr(refusal, "result", None)
    measured = getattr(refusal, "relative_error", None)
    bound = getattr(result, "bound", None)
    mismatch = measured is not None and not getattr(result, "reference_undetermined", False)
    lines = str(refusal).strip().splitlines()
    return ExactnessVerdict(
        VALUE_MISMATCH if mismatch else REFUSED,
        relative_error=None if measured is None else float(measured),
        bound=None if measured is None or bound is None else float(bound),
        refusal=type(refusal).__name__, refusal_type=type(refusal), reason=lines[0] if lines else None,
        root_refusal=type(root).__name__, root_refusal_type=type(root), value_mismatch=mismatch,
    )


def c0_verdict(model: nn.Module, input_shape, *, packaging: PackagingContract,
               decompose_leaf_modules: bool = False) -> ExactnessVerdict:
    """Every probe stage ``packaging`` requires (stage 2 for a value-domain contract) on an fp64 witness, converted
    under ``decompose_leaf_modules`` as the census converts the member.

    A refusal by name becomes the verdict (a measured mismatch is marked, never counted a refusal); a verdict
    missing a required stage is incomplete; an anonymous crash propagates, loud.
    """
    results, exc = guarded(lambda: probe_witness_conversion(model, tuple(input_shape), packaging=packaging,
                                                            decompose_leaf_modules=decompose_leaf_modules))
    if exc is not None:
        refusal = refusal_in_chain(exc)
        if refusal is None:
            raise exc
        return _refused(refusal)
    assert results is not None
    stages = tuple((r.stage, float(r.relative_error or 0.0)) for r in results)
    missing = tuple(stage for stage in probe_stages(packaging) if stage not in {name for name, _ in stages})
    if missing:
        return ExactnessVerdict(INCOMPLETE, stages=stages, missing_stages=missing,
                                reason=f"the probe returned without the required stage(s) {', '.join(missing)}")
    return ExactnessVerdict(AGREEMENT, relative_error=max(error for _, error in stages),
                            bound=max(float(r.bound) for r in results), stages=stages)


@dataclass(frozen=True)
class ConvertedMember:
    ir: IRGraph
    mac_counters: MacCounters


def has_multi_input_op(ir_graph: IRGraph) -> bool:
    return any(isinstance(n, ComputeOp) and n.params.get("input_shapes") is not None for n in ir_graph.nodes)


def convert_for_census(
    model: nn.Module, input_shape, num_classes: int, *, packaging: PackagingContract,
    decompose_leaf_modules: bool = False,
) -> ConvertedMember:
    """The member's IR as stage 2 lowers it (unbounded cores, uncoalesced) and its two MAC counters;
    ``decompose_leaf_modules`` replaces the registered leaves and decomposes attention calls before the lowering."""
    flow = trace_and_convert(model, tuple(input_shape), "cpu", packaging=packaging,
                             decompose_leaf_modules=decompose_leaf_modules).flow
    mapper_repr = flow.get_mapper_repr()
    try:
        ir = map_model_to_ir(mapper_repr)
    finally:
        mapper_repr.clear_ir_caches()
    estimate = estimate_onchip_fraction(flow, tuple(input_shape), int(num_classes), metric="macs")
    counters = MacCounters(count_host_ops(ir), int(estimate.host), int(estimate.total), has_multi_input_op(ir))
    return ConvertedMember(ir, counters)


@dataclass(frozen=True)
class MemberCensus:
    """A member's verdict, counters and rows; a member without an agreement (or a refused census) has no rows."""

    verdict: ExactnessVerdict
    mac_counters: Optional[MacCounters]
    rows: Tuple[LevelCensus, ...]
    census_refusal: Optional[str] = None


def member_census(
    model: nn.Module, input_shape, num_classes: int, *, packaging: PackagingContract, value_bits: int,
    levels: Sequence[CapabilityLevel] = LADDER, verdict: Optional[ExactnessVerdict] = None,
    decompose_leaf_modules: bool = False,
) -> MemberCensus:
    """The verdict first (or ``verdict``, judged by the caller, possibly on a smaller input of the same model); an
    agreeing member is then converted once more at ``input_shape`` for its IR and placed on every level. The verdict
    and the census conversion read the same ``decompose_leaf_modules``."""
    verdict = verdict if verdict is not None else c0_verdict(model, input_shape, packaging=packaging,
                                                             decompose_leaf_modules=decompose_leaf_modules)
    if verdict.status != AGREEMENT:
        return MemberCensus(verdict, None, ())
    converted = convert_for_census(model, input_shape, num_classes, packaging=packaging,
                                   decompose_leaf_modules=decompose_leaf_modules)
    rows, exc = guarded(lambda: ladder_census(converted.ir, value_bits=value_bits, levels=levels))
    if exc is not None:
        refusal = refusal_in_chain(exc)
        if refusal is None:
            raise exc
        return MemberCensus(verdict, converted.mac_counters, (), f"{type(refusal).__name__}: {refusal}")
    assert rows is not None
    return MemberCensus(verdict, converted.mac_counters, rows)
