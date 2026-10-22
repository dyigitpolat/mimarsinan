"""Public API for converting native PyTorch models to a mimarsinan ConvertedModelFlow."""

from __future__ import annotations

import dataclasses
import warnings
from typing import Callable, FrozenSet, List, NamedTuple, Optional, Tuple, Union

import torch
import torch.fx as fx
import torch.nn as nn

from mimarsinan.mapping.platform.packaging_contract import (
    SPIKING_PACKAGING,
    PackagingContract,
)
from mimarsinan.torch_mapping.conversion_probe import guarded
from mimarsinan.torch_mapping.torch_graph_tracer import trace_model
from mimarsinan.torch_mapping.representability_analyzer import (
    RepresentabilityAnalyzer,
    RepresentabilityReport,
    RepresentabilityError,
)
from mimarsinan.torch_mapping.mapper_graph_converter import MapperGraphConverter
from mimarsinan.torch_mapping.converted_model_flow import ConvertedModelFlow
from mimarsinan.torch_mapping.converter_handlers.symbolic.purity import program_purity
from mimarsinan.torch_mapping.encoding_layers import mark_encoding_layers
from mimarsinan.torch_mapping.graph_normalization import normalize_fx_graph
from mimarsinan.torch_mapping.value_parity import (
    NATIVE_EVAL_IDENTITY,
    STAGE_CONVERTED,
    STAGE_VALUE_PROGRAM,
    FunctionContract,
    ValueParityResult,
    WitnessConversion,
    fp64_snapshot,
    probe_conversion,
)
from mimarsinan.torch_mapping.value_parity.certificate import (
    ConversionCertificate,
    UnverifiedConversionWarning,
)
from mimarsinan.torch_mapping.value_parity.program_binding import (
    NodeRecord,
    program_difference,
    program_record,
    state_differences,
)
from mimarsinan.torch_mapping.value_parity.value_program import identity_value_program_stage
from mimarsinan.torch_mapping.value_parity.witness import fp64_program, native_reference
from mimarsinan.torch_mapping.value_parity.verdict import (
    WitnessProgramError,
    WitnessProgramMismatchError,
    describe_failure,
    literal_differences,
)


class TracedConversion(NamedTuple):
    """A converted flow, the representability report (absorption plan) it was converted under, its traced program as
    the witness binding compares it (``program_binding.program_record``) and, when asked for (``witness=True``), the
    probe's fp64 witness program of the same traced graph (``witness.fp64_program``; ``witness_error`` when it does
    not run in fp64) with the tensors it holds otherwise than the traced program (``altered``)."""

    flow: ConvertedModelFlow
    report: RepresentabilityReport
    program: Tuple[NodeRecord, ...] = ()
    witness: Optional[fx.GraphModule] = None
    witness_error: Optional[BaseException] = None
    altered: FrozenSet[str] = frozenset()


class TracedProgram(NamedTuple):
    """A program already traced (``trace_model``'s ``GraphModule``): ``trace_and_convert`` converts it as it is."""

    graph_module: fx.GraphModule


def check_representability(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: Union[torch.device, str] = "cpu",
    *,
    packaging: PackagingContract = SPIKING_PACKAGING,
) -> RepresentabilityReport:
    """Trace a native PyTorch model and classify every op as supported, absorbable, or unsupported."""
    gm = trace_model(model, input_shape, device=device)
    foreign = program_purity(gm).foreign_modules
    gm = normalize_fx_graph(gm, foreign)
    analyzer = RepresentabilityAnalyzer(gm, packaging=packaging, foreign_modules=foreign)
    return analyzer.analyze()


def trace_and_convert(
    model: Union[nn.Module, TracedProgram],
    input_shape: Tuple[int, ...],
    device: Union[torch.device, str] = "cpu",
    *,
    encoding_layer_placement: str = "subsume",
    packaging: PackagingContract = SPIKING_PACKAGING,
    witness: bool = False,
) -> TracedConversion:
    """Trace → purity (before any rewrite) → normalize → analyze → convert → mark encoding layers, without any probe.
    ``witness``: also derive the probe's fp64 witness program from the traced graph before the conversion rewrites it
    (RNG-neutral)."""
    traced = model.graph_module if isinstance(model, TracedProgram) else trace_model(
        model, input_shape, device=torch.device(device))
    program, unrun, altered = None, None, frozenset()
    if witness:
        program, unrun = guarded(lambda: fp64_program(traced, tuple(input_shape)))
        altered = state_differences(traced, program) if program is not None else frozenset()
    converted = convert_traced(traced, input_shape, device, encoding_layer_placement=encoding_layer_placement,
                               packaging=packaging)
    return converted._replace(witness=program, witness_error=unrun, altered=altered)


def convert_traced(
    gm: fx.GraphModule,
    input_shape: Tuple[int, ...],
    device: Union[torch.device, str] = "cpu",
    *,
    encoding_layer_placement: str = "subsume",
    packaging: PackagingContract = SPIKING_PACKAGING,
) -> TracedConversion:
    """``trace_and_convert`` of a traced program (``trace_model``'s ``GraphModule``, which the conversion rewrites)."""
    device = torch.device(device)
    program = program_record(gm)
    purity = program_purity(gm)
    gm = normalize_fx_graph(gm, purity.foreign_modules)

    analyzer = RepresentabilityAnalyzer(gm, packaging=packaging, foreign_modules=purity.foreign_modules)
    report = analyzer.analyze()

    if not report.is_representable:
        raise RepresentabilityError(report)

    converter = MapperGraphConverter(gm, input_shape, packaging=packaging, purity=purity)
    mapper_repr = converter.convert(report)

    flow = ConvertedModelFlow(device, mapper_repr)
    flow.shape_literals = converter.shape_literals()
    flow.shape_proof = converter.shape_proof()
    flow.shape_impurity = purity.impurity
    flow.shape_opaque = converter.shape_opaque()
    # Flow birth for a torch module: the placement is resolved here, once, for
    # EVERY packaging contract. Encoding layers are an event-domain concept
    # (host-side spike-train generation), so a value-domain flow marks none —
    # but it is still STAMPED (not-applicable), because an unstamped graph is
    # indistinguishable from one nobody ever applied the placement to.
    mark_encoding_layers(flow.get_mapper_repr(), placement=encoding_layer_placement, packaging=packaging)

    return TracedConversion(flow.to(device), report, program)


def witness_converter(
    input_shape: Tuple[int, ...],
    *,
    encoding_layer_placement: str,
    packaging: PackagingContract,
) -> Callable[[nn.Module], WitnessConversion]:
    """The conversion the value probe applies to its fp64 witness (CPU, same placement and packaging): a witness
    program (``fp64_program``) is converted as traced, any other witness traced first."""

    def convert(witness: nn.Module) -> WitnessConversion:
        program = TracedProgram(witness) if isinstance(witness, fx.GraphModule) else witness
        traced = trace_and_convert(program, input_shape, "cpu", encoding_layer_placement=encoding_layer_placement,
                                   packaging=packaging)
        return WitnessConversion(traced.flow, traced.report.absorption_plan, traced.program)

    return convert


def _recording(convert: Callable[[nn.Module], WitnessConversion], into: List[WitnessConversion]):
    def record(witness: nn.Module) -> WitnessConversion:
        into.append(convert(witness))
        return into[-1]

    return record


def probe_stages(packaging: PackagingContract) -> Tuple[str, ...]:
    """The probe stages a conversion under ``packaging`` must pass: stage 2 for value-domain targets."""
    return (STAGE_CONVERTED, STAGE_VALUE_PROGRAM) if packaging.is_value_domain else (STAGE_CONVERTED,)


def _later_stages(packaging: PackagingContract):
    return (identity_value_program_stage,) if packaging.is_value_domain else ()


def probe_witness_conversion(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    *,
    packaging: PackagingContract,
    encoding_layer_placement: str = "subsume",
    contract: FunctionContract = NATIVE_EVAL_IDENTITY,
    strict: bool = True,
    context: str = "probe_witness_conversion",
) -> Tuple[ValueParityResult, ...]:
    """Every probe stage ``packaging`` requires, on an fp64 witness of ``model`` (the model is not touched)."""
    convert = witness_converter(tuple(input_shape), encoding_layer_placement=encoding_layer_placement,
                                packaging=packaging)
    return probe_conversion(fp64_snapshot(model), convert, tuple(input_shape), later_stages=_later_stages(packaging),
                            contract=contract, strict=strict, context=context)


def _witness_mismatch(traced: TracedConversion, witnessed: List[WitnessConversion],
                      altered: FrozenSet[str]) -> Optional[WitnessProgramMismatchError]:
    """The certificate covers the witness's program, derived from the returned conversion's own traced graph (review
    round 14): the two must still be the same program, node by node and in every tensor a node reads, by value
    (``program_binding``: the witness program is the traced graph's deep copy, the only copy made), and fold the same
    literals."""
    if witnessed and witnessed[0].program is not None:
        found = program_difference(traced.program, witnessed[0].program, altered)
        if found is not None:
            return WitnessProgramMismatchError.of_program(*found)
    flow = traced.flow
    witness_literals = getattr(witnessed[0].flow, "shape_literals", {}) if witnessed else flow.shape_literals
    differences = literal_differences(flow.shape_literals, witness_literals)
    return WitnessProgramMismatchError(differences) if differences else None


def _uncertified(flow: ConvertedModelFlow, reason: str, required: Tuple[str, ...]) -> ConvertedModelFlow:
    flow.conversion_certificate = ConversionCertificate.refused(reason, required, flow.shape_proof)
    warnings.warn(f"[convert_torch_model] {reason}", UnverifiedConversionWarning, stacklevel=3)
    return flow


def convert_torch_model(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    num_classes: int,
    device: Union[torch.device, str] = "cpu",
    Tq: int = 64,
    *,
    strict: bool = True,
    encoding_layer_placement: str = "subsume",
    packaging: PackagingContract = SPIKING_PACKAGING,
    value_program_stage: bool = False,
) -> ConvertedModelFlow:
    """Convert a trained native PyTorch model to a ``ConvertedModelFlow`` stamped with its conversion certificate.

    ``packaging`` selects the target family's packaging contract (default: today's spiking perceptron rule). ``Tq`` is
    unused (kept for API compatibility). The probes follow the conversion: the zero-input structural forward, then the
    value-parity probe on the fp64 witness program of the same traced graph (``witness.fp64_program``), with randomized
    BatchNorm statistics and all-zero parameters (stage 1, the converted model; with ``value_program_stage`` under a
    value-domain contract, stage 2, its identity value program, which the pipeline's conversion always runs). With
    ``strict=True`` a failure raises a ``ConversionProbeError``; otherwise the flow is returned with an uncertified
    stamp and an ``UnverifiedConversionWarning``. A certificate without every stage ``probe_stages(packaging)`` lists is
    incomplete, and a pipeline cache refuses to resume from it.
    """
    # Imported lazily so tests can monkeypatch conversion_probe.probe_forward at call time.
    from mimarsinan.torch_mapping.conversion_probe import probe_forward

    device = torch.device(device)
    required = probe_stages(packaging)
    traced = trace_and_convert(
        model, input_shape, device,
        encoding_layer_placement=encoding_layer_placement, packaging=packaging, witness=True,
    )
    flow, snapshot, unrun, altered = traced.flow, traced.witness, traced.witness_error, traced.altered

    model_param = next(model.parameters(), None)
    result = probe_forward(
        flow, input_shape, device=device, batch=1,
        strict=strict, context="convert_torch_model",
        dtype=model_param.dtype if model_param is not None else None,
    )
    if not result.ok:
        return _uncertified(flow, result.format(), required)
    if snapshot is None:
        refusal = WitnessProgramError(describe_failure(unrun) if unrun is not None else "no witness program")
        if strict:
            raise refusal from unrun
        return _uncertified(flow, str(refusal), required)

    witnessed: List[WitnessConversion] = []
    convert = witness_converter(tuple(input_shape), encoding_layer_placement=encoding_layer_placement,
                                packaging=packaging)
    results = probe_conversion(
        snapshot, _recording(convert, witnessed), tuple(input_shape),
        later_stages=_later_stages(packaging) if value_program_stage else (), strict=strict,
        context="convert_torch_model", native=native_reference(model),
    )
    for stage_result in results:
        print(stage_result.format())
    flow.conversion_certificate = ConversionCertificate.from_results(results, required, flow.shape_proof)
    mismatch = _witness_mismatch(traced, witnessed, altered)
    if mismatch is not None:
        if strict:
            raise mismatch
        flow.conversion_certificate = dataclasses.replace(flow.conversion_certificate, reason=str(mismatch))
    if not flow.conversion_certificate.certified:
        warnings.warn(
            f"[convert_torch_model] {flow.conversion_certificate.summary()}", UnverifiedConversionWarning,
            stacklevel=2,
        )
    return flow
