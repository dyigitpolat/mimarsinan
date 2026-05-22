"""Canonical TTFS neural-segment execution (HCM, SANA-FE, Nevresim alignment)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from mimarsinan.chip_simulation.hybrid_execution import (
    assemble_segment_input_numpy,
    execute_compute_op_numpy,
    resolve_stage_compute_scales,
    store_segment_output_numpy,
)
from mimarsinan.chip_simulation.hybrid_semantics import (
    NeuralSegmentResult,
    is_ttfs_spiking_mode,
    store_neural_segment_output,
)
from mimarsinan.chip_simulation.hybrid_stage_runner import run_hybrid_stages
from mimarsinan.chip_simulation.ttfs_segment import (
    run_ttfs_continuous_segment,
    run_ttfs_quantized_segment,
    segment_ttfs_arrays_from_mapping,
)
from mimarsinan.mapping.core_geometry import used_neurons

_RAW_INPUT_NODE_ID = -2
_CONTRACT_DTYPE = np.float64


@dataclass
class TtfsContractNeuralStageResult:
    """Contract outputs for one hybrid neural stage (analytical TTFS only)."""

    segment_record: Any  # SegmentTtfsRecord
    neural_result: NeuralSegmentResult
    membrane_voltages: List[np.ndarray]
    seg_input: np.ndarray


@dataclass
class TtfsContractRunResult:
    """Full hybrid TTFS contract run (numpy/float64 compute path)."""

    record: Any  # TtfsRunRecord
    state_buffer: Dict[int, np.ndarray]


@dataclass(frozen=True)
class TtfsContractComputeStageResult:
    """Contract outputs for one hybrid compute stage."""

    op_id: int
    output: np.ndarray


class TtfsAnalyticalExecutor:
    """Closed-form TTFS segment semantics shared across simulators."""

    def run_segment(
        self,
        hcm: Any,
        seg_input: np.ndarray,
        *,
        simulation_length: int,
        spiking_mode: str,
    ) -> NeuralSegmentResult:
        if not is_ttfs_spiking_mode(spiking_mode):
            raise ValueError(
                f"TtfsAnalyticalExecutor requires TTFS mode, got {spiking_mode!r}"
            )
        seg_in = np.asarray(seg_input, dtype=np.float64)
        if seg_in.ndim != 2:
            raise ValueError(f"seg_input must be 2-D; got shape {seg_in.shape}")
        seg_arrays = segment_ttfs_arrays_from_mapping(hcm)
        quantized = spiking_mode == "ttfs_quantized"
        if quantized:
            seg_out, bufs = run_ttfs_quantized_segment(
                seg_arrays, seg_in, int(simulation_length),
            )
        else:
            seg_out, bufs = run_ttfs_continuous_segment(seg_arrays, seg_in)

        per_core: List[np.ndarray] = []
        for ci, core in enumerate(hcm.cores):
            n_out = used_neurons(core, min_one=True)
            if n_out <= 0:
                per_core.append(
                    np.zeros((seg_in.shape[0], 0), dtype=np.float64)
                )
                continue
            per_core.append(bufs[ci][:, :n_out].astype(np.float64, copy=False))

        return NeuralSegmentResult(
            inter_stage=np.asarray(seg_out, dtype=np.float64),
            per_core_activations=per_core,
        )

    def membrane_voltages(
        self,
        hcm: Any,
        seg_input: np.ndarray,
        *,
        simulation_length: int,
        spiking_mode: str,
    ) -> List[np.ndarray]:
        """Per-core ``V = W @ a + b`` before activation (SANA-FE preset injection)."""
        from mimarsinan.chip_simulation.ttfs_segment import ttfs_core_membrane_voltages

        seg_arrays = segment_ttfs_arrays_from_mapping(hcm)
        return ttfs_core_membrane_voltages(
            seg_arrays,
            np.asarray(seg_input, dtype=np.float64),
            simulation_length=simulation_length,
            spiking_mode=spiking_mode,
        )


def _segment_record_from_neural_result(
    *,
    stage_index: int,
    stage: Any,
    hcm: Any,
    result: NeuralSegmentResult,
) -> Any:
    from mimarsinan.chip_simulation.ttfs_recorder import (
        CoreTtfsActivations,
        SegmentTtfsRecord,
    )

    cores_rec = []
    per_core = result.per_core_activations or []
    for ci, core in enumerate(hcm.cores):
        n_out = used_neurons(core, min_one=True)
        if n_out <= 0:
            continue
        if ci >= len(per_core):
            continue
        act = per_core[ci]
        if act is None or act.size == 0:
            continue
        from mimarsinan.chip_simulation.ttfs_recorder import (
            normalize_core_output_activation,
        )

        cores_rec.append(CoreTtfsActivations(
            core_index=ci,
            n_out_used=n_out,
            output_activation=normalize_core_output_activation(
                act, n_out_used=n_out,
            ),
        ))
    inter = np.asarray(result.inter_stage, dtype=np.float64)
    seg_out = inter[0] if inter.ndim >= 2 else inter
    return SegmentTtfsRecord(
        stage_index=stage_index,
        stage_name=stage.name,
        schedule_segment_index=getattr(stage, "schedule_segment_index", None),
        schedule_pass_index=getattr(stage, "schedule_pass_index", None),
        seg_output=np.asarray(seg_out, dtype=np.float64).reshape(-1),
        cores=cores_rec,
    )


def run_ttfs_contract_neural_stage(
    mapping: Any,
    stage: Any,
    stage_index: int,
    state_buffer: Dict[int, np.ndarray],
    *,
    simulation_length: int,
    spiking_mode: str,
    executor: TtfsAnalyticalExecutor | None = None,
) -> TtfsContractNeuralStageResult:
    """Run one neural stage on the shared TTFS contract path (float64 numpy)."""
    hcm = stage.hard_core_mapping
    assert hcm is not None
    exec_ = executor or TtfsAnalyticalExecutor()
    num_samples = 1
    for buf in state_buffer.values():
        if getattr(buf, "ndim", 0) >= 2:
            num_samples = int(buf.shape[0])
            break
    seg_in = assemble_segment_input_numpy(
        stage.input_map, state_buffer, num_samples=num_samples, dtype=_CONTRACT_DTYPE,
    )
    result = exec_.run_segment(
        hcm, seg_in, simulation_length=simulation_length, spiking_mode=spiking_mode,
    )
    membrane_V = exec_.membrane_voltages(
        hcm, seg_in, simulation_length=simulation_length, spiking_mode=spiking_mode,
    )
    segment_record = _segment_record_from_neural_result(
        stage_index=stage_index, stage=stage, hcm=hcm, result=result,
    )
    store_neural_segment_output(
        spiking_mode, stage.output_map, state_buffer, result,
    )
    return TtfsContractNeuralStageResult(
        segment_record=segment_record,
        neural_result=result,
        membrane_voltages=membrane_V,
        seg_input=seg_in,
    )


def run_ttfs_contract_compute_stage(
    mapping: Any,
    stage: Any,
    state_buffer: Dict[int, np.ndarray],
    sample_input: np.ndarray,
) -> TtfsContractComputeStageResult:
    """Run one compute stage on the shared TTFS contract path (float64 numpy)."""
    op = stage.compute_op
    assert op is not None
    in_scale, out_scale = resolve_stage_compute_scales(
        mapping, op.id, apply_ttfs=True,
    )
    result = execute_compute_op_numpy(
        op,
        np.asarray(sample_input, dtype=_CONTRACT_DTYPE),
        state_buffer,
        in_scale=in_scale,
        out_scale=out_scale,
        dtype=_CONTRACT_DTYPE,
    )
    out = np.asarray(result, dtype=_CONTRACT_DTYPE)
    state_buffer[op.id] = out
    return TtfsContractComputeStageResult(op_id=int(op.id), output=out)


def run_ttfs_hybrid_contract(
    mapping: Any,
    sample_input: np.ndarray,
    *,
    simulation_length: int,
    spiking_mode: str,
    sample_index: int = 0,
) -> TtfsContractRunResult:
    """Execute the full hybrid mapping on the canonical TTFS contract path."""
    from mimarsinan.chip_simulation.ttfs_recorder import TtfsRunRecord

    if not is_ttfs_spiking_mode(spiking_mode):
        raise ValueError(
            f"run_ttfs_hybrid_contract requires TTFS mode, got {spiking_mode!r}"
        )
    x = np.asarray(sample_input, dtype=_CONTRACT_DTYPE)
    if x.ndim != 2 or x.shape[0] != 1:
        raise ValueError(f"sample_input must be shape (1, D); got {x.shape}")

    state_buffer: Dict[int, np.ndarray] = {_RAW_INPUT_NODE_ID: x}
    record = TtfsRunRecord(
        sample_index=int(sample_index),
        simulation_length=int(simulation_length),
        spiking_mode=str(spiking_mode),
    )
    executor = TtfsAnalyticalExecutor()
    T = int(simulation_length)

    def _on_neural(stage_index: int, stage: Any, buf: Dict[int, np.ndarray]) -> None:
        out = run_ttfs_contract_neural_stage(
            mapping,
            stage,
            stage_index,
            buf,
            simulation_length=T,
            spiking_mode=spiking_mode,
            executor=executor,
        )
        record.segments[stage_index] = out.segment_record

    def _on_compute(_stage_index: int, stage: Any, buf: Dict[int, np.ndarray]) -> None:
        result = run_ttfs_contract_compute_stage(mapping, stage, buf, x)
        record.compute_outputs[result.op_id] = result.output

    run_hybrid_stages(
        mapping,
        state_buffer,
        on_neural=_on_neural,
        on_compute=_on_compute,
    )
    return TtfsContractRunResult(record=record, state_buffer=state_buffer)
