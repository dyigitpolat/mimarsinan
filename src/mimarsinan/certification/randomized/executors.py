"""The executors an X3a instance runs on, under its declared semantics, captured per neuron and window."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import torch

from mimarsinan.certification.randomized.instance import (
    ANALYTICAL_TTFS,
    CASCADED_TTFS,
    STREAMED_LIF,
    SYNCHRONIZED_LIF,
    SYNCHRONIZED_TTFS,
    VALUE,
    WINDOWED_LIF,
    Instance,
)
from mimarsinan.certification.randomized.program import VALUE_LAW
from mimarsinan.chip_simulation.deployed_host import DEPLOYED_HOST_DEVICE
from mimarsinan.chip_simulation.execution_bounds import resolve_simulation_step_timeout_s
from mimarsinan.chip_simulation.nevresim.nevresim_driver import resolve_nevresim_path
from mimarsinan.chip_simulation.recording.spike_modes import comb_spike_count
from mimarsinan.chip_simulation.simulation_runner.core import SimulationRunner
from mimarsinan.chip_simulation.spiking_mode_policy import policy_for_spiking_mode
from mimarsinan.chip_simulation.spiking_point import resolve_spiking_point
from mimarsinan.chip_simulation.spiking_semantics import lif_per_hop_retiming_enabled
from mimarsinan.chip_simulation.ttfs.ttfs_executor import run_ttfs_hybrid_contract
from mimarsinan.chip_simulation.value_run import ValueHybridCoreFlow
from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.ir import IRGraph
from mimarsinan.mapping.ir.deployment_dtype import SPIKING_ARITHMETIC, VALUE_ARITHMETIC
from mimarsinan.models.spiking.hybrid.carry import run_pass_transfer
from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow


@dataclass(frozen=True)
class SemanticsRun:
    """One declared semantics: its authored axes (a flat config) and the executor arguments they fold to."""

    name: str
    config: Mapping[str, Any]
    flow: Mapping[str, Any] = field(default_factory=dict)
    #: The integer readout law the semantics' programs carry (``activation_bits``, ``readout_law``): a
    #: function-bearing platform declaration held fixed (``screen.FIXED_AXES``), never drawn.
    law: Optional[Mapping[str, Any]] = None

    @property
    def transfer(self) -> str:
        """The pass-boundary discipline the toolchain's one rule gives this semantics (``run_pass_transfer``)."""
        return run_pass_transfer(dict(self.config))

    @property
    def retimed(self) -> bool:
        return lif_per_hop_retiming_enabled(dict(self.config))

    @property
    def decode(self) -> str:
        if self.name == VALUE:
            return "value"
        return policy_for_spiking_mode(resolve_spiking_point(self.config)).decode_mode()  # [U-LIF2 r2] its config's point


_TTFS_FLOW = dict(firing_mode="TTFS", spike_mode="TTFS")
RUNS: Dict[str, SemanticsRun] = {
    STREAMED_LIF: SemanticsRun(STREAMED_LIF, dict(spiking_family="lif", spiking_variant="streamed",
                                                  spiking_mode="lif"),
                               dict(spiking_mode="lif", cycle_accurate_lif_forward=True)),
    # [U-LIF2 W14] each synchronized LIF semantics declares the window firing its flow executes (the flow follows the
    # point: ``run_flow_point``)
    WINDOWED_LIF: SemanticsRun(WINDOWED_LIF, dict(spiking_family="lif", spiking_variant="synchronized",
                                                  spiking_mode="lif", lif_per_hop_retiming=True,
                                                  lif_window_firing="fire_while_integrating"),
                               dict(spiking_mode="lif", cycle_accurate_lif_forward=True)),
    SYNCHRONIZED_LIF: SemanticsRun(SYNCHRONIZED_LIF, dict(spiking_family="lif", spiking_variant="synchronized",
                                                          spiking_mode="lif", lif_window_firing="integrate_then_emit"),
                                   dict(spiking_mode="lif", cycle_accurate_lif_forward=True)),
    CASCADED_TTFS: SemanticsRun(CASCADED_TTFS, dict(spiking_family="ttfs", spiking_variant="cascaded",
                                                    spiking_mode="ttfs_cycle_based"),
                                dict(spiking_mode="ttfs_cycle_based", ttfs_cycle_schedule="cascaded", **_TTFS_FLOW)),
    SYNCHRONIZED_TTFS: SemanticsRun(SYNCHRONIZED_TTFS, dict(spiking_family="ttfs", spiking_variant="synchronized",
                                                            spiking_mode="ttfs_cycle_based"),
                                    dict(spiking_mode="ttfs_cycle_based", ttfs_cycle_schedule="synchronized",
                                         **_TTFS_FLOW)),
    ANALYTICAL_TTFS: SemanticsRun(ANALYTICAL_TTFS, dict(spiking_family="ttfs", spiking_variant="analytical",
                                                        spiking_mode="ttfs"),
                                  dict(spiking_mode="ttfs", **_TTFS_FLOW)),
    VALUE: SemanticsRun(VALUE, dict(core_semantics="mvm"), law=VALUE_LAW),
}


def input_tensor(instance: Instance, dtype: torch.dtype = SPIKING_ARITHMETIC.input_dtype) -> torch.Tensor:
    """The executor input: under a spiking semantics the midpoint rate ``(k + 1/2) / T`` of each count ``k``,
    the integer values under the value one; [U-DAR] at the record's spiking input dtype unless declared (the value
    executors declare theirs).

    The identity encoder integrates its rate for ``T`` cycles (threshold 1, subtractive reset, float32), so it
    emits ``floor(T x)`` spikes up to rounding: the rate ``k / T`` itself sits on that floor's knife edge and
    loses a spike where it is inexact (``float32(5/6) * 6 < 5``), while the midpoint keeps half a spike of
    margin, so the encoder emits exactly ``k`` (and ``T`` at ``k = T``, one spike per cycle at most)."""
    x = torch.tensor(instance.inputs, dtype=torch.float64)
    if instance.semantics != VALUE:
        x = (x + 0.5) / float(instance.timesteps)
    return x.to(dtype)


def encoder_counts(mapping, instance: Instance) -> torch.Tensor:
    """``(B, input_width)`` int64: the spike counts the program's own input encoder emits on the executor input.

    The encoder is the compute stage that reads the raw input (node -2); it runs here exactly as the flow
    runs it (``ComputeOp.execute``), and the comb count of its window rate (``comb_spike_count``, the chip's
    llround rule) is the count every segment entry re-encodes. Refuses a program with no such stage (the
    value semantics reads its input raw)."""
    x = input_tensor(instance)
    for stage in mapping.stages:
        op = getattr(stage, "compute_op", None)
        if stage.kind != "compute" or op is None:
            continue
        reads = {int(s.node_id) for s in op.input_sources.flatten() if hasattr(s, "node_id")}
        if reads == {-2}:
            with torch.no_grad():
                rate = op.execute(x, {}).detach().cpu()
            return comb_spike_count(rate, instance.timesteps)
    raise CoverageError("the program has no input encoder stage (a compute stage reading node -2 only)")


def run_nodes(mapping, instance: Instance, *, transfer: Optional[str] = None,
              value_dtype: torch.dtype = VALUE_ARITHMETIC.core_dtype, spike_mode: str = "Uniform",
              phase_dither: bool = False) -> Dict[int, torch.Tensor]:
    """``{node_id: (B, width)}`` for every neural node the program executes, and the final output (key -1).

    Rate-decoded runs report each stage's raw window counts through the flow's ``stage_count_recorder``
    seam (``phase_dither`` declares the phase-dithered Uniform encode); timing-decoded runs report the TTFS
    contract state buffer; the value run its retained state.
    """
    run = RUNS[instance.semantics]
    if run.decode == "value":
        flow = ValueHybridCoreFlow(mapping, device=VALUE_ARITHMETIC.device, dtype=value_dtype)
        with torch.no_grad():
            out, state, _x = flow.run_retaining_state(input_tensor(instance, value_dtype))
        nodes = {int(k): v.detach().cpu().double() for k, v in state.items() if int(k) >= 0}
        nodes[-1] = out.detach().cpu().double()
        return nodes
    if run.decode == "timing":
        return _timing_nodes(mapping, instance, run)
    kwargs = dict(run.flow)
    kwargs.setdefault("spike_mode", spike_mode)
    # [U-LIF2 W14, n-7] the flow runs the semantics' point: its two-window flag follows the point's window firing
    point = resolve_spiking_point({**run.config, **({"spike_phase_dither": True} if phase_dither else {})})
    flow = SpikingHybridCoreFlow((instance.network.input_width,), mapping, simulation_length=instance.timesteps,
                                 thresholding_mode="<=", pass_transfer=transfer or run.transfer,
                                 phase_dither=phase_dither, lif_execution_synchronized=point.integrates_then_emits,
                                 spiking_point=point, **kwargs)
    captured: Dict[int, torch.Tensor] = {}

    def record(stage, counts):
        for sl in stage.output_map:
            captured[int(sl.node_id)] = counts[:, sl.offset:sl.offset + sl.size].detach().cpu().double()

    flow.stage_count_recorder = record
    with torch.no_grad():
        captured[-1] = flow(input_tensor(instance)).detach().cpu().double()
    return captured


#: The semantics nevresim runs here: the LIF record build (window counts, verbatim pass carry) and cascaded
#: TTFS (single-spike cascade, decoded segment values).
NEVRESIM_SEMANTICS = (STREAMED_LIF, CASCADED_TTFS)


class _ChipProbe(SimulationRunner):
    """The nevresim hybrid runner over in-memory samples: no pipeline, no data provider, counts captured."""

    def __init__(self, mapping, instance: Instance, workdir: str, transfer: str):
        self.stage_count_recorder = None
        self.stage_timer = None
        self.nevresim_total_spikes = None
        x = input_tensor(instance).numpy()
        self.test_data = [(x[i], np.zeros(1)) for i in range(x.shape[0])]
        self.test_targets = [torch.zeros(1, dtype=torch.long) for _ in range(x.shape[0])]
        self.num_classes = max(1, len(mapping.output_sources.flatten()))
        self.working_directory = workdir
        run = RUNS[instance.semantics]
        ttfs = run.flow["spiking_mode"] != "lif"
        self.weight_type = int
        self.threshold_type = float if ttfs else int
        self.spike_generation_mode = "TTFS" if ttfs else "Uniform"
        self.firing_mode = "TTFS" if ttfs else "Default"
        self.thresholding_mode = "<="
        self.spiking_mode = run.flow["spiking_mode"]
        self.spiking_point = resolve_spiking_point(run.config)  # [U-LIF2 r2] the semantics' record, never a mode's
        self.host_compute_device = DEPLOYED_HOST_DEVICE  # [U-HDA] the deployed host's device
        self.nevresim_connectivity_mode = "compile_time"
        self.pass_transfer = transfer
        self.simulation_step_timeout_s = resolve_simulation_step_timeout_s(None)
        self.membrane_readout = False
        self.membrane_half_step_charge = 0.0
        self.simulation_length = instance.timesteps
        self.mapping = mapping

        self.compile_s = 0.0

    def _prepare_all_segments(self, hybrid):
        started = time.perf_counter()
        try:
            return super()._prepare_all_segments(hybrid)
        finally:
            self.compile_s += time.perf_counter() - started

    def _evaluate_chip_output(self, predictions) -> float:
        return float("nan")


def run_nevresim_nodes(mapping, instance: Instance, workdir: str, *,
                       transfer: Optional[str] = None) -> Tuple[Dict[int, torch.Tensor], float]:
    """``({node_id: (B, width)} window counts of every neural stage, compile seconds)``, run on nevresim."""
    if instance.semantics not in NEVRESIM_SEMANTICS:
        raise ValueError(f"nevresim runs {NEVRESIM_SEMANTICS} here, not {instance.semantics!r}")
    resolve_nevresim_path()
    probe = _ChipProbe(mapping, instance, workdir, transfer or RUNS[instance.semantics].transfer)
    captured: Dict[int, torch.Tensor] = {}

    def record(stage, raw):
        counts = torch.as_tensor(np.asarray(raw, dtype=np.float64))
        for sl in stage.output_map:
            captured[int(sl.node_id)] = counts[:, sl.offset:sl.offset + sl.size]

    probe.stage_count_recorder = record
    probe._run_hybrid(mapping)
    return captured, probe.compile_s


def _timing_nodes(mapping, instance: Instance, run: SemanticsRun) -> Dict[int, torch.Tensor]:
    rows: Dict[int, list] = {}
    x = input_tensor(instance, torch.float64).numpy()
    neural = {int(sl.node_id) for s in mapping.stages if s.kind == "neural" for sl in s.output_map}
    outputs = []
    for b in range(x.shape[0]):
        result = run_ttfs_hybrid_contract(mapping, x[b:b + 1], simulation_length=instance.timesteps,
                                          spiking_mode=run.flow["spiking_mode"], sample_index=b,
                                          ttfs_cycle_schedule=run.flow.get("ttfs_cycle_schedule"))
        for node_id in neural:
            rows.setdefault(node_id, []).append(np.asarray(result.state_buffer[node_id], dtype=np.float64))
        outputs.append(_gathered_output(mapping, result.state_buffer, x[b:b + 1]))
    nodes = {k: torch.from_numpy(np.concatenate(v, axis=0)) for k, v in rows.items()}
    nodes[-1] = torch.from_numpy(np.concatenate(outputs, axis=0))
    return nodes


def _gathered_output(mapping, state_buffer, x) -> np.ndarray:
    from_sources = []
    for src in mapping.output_sources.flatten():
        node_id, index = int(src.node_id), int(src.index)
        if node_id == -2:
            from_sources.append(float(x[0, index]))
        elif node_id == -3:
            from_sources.append(1.0)
        elif node_id < 0:
            from_sources.append(0.0)
        else:
            from_sources.append(float(np.asarray(state_buffer[node_id]).reshape(-1)[index]))
    return np.asarray([from_sources], dtype=np.float64)


def per_neuron(ir: IRGraph, nodes: Mapping[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
    """``{perceptron_index: (B, n)}``: node outputs placed by IR provenance (output column x channels + slice).

    Unlike the certificate assembler this keeps values unrounded (the value semantics) and refuses an
    incompletely covered perceptron instead of dropping it, so no comparison is ever vacuous.
    """
    placed: Dict[int, list] = {}
    for node in ir.get_neural_cores():
        if node.perceptron_index is None or int(node.id) not in nodes:
            continue
        column = int(getattr(node, "perceptron_output_column", None) or 0)
        start, end = getattr(node, "perceptron_output_slice", None) or (0, nodes[int(node.id)].shape[1])
        placed.setdefault(int(node.perceptron_index), []).append((column, int(start), int(end),
                                                                   nodes[int(node.id)]))
    out: Dict[int, torch.Tensor] = {}
    for index, pieces in placed.items():
        channels = max(end for _c, _s, end, _t in pieces)
        columns = max(c for c, _s, _e, _t in pieces) + 1
        batch = pieces[0][3].shape[0]
        buf = torch.full((batch, columns * channels), float("nan"), dtype=torch.float64)
        writes = torch.zeros(columns * channels, dtype=torch.int64)
        for column, start, end, tensor in pieces:
            buf[:, column * channels + start:column * channels + end] = tensor.double()
            writes[column * channels + start:column * channels + end] += 1
        if int(writes.min()) != 1 or int(writes.max()) != 1:
            raise CoverageError(f"perceptron {index}: neuron coverage min {int(writes.min())} max "
                                f"{int(writes.max())}")
        out[index] = buf
    return out


class CoverageError(ToolchainRefusal, AssertionError):
    """A perceptron whose neurons the run did not report exactly once (a capture defect, never compared)."""
