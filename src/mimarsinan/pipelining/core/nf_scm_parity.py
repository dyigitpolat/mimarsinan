"""NF↔SCM per-neuron parity gate (rung 1 ↔ rung 2) for analytic TTFS schedules."""

from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import torch

from mimarsinan.common.env import nf_scm_parity_debug_enabled


class NfScmParityError(AssertionError):
    """Per-neuron divergence between the analytical NF and the identity mapping.

    ``mismatch_fraction`` is the measured per-neuron flip fraction (``None`` for
    decision-level and structural raises that have no such scalar).
    """

    def __init__(self, message: str, *, mismatch_fraction: float | None = None):
        super().__init__(message)
        self.mismatch_fraction = mismatch_fraction


def _unify_model_device(model):
    """Place the WHOLE model on one device and return it (``None`` if param-less).

    Prevents a cross-device matmul when the mapper graph left modules on different
    devices; prefers a CUDA device when the model holds any CUDA parameter.
    """
    params = list(model.parameters())
    if not params:
        return None
    device = next(
        (p.device for p in params if p.device.type == "cuda"), params[0].device,
    )
    model.to(device)
    return device


def nf_scm_parity_enabled(contract: Any) -> bool:
    """Whether this mode's NF can be held per-neuron against the deployed executor.

    Continuous ttfs gets the per-neuron gate; cascaded gets a decision-level gate;
    the floor+half-step-bias convention modes (ttfs_quantized, synchronized floor-collapse) are excluded.
    """
    if contract.is_streamed_lif():
        # [P3] streamed lif holds EXACTLY: the NF train forward IS the
        # deployed streaming cascade, so the gate compares at atol=0.
        return True
    if contract.uses_ttfs_floor_ceil_convention():
        return False
    if contract.is_cascaded():
        return True
    return contract.training_forward_kind() == "analytical_staircase"


def readout_decision_drift_enabled(contract: Any) -> bool:
    """Whether the diagnostic readout-drift statistic is measured for this mode.

    Every mode with a faithful identity executor gets it: analytic/cascaded TTFS,
    the synchronized grid-snap, and windowed LIF rate cascades (W1c: the LIF gap
    let the t0_03 NF↔SCM defect surface as a retention abort). Streamed lif is
    EXCLUDED: there the NF train forward IS the deployed streaming cascade, so
    every readout count is already held at atol=0 by the exactness gate
    (``assert_streamed_nf_scm_exact_or_raise``) and a second, weaker read of the
    same hop can only add noise to a question already answered.
    """
    from mimarsinan.chip_simulation.spiking_semantics import is_lif

    if contract.is_streamed_lif():
        return False
    return (
        nf_scm_parity_enabled(contract)
        or contract.is_synchronized()
        or is_lif(contract.spiking_mode)
    )


def assert_nf_scm_parity_or_raise(
    pipeline,
    model,
    ir_graph,
    samples: torch.Tensor,
    *,
    atol: float = 1e-9,
    max_mismatch_fraction: float = 0.0,
) -> float:
    """Compare per-neuron NF activations against the identity-mapped contract run.

    Both sides are compared in the normalized [0, 1] TTFS domain; ``max_mismatch_fraction``
    budgets the honest mapping-level wire residual. Returns the measured fraction.
    """
    from mimarsinan.chip_simulation.deployment_contract import SpikingDeploymentContract
    from mimarsinan.pipelining.core.simulation_factory import (
        build_identity_mapping_for_pipeline,
    )

    contract = SpikingDeploymentContract.from_pipeline_config(pipeline.config)
    if contract.is_synchronized() and "forward" in getattr(model, "__dict__", {}):
        raise NfScmParityError(
            "NF↔SCM parity: synchronized NF must run the class-level analytical "
            "forward, but the model carries an instance forward override (a "
            "pre-schedule-aware-tuner cascade forward from a legacy cache?). "
            "Re-run TTFS Cycle Fine-Tuning or strip model.__dict__['forward']."
        )
    identity_mapping = build_identity_mapping_for_pipeline(
        ir_graph, pipeline_config=pipeline.config,
    )

    from mimarsinan.mapping.pruning import derive_deployed_neuron_survival

    nf = _capture_nf_normalized(model, samples)
    # Project the NF onto neurons actually deployed after pruning (the pruned ir_graph is the survival authority; no-op when nothing was pruned).
    nf = derive_deployed_neuron_survival(ir_graph).project(nf)
    scm = _collect_scm_normalized(identity_mapping, model, samples, contract)

    shared = sorted(set(nf) & set(scm))
    if not shared:
        raise NfScmParityError(
            "NF↔SCM parity: no comparable perceptrons (no on-chip cores carry "
            "a perceptron_index; run assign_perceptron_indices before mapping)"
        )

    debug = nf_scm_parity_debug_enabled()
    if debug:
        print(
            f"[nf_scm_parity] nf-only perceptrons: {sorted(set(nf) - set(scm))} "
            f"scm-only: {sorted(set(scm) - set(nf))}"
        )

    mismatches, total, worst = compare_normalized_records(
        nf, scm, atol=atol, debug=debug,
    )

    fraction = mismatches / max(total, 1)
    if fraction > max_mismatch_fraction:
        assert worst is not None, "mismatches > 0 implies a worst record"
        d, pi, s_idx, rank, nf_v, scm_v = worst
        raise NfScmParityError(
            f"NF↔SCM per-neuron parity failed: {mismatches}/{total} values "
            f"differ beyond atol={atol} (fraction {fraction:.4f} > budget "
            f"{max_mismatch_fraction}). Worst: perceptron {pi} sample {s_idx} "
            f"sorted-rank {rank}: nf={nf_v!r} scm={scm_v!r} (|Δ|={d!r})",
            mismatch_fraction=fraction,
        )
    return fraction


def _build_cascaded_identity_executor(pipeline, model, ir_graph):
    from mimarsinan.chip_simulation.deployment_contract import SpikingDeploymentContract
    from mimarsinan.models.spiking.hybrid.identity_flow import (
        build_identity_spiking_flow,
    )

    cfg = pipeline.config
    contract = SpikingDeploymentContract.from_pipeline_config(cfg)
    return build_identity_spiking_flow(
        cfg["input_shape"],
        ir_graph,
        contract.simulation_steps,
        getattr(model, "preprocessor", None),
        contract.firing_mode,
        contract.spike_generation_mode,
        thresholding_mode=contract.thresholding_mode,
        spiking_mode=contract.spiking_mode,
        ttfs_cycle_schedule=contract.ttfs_cycle_schedule,
    ).eval()


def _build_streamed_identity_executor(pipeline, model, ir_graph):
    from mimarsinan.chip_simulation.deployment_contract import SpikingDeploymentContract
    from mimarsinan.models.spiking.hybrid.identity_flow import (
        build_identity_spiking_flow,
    )
    from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan

    cfg = pipeline.config
    contract = SpikingDeploymentContract.from_pipeline_config(cfg)
    plan = DeploymentPlan.of(pipeline)
    return build_identity_spiking_flow(
        cfg["input_shape"],
        ir_graph,
        contract.simulation_steps,
        getattr(model, "preprocessor", None),
        contract.firing_mode,
        contract.spike_generation_mode,
        thresholding_mode=contract.thresholding_mode,
        spiking_mode=contract.spiking_mode,
        ttfs_cycle_schedule=contract.ttfs_cycle_schedule,
        cycle_accurate_lif_forward=True,
        phase_dither=contract.spike_phase_dither,
        lif_membrane_init=contract.lif_membrane_init,
        membrane_integer_lattice=bool(plan.weight_quantization),
        soma_law=contract.soma_law(),
    ).eval()


def assert_streamed_nf_scm_exact_or_raise(
    pipeline,
    model,
    ir_graph,
    samples: torch.Tensor,
) -> None:
    """[P3] streamed-lif exactness gate: per-neuron WINDOW COUNTS of the NF
    forward must equal the identity-mapped streaming executor at atol=0 —
    no mismatch budget; parity holds by construction or the deployment is
    wrong."""
    from mimarsinan.chip_simulation.deployment_contract import SpikingDeploymentContract
    from mimarsinan.mapping.pruning import derive_deployed_neuron_survival
    from mimarsinan.models.nn.lif_kernels import measurement_plane
    from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
    from mimarsinan.spiking.lif_utils import arm_integer_membrane_lattice

    executor = _build_streamed_identity_executor(pipeline, model, ir_graph)
    if DeploymentPlan.of(pipeline).weight_quantization:
        # Tuning stages recreate activations; re-arm from parameter_scale so
        # the NF decides ties by the exact lattice value (like the chip).
        # The snap fires only inside the measurement plane below — tuning
        # telemetry keeps the continuous membrane.
        arm_integer_membrane_lattice(model)
    device = _unify_model_device(model)
    if device is not None:
        samples = samples.to(device)
        executor = executor.to(device)
    T = float(executor.simulation_length)

    # Under a per-event law the window count is a PROJECTION of the record:
    # equal counts with a different per-cycle rhythm are different
    # computations at the next hop, so the gate compares rasters too.
    per_event = SpikingDeploymentContract.from_pipeline_config(
        pipeline.config).soma_law().is_per_event

    def _counts(core_record):
        return core_record.output_spike_count[: core_record.n_out_used]

    def _raster(core_record):
        raster = core_record.output_spike_raster
        assert raster is not None, (
            "streamed NF↔SCM raster gate: the executor recorded no per-cycle "
            "raster under firing_granularity='per_event'"
        )
        # Neuron-major flatten so a perceptron's tiles concatenate in the same
        # order the NF's own (T, n) train does.
        return raster[:, : core_record.n_out_used].T.reshape(-1)

    # BOTH twins read inside the SAME measurement plane: an asymmetric wrap
    # re-introduces tie mismatches in opposite directions (n8e canary).
    per_sample: List[Dict[int, np.ndarray]] = []
    per_sample_raster: List[Dict[int, np.ndarray]] = []
    with measurement_plane():
        nf_counts_by_pi = _capture_nf_streamed_counts(model, samples)
        # Project the NF onto neurons actually deployed after pruning (the pruned ir_graph is the survival authority; no-op when nothing was pruned).
        nf_counts_by_pi = derive_deployed_neuron_survival(ir_graph).project(
            nf_counts_by_pi,
        )
        nf_rasters_by_pi = (
            _capture_nf_streamed_rasters(model, samples) if per_event else {}
        )
        with torch.no_grad():
            for i in range(samples.shape[0]):
                _, record = executor.forward_with_recording(
                    samples[i : i + 1], sample_index=i,
                )
                per_sample.append(_group_record_by_perceptron(
                    record, executor.hybrid_mapping, values_of=_counts,
                ))
                if per_event:
                    per_sample_raster.append(_group_record_by_perceptron(
                        record, executor.hybrid_mapping, values_of=_raster,
                    ))
    scm_counts = {
        pi: np.stack([sample_vals[pi] for sample_vals in per_sample])
        for pi in per_sample[0]
    }
    if per_event:
        _assert_raster_exact_or_raise(
            nf_rasters_by_pi, per_sample_raster, n_samples=int(samples.shape[0]),
        )
    # Per-cycle spike sums ÷ scale are the integer window counts; rint
    # recovers them exactly (float noise ≪ 0.5, a real miss is ≥ 1 count).
    nf_counts = {pi: np.rint(nf_counts_by_pi[pi]) for pi in scm_counts}
    mismatches, total, worst = compare_normalized_records(
        nf_counts, scm_counts, atol=0.0,
    )
    if mismatches:
        per_pi = {
            pi: int((np.sort(nf_counts[pi], axis=1)
                     != np.sort(scm_counts[pi], axis=1)).sum())
            for pi in scm_counts
        }
        raise NfScmParityError(
            f"streamed NF↔SCM exactness violated: {mismatches}/{total} "
            f"neuron-window count mismatches over {int(samples.shape[0])} "
            f"samples (atol=0; worst={worst}; per-perceptron={per_pi}). "
            f"Streamed lif admits NO tolerance — the NF train forward must BE "
            f"the deployed streaming cascade (check depth-balancing relays / "
            f"latency +1 invariant / boundary config drift)."
        )


def assert_cascaded_nf_scm_agreement_or_raise(
    pipeline,
    model,
    ir_graph,
    samples: torch.Tensor,
    *,
    min_agreement: float = 0.98,
) -> float:
    """Decision-level cascaded gate: NF argmax must agree with the identity-mapped executor on ``min_agreement`` of samples.

    Healthy agreement is ~1.0 (driver==executor bit-exact once bias references stay
    live); a wrong-NF-dynamics regression craters it. Returns the measured agreement.
    """
    executor = _build_cascaded_identity_executor(pipeline, model, ir_graph)
    device = _unify_model_device(model)
    if device is not None:
        samples = samples.to(device)
        executor = executor.to(device)
    with torch.no_grad():
        nf_pred = model(samples).argmax(dim=1)
        scm_pred = executor(samples).argmax(dim=1)
    agreement = float((nf_pred == scm_pred).double().mean())
    if agreement < float(min_agreement):
        raise NfScmParityError(
            f"NF↔SCM cascaded decision agreement failed: {agreement:.4f} < "
            f"min_agreement={min_agreement} over {int(samples.shape[0])} "
            f"samples (healthy WQ tie-flip residual measures ~0.95; the "
            f"wrong-NF-dynamics incident class craters this)"
        )
    return agreement


def torch_parity_reference(model):
    """The torch side of the deployed-sim parity check, corrected for the
    floor-convention double shift (T6 Part A).

    The mapping-time half-step bias compensation is baked BEFORE this gate runs,
    while the trained activation stack still carries the training-time half-step
    ShiftDecorator — the raw reference would read half a level high. Neutralize
    the trained shift on the comp-baked perceptrons of a DEEPCOPY (encoders keep
    their trained convention: they deploy as host ops running the trained torch
    module). Returns ``model`` itself when nothing is baked.
    """
    # Lazy: mapping.support pulls transformations/tuning; keep this module's
    # import surface minimal like its chip_simulation imports.
    from mimarsinan.mapping.support.bias_compensation import TTFS_COMP_BAKED_FLAG

    perceptrons = list(model.get_perceptrons())
    baked = [bool(getattr(p, TTFS_COMP_BAKED_FLAG, False)) for p in perceptrons]
    if not any(baked):
        return model
    reference = copy.deepcopy(model)
    for perceptron, is_baked in zip(reference.get_perceptrons(), baked):
        if is_baked:
            _neutralize_trained_halfstep(perceptron)
    return reference


def _neutralize_trained_halfstep(perceptron) -> None:
    """Zero every ShiftDecorator in the perceptron's activation stack (on the
    floor path the only shift is the quantize decorator's trained half-step)."""
    from mimarsinan.models.nn.decorators.adjustment import iter_activation_tree
    from mimarsinan.models.nn.decorators.transforms import ShiftDecorator

    for node in iter_activation_tree(perceptron.activation):
        if isinstance(node, ShiftDecorator):
            node.shift = torch.zeros_like(torch.as_tensor(node.shift))


def _readout_count_records(nf: torch.Tensor, sim: torch.Tensor):
    """Both readouts as per-CLASS integer count records, on ONE lattice.

    The deployed flow's logits ARE the readout's integer lattice — window
    counts for a rate decode, TTFS levels for a timing decode, both emitted as
    ``value * T``. The torch twin holds the same integers times one positive
    readout gauge (the firing gain and the window decode are a single
    constant), recovered as the ratio of the two sides' total emitted
    magnitude: exact when the twins agree, and unmoved by a few large
    disagreements the way a squared fit is not. A scalar gauge cannot absorb a
    per-neuron count difference, which is the thing being measured. Grouping BY
    CLASS keeps the comparison positional — ``compare_normalized_records``
    sorts each record row, and a one-column row is already sorted.
    """
    if nf.shape != sim.shape:
        raise NfScmParityError(
            f"readout decision drift: the torch twin returned shape "
            f"{tuple(nf.shape)} and the deployed sim {tuple(sim.shape)}; the "
            f"two readouts are not the same neurons."
        )
    nf_np = nf.detach().to(torch.float64).cpu().numpy().reshape(nf.shape[0], -1)
    sim_np = sim.detach().to(torch.float64).cpu().numpy().reshape(sim.shape[0], -1)
    magnitude = float(np.abs(sim_np).sum())
    gauge = float(np.abs(nf_np).sum()) / magnitude if magnitude > 0.0 else 1.0
    if not gauge > 0.0:
        gauge = 1.0
    nf_counts = np.rint(nf_np / gauge)
    sim_counts = np.rint(sim_np)
    return (
        {c: nf_counts[:, c : c + 1] for c in range(nf_counts.shape[1])},
        {c: sim_counts[:, c : c + 1] for c in range(sim_counts.shape[1])},
    )


def _report_readout_decision_moves(nf: torch.Tensor, sim: torch.Tensor, labels) -> None:
    """Which way the readout decisions that moved went (a labelled side-report)."""
    nf_pred = nf.argmax(dim=1)
    sim_pred = sim.argmax(dim=1)
    moved = (nf_pred != sim_pred).nonzero(as_tuple=True)[0]
    y = labels.to(nf_pred.device)
    tr = int(((nf_pred[moved] == y[moved]) & (sim_pred[moved] != y[moved])).sum())
    sr = int(((sim_pred[moved] == y[moved]) & (nf_pred[moved] != y[moved])).sum())
    print(
        f"[readout_decision_drift] decisions_moved={int(len(moved))} "
        f"torch-right-sim-wrong={tr} sim-right-torch-wrong={sr} "
        f"both-wrong={int(len(moved)) - tr - sr}",
        flush=True,
    )


def measure_readout_decision_drift(
    model,
    flow,
    samples: torch.Tensor,
    *,
    min_agreement: float | None = None,
    labels: torch.Tensor | None = None,
) -> float:
    """[diagnostic] Fraction of readout neurons whose INTEGER count the trained
    torch twin and the deployed sim agree on, both read inside the chip-lattice
    measurement plane. Returns that fraction; 1.0 means every count matched.

    It never gates. It measures how far apart two DIFFERENT float programs land
    on one integer readout lattice; the verdict on deployment faithfulness is
    the count-exactness gate (``assert_streamed_nf_scm_exact_or_raise``) or the
    per-neuron gate, not this. Reading it as an argmax agreement measured
    float32 tie-breaking instead: on a tie-dense integer readout the two twins
    resolve an exact tie in opposite directions from identical counts, and
    reading it OUTSIDE the plane compared two computations neither of which is
    the deployed one (t0_55: 0 in-plane count mismatches, up to 11 counts out).
    ``min_agreement`` is accepted for the pre-rename call sites and IGNORED.
    """
    from mimarsinan.models.nn.lif_kernels import measurement_plane
    from mimarsinan.spiking.lif_utils import arm_integer_membrane_lattice

    del min_agreement
    device = _unify_model_device(model)
    if device is not None:
        samples = samples.to(device)
        flow = flow.to(device)
    if hasattr(model, "get_perceptrons"):
        # Tuning stages recreate activations; re-arm from parameter_scale so the
        # twin decides ties by the exact lattice value, like the chip. A twin
        # with no perceptrons (a fixture, a flow) has nothing to arm.
        arm_integer_membrane_lattice(model)
    # BOTH twins inside the SAME plane: outside it neither snaps its membrane
    # onto the integer chip lattice, so neither one is the deployed computation.
    with measurement_plane(), torch.no_grad():
        torch_out = model(samples)
        sim_out = flow(samples)
    nf_counts, sim_counts = _readout_count_records(torch_out, sim_out)
    mismatches, total, _worst = compare_normalized_records(
        nf_counts, sim_counts, atol=0.0,
    )
    if labels is not None:
        _report_readout_decision_moves(torch_out, sim_out, labels)
    return 1.0 - mismatches / max(total, 1)


# The pinned observable test imports the pre-rename name; keep it bound.


def compare_normalized_records(
    nf: Dict[int, np.ndarray],
    scm: Dict[int, np.ndarray],
    *,
    atol: float,
    debug: bool = False,
):
    """Order-insensitive per-perceptron comparison: ``(mismatches, total, worst)``.

    Each (perceptron, sample) row is compared as a sorted multiset (conv core emission
    order need not match the torch flatten order); positional wiring is enforced transitively.
    """
    total = 0
    mismatches = 0
    worst = None
    for pi in sorted(set(nf) & set(scm)):
        nf_vals, scm_vals = nf[pi], scm[pi]
        if nf_vals.shape != scm_vals.shape:
            raise NfScmParityError(
                f"NF↔SCM parity: perceptron {pi} neuron-count mismatch "
                f"{nf_vals.shape} vs {scm_vals.shape}"
            )
        nf_sorted = np.sort(nf_vals, axis=1)
        scm_sorted = np.sort(scm_vals, axis=1)
        diff = np.abs(nf_sorted - scm_sorted)
        if debug:
            frac_pi = float((diff > atol).mean())
            print(
                f"[nf_scm_parity] perceptron {pi}: shape={nf_vals.shape} "
                f"mismatch={frac_pi:.4%} max|Δ|={float(diff.max()):.4f} "
                f"nf[mean={float(nf_vals.mean()):.4f}] "
                f"scm[mean={float(scm_vals.mean()):.4f}]"
            )
        total += diff.size
        bad = diff > atol
        mismatches += int(bad.sum())
        if bad.any():
            s_idx, rank = np.unravel_index(int(diff.argmax()), diff.shape)
            candidate = (
                float(diff[s_idx, rank]), pi, int(s_idx), int(rank),
                float(nf_sorted[s_idx, rank]), float(scm_sorted[s_idx, rank]),
            )
            if worst is None or candidate[0] > worst[0]:
                worst = candidate
    return mismatches, total, worst


def _assert_raster_exact_or_raise(
    nf_rasters: Dict[int, np.ndarray],
    per_sample_raster: List[Dict[int, np.ndarray]],
    *,
    n_samples: int,
) -> None:
    """[ODIN P2] the per-CYCLE multiplicity comparison, atol=0."""
    if not per_sample_raster:
        return
    scm_rasters = {
        pi: np.stack([sample_vals[pi] for sample_vals in per_sample_raster])
        for pi in per_sample_raster[0]
    }
    if not scm_rasters:
        raise NfScmParityError(
            "streamed NF↔SCM raster gate armed under "
            "firing_granularity='per_event' but the executor recorded NO "
            "per-cycle rasters, so the window-count agreement above proves "
            "nothing about rhythm."
        )
    # COVERAGE, not intersection: a hop the executor recorded but the NF twin
    # did not stack is a hop whose rhythm nothing compares, and the count arm
    # cannot see a rhythm difference. Name it instead of dropping it.
    uncovered = sorted(pi for pi in scm_rasters if pi not in nf_rasters)
    if uncovered:
        raise NfScmParityError(
            f"streamed NF↔SCM raster gate armed under "
            f"firing_granularity='per_event' but the NF twin captured no "
            f"per-cycle train for perceptron(s) {uncovered} that the executor "
            f"recorded (captured: {sorted(nf_rasters)}): those hops did not "
            f"run the event-serial fold under the streamed walk, so their "
            f"per-cycle multiplicity is UNCHECKED while their window counts "
            f"are compared as if it were."
        )
    mismatches, total, worst = compare_normalized_records(
        {pi: np.rint(nf_rasters[pi]) for pi in scm_rasters}, scm_rasters,
        atol=0.0,
    )
    if mismatches:
        raise NfScmParityError(
            f"streamed NF↔SCM RASTER exactness violated: {mismatches}/{total} "
            f"per-cycle emission mismatches over {n_samples} samples (atol=0; "
            f"worst={worst}). Under firing_granularity='per_event' the window "
            f"count is a projection — equal counts with a different per-cycle "
            f"multiplicity are a DIFFERENT computation at the next hop."
        )


def _capture_nf_streamed_rasters(model, samples: torch.Tensor) -> Dict[int, np.ndarray]:
    """Per-perceptron NF per-cycle emission multiplicities, neuron-major.

    The streamed NF calls each spiking node once PER CYCLE; stacking (rather
    than accumulating) those emissions is the rhythm the count hides.
    """
    return _capture_nf_streamed(model, samples, accumulate=False)


def _capture_nf_streamed_counts(model, samples: torch.Tensor) -> Dict[int, np.ndarray]:
    """Per-perceptron NF WINDOW COUNTS over the batch for the streaming walk.

    The streaming NF calls each spiking node once PER CYCLE, so the hook
    ACCUMULATES the per-cycle emissions (spike × scale); the sum ÷ scale is
    the window count. (The single-shot ``_capture_nf_normalized`` would keep
    only the last cycle.)
    """
    return _capture_nf_streamed(model, samples, accumulate=True)


def _capture_nf_streamed(
    model, samples: torch.Tensor, *, accumulate: bool,
) -> Dict[int, np.ndarray]:
    """One streamed capture: window counts (accumulate) or the raster (stack)."""
    from mimarsinan.models.nn.activations.ttfs_spiking import _channel_broadcast_view

    device = _unify_model_device(model)
    if device is not None:
        samples = samples.to(device)
    perceptrons = list(model.get_perceptrons())
    captured: Dict[int, torch.Tensor] = {}
    cycles: Dict[int, list] = defaultdict(list)

    def _make_hook(index, perceptron):
        def hook(_module, _inp, out):
            scale = torch.as_tensor(
                perceptron.activation_scale, device=out.device, dtype=out.dtype,
            )
            if scale.dim() == 0:
                normalized = out / scale.clamp(min=1e-12)
            else:
                normalized = out / _channel_broadcast_view(scale, out).clamp(min=1e-12)
            flat = normalized.detach().reshape(out.shape[0], -1)
            if not accumulate:
                cycles[index].append(flat)
                return
            prev = captured.get(index)
            captured[index] = flat if prev is None else prev + flat
        return hook

    handles = [
        p.activation.register_forward_hook(_make_hook(i, p))
        for i, p in enumerate(perceptrons)
    ]
    try:
        with torch.no_grad():
            model(samples)
    finally:
        for handle in handles:
            handle.remove()
    if not accumulate:
        # (B, T*n) neuron-major, mirroring the record's raster flatten.
        captured = {
            index: torch.stack(trains, dim=1).permute(0, 2, 1).reshape(
                trains[0].shape[0], -1)
            for index, trains in cycles.items() if len(trains) > 1
        }
    return {i: v.cpu().numpy().astype(np.float64) for i, v in captured.items()}


def _capture_nf_normalized(model, samples: torch.Tensor) -> Dict[int, np.ndarray]:
    """Per-perceptron NF outputs over the batch, normalized to [0, 1]."""
    from mimarsinan.models.nn.activations.ttfs_spiking import _channel_broadcast_view

    device = _unify_model_device(model)
    if device is not None:
        samples = samples.to(device)
    perceptrons = list(model.get_perceptrons())
    captured: Dict[int, torch.Tensor] = {}

    def _make_hook(index, perceptron):
        def hook(_module, _inp, out):
            scale = torch.as_tensor(
                perceptron.activation_scale, device=out.device, dtype=out.dtype,
            )
            if scale.dim() == 0:
                normalized = out / scale.clamp(min=1e-12)
            else:
                normalized = out / _channel_broadcast_view(scale, out).clamp(min=1e-12)
            captured[index] = normalized.detach().reshape(out.shape[0], -1)
        return hook

    handles = [
        p.activation.register_forward_hook(_make_hook(i, p))
        for i, p in enumerate(perceptrons)
    ]
    try:
        with torch.no_grad():
            model(samples)
    finally:
        for handle in handles:
            handle.remove()
    return {i: v.cpu().numpy().astype(np.float64) for i, v in captured.items()}


def _collect_scm_normalized(
    identity_mapping,
    model,
    samples: torch.Tensor,
    contract,
) -> Dict[int, np.ndarray]:
    """Per-perceptron contract-runner outputs on the identity mapping.

    Cores group by ``perceptron_index`` and concatenate in IR-id order; psum partials are excluded.
    """
    from mimarsinan.chip_simulation.ttfs.ttfs_executor import run_ttfs_hybrid_contract

    preprocessor = getattr(model, "preprocessor", None)
    per_sample: List[Dict[int, np.ndarray]] = []
    for i in range(samples.shape[0]):
        x = samples[i : i + 1]
        if preprocessor is not None:
            with torch.no_grad():
                x = preprocessor(x)
        x_np = x.reshape(1, -1).detach().cpu().to(torch.float64).numpy()
        run = run_ttfs_hybrid_contract(
            identity_mapping, x_np, sample_index=i, contract=contract,
        )
        per_sample.append(_group_record_by_perceptron(run.record, identity_mapping))

    grouped: Dict[int, np.ndarray] = {}
    for pi in per_sample[0]:
        grouped[pi] = np.stack([sample_vals[pi] for sample_vals in per_sample])
    return grouped


def _group_record_by_perceptron(
    record, identity_mapping, *, values_of=None,
) -> Dict[int, np.ndarray]:
    # Order a perceptron's tiles by tile_offset (perceptron_output_slice start), not ir_id: id assignment is not monotone in the slice after compaction.
    per_core: Dict[int, tuple[int, int, np.ndarray]] = {}
    for stage_index, segment in record.segments.items():
        stage = identity_mapping.stages[stage_index]
        placements = stage.hard_core_mapping.soft_core_placements_per_hard_core
        for core_record in segment.cores:
            core_placements = placements[core_record.core_index]
            assert len(core_placements) == 1, (
                "NF↔SCM parity gate requires an identity mapping (1 placement/core)"
            )
            placement = core_placements[0]
            assert placement.get("split_group_id") is None, (
                "identity mappings must not contain neuron-split fragments"
            )
            perceptron_index = placement.get("perceptron_index")
            if perceptron_index is None or perceptron_index < 0:
                continue
            if placement.get("psum_role") not in (None, "accum"):
                continue
            values = np.asarray(
                (values_of(core_record) if values_of is not None
                 else core_record.output_activation[: core_record.n_out_used]),
                dtype=np.float64,
            )
            out_slice = placement.get("perceptron_output_slice")
            tile_offset = int(out_slice[0]) if out_slice is not None else 0
            per_core[placement["ir_node_id"]] = (
                int(perceptron_index), tile_offset, values,
            )

    by_perceptron: Dict[int, list[int]] = defaultdict(list)
    for ir_id, (perceptron_index, _, _) in per_core.items():
        by_perceptron[perceptron_index].append(ir_id)
    return {
        pi: np.concatenate(
            [
                per_core[ir_id][2]
                for ir_id in sorted(ir_ids, key=lambda c: (per_core[c][1], c))
            ]
        )
        for pi, ir_ids in by_perceptron.items()
    }
