"""SOLUTION study: candidate fixes on the SHARED deep-residual collapse probe.

Locks the collapse vehicle (depth, seed, T) and the production deploy SSOT from
``probe_residual_genuine_spiking_sweep`` (P.deploy / P.measure / P.make_task /
P.ResidualStack / P.train) so retention is byte-identical-comparable across
fixes. Each fix is a hook that EITHER mutates the deployed flow's calibration
(post-train, pre-map) OR retrains the weights through the genuine spike forward
(QAT) before deploying through the SAME path.

Usage:
  PYTHONPATH=src:spikingjelly env/bin/python probe_residual_fix_study.py <fix> <depth> <seed> <T> <modes>
    fix    in {baseline, gain, highT, qat, revive_refine, scale_dfq_strong}
    modes  comma list subset of {lif, ttfs_cascaded, ttfs_sync}; default all

Retention = deployed_HCM_top1 / ann_top1 on the FIXED eval slice.
"""
from __future__ import annotations

import copy
import sys

import torch
import torch.nn as nn

import probe_residual_genuine_spiking_sweep as P

P.N_EVAL = 200  # fixed eval slice, identical across fixes


def _build_flow_with_activation(base, mode, T):
    """Convert ``base`` to a flow, mark encoding, install the mode's deployed
    spiking activation, assign indices + per-source scales. Returns
    ``(flow, teacher, is_lif)`` with the flow in the deployed activation state."""
    from mimarsinan.torch_mapping.converter import convert_torch_model
    from mimarsinan.torch_mapping.encoding_layers import mark_encoding_layers
    from mimarsinan.models.nn.activations import LIFActivation
    from mimarsinan.models.nn.activations.ttfs_spiking import TTFSActivation
    from mimarsinan.mapping.support.per_source_scales import compute_per_source_scales

    is_lif = mode == "lif"
    flow = convert_torch_model(copy.deepcopy(base), (P.IN,), P.NC, device="cpu")
    flow.eval()
    repr_ = flow.get_mapper_repr()
    mark_encoding_layers(repr_)
    teacher = copy.deepcopy(flow).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    if is_lif:
        for p in flow.get_perceptrons():
            p.set_activation(LIFActivation(
                T=T, activation_scale=p.activation_scale, thresholding_mode="<="))
    else:
        for p in flow.get_perceptrons():
            p.set_activation(TTFSActivation(
                T=T, activation_scale=p.activation_scale,
                input_scale=p.input_activation_scale, bias=p.layer.bias,
                thresholding_mode="<=", encoding=getattr(p, "is_encoding_layer", False)))
        flow = flow.double()
    repr_ = flow.get_mapper_repr()
    repr_.assign_perceptron_indices()
    compute_per_source_scales(repr_)
    return flow, teacher, is_lif


def _qat_adapt_flow(flow, teacher, is_lif, Xtr, ytr, T, *, steps, lr, ramp_frac, seed):
    """Retrain the flow's weights IN PLACE through the genuine spike forward.

    Blends a frozen float teacher -> the differentiable genuine cascade
    (chip_aligned_segment_forward for LIF, TTFSSegmentForward for TTFS), ramping
    the genuine rate 0->1 over the first ``ramp_frac`` of steps, then pure
    genuine. Mirrors BlendedGenuineForward semantics on the probe substrate."""
    from mimarsinan.spiking.chip_aligned_nf import chip_aligned_segment_forward
    from mimarsinan.models.spiking.training.ttfs_segment_forward import TTFSSegmentForward

    flow.train()

    def genuine(x):
        if is_lif:
            return chip_aligned_segment_forward(flow, x, T)
        return TTFSSegmentForward(flow.get_mapper_repr(), T, boundary_surrogate_temp=1.0)(x)

    Xt = Xtr.double() if not is_lif else Xtr
    teach = teacher.double() if not is_lif else teacher
    yt = ytr
    opt = torch.optim.Adam([p for p in flow.parameters() if p.requires_grad], lr=lr)
    lossf = nn.CrossEntropyLoss()
    torch.manual_seed(seed)
    n_ramp = max(1, int(steps * ramp_frac))
    for step in range(steps):
        rate = min(1.0, step / n_ramp)
        opt.zero_grad()
        g = genuine(Xt)
        with torch.no_grad():
            t = teach(Xt)
        # genuine cascade logits are spike-count scale (~[0,T]); teacher is value
        # scale. Bring both to value scale for a stable blended CE.
        g_val = g.float() / float(T)
        out = (1.0 - rate) * t.float() + rate * g_val if rate < 1.0 else g_val
        loss = lossf(out, yt)
        loss.backward()
        opt.step()
    flow.eval()
    return flow


def _map_flow(flow, teacher, is_lif, cal_x, sched, T, *, calibrate=True):
    """Calibrate + IRMap + build the deployed HCM from an already-activated flow
    (the second half of P.deploy, on the SAME flow object QAT adapted in place)."""
    from mimarsinan.mapping.ir_mapping_class import IRMapping
    from mimarsinan.mapping.packing.hybrid_hardcore_mapping import build_hybrid_hard_core_mapping
    from mimarsinan.mapping.platform.mapping_structure import MappingStrategy
    from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow
    from mimarsinan.spiking.distribution_matching import match_activation_distributions
    from mimarsinan.spiking.lif_distribution_matching import match_lif_activation_distributions

    repr_ = flow.get_mapper_repr()
    firing, spike = ("Default", "Uniform") if is_lif else ("TTFS", "TTFS")
    if calibrate:
        if is_lif:
            match_lif_activation_distributions(flow, teacher, cal_x, T)
        else:
            match_activation_distributions(flow, teacher.double(), cal_x.double(), T, quantile=0.99)
    ir = IRMapping(q_max=127.0, firing_mode=firing, max_axons=8192, max_neurons=8192,
                   allow_coalescing=False).map(repr_)
    hybrid = build_hybrid_hard_core_mapping(
        ir_graph=ir, cores_config=[{"max_axons": 8192, "max_neurons": 8192, "count": 8000}],
        strategy=MappingStrategy.from_permissions(allow_neuron_splitting=False, allow_coalescing=False))
    flow_kwargs = (dict(spiking_mode="lif", cycle_accurate_lif_forward=True) if is_lif
                   else dict(spiking_mode="ttfs_cycle_based", ttfs_cycle_schedule=(sched or "cascaded")))
    hcm = SpikingHybridCoreFlow(
        (P.IN,), hybrid, simulation_length=T, preprocessor=nn.Identity(),
        firing_mode=firing, spike_mode=spike, thresholding_mode="<=", **flow_kwargs)
    nseg = sum(1 for s in hybrid.stages if s.hard_core_mapping is not None)
    return flow, hcm, nseg


def _deploy_qat(base, cal_x, Xtr, ytr, mode, sched, T, *, steps, lr, ramp_frac):
    """QAT: build flow -> adapt weights through genuine forward IN PLACE ->
    calibrate + map the SAME flow. The flow IS the deployed source (no torch
    weight copy-back; the converted graph has more perceptron nodes than the
    torch module's Linears, so a 1:1 copy-back is structurally impossible)."""
    flow, teacher, is_lif = _build_flow_with_activation(base, mode, T)
    _qat_adapt_flow(flow, teacher, is_lif, Xtr, ytr, T,
                    steps=steps, lr=lr, ramp_frac=ramp_frac, seed=0)
    flow, hcm, ns = _map_flow(flow, teacher, is_lif, cal_x, sched, T)
    return flow, hcm, teacher, ns


def deploy_with_fix(fix, base, cal_x, Xtr, ytr, mode, sched, T):
    """Deploy ``base`` in ``mode`` applying ``fix``. Returns (flow, hcm, teacher, ns)."""
    if fix in ("baseline", "highT"):
        # highT just raises P.T (set by caller).
        flow, hcm, teacher, ns = P.deploy(
            copy.deepcopy(base), cal_x, mode, ttfs_cycle_schedule=(sched or "cascaded"))
        return flow, hcm, teacher, ns

    if fix == "gain":
        return _deploy_gain(base, cal_x, mode, sched, T)

    if fix == "qat":
        return _deploy_qat(base, cal_x, Xtr, ytr, mode, sched, T,
                           steps=200, lr=1.5e-3, ramp_frac=0.5)

    if fix == "revive_refine":
        # revive = strong DFQ calibration init, then a longer genuine refine.
        return _deploy_qat(base, cal_x, Xtr, ytr, mode, sched, T,
                           steps=300, lr=2e-3, ramp_frac=0.35)

    if fix in ("scale_dfq_strong", "resmerge"):
        return _deploy_calib(base, cal_x, mode, sched, T,
                             strong_dfq=(fix == "scale_dfq_strong"),
                             onchip_residual_merge=(fix == "resmerge"))

    raise ValueError(f"unknown fix {fix}")


def _deploy_calib(base, cal_x, mode, sched, T, *, strong_dfq=False, onchip_residual_merge=False):
    """Calibration-only fixes: stronger scale-aware+DFQ (lower quantile widens the
    [0,1] window, more iters + higher eta drive the first moment harder) and/or
    the on-chip residual merge (IRMapping flag). No retraining."""
    from mimarsinan.mapping.ir_mapping_class import IRMapping
    from mimarsinan.mapping.packing.hybrid_hardcore_mapping import build_hybrid_hard_core_mapping
    from mimarsinan.mapping.platform.mapping_structure import MappingStrategy
    from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow
    from mimarsinan.spiking.distribution_matching import match_activation_distributions
    from mimarsinan.spiking.lif_distribution_matching import match_lif_activation_distributions

    flow, teacher, is_lif = _build_flow_with_activation(base, mode, T)
    repr_ = flow.get_mapper_repr()
    firing, spike = ("Default", "Uniform") if is_lif else ("TTFS", "TTFS")
    if is_lif:
        match_lif_activation_distributions(flow, teacher, cal_x, T)
    elif strong_dfq:
        match_activation_distributions(flow, teacher.double(), cal_x.double(), T,
                                       quantile=0.95, bias_iters=40, eta=1.0)
    else:
        match_activation_distributions(flow, teacher.double(), cal_x.double(), T, quantile=0.99)
    ir = IRMapping(q_max=127.0, firing_mode=firing, max_axons=8192, max_neurons=8192,
                   allow_coalescing=False, onchip_residual_merge=onchip_residual_merge).map(repr_)
    hybrid = build_hybrid_hard_core_mapping(
        ir_graph=ir, cores_config=[{"max_axons": 8192, "max_neurons": 8192, "count": 8000}],
        strategy=MappingStrategy.from_permissions(allow_neuron_splitting=False, allow_coalescing=False))
    flow_kwargs = (dict(spiking_mode="lif", cycle_accurate_lif_forward=True) if is_lif
                   else dict(spiking_mode="ttfs_cycle_based", ttfs_cycle_schedule=(sched or "cascaded")))
    hcm = SpikingHybridCoreFlow(
        (P.IN,), hybrid, simulation_length=T, preprocessor=nn.Identity(),
        firing_mode=firing, spike_mode=spike, thresholding_mode="<=", **flow_kwargs)
    nseg = sum(1 for s in hybrid.stages if s.hard_core_mapping is not None)
    return flow, hcm, teacher, nseg


def _deploy_gain(base, cal_x, mode, sched, T):
    """Gain-correction fix: deploy then apply per-depth gain trim on the flow
    BEFORE mapping. Reimplements P.deploy inline so we can inject the gain trim
    after calibration but before IRMapping (cascaded TTFS only by physics)."""
    from mimarsinan.torch_mapping.converter import convert_torch_model
    from mimarsinan.torch_mapping.encoding_layers import mark_encoding_layers
    from mimarsinan.mapping.ir_mapping_class import IRMapping
    from mimarsinan.mapping.packing.hybrid_hardcore_mapping import build_hybrid_hard_core_mapping
    from mimarsinan.mapping.platform.mapping_structure import MappingStrategy
    from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow
    from mimarsinan.models.nn.activations import LIFActivation
    from mimarsinan.models.nn.activations.ttfs_spiking import TTFSActivation
    from mimarsinan.mapping.support.per_source_scales import compute_per_source_scales
    from mimarsinan.spiking.distribution_matching import match_activation_distributions
    from mimarsinan.spiking.lif_distribution_matching import match_lif_activation_distributions
    from mimarsinan.spiking.gain_correction import apply_cascaded_gain_correction

    is_lif = mode == "lif"
    m = copy.deepcopy(base)
    flow = convert_torch_model(m, (P.IN,), P.NC, device="cpu")
    flow.eval()
    repr_ = flow.get_mapper_repr()
    mark_encoding_layers(repr_)
    teacher = copy.deepcopy(flow).eval()
    if is_lif:
        for p in flow.get_perceptrons():
            p.set_activation(LIFActivation(T=T, activation_scale=p.activation_scale, thresholding_mode="<="))
        firing, spike = "Default", "Uniform"
    else:
        for p in flow.get_perceptrons():
            p.set_activation(TTFSActivation(
                T=T, activation_scale=p.activation_scale, input_scale=p.input_activation_scale,
                bias=p.layer.bias, thresholding_mode="<=", encoding=getattr(p, "is_encoding_layer", False)))
        firing, spike = "TTFS", "TTFS"
        flow = flow.double()
    repr_.assign_perceptron_indices()
    compute_per_source_scales(repr_)
    if is_lif:
        match_lif_activation_distributions(flow, teacher, cal_x, T)
    else:
        match_activation_distributions(flow, teacher.double(), cal_x.double(), T, quantile=0.99)
        apply_cascaded_gain_correction(flow, T, rule="relative")  # the fix
    ir = IRMapping(q_max=127.0, firing_mode=firing, max_axons=8192, max_neurons=8192,
                   allow_coalescing=False).map(repr_)
    hybrid = build_hybrid_hard_core_mapping(
        ir_graph=ir, cores_config=[{"max_axons": 8192, "max_neurons": 8192, "count": 8000}],
        strategy=MappingStrategy.from_permissions(allow_neuron_splitting=False, allow_coalescing=False))
    flow_kwargs = (dict(spiking_mode="lif", cycle_accurate_lif_forward=True) if is_lif
                   else dict(spiking_mode="ttfs_cycle_based", ttfs_cycle_schedule=(sched or "cascaded")))
    hcm = SpikingHybridCoreFlow(
        (P.IN,), hybrid, simulation_length=T, preprocessor=nn.Identity(),
        firing_mode=firing, spike_mode=spike, thresholding_mode="<=", **flow_kwargs)
    nseg = sum(1 for s in hybrid.stages if s.hard_core_mapping is not None)
    return flow, hcm, teacher, nseg


def run(fix, depth, seed, T, modes):
    P.T = T
    Xtr, ytr, Xe, ye = P.make_task(seed)
    cal_x = Xtr[:P.N_CAL]
    torch.manual_seed(seed)
    base = P.train(P.ResidualStack(depth), Xtr, ytr, seed=seed)
    sel = [m for m in P.MODES if m[0] in modes]
    print(f"FIX={fix} depth={depth} seed={seed} T={T} W={P.W} chance={1.0/P.NC:.3f}", flush=True)
    print(f"{'mode':<16}{'ANN':>7}{'NF':>8}{'deployed':>10}{'ret%':>8}{'nseg':>6}", flush=True)
    results = {}
    for name, mode, sched in sel:
        flow, hcm, teacher, ns = deploy_with_fix(fix, base, cal_x, Xtr, ytr, mode, sched, T)
        ann, nf, hc = P.measure(flow, hcm, teacher, Xe, ye, mode)
        ret = hc / ann if ann > 0 else 0.0
        results[name] = (ann, nf, hc, ret)
        print(f"{name:<16}{ann:>7.3f}{nf:>8.3f}{hc:>10.3f}{100*ret:>7.1f}%{ns:>6}", flush=True)
    return results


if __name__ == "__main__":
    fix = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    depth = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    T = int(sys.argv[4]) if len(sys.argv) > 4 else 16
    modes = sys.argv[5].split(",") if len(sys.argv) > 5 else ["lif", "ttfs_cascaded", "ttfs_sync"]
    run(fix, depth, seed, T, modes)
