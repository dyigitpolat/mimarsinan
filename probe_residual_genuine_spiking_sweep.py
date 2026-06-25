"""Probe: GENUINE-SPIKING mode sweep on ONE deep residual net.

Sweeps the three genuine-spiking deployment modes on the SAME trained residual
net + SAME eval set, so the only variable is the spiking mode:

  - lif                       (rate-coded, cycle-accurate LIF cascade)
  - ttfs_cycle_based cascaded (single-spike timing cascade)
  - ttfs_cycle_based sync     (synchronized single-spike, latency-grouped)

Same residual spec the analytical-sweep agent builds: a small TRAINED residual
stack of N BasicBlock-style residual blocks (y = relu(x + relu(F(x)))), fixed
seed, recorded ann_top1. Each block's on-chip skip lowers to a host rate-domain
ComputeOp (Tier-0), so an N-block net deploys as an N-segment cascade — the
depth axis we sweep.

Deploy path is the production one (the same one probe_residual_sync_ttfs.py and
the torch_sim fidelity harness use): convert -> install the mode's spiking
activation -> compute_per_source_scales -> per-mode distribution-match
calibration -> IRMapping -> hybrid hard-core mapping -> SpikingHybridCoreFlow.

  ANN(float ref)  ->  deployed HCM   (retention = deployed_top1 / ann_top1)

Every model/seed is built deterministically and reused across all three modes.
"""
from __future__ import annotations

import copy
import os as _os
import sys
import torch
import torch.nn as nn

torch.set_num_threads(int(_os.environ.get("PROBE_THREADS", "8")))

from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.torch_mapping.encoding_layers import mark_encoding_layers
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import build_hybrid_hard_core_mapping
from mimarsinan.mapping.platform.mapping_structure import MappingStrategy
from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow
from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.models.nn.activations.ttfs_spiking import TTFSActivation
from mimarsinan.mapping.support.per_source_scales import compute_per_source_scales
from mimarsinan.mapping.support.neg_shift_bias import calibration_forward_for_mode
from mimarsinan.spiking.chip_aligned_nf import chip_aligned_segment_forward
from mimarsinan.spiking.distribution_matching import match_activation_distributions
from mimarsinan.spiking.lif_distribution_matching import match_lif_activation_distributions

import os

IN = 16
NC = 4
W = 24
T = 16
N_TRAIN = 1500
N_EVAL = int(os.environ.get("PROBE_N_EVAL", "400"))
N_CAL = int(os.environ.get("PROBE_N_CAL", "256"))


def make_task(seed=0):
    g = torch.Generator().manual_seed(seed)
    Wt = torch.randn(NC, IN, generator=g)
    X = torch.rand(N_TRAIN + N_EVAL, IN, generator=g)
    logits = X @ Wt.t() + 0.3 * torch.randn(N_TRAIN + N_EVAL, NC, generator=g)
    y = logits.argmax(1)
    return X[:N_TRAIN], y[:N_TRAIN], X[N_TRAIN:], y[N_TRAIN:]


class ResidualStack(nn.Module):
    """N residual blocks: y = relu(x + relu(F(x))), F a width->width Linear.

    The on-chip skip in each block is a Tier-0 host rate-domain add, so an
    N-block net deploys as an N-segment genuine-spiking cascade.
    """

    def __init__(self, depth, width=W):
        super().__init__()
        self.stem = nn.Linear(IN, width)
        self.sa = nn.ReLU()
        self.f = nn.ModuleList([nn.Linear(width, width) for _ in range(depth)])
        self.fa = nn.ModuleList([nn.ReLU() for _ in range(depth)])
        self.ba = nn.ModuleList([nn.ReLU() for _ in range(depth)])
        self.head = nn.Linear(width, NC)

    def forward(self, x):
        x = self.sa(self.stem(x))
        for f, fa, ba in zip(self.f, self.fa, self.ba):
            b = fa(f(x))
            x = ba(x + b)
        return self.head(x)


def train(model, X, y, epochs=150, lr=3e-3, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(model(X), y)
        loss.backward()
        opt.step()
    return model.eval()


def _install_lif(flow):
    for p in flow.get_perceptrons():
        lif = LIFActivation(
            T=T, activation_scale=p.activation_scale, thresholding_mode="<=")
        p.set_activation(lif)


def _install_ttfs(flow):
    for p in flow.get_perceptrons():
        p.set_activation(TTFSActivation(
            T=T, activation_scale=p.activation_scale,
            input_scale=p.input_activation_scale, bias=p.layer.bias,
            thresholding_mode="<=", encoding=getattr(p, "is_encoding_layer", False),
        ))


def deploy(model, cal_x, spiking_mode, ttfs_cycle_schedule="cascaded", calibrate=True):
    """Production deploy of ``model`` in one genuine-spiking mode.

    Returns ``(flow, hcm, teacher, nseg)``: ``flow`` is the calibrated torch NF,
    ``hcm`` the deployed SpikingHybridCoreFlow, ``teacher`` the float reference.
    """
    is_lif = spiking_mode == "lif"
    flow = convert_torch_model(model, (IN,), NC, device="cpu")
    flow.eval()
    repr_ = flow.get_mapper_repr()
    mark_encoding_layers(repr_)
    teacher = copy.deepcopy(flow).eval()

    if is_lif:
        _install_lif(flow)
        firing, spike = "Default", "Uniform"
    else:
        _install_ttfs(flow)
        firing, spike = "TTFS", "TTFS"
        flow = flow.double()

    repr_.assign_perceptron_indices()
    compute_per_source_scales(repr_)

    if calibrate:
        cal = cal_x.double() if not is_lif else cal_x
        tch = teacher.double() if not is_lif else teacher
        if is_lif:
            match_lif_activation_distributions(flow, teacher, cal_x, T)
        else:
            match_activation_distributions(flow, tch, cal, T, quantile=0.99)

    ir = IRMapping(
        q_max=127.0, firing_mode=firing, max_axons=8192, max_neurons=8192,
        allow_coalescing=False,
    ).map(repr_)
    hybrid = build_hybrid_hard_core_mapping(
        ir_graph=ir,
        cores_config=[{"max_axons": 8192, "max_neurons": 8192, "count": 8000}],
        strategy=MappingStrategy.from_permissions(
            allow_neuron_splitting=False, allow_coalescing=False),
    )
    flow_kwargs = (
        dict(spiking_mode="lif", cycle_accurate_lif_forward=True) if is_lif
        else dict(spiking_mode=spiking_mode, ttfs_cycle_schedule=ttfs_cycle_schedule)
    )
    hcm = SpikingHybridCoreFlow(
        (IN,), hybrid, simulation_length=T, preprocessor=nn.Identity(),
        firing_mode=firing, spike_mode=spike, thresholding_mode="<=", **flow_kwargs,
    )
    nseg = sum(1 for s in hybrid.stages if s.hard_core_mapping is not None)
    return flow, hcm, teacher, nseg


def _nf_forward(flow, x, spiking_mode):
    if spiking_mode == "lif":
        return chip_aligned_segment_forward(flow, x, T)
    return calibration_forward_for_mode(spiking_mode)(flow, x.double(), T)


def measure(flow, hcm, teacher, Xe, ye, spiking_mode):
    with torch.no_grad():
        ann = (teacher(Xe.to(next(teacher.parameters()).dtype)).argmax(1) == ye).float().mean().item()
        nf = _nf_forward(flow, Xe, spiking_mode)
        nf_acc = (nf.argmax(1) == ye).float().mean().item()
        hc = hcm(Xe.to(next(hcm.parameters()).dtype if any(True for _ in hcm.parameters()) else Xe.dtype)).float() / float(T)
        hcm_acc = (hc.argmax(1) == ye).float().mean().item()
    return ann, nf_acc, hcm_acc


MODES = [
    ("lif", "lif", None),
    ("ttfs_cascaded", "ttfs_cycle_based", "cascaded"),
    ("ttfs_sync", "ttfs_cycle_based", "synchronized"),
]


def main():
    depths = [int(d) for d in sys.argv[1].split(",")] if len(sys.argv) > 1 else [2, 4, 8, 12]
    seeds = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0, 1, 2]
    print(f"T={T} W={W} IN={IN} NC={NC} chance={1.0/NC:.3f} calib=ON seeds={seeds}")
    print(f"{'mode':<16}{'d':>3}{'blocks':>7} {'ANN':>7} {'NF':>7} {'deployed':>9} {'ret%':>7} {'nseg':>5}")
    for d in depths:
        print(f"# depth={d} ...", flush=True)
        agg = {name: [] for name, _, _ in MODES}
        nsegs = {}
        for s in seeds:
            Xtr, ytr, Xe, ye = make_task(s)
            cal_x = Xtr[:N_CAL]
            torch.manual_seed(s)
            base = train(ResidualStack(d), Xtr, ytr, seed=s)
            for name, mode, sched in MODES:
                m = copy.deepcopy(base)
                flow, hcm, teacher, ns = deploy(
                    m, cal_x, mode, ttfs_cycle_schedule=(sched or "cascaded"))
                r = measure(flow, hcm, teacher, Xe, ye, mode)
                agg[name].append(r)
                nsegs[name] = ns
                if _os.environ.get("PROBE_PER_SEED"):
                    print(f"  seed={s} {name:<16} ann={r[0]:.3f} nf={r[1]:.3f} dep={r[2]:.3f} ret={100*r[2]/r[0]:.1f}%", flush=True)
        for name, _, _ in MODES:
            arr = torch.tensor(agg[name])
            ann, nf, hc = arr.mean(0).tolist()
            ret = hc / ann if ann > 0 else 0.0
            print(f"{name:<16}{d:>3}{d:>7} {ann:>7.3f} {nf:>7.3f} {hc:>9.3f} {100*ret:>6.1f}% {nsegs[name]:>5}", flush=True)
        print(flush=True)


if __name__ == "__main__":
    main()
