"""Probe: where does synchronized-TTFS lose on a DEEP RESIDUAL net?

Clean substrate (Linear stacks, every layer on-chip), tiny synthetic task so
retention is meaningful. Deploys the GENUINE synchronized ttfs_cycle_based HCM
(the deployed-on-chip metric) with the production scale-aware distribution-match
calibration, and measures deployed-HCM accuracy vs the float ANN:

  ANN(float ref)  ->  sync-TTFS deployed HCM   (retention = HCM/ANN)

Isolates residual-ADD boundaries vs raw depth: PLAIN N-layer stack vs RESIDUAL
N-block stack at MATCHED main-path depth, residual add as host ComputeOp (Tier-0)
and as on-chip merge (Tier-1, onchip_residual_merge). Multi-seed to kill noise.
"""
from __future__ import annotations

import copy
import sys
import torch
import torch.nn as nn

from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.torch_mapping.encoding_layers import mark_encoding_layers
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import build_hybrid_hard_core_mapping
from mimarsinan.mapping.platform.mapping_structure import MappingStrategy
from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow
from mimarsinan.models.nn.activations.ttfs_spiking import TTFSActivation
from mimarsinan.mapping.support.per_source_scales import compute_per_source_scales
from mimarsinan.mapping.support.neg_shift_bias import _analytical_segment_calibration_forward
from mimarsinan.spiking.distribution_matching import match_activation_distributions

IN = 16
NC = 4
W = 24
T = 16
N_TRAIN = 1500
N_EVAL = 400
N_CAL = 256


def make_task(seed=0):
    g = torch.Generator().manual_seed(seed)
    Wt = torch.randn(NC, IN, generator=g)
    X = torch.rand(N_TRAIN + N_EVAL, IN, generator=g)
    logits = X @ Wt.t() + 0.3 * torch.randn(N_TRAIN + N_EVAL, NC, generator=g)
    y = logits.argmax(1)
    return X[:N_TRAIN], y[:N_TRAIN], X[N_TRAIN:], y[N_TRAIN:]


class PlainStack(nn.Module):
    def __init__(self, depth, width=W):
        super().__init__()
        self.stem = nn.Linear(IN, width)
        self.sa = nn.ReLU()
        self.body = nn.ModuleList()
        for _ in range(depth):
            self.body.append(nn.Linear(width, width))
            self.body.append(nn.ReLU())
        self.head = nn.Linear(width, NC)

    def forward(self, x):
        x = self.sa(self.stem(x))
        for layer in self.body:
            x = layer(x)
        return self.head(x)


class ResidualStack(nn.Module):
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


def deploy_sync(model, cal_x, onchip_residual_merge=False, calibrate=True):
    flow = convert_torch_model(model, (IN,), NC, device="cpu")
    flow.eval()
    repr_ = flow.get_mapper_repr()
    mark_encoding_layers(repr_)
    teacher = copy.deepcopy(flow).eval()
    for p in flow.get_perceptrons():
        p.set_activation(TTFSActivation(
            T=T, activation_scale=p.activation_scale,
            input_scale=p.input_activation_scale, bias=p.layer.bias,
            thresholding_mode="<=", encoding=getattr(p, "is_encoding_layer", False),
        ))
    repr_.assign_perceptron_indices()
    compute_per_source_scales(repr_)

    cs = None
    if calibrate:
        cs = match_activation_distributions(flow, teacher, cal_x, T, quantile=0.99)

    ir = IRMapping(
        q_max=127.0, firing_mode="TTFS", max_axons=8192, max_neurons=8192,
        allow_coalescing=False, onchip_residual_merge=onchip_residual_merge,
    ).map(repr_)
    hybrid = build_hybrid_hard_core_mapping(
        ir_graph=ir,
        cores_config=[{"max_axons": 8192, "max_neurons": 8192, "count": 8000}],
        strategy=MappingStrategy.from_permissions(
            allow_neuron_splitting=False, allow_coalescing=False),
    )
    hcm = SpikingHybridCoreFlow(
        (IN,), hybrid, simulation_length=T, preprocessor=nn.Identity(),
        firing_mode="TTFS", spike_mode="TTFS", thresholding_mode="<=",
        spiking_mode="ttfs_cycle_based", ttfs_cycle_schedule="synchronized",
    )
    nseg = sum(1 for s in hybrid.stages if s.hard_core_mapping is not None)
    return flow, hcm, teacher, nseg, cs


def measure(flow, hcm, teacher, Xe, ye):
    with torch.no_grad():
        ann = (teacher(Xe).argmax(1) == ye).float().mean().item()
        nf = _analytical_segment_calibration_forward(flow, Xe.float(), T)
        nf_acc = (nf.argmax(1) == ye).float().mean().item()
        hc = hcm(Xe.float()).float() / float(T)
        hcm_acc = (hc.argmax(1) == ye).float().mean().item()
    return ann, nf_acc, hcm_acc


def main():
    depths = [int(d) for d in sys.argv[1].split(",")] if len(sys.argv) > 1 else [2, 4, 6, 8, 12]
    seeds = [0, 1, 2]
    print(f"T={T} W={W} chance={1.0/NC:.3f} q=0.99 calib=ON seeds={seeds}")
    print(f"{'arch':<22}{'d':>3} {'ANN':>7} {'NF(stair)':>10} {'HCM(sync)':>10} {'ret%':>7} {'nseg':>5}")
    for d in depths:
        rows = {"plain": [], "res_T0": [], "res_T1": []}
        nsegs = {}
        for s in seeds:
            Xtr, ytr, Xe, ye = make_task(s)
            cal_x = Xtr[:N_CAL]

            mp = train(PlainStack(d), Xtr, ytr, seed=s)
            flow, hcm, teacher, ns, cs = deploy_sync(mp, cal_x)
            rows["plain"].append(measure(flow, hcm, teacher, Xe, ye)); nsegs["plain"] = ns

            mr = train(ResidualStack(d), Xtr, ytr, seed=s)
            flow, hcm, teacher, ns, cs = deploy_sync(copy.deepcopy(mr), cal_x, onchip_residual_merge=False)
            rows["res_T0"].append(measure(flow, hcm, teacher, Xe, ye)); nsegs["res_T0"] = ns
            flow, hcm, teacher, ns, cs = deploy_sync(copy.deepcopy(mr), cal_x, onchip_residual_merge=True)
            rows["res_T1"].append(measure(flow, hcm, teacher, Xe, ye)); nsegs["res_T1"] = ns

        labels = {"plain": "plain", "res_T0": "residual_T0(hostadd)", "res_T1": "residual_T1(onchip)"}
        for key in ["plain", "res_T0", "res_T1"]:
            arr = torch.tensor(rows[key])
            ann, nf, hc = arr.mean(0).tolist()
            ret = hc / ann if ann > 0 else 0.0
            print(f"{labels[key]:<22}{d:>3} {ann:>7.3f} {nf:>10.3f} {hc:>10.3f} {100*ret:>6.1f}% {nsegs[key]:>5}")
        print()


if __name__ == "__main__":
    main()
