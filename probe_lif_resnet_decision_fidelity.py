"""STRONG probe: deep conv/BN RESIDUAL LIF deployed DECISION-FIDELITY on CIFAR10.

Builds a REAL ResNet-style conv/BN residual net, trains it to a decent ANN
top-1 on CIFAR10, then measures the LIF-deployed neuromorphic forward's
DECISION-FIDELITY = argmax-agreement-with-the-float-ANN (NOT accuracy-retention).

Reuses the PRODUCTION deploy primitives (convert -> install LIF ->
compute_per_source_scales -> activation-quantile scale calibration -> optional
DFQ bias match -> IRMapping -> hybrid HCM). The torch chip-aligned NF is the
deployment proxy; a bit-exactness check vs the packed HCM sim confirms NF==HCM.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn

from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.torch_mapping.encoding_layers import mark_encoding_layers
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import build_hybrid_hard_core_mapping
from mimarsinan.mapping.platform.mapping_structure import MappingStrategy
from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow
from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.mapping.support.per_source_scales import compute_per_source_scales
from mimarsinan.spiking.chip_aligned_nf import chip_aligned_segment_forward
from mimarsinan.spiking.lif_distribution_matching import match_lif_activation_distributions

NUM_CLASSES = 10
INPUT_SHAPE = (3, 32, 32)


class _ConvBNReLU(nn.Module):
    """Conv-BN-ReLU as MODULES so the converter folds it into ONE on-chip neuron."""

    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, 3, padding=1)
        self.bn = nn.BatchNorm2d(ch_out)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class BasicBlock(nn.Module):
    """Mappable ResNet block: y = z + relu(bn(conv2(relu(bn(conv1(z)))))).

    Both branches are post-ReLU rate-domain neurons; the residual add is a
    param-free equal-width add (host sync-point ComputeOp by default, or an
    on-chip merge under ``onchip_residual_merge``)."""

    def __init__(self, ch):
        super().__init__()
        self.f1 = _ConvBNReLU(ch, ch)
        self.f2 = _ConvBNReLU(ch, ch)

    def forward(self, x):
        return x + self.f2(self.f1(x))


class PlainBlock(nn.Module):
    """Same two conv-bn-relu neurons, NO skip: y = F(x). The residual-ablated twin."""

    def __init__(self, ch):
        super().__init__()
        self.f1 = _ConvBNReLU(ch, ch)
        self.f2 = _ConvBNReLU(ch, ch)

    def forward(self, x):
        return self.f2(self.f1(x))


class ResNet(nn.Module):
    """Stem conv-bn-relu -> /2 pool -> N (Basic|Plain)Blocks at fixed width -> GAP -> fc."""

    def __init__(self, depth, width=32, residual=True):
        super().__init__()
        self.stem = _ConvBNReLU(3, width)
        self.pool = nn.MaxPool2d(2)  # 32 -> 16
        block = BasicBlock if residual else PlainBlock
        self.blocks = nn.ModuleList([block(width) for _ in range(depth)])
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(width, NUM_CLASSES)

    def forward(self, x):
        x = self.pool(self.stem(x))
        for blk in self.blocks:
            x = blk(x)
        x = self.gap(x).flatten(1)
        return self.fc(x)


def cifar_loaders(data_root, batch=128, n_train=None):
    import torchvision
    import torchvision.transforms as TT

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    train_tf = TT.Compose([
        TT.RandomCrop(32, padding=4),
        TT.RandomHorizontalFlip(),
        TT.ToTensor(),
        TT.Normalize(mean, std),
    ])
    test_tf = TT.Compose([TT.ToTensor(), TT.Normalize(mean, std)])
    tr = torchvision.datasets.CIFAR10(data_root, train=True, download=True, transform=train_tf)
    te = torchvision.datasets.CIFAR10(data_root, train=False, download=True, transform=test_tf)
    if n_train is not None:
        tr = torch.utils.data.Subset(tr, list(range(n_train)))
    trl = torch.utils.data.DataLoader(tr, batch_size=batch, shuffle=True, num_workers=4, drop_last=True)
    tel = torch.utils.data.DataLoader(te, batch_size=256, shuffle=False, num_workers=4)
    return trl, tel


def train(model, trl, tel, device, epochs=30, lr=0.1):
    model = model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, epochs=epochs, steps_per_epoch=len(trl))
    lossf = nn.CrossEntropyLoss()
    for ep in range(epochs):
        model.train()
        for x, y in trl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = lossf(model(x), y)
            loss.backward()
            opt.step()
            sched.step()
        if ep % 5 == 0 or ep == epochs - 1:
            acc = evaluate(model, tel, device)
            print(f"  ep {ep:>2} top1={acc:.4f}", flush=True)
    return model.eval()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.numel()
    return correct / total


@torch.no_grad()
def calibrate_activation_scales(teacher, cal_x, quantile=0.99):
    """Per-perceptron activation_scale = q-quantile of post-activation positive outputs."""
    perceptrons = list(teacher.get_perceptrons())
    captured = {id(p): [] for p in perceptrons}
    handles = []
    for p in perceptrons:
        def hook(mod, inp, out, key=id(p)):
            captured[key].append(out.detach().reshape(-1).float().cpu())
        handles.append(p.activation.register_forward_hook(hook))
    teacher.eval()
    teacher(cal_x)
    for h in handles:
        h.remove()
    scales = []
    for p in perceptrons:
        vals = torch.cat(captured[id(p)]) if captured[id(p)] else torch.zeros(1)
        pos = vals[vals > 1e-9]
        if pos.numel() == 0:
            scales.append(1.0)
        else:
            q = torch.quantile(pos, float(quantile), interpolation="higher").item()
            scales.append(max(float(q), 1e-6))
    return scales


@torch.no_grad()
def fold_perlayer_scales(model, cal_x, quantile=0.99):
    """Fold a per-layer activation scale into the weights so every Conv-BN-ReLU
    neuron's post-activation output lands in [0,1] (LIF scale stays 1.0 -> NF==HCM
    bit-exact) WHILE giving each layer full rate resolution.

    Skip-chain constraint: the stem, every block's ``f2`` (the F branch added to
    the identity skip) and the add output all live in ONE rate domain, so they
    must share a single scale ``s_skip``. The internal ``f1`` layers are free
    per-layer. The fold:
      - divide a neuron's output by its scale (BN affine gamma,beta /= s);
      - multiply the consuming conv's input-channel weight slices by the upstream
        scale (so ``W @ (a/s) * s == W @ a``).
    Returns the per-layer scales applied (diagnostic).
    """
    # 1) per-layer post-activation 0.99-quantiles
    def act_quantile(m):
        outs = {}
        hs = []
        for name, mod in m.named_modules():
            if isinstance(mod, _ConvBNReLU):
                def hook(mm, i, o, key=name):
                    outs.setdefault(key, []).append(o.detach())
                hs.append(mod.register_forward_hook(hook))
        m.eval()
        m(cal_x)
        for h in hs:
            h.remove()
        q = {}
        for k, v in outs.items():
            a = torch.cat([t.reshape(-1) for t in v])
            pos = a[a > 1e-9]
            q[k] = max(float(torch.quantile(pos, quantile, interpolation="higher")), 1e-6) if pos.numel() else 1.0
        return q

    q = act_quantile(model)
    # 2) skip-domain scale = max over {stem, all f2}
    skip_keys = ["stem"] + [f"blocks.{i}.f2" for i in range(len(model.blocks))]
    s_skip = max(q[k] for k in skip_keys if k in q)

    def set_scale(cbr_name, s):
        cbr = dict(model.named_modules())[cbr_name]
        cbr.bn.weight.data /= s
        cbr.bn.bias.data /= s

    # stem & all f2 -> s_skip
    for k in skip_keys:
        set_scale(k, s_skip)
    # f1 internal -> own scale; the f2 conv that consumes f1 must be re-scaled by s_f1
    applied = {}
    for k in skip_keys:
        applied[k] = s_skip
    for i in range(len(model.blocks)):
        f1_name = f"blocks.{i}.f1"
        s_f1 = q[f1_name]
        set_scale(f1_name, s_f1)
        applied[f1_name] = s_f1
        # f2's conv reads f1's output (now /s_f1) -> multiply f2.conv weight by s_f1
        model.blocks[i].f2.conv.weight.data *= s_f1
        # f1's conv reads the block input (skip domain, /s_skip) -> *= s_skip
        model.blocks[i].f1.conv.weight.data *= s_skip
        # f2's input from f1 is handled; f2's own output set to s_skip above.

    # stem reads the raw image (no upstream neuron) -> no input rescale.
    # block0.f1 reads stem output (/s_skip) -> already *= s_skip above.
    # the residual add output (block_i out) is in /s_skip domain; the NEXT block's
    # f1 already multiplies its conv weight by s_skip (handled in the loop).
    # fc reads the final block output (/s_skip) -> *= s_skip on fc weight.
    model.fc.weight.data *= s_skip
    return applied


def deploy_lif(model, cal_x, T, *, onchip_residual_merge=False, quantile=0.99,
               dfq=False, scale_mode="global", global_scale=None, fold=False, device="cpu"):
    """Production LIF deploy. Returns (flow, hybrid, teacher, nseg, scales).

    ``scale_mode``:
      - ``"one"``    : activation_scale=1.0 everywhere (bit-exact NF==HCM).
      - ``"global"`` : ONE uniform scalar scale on every layer = the chosen
        ``global_scale`` (default = max over per-layer q-quantiles). A uniform
        threshold change stays bit-exact NF==HCM while giving the rate code
        headroom. This is the calibrated, bit-exact deploy.
      - ``"perlayer"``: per-perceptron q-quantile scale (the production
        ActivationAnalysisStep value). NOT bit-exact under this probe's
        re-encode (heterogeneous boundary scales) -- used only to show the
        production-calibrated point with its caveat.
    """
    if fold:
        # Per-layer scale folded into weights -> LIF scale stays 1.0 (bit-exact
        # NF==HCM) AND every layer gets full [0,1] rate resolution.
        model = copy.deepcopy(model)
        fold_perlayer_scales(model, cal_x, quantile=quantile)
        scale_mode = "one"

    flow = convert_torch_model(model, INPUT_SHAPE, NUM_CLASSES, device=device)
    flow.eval()
    repr_ = flow.get_mapper_repr()
    mark_encoding_layers(repr_)
    teacher = copy.deepcopy(flow).eval()

    n_perc = len(list(flow.get_perceptrons()))
    perlayer = calibrate_activation_scales(teacher, cal_x, quantile=quantile)
    if scale_mode == "one":
        scales = [1.0] * n_perc
    elif scale_mode == "perlayer":
        scales = perlayer
    else:  # global
        g = float(global_scale) if global_scale is not None else max(perlayer)
        scales = [g] * n_perc
    for p, s in zip(flow.get_perceptrons(), scales):
        p.set_activation_scale(float(s))
        lif = LIFActivation(T=T, activation_scale=p.activation_scale, thresholding_mode="<=")
        p.set_activation(lif)

    repr_.assign_perceptron_indices()
    compute_per_source_scales(repr_)

    if dfq:
        match_lif_activation_distributions(flow, teacher, cal_x, T)

    ir = IRMapping(
        q_max=127.0, firing_mode="Default", max_axons=100000, max_neurons=100000,
        allow_coalescing=False, onchip_residual_merge=onchip_residual_merge,
    ).map(repr_)
    hybrid = build_hybrid_hard_core_mapping(
        ir_graph=ir,
        cores_config=[{"max_axons": 100000, "max_neurons": 100000, "count": 20000}],
        strategy=MappingStrategy.from_permissions(
            allow_neuron_splitting=False, allow_coalescing=False),
    )
    nseg = sum(1 for s in hybrid.stages if s.hard_core_mapping is not None)
    return flow, hybrid, teacher, nseg, scales


def build_hcm(flow, hybrid, T):
    return SpikingHybridCoreFlow(
        INPUT_SHAPE, hybrid, simulation_length=T, preprocessor=nn.Identity(),
        firing_mode="Default", spike_mode="Uniform", thresholding_mode="<=",
        spiking_mode="lif", cycle_accurate_lif_forward=True,
    )


@torch.no_grad()
def decision_fidelity(flow, teacher, Xe, T, device, batch=64):
    """argmax-agreement of LIF chip-aligned NF vs the float ANN teacher."""
    teacher = teacher.to(device)
    flow = flow.to(device)
    agree = 0
    n = 0
    for i in range(0, len(Xe), batch):
        xb = Xe[i:i + batch].to(device)
        ann = teacher(xb)
        nf = chip_aligned_segment_forward(flow, xb, T)
        agree += (ann.argmax(1) == nf.argmax(1)).sum().item()
        n += xb.shape[0]
    return agree / n


@torch.no_grad()
def nf_logits(flow, Xe, T, device, batch=64):
    flow = flow.to(device)
    out = []
    for i in range(0, len(Xe), batch):
        out.append(chip_aligned_segment_forward(flow, Xe[i:i + batch].to(device), T).cpu())
    return torch.cat(out)


@torch.no_grad()
def ann_logits(teacher, Xe, device, batch=256):
    teacher = teacher.to(device)
    out = []
    for i in range(0, len(Xe), batch):
        out.append(teacher(Xe[i:i + batch].to(device)).cpu())
    return torch.cat(out)


@torch.no_grad()
def verify_nf_equals_hcm(flow, hybrid, Xe, T, n=8):
    """Confirm NF == packed HCM (not a sim bug). Returns max|Δ| over n samples (CPU)."""
    hcm = build_hcm(flow, hybrid, T)
    x = Xe[:n].cpu().float()
    nf = chip_aligned_segment_forward(flow.cpu(), x, T).double()
    hc = hcm(x).double() / float(T)
    return float((nf - hc).abs().max())
