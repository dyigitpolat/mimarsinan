"""Mixer per-channel scale pathology prototype: measure, equalize, project AQ/WQ.

Mirrors the pipeline semantics exactly where it matters:
- pretrain recipe: Adam lr=3e-3 wd=5e-5, CosineAnnealing over N epochs + 5 warmup epochs
  (basic_trainer.py:117-126, basic_trainer_epochs.py:31-56, pretraining_step.py:19-23),
  CE label_smoothing=0.1 (training_utilities.py:24), MNIST raw [0,1], 95/5 split.
- theta: count-quantile 0.99 over positive activations, interpolation='higher'
  (activation_analysis_step.py:63-82), pooled over all channels.
- per-channel q99: positives-only count quantile (install_resolution/capture.py:46-58).
- gauge math: median_effective_levels / starved_mass (install_resolution/gauges.py:49-67).
- AQ kernels: floor staircase (clamp_quantize.py:52-63 + clamp), ttfs ceil kernel
  (clamp_quantize.py:30-49 -> wire_semantics.ttfs_quantized_staircase), nearest grid
  (wire_semantics.ttfs_grid_quantize).
- WQ: per-perceptron symmetric grid, scale = q_max / max(|w|,|b|)
  (normalization_aware_perceptron_quantization.py:36-64), weight_bits=5 (tier0).
"""

import copy
import json
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/home/yigit/repos/research_stuff/mimarsinan/src")
from mimarsinan.models.spiking.wire_semantics import (  # noqa: E402
    ttfs_quantized_staircase,
    ttfs_grid_quantize,
)
from mimarsinan.models.torch_mlp_mixer_core import TorchMLPMixerCore  # noqa: E402
from mimarsinan.models.lenet5 import LeNet5  # noqa: E402

DEVICE = torch.device("cuda:0")
OUT = os.path.dirname(os.path.abspath(__file__))
DATA = "/home/yigit/repos/research_stuff/mimarsinan/datasets"
torch.manual_seed(0)

# ---------------------------------------------------------------- data
import torchvision  # noqa: E402
from torchvision import transforms  # noqa: E402


def loaders(batch_size=128):
    tf = transforms.ToTensor()
    full = torchvision.datasets.MNIST(DATA, train=True, download=False, transform=tf)
    n = len(full)
    tr_n = int(n * 0.95)
    g = torch.Generator().manual_seed(0)
    tr, va = torch.utils.data.random_split(full, (tr_n, n - tr_n), generator=g)
    te = torchvision.datasets.MNIST(DATA, train=False, download=False, transform=tf)
    mk = lambda ds, sh: torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=sh, num_workers=4, pin_memory=True)
    return mk(tr, True), mk(va, False), mk(te, False)


# ---------------------------------------------------------------- comparison model
class SimpleMLPProto(nn.Module):
    """simple_mlp topology (simple_mlp.py:36-38): 784->w1->w2->w1->10, BN on L0/L2."""

    def __init__(self, w1=128, w2=64):
        super().__init__()
        self.l0 = nn.Linear(784, w1)
        self.bn0 = nn.BatchNorm1d(w1)
        self.l1 = nn.Linear(w1, w2)
        self.l2 = nn.Linear(w2, w1)
        self.bn2 = nn.BatchNorm1d(w1)
        self.l3 = nn.Linear(w1, 10)

    def forward(self, x):
        x = x.flatten(1)
        x = F.relu(self.bn0(self.l0(x)))
        x = F.relu(self.l1(x))
        x = F.relu(self.bn2(self.l2(x)))
        return self.l3(x)


# ---------------------------------------------------------------- training
def train(model, epochs, lr=3e-3, warmup_epochs=5, tag=""):
    tr, va, te = loaders()
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=5e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 1e-3)
    import warmup_scheduler
    sched = warmup_scheduler.GradualWarmupScheduler(
        opt, multiplier=1.0, total_epoch=warmup_epochs, after_scheduler=sched)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    for ep in range(epochs + warmup_epochs):
        model.train()
        for x, y in tr:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss_fn(model(x), y).backward()
            opt.step()
        sched.step()
        acc = evaluate(model, va)
        print(f"  [{tag}] epoch {ep+1}/{epochs+warmup_epochs} val={acc:.4f}", flush=True)
    return evaluate(model, te)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    hits = tot = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        hits += (model(x).argmax(-1) == y).sum().item()
        tot += y.numel()
    return hits / tot


# ---------------------------------------------------------------- hop instrumentation
# A "hop" = one perceptron = (linear-or-conv [+BN]) + ReLU. For the mixer the
# converted flow's hops are: patch_embed_full, mixer_blocks_{0..3}_{fc1,fc2}
# (Ensure2DMapper flattens leading dims -> channel axis is the linear's out dim;
# conv hop channel axis is dim 1).

class HopSpec:
    def __init__(self, name, act_parent, channel_dim, producer, consumers):
        self.name = name
        self.act_parent = act_parent      # module whose forward output is the post-act tensor
        self.channel_dim = channel_dim    # channel axis of that output tensor
        self.producer = producer          # nn.Linear/nn.Conv2d producing this hop's channels
        self.consumers = consumers        # list of nn.Linear consuming this axis as IN features (exact pairs), or []


def mixer_hops(model):
    hops = []
    b = model.mixer_blocks
    # patch stem: post-act tensor produced inside model.forward after patch_bn;
    # we hook model.act on the stem call via a wrapper below instead.
    hops.append(HopSpec("patch_embed_full", ("stem", model), 1, model.patch_embed, []))
    for i, blk in enumerate(b):
        hops.append(HopSpec(f"mixer_blocks_{i}_fc1", ("mod", blk.fc1), -1, blk.fc1, [blk.fc2]))
        last = (i == len(b) - 1)
        consumers = [model.classifier] if (last and isinstance(blk, type(b[1]))) else []
        # token mixer fc2 channels = patches (not feature-adjacent to anything);
        # channel mixer fc2 channels = C, consumed (for the LAST block) by classifier via mean.
        if not last:
            consumers = []
        hops.append(HopSpec(f"mixer_blocks_{i}_fc2", ("mod", blk.fc2), -1, blk.fc2, consumers))
    return hops


def lenet_hops(model):
    f, c = model.features, model.classifier
    return [
        HopSpec("features_0_full", ("mod", f[0]), 1, f[0], []),
        HopSpec("features_3_full", ("mod", f[3]), 1, f[3], []),
        HopSpec("classifier_0", ("mod", c[0]), -1, c[0], [c[2]]),
        HopSpec("classifier_2", ("mod", c[2]), -1, c[2], [c[4]]),
    ]


def simplemlp_hops(model):
    return [
        HopSpec("l0", ("bn", model.bn0), -1, model.l0, []),          # BN precedes act
        HopSpec("l1", ("mod", model.l1), -1, model.l1, []),
        HopSpec("l2", ("bn", model.bn2), -1, model.l2, []),
    ]


class ActTap:
    """Captures the post-ReLU tensor for a hop by hooking the producing linear/conv
    and re-applying BN(if stem)+ReLU functionally — avoids double-hook issues with
    the shared inplace ReLU module."""

    def __init__(self, model, hop):
        self.rows = []
        self.channel_dim = hop.channel_dim
        kind, mod = hop.act_parent
        self.hop = hop
        self.model = model
        target = hop.producer
        self.handle = target.register_forward_hook(self._grab)
        self.kind = kind

    def _grab(self, module, inp, out):
        with torch.no_grad():
            if self.kind == "stem":
                out = self.model.patch_bn(out)
            elif self.kind == "bn":
                out = self.hop.act_parent[1](out)
            a = F.relu(out)
            d = self.channel_dim % a.dim()
            rows = a.detach().moveaxis(d, -1).reshape(-1, a.shape[d])
            if rows.shape[0] > 4096:
                idx = torch.linspace(0, rows.shape[0] - 1, 4096, device=rows.device).round().long()
                rows = rows.index_select(0, idx)
            self.rows.append(rows.float().cpu())

    def stacked(self):
        return torch.cat(self.rows, 0)

    def remove(self):
        self.handle.remove()


def q_count_higher(t, q):
    """Count quantile with interpolation='higher' (matches torch.quantile usage)."""
    if t.numel() == 0:
        return 0.0
    return float(torch.quantile(t, q, interpolation="higher"))


@torch.no_grad()
def collect_stats(model, hops, loader, n_batches=32):
    taps = [ActTap(model, h) for h in hops]
    model.eval()
    it = iter(loader)
    for _ in range(min(n_batches, len(loader))):
        x, _ = next(it)
        model(x.to(DEVICE))
    stats = []
    for h, tap in zip(hops, taps):
        rows = tap.stacked()
        pos = rows[rows > 1e-9]
        theta = max(q_count_higher(pos, 0.99), 1e-6) if pos.numel() else 1.0
        ch_q99 = []
        for c in range(rows.shape[1]):
            p = rows[:, c][rows[:, c] > 0]
            ch_q99.append(q_count_higher(p, 0.99) if p.numel() else 0.0)
        tap.remove()
        stats.append(dict(name=h.name, theta=theta, ch_q99=ch_q99,
                          pos=pos, rows_shape=tuple(rows.shape)))
    return stats


def gauge_row(st, S):
    theta, ch = st["theta"], [c for c in st["ch_q99"] if c > 0]
    delta = theta / S
    med_lvl = float(torch.tensor([c / delta for c in ch]).median()) if ch else 0.0
    pos = st["pos"]
    sm = float((pos < delta).float().mean()) if pos.numel() else 1.0
    zero_ch = sum(1 for c in ch if c < delta) / max(len(ch), 1)
    gm = math.exp(sum(math.log(c) for c in ch) / len(ch)) if ch else 0.0
    return dict(theta=theta, median_levels=med_lvl, starved_mass=sm,
                frac_ch_below_1lvl=zero_ch,
                ch_max=max(ch) if ch else 0.0, ch_med=float(torch.tensor(ch).median()) if ch else 0.0,
                ch_gmean=gm, ch_min=min(ch) if ch else 0.0, live=len(ch), total=len(st["ch_q99"]))


# ---------------------------------------------------------------- equalization
@torch.no_grad()
def equalize(model, hops, stats, alpha=1.0):
    """Cross-layer scale migration on exact pairs: divide producer row c (w,b) by s_c,
    multiply each consumer's column c by s_c. s_c = (q99_c / gmean)^alpha, dead ch -> 1."""
    n_pairs = 0
    for h, st in zip(hops, stats):
        if not h.consumers:
            continue
        ch = torch.tensor(st["ch_q99"])
        live = ch > 0
        if live.sum() == 0:
            continue
        gm = torch.exp(torch.log(ch[live]).mean())
        s = torch.ones_like(ch)
        s[live] = (ch[live] / gm) ** alpha
        s = s.clamp(min=1e-3)
        A = h.producer
        A.weight.data.div_(s.view(-1, *([1] * (A.weight.dim() - 1))).to(A.weight.device))
        if A.bias is not None:
            A.bias.data.div_(s.to(A.bias.device))
        for B in h.consumers:
            B.weight.data.mul_(s.view(1, -1).to(B.weight.device))
        n_pairs += 1
    return n_pairs


# ---------------------------------------------------------------- projection kernels
class KernelWrap(nn.Module):
    """Wraps a linear/conv producer: recompute post-act with clamp+staircase at theta."""


@torch.no_grad()
def project_model(model, hops, thetas, kernel, S):
    """Return a deep-copied model whose hop activations pass through the value kernel.
    kernel in {floor, ceil_ttfs, nearest}. Implemented via forward hooks that REPLACE
    the producer output so that downstream sees quantized-activation values:
    we hook the producer and return (quantized_act) composed with the -ReLU inverse?
    Simpler: hook producer output, apply BN(if stem)+ReLU+kernel, and mark so the
    outer ReLU is a no-op (ReLU is idempotent on the non-negative kernel output;
    stem BN must be bypassed on second application)."""
    m2 = copy.deepcopy(model)
    if isinstance(m2, TorchMLPMixerCore):
        hops2 = mixer_hops(m2)
        # neutralize the stem BN double-application: fold BN into patch_embed
        fold_bn_into_conv(m2.patch_embed, m2.patch_bn)
        m2.patch_bn = nn.Identity()
    elif isinstance(m2, LeNet5):
        hops2 = lenet_hops(m2)
    else:
        hops2 = simplemlp_hops(m2)
        fold_bn_into_linear(m2.l0, m2.bn0); m2.bn0 = nn.Identity()
        fold_bn_into_linear(m2.l2, m2.bn2); m2.bn2 = nn.Identity()

    def mk_hook(theta, S=S, kernel=kernel):
        def hook(module, inp, out):
            a = F.relu(out)
            th = torch.tensor(theta, device=a.device, dtype=a.dtype)
            if kernel == "floor":
                a = torch.clamp(a, torch.zeros_like(th), th)
                a = torch.floor(a * (S / th)) / (S / th)
            elif kernel == "ceil_ttfs":
                a = ttfs_quantized_staircase(a, th, S) * th
            elif kernel == "nearest":
                r = (a / th).clamp(0.0, 1.0)
                a = ttfs_grid_quantize(r, S) * th
            # returning `a` replaces linear output; the model's own ReLU then
            # re-applies on a >= 0 which is identity.
            return a
        return hook

    handles = []
    for h, th in zip(hops2, thetas):
        handles.append(h.producer.register_forward_hook(mk_hook(th)))
    return m2, handles


@torch.no_grad()
def fold_bn_into_conv(conv, bn):
    u = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    conv.weight.data.mul_(u.view(-1, 1, 1, 1))
    b = conv.bias.data if conv.bias is not None else torch.zeros_like(bn.running_mean)
    conv.bias = nn.Parameter((b - bn.running_mean) * u + bn.bias)


@torch.no_grad()
def fold_bn_into_linear(lin, bn):
    u = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    lin.weight.data.mul_(u.view(-1, 1))
    b = lin.bias.data if lin.bias is not None else torch.zeros_like(bn.running_mean)
    lin.bias = nn.Parameter((b - bn.running_mean) * u + bn.bias)


@torch.no_grad()
def wq_project(model, bits=5):
    """Per-perceptron symmetric grid on (w,b) jointly, mirroring NAPQ rate=1."""
    m2 = copy.deepcopy(model)
    q_max = 2 ** (bits - 1) - 1
    mods = [m for m in m2.modules() if isinstance(m, (nn.Linear, nn.Conv2d))]
    for m in mods:
        w = m.weight.data
        b = m.bias.data if m.bias is not None else torch.zeros(1, device=w.device)
        p_max = torch.clamp(torch.maximum(w.abs().max(), b.abs().max()), min=1e-12)
        scale = q_max / p_max
        m.weight.data = torch.round(w * scale).clamp(-q_max - 1, q_max) / scale
        if m.bias is not None:
            m.bias.data = torch.round(m.bias.data * scale).clamp(-q_max - 1, q_max) / scale
    return m2


# ---------------------------------------------------------------- main
def main():
    tr, va, te = loaders()
    results = {}

    ckpt = os.path.join(OUT, "models.pt")
    if os.path.exists(ckpt):
        blob = torch.load(ckpt, map_location=DEVICE, weights_only=False)
        mixer, lenet, smlp = blob["mixer"], blob["lenet"], blob["smlp"]
        results["float_test"] = blob["float_test"]
        print("loaded checkpoint:", blob["float_test"])
    else:
        mixer = TorchMLPMixerCore((1, 28, 28), 10, 4, 4, 32, 64, 64, "ReLU", 2)
        lenet = LeNet5((1, 28, 28), 10, "ReLU")
        smlp = SimpleMLPProto()
        accs = {}
        accs["mixer"] = train(mixer, 2, tag="mixer")
        accs["lenet"] = train(lenet, 2, tag="lenet")
        accs["smlp"] = train(smlp, 2, tag="smlp")
        results["float_test"] = accs
        torch.save(dict(mixer=mixer, lenet=lenet, smlp=smlp, float_test=accs), ckpt)
        print("float test:", accs)

    for m in (mixer, lenet, smlp):
        m.to(DEVICE).eval()

    specs = {"mixer": (mixer, mixer_hops(mixer)),
             "lenet": (lenet, lenet_hops(lenet)),
             "smlp": (smlp, simplemlp_hops(smlp))}

    # ---- stage 1: stats + gauges pre-equalization
    all_stats = {}
    results["gauges_pre"] = {}
    for name, (model, hops) in specs.items():
        st = collect_stats(model, hops, va)
        all_stats[name] = st
        results["gauges_pre"][name] = {
            s["name"]: {f"S{S}": gauge_row(s, S) for S in (4, 8, 16)} for s in st}
        print(f"\n== {name} per-hop stats (pre-eq)")
        for s in st:
            g4 = gauge_row(s, 4)
            print(f"  {s['name']:24s} theta={g4['theta']:7.3f} chq99[max/med/gmean/min]="
                  f"{g4['ch_max']:.3f}/{g4['ch_med']:.3f}/{g4['ch_gmean']:.3f}/{g4['ch_min']:.4f} "
                  f"live={g4['live']}/{g4['total']} medlvl@4={g4['median_levels']:.2f} "
                  f"sm@4={g4['starved_mass']:.2f}", flush=True)

    # ---- stage 2: projections pre-equalization
    results["proj_pre"] = {}
    for name, (model, hops) in specs.items():
        thetas = [s["theta"] for s in all_stats[name]]
        row = {}
        for kernel in ("floor", "ceil_ttfs", "nearest"):
            for S in (4, 8, 16):
                m2, hd = project_model(model, hops, thetas, kernel, S)
                acc = evaluate(m2, te)
                for h in hd:
                    h.remove()
                row[f"{kernel}_S{S}"] = acc
        m2 = wq_project(model)
        row["wq5"] = evaluate(m2, te)
        results["proj_pre"][name] = row
        print(f"\n== {name} projections (pre-eq): {json.dumps(row, indent=None)}", flush=True)

    # ---- stage 3: equalize the mixer (alpha=1.0 and 0.5), verify exactness
    for alpha in (1.0, 0.5):
        mx = copy.deepcopy(mixer)
        hops = mixer_hops(mx)
        st = collect_stats(mx, hops, va)
        # reference logits
        xref = torch.cat([te.dataset[i][0].unsqueeze(0) for i in range(2048)]).to(DEVICE)
        mx.eval()
        with torch.no_grad():
            ref = mx(xref)
        n_pairs = equalize(mx, hops, st, alpha=alpha)
        with torch.no_grad():
            new = mx(xref)
        max_dev = float((ref - new).abs().max())
        rel_dev = float(((ref - new).abs() / ref.abs().clamp(min=1e-6)).max())
        agree = float((ref.argmax(-1) == new.argmax(-1)).float().mean())
        print(f"\n== equalize alpha={alpha}: pairs={n_pairs} max|dlogit|={max_dev:.3e} "
              f"maxrel={rel_dev:.3e} argmax_agree={agree:.6f}", flush=True)

        st2 = collect_stats(mx, hops, va)
        gpost = {s["name"]: {f"S{S}": gauge_row(s, S) for S in (4, 8, 16)} for s in st2}
        results[f"gauges_post_a{alpha}"] = gpost
        print(f"== mixer per-hop stats (post-eq alpha={alpha})")
        for s in st2:
            g4 = gauge_row(s, 4)
            print(f"  {s['name']:24s} theta={g4['theta']:7.3f} chq99[max/med/gmean/min]="
                  f"{g4['ch_max']:.3f}/{g4['ch_med']:.3f}/{g4['ch_gmean']:.3f}/{g4['ch_min']:.4f} "
                  f"medlvl@4={g4['median_levels']:.2f} sm@4={g4['starved_mass']:.2f}", flush=True)

        thetas2 = [s["theta"] for s in st2]
        row = {}
        for kernel in ("floor", "ceil_ttfs", "nearest"):
            for S in (4, 8, 16):
                m2, hd = project_model(mx, hops, thetas2, kernel, S)
                acc = evaluate(m2, te)
                for h in hd:
                    h.remove()
                row[f"{kernel}_S{S}"] = acc
        m2 = wq_project(mx)
        row["wq5"] = evaluate(m2, te)
        # combined AQ+WQ at the tier0 operating points
        for S in (4, 8):
            m3 = wq_project(mx)
            hops3 = mixer_hops(m3)
            m4, hd = project_model(m3, hops3, thetas2, "ceil_ttfs", S)
            row[f"wq5+ceil_S{S}"] = evaluate(m4, te)
            for h in hd:
                h.remove()
        results[f"proj_post_a{alpha}"] = row
        results[f"exactness_a{alpha}"] = dict(max_abs_dlogit=max_dev, max_rel=rel_dev, argmax_agree=agree)
        print(f"== mixer projections (post-eq alpha={alpha}): {json.dumps(row)}", flush=True)

    # combined pre-eq for reference
    row = {}
    hops = specs["mixer"][1]
    thetas = [s["theta"] for s in all_stats["mixer"]]
    for S in (4, 8):
        m3 = wq_project(mixer)
        hops3 = mixer_hops(m3)
        m4, hd = project_model(m3, hops3, thetas, "ceil_ttfs", S)
        row[f"wq5+ceil_S{S}"] = evaluate(m4, te)
        for h in hd:
            h.remove()
    results["proj_pre"]["mixer"].update(row)
    print("mixer pre-eq combined:", row, flush=True)

    with open(os.path.join(OUT, "results.json"), "w") as f:
        def clean(o):
            if isinstance(o, dict):
                return {k: clean(v) for k, v in o.items() if k != "pos"}
            if isinstance(o, list):
                return [clean(v) for v in o]
            return o
        json.dump(clean(results), f, indent=1)
    print("\nsaved results.json")


if __name__ == "__main__":
    main()
