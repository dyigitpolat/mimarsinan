"""QAT-exit probe: does equalization raise the AQ-QAT recovery ceiling?

Arms per (S, seed): pre-eq vs post-eq (s clipped to [1/4, 4]) mixer, identical
QAT geometry mirroring the pipeline endpoint: STE ceil kernel installed at full
rate, Adam lr=2e-3 (tuning_policy.endpoint_floor_lr), cosine over budget,
CE label_smoothing=0.1, batch 128, keep-best on val, test read at exit.
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
from mimarsinan.models.nn.activations.autograd import TTFSStaircaseFunction  # noqa: E402
from mimarsinan.models.torch_mlp_mixer_core import TorchMLPMixerCore  # noqa: E402

from proto import (  # noqa: E402
    DEVICE, OUT, loaders, evaluate, mixer_hops, collect_stats, gauge_row,
    fold_bn_into_conv, wq_project, SimpleMLPProto,
)
import __main__
__main__.SimpleMLPProto = SimpleMLPProto  # models.pt pickled under __main__


@torch.no_grad()
def equalize_clipped(model, hops, stats, clip=4.0, alpha=1.0):
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
        s[live] = ((ch[live] / gm) ** alpha).clamp(1.0 / clip, clip)
        A = h.producer
        A.weight.data.div_(s.view(-1, *([1] * (A.weight.dim() - 1))).to(A.weight.device))
        if A.bias is not None:
            A.bias.data.div_(s.to(A.bias.device))
        for B in h.consumers:
            B.weight.data.mul_(s.view(1, -1).to(B.weight.device))
        n_pairs += 1
    return n_pairs


class STEHook:
    """Trainable ceil-kernel install on a hop producer (autograd-friendly)."""

    def __init__(self, producer, theta, S):
        self.theta, self.S = float(theta), int(S)
        self.handle = producer.register_forward_hook(self)

    def __call__(self, module, inp, out):
        a = F.relu(out)
        th = torch.tensor(self.theta, device=a.device, dtype=a.dtype).clamp(min=1e-12)
        return TTFSStaircaseFunction.apply(a / th, self.S) * th

    def remove(self):
        self.handle.remove()


def qat(model, thetas, S, steps=600, lr=2e-3, seed=0):
    torch.manual_seed(seed)
    m = copy.deepcopy(model)
    hops = mixer_hops(m)
    fold_bn_into_conv(m.patch_embed, m.patch_bn)
    m.patch_bn = nn.Identity()
    installs = [STEHook(h.producer, th, S) for h, th in zip(hops, thetas)]

    tr, va, te = loaders()
    opt = torch.optim.Adam(m.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=5e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 1e-3)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    entry = evaluate(m, te)
    best_val, best_state = -1.0, None
    it = iter(tr)
    m.train()
    for step in range(steps):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(tr)
            x, y = next(it)
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad(set_to_none=True)
        loss_fn(m(x), y).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        sched.step()
        if (step + 1) % 100 == 0:
            va_acc = evaluate(m, va)
            if va_acc > best_val:
                best_val = va_acc
                best_state = copy.deepcopy(m.state_dict())
            m.train()
    if best_state is not None:
        m.load_state_dict(best_state)
    exit_acc = evaluate(m, te)
    for i in installs:
        i.remove()
    return entry, exit_acc


def main():
    blob = torch.load(os.path.join(OUT, "models.pt"), map_location=DEVICE, weights_only=False)
    mixer = blob["mixer"].to(DEVICE).eval()
    tr, va, te = loaders()
    results = {"float": blob["float_test"]["mixer"]}

    # --- arms
    arms = {}
    arms["pre"] = copy.deepcopy(mixer)
    post = copy.deepcopy(mixer)
    hops = mixer_hops(post)
    st = collect_stats(post, hops, va)
    n = equalize_clipped(post, hops, st, clip=4.0, alpha=1.0)
    # exactness check
    xref = torch.cat([te.dataset[i][0].unsqueeze(0) for i in range(2048)]).to(DEVICE)
    with torch.no_grad():
        d = (mixer(xref) - post(xref)).abs().max().item()
        agree = (mixer(xref).argmax(-1) == post(xref).argmax(-1)).float().mean().item()
    print(f"clipped equalization: pairs={n} max|dlogit|={d:.3e} agree={agree:.6f}", flush=True)
    arms["post"] = post

    # post-eq gauges + WQ sanity
    st2 = collect_stats(post, mixer_hops(post), va)
    for s in st2:
        g4 = gauge_row(s, 4)
        print(f"  postclip {s['name']:24s} theta={g4['theta']:7.3f} "
              f"chq99[max/med/gmean/min]={g4['ch_max']:.3f}/{g4['ch_med']:.3f}/"
              f"{g4['ch_gmean']:.3f}/{g4['ch_min']:.4f} medlvl@4={g4['median_levels']:.2f} "
              f"sm@4={g4['starved_mass']:.2f}", flush=True)
    results["wq5_postclip"] = evaluate(wq_project(post), te)
    results["wq5_pre"] = evaluate(wq_project(mixer), te)
    print(f"wq5: pre={results['wq5_pre']:.4f} postclip={results['wq5_postclip']:.4f}", flush=True)

    thetas = {}
    for name, m in arms.items():
        stt = collect_stats(m, mixer_hops(m), va)
        thetas[name] = [s["theta"] for s in stt]

    # --- QAT probe
    results["qat"] = {}
    for S in (4, 8):
        for name, m in arms.items():
            reads = []
            for seed in (0, 1, 2):
                entry, exit_acc = qat(m, thetas[name], S, steps=600, seed=seed)
                reads.append((entry, exit_acc))
                print(f"S={S} arm={name} seed={seed}: entry={entry:.4f} exit={exit_acc:.4f}",
                      flush=True)
            results["qat"][f"S{S}_{name}"] = reads

    with open(os.path.join(OUT, "qat_results.json"), "w") as f:
        json.dump(results, f, indent=1)
    print(json.dumps(results["qat"], indent=1))


if __name__ == "__main__":
    main()
