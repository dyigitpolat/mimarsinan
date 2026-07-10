"""Raw projections + ramped-QAT probe for the CLIPPED equalization variant."""

import copy
import json
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/home/yigit/repos/research_stuff/mimarsinan/src")
from mimarsinan.models.nn.activations.autograd import TTFSStaircaseFunction  # noqa: E402

from proto import (  # noqa: E402
    DEVICE, OUT, loaders, evaluate, mixer_hops, collect_stats,
    fold_bn_into_conv, wq_project, project_model, SimpleMLPProto,
)
from qat_probe import equalize_clipped  # noqa: E402
import __main__
__main__.SimpleMLPProto = SimpleMLPProto


class RampSTEHook:
    """Pipeline-style rate-ramped ceil-kernel install: mixes quantized and float
    activations by a random element mask at rate r (RandomMaskAdjustmentStrategy
    + MixAdjustmentStrategy analogue), r ramps 0->1 over ramp_steps."""

    def __init__(self, producer, theta, S):
        self.theta, self.S = float(theta), int(S)
        self.rate = 1.0
        self.handle = producer.register_forward_hook(self)

    def __call__(self, module, inp, out):
        a = F.relu(out)
        th = torch.tensor(self.theta, device=a.device, dtype=a.dtype).clamp(min=1e-12)
        q = TTFSStaircaseFunction.apply(a / th, self.S) * th
        if self.rate >= 1.0:
            return q
        mask = (torch.rand_like(a) < self.rate).to(a.dtype)
        return mask * q + (1.0 - mask) * a

    def remove(self):
        self.handle.remove()


def ramped_qat(model, thetas, S, steps=1200, ramp_steps=600, lr=2e-3, seed=0):
    torch.manual_seed(seed)
    m = copy.deepcopy(model)
    hops = mixer_hops(m)
    fold_bn_into_conv(m.patch_embed, m.patch_bn)
    m.patch_bn = nn.Identity()
    installs = [RampSTEHook(h.producer, th, S) for h, th in zip(hops, thetas)]

    tr, va, te = loaders()
    opt = torch.optim.Adam(m.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=5e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 1e-3)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_val, best_state = -1.0, None
    it = iter(tr)
    m.train()
    for step in range(steps):
        r = min(1.0, (step + 1) / ramp_steps)
        for h in installs:
            h.rate = r
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
        if step + 1 >= ramp_steps and (step + 1) % 100 == 0:
            for h in installs:
                h.rate = 1.0
            va_acc = evaluate(m, va)
            if va_acc > best_val:
                best_val = va_acc
                best_state = copy.deepcopy(m.state_dict())
            m.train()
    if best_state is not None:
        m.load_state_dict(best_state)
    for h in installs:
        h.rate = 1.0
    exit_acc = evaluate(m, te)
    for i in installs:
        i.remove()
    return exit_acc


def main():
    blob = torch.load(os.path.join(OUT, "models.pt"), map_location=DEVICE, weights_only=False)
    mixer = blob["mixer"].to(DEVICE).eval()
    tr, va, te = loaders()
    results = {}

    post = copy.deepcopy(mixer)
    hops = mixer_hops(post)
    st = collect_stats(post, hops, va)
    equalize_clipped(post, hops, st, clip=4.0, alpha=1.0)

    arms = {"pre": mixer, "post": post}
    thetas = {}
    for name, m in arms.items():
        stt = collect_stats(m, mixer_hops(m), va)
        thetas[name] = [s["theta"] for s in stt]

    # raw projections for the clipped variant
    for name, m in arms.items():
        row = {}
        for kernel in ("floor", "ceil_ttfs", "nearest"):
            for S in (4, 8, 16):
                m2, hd = project_model(m, mixer_hops(m), thetas[name], kernel, S)
                row[f"{kernel}_S{S}"] = evaluate(m2, te)
                for h in hd:
                    h.remove()
        results[f"raw_{name}"] = row
        print(f"raw {name}: {json.dumps(row)}", flush=True)

    # ramped QAT (pipeline-style) at S=4 and S=8
    results["ramped_qat"] = {}
    for S in (4, 8):
        for name, m in arms.items():
            reads = []
            for seed in (0, 1, 2):
                acc = ramped_qat(m, thetas[name], S, seed=seed)
                reads.append(acc)
                print(f"ramped S={S} arm={name} seed={seed}: exit={acc:.4f}", flush=True)
            results["ramped_qat"][f"S{S}_{name}"] = reads

    # WQ-then-AQ combined on the ramped-QAT-exit? (out of scope; raw wq already measured)
    with open(os.path.join(OUT, "clipped_results.json"), "w") as f:
        json.dump(results, f, indent=1)
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
