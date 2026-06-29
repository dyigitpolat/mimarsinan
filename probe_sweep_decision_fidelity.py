"""Isolation sweep: deep LIF residual decision-fidelity on the STRONG CIFAR10 probe.

Loads a trained ResNet checkpoint (res or plain), calibrates on REAL CIFAR10
train images, and reports DECISION-FIDELITY = argmax-agreement(LIF-NF, float ANN)
on REAL CIFAR10 test images, across the isolation axes + candidate fixes.

The LIF chip-aligned NF is the deployment proxy (GPU-fast); a one-shot
bit-exact NF==HCM check on the packed sim confirms it is not a sim bug.
"""
from __future__ import annotations

import argparse
import os

import torch

import probe_lif_resnet_decision_fidelity as P


def cifar_tensors(data_root, n_cal, n_eval, device):
    import torchvision
    import torchvision.transforms as TT

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    tf = TT.Compose([TT.ToTensor(), TT.Normalize(mean, std)])
    tr = torchvision.datasets.CIFAR10(data_root, train=True, download=True, transform=tf)
    te = torchvision.datasets.CIFAR10(data_root, train=False, download=True, transform=tf)

    def stack(ds, n):
        xs = torch.stack([ds[i][0] for i in range(n)])
        ys = torch.tensor([ds[i][1] for i in range(n)])
        return xs, ys

    cal_x, _ = stack(tr, n_cal)
    eval_x, eval_y = stack(te, n_eval)
    return cal_x.to(device), eval_x.to(device), eval_y.to(device)


def load_model(ckpt_path, depth_override=None):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m = P.ResNet(ck["depth"], width=ck["width"], residual=ck["residual"])
    m.load_state_dict(ck["state_dict"])
    return m.eval(), ck


def measure(model, cal_x, eval_x, eval_y, T, device, *, label, ann_top1,
            verify=False, **deploy_kw):
    flow, hybrid, teacher, nseg, scales = P.deploy_lif(
        model, cal_x, T, device=device, **deploy_kw)
    df = P.decision_fidelity(flow, teacher, eval_x, T, device)
    # deployed top1 and retention, for reference
    nf = P.nf_logits(flow, eval_x, T, device)
    dep_top1 = (nf.argmax(1) == eval_y.cpu()).float().mean().item()
    ret = dep_top1 / ann_top1 if ann_top1 > 0 else 0.0
    bit = ""
    if verify:
        d = P.verify_nf_equals_hcm(flow, hybrid, eval_x[:4], T, n=4)
        bit = f" NF==HCM|Δ|={d:.2e}"
    sc = scales[0] if scales else 0.0
    print(f"  {label:38} T={T:>3} nseg={nseg:>2} scale={sc:>5.2f} "
          f"dec_fid={df:.4f} dep_top1={dep_top1:.4f} ret={100*ret:5.1f}%{bit}", flush=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--data", default="/home/yigit/data")
    ap.add_argument("--n-cal", type=int, default=256)
    ap.add_argument("--n-eval", type=int, default=2000)
    ap.add_argument("--axes", default="all")
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model, ck = load_model(args.ckpt)
    ann_top1 = ck["ann_top1"]
    cal_x, eval_x, eval_y = cifar_tensors(args.data, args.n_cal, args.n_eval, device)
    model = model.to(device)
    # recompute ANN top1 on the eval slice (the deploy teacher reference)
    with torch.no_grad():
        ann_eval = (model(eval_x).argmax(1) == eval_y).float().mean().item()
    print(f"CKPT {os.path.basename(args.ckpt)} depth={ck['depth']} width={ck['width']} "
          f"residual={ck['residual']} ANN(train)={ann_top1:.4f} ANN(eval-slice)={ann_eval:.4f} "
          f"chance=0.1 n_eval={args.n_eval}", flush=True)

    # baseline (NF==HCM bit-exactness is confirmed separately on a fast model;
    # the production residual-LIF fidelity test also locks atol=0).
    print("[baseline]", flush=True)
    measure(model, cal_x, eval_x, eval_y, 16, device,
            label="baseline global-scale T16", ann_top1=ann_eval,
            scale_mode="global")

    print("[T sweep, global-scale, host residual add]", flush=True)
    for T in (8, 16, 32, 64, 128):
        measure(model, cal_x, eval_x, eval_y, T, device,
                label=f"global-scale", ann_top1=ann_eval, scale_mode="global")

    print("[scale-mode sweep @ T=32]", flush=True)
    for sm in ("one", "global", "perlayer"):
        measure(model, cal_x, eval_x, eval_y, 32, device,
                label=f"scale_mode={sm}", ann_top1=ann_eval, scale_mode=sm)

    print("[FIX: on-chip residual merge (fewer boundaries) @ T=32]", flush=True)
    for ocm in (False, True):
        measure(model, cal_x, eval_x, eval_y, 32, device,
                label=f"onchip_merge={ocm}", ann_top1=ann_eval,
                scale_mode="global", onchip_residual_merge=ocm)

    print("[FIX: DFQ bias correction @ T=32]", flush=True)
    for dfq in (False, True):
        measure(model, cal_x, eval_x, eval_y, 32, device,
                label=f"dfq={dfq}", ann_top1=ann_eval, scale_mode="global", dfq=dfq)

    print("[FIX: higher quantile (less clip) perlayer @ T=32] (NOT bit-exact)", flush=True)
    for q in (0.99, 0.999, 1.0):
        # global scale derived from the higher quantile = less saturation clip
        measure(model, cal_x, eval_x, eval_y, 32, device,
                label=f"global q={q}", ann_top1=ann_eval, scale_mode="global", quantile=q)


if __name__ == "__main__":
    main()
