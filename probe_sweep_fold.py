"""CLEAN isolation sweep: FOLD-calibrated (bit-exact, per-layer) LIF decision-fidelity.

Uses the fold path: per-layer activation scale folded into weights -> LIF scale
1.0 -> NF==HCM bit-exact AND every layer keeps full [0,1] rate resolution. This
removes the global-scale quantization artifact, so the residual decision-fidelity
loss measured here is the genuine deployment loss (boundary re-encode + inherent
per-layer T-quantization), not a calibration artifact.
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
    cal_x = torch.stack([tr[i][0] for i in range(n_cal)])
    eval_x = torch.stack([te[i][0] for i in range(n_eval)])
    eval_y = torch.tensor([te[i][1] for i in range(n_eval)])
    return cal_x.to(device), eval_x.to(device), eval_y.to(device)


def measure(model, cal_x, eval_x, eval_y, T, device, *, label, ann_eval, **kw):
    flow, hybrid, teacher, nseg, scales = P.deploy_lif(model, cal_x, T, device=device, **kw)
    df = P.decision_fidelity(flow, teacher, eval_x, T, device)
    nf = P.nf_logits(flow, eval_x, T, device)
    dep = (nf.argmax(1) == eval_y.cpu()).float().mean().item()
    print(f"  {label:34} T={T:>3} nseg={nseg:>2} dec_fid={df:.4f} dep_top1={dep:.4f} "
          f"ret={100*dep/ann_eval:5.1f}%", flush=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--data", default="/home/yigit/data")
    ap.add_argument("--n-cal", type=int, default=256)
    ap.add_argument("--n-eval", type=int, default=2000)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = P.ResNet(ck["depth"], width=ck["width"], residual=ck["residual"])
    model.load_state_dict(ck["state_dict"])
    model = model.eval().to(device)
    cal_x, eval_x, eval_y = cifar_tensors(args.data, args.n_cal, args.n_eval, device)
    with torch.no_grad():
        ann_eval = (model(eval_x).argmax(1) == eval_y).float().mean().item()
    print(f"CKPT {os.path.basename(args.ckpt)} depth={ck['depth']} width={ck['width']} "
          f"residual={ck['residual']} ANN(eval)={ann_eval:.4f} chance=0.1 n_eval={args.n_eval}",
          flush=True)

    print("[FOLD (bit-exact per-layer) T sweep -- host residual add]", flush=True)
    for T in (8, 16, 32, 64, 128, 256):
        measure(model, cal_x, eval_x, eval_y, T, device,
                label="fold host-add", ann_eval=ann_eval, fold=True)

    print("[FIX: FOLD + on-chip residual merge (fewer boundaries)]", flush=True)
    for T in (16, 64, 256):
        measure(model, cal_x, eval_x, eval_y, T, device,
                label="fold onchip-merge", ann_eval=ann_eval, fold=True,
                onchip_residual_merge=True)

    print("[reference: fold higher quantile (1.0 = no clip) @ T=64]", flush=True)
    for q in (0.99, 1.0):
        measure(model, cal_x, eval_x, eval_y, 64, device,
                label=f"fold q={q}", ann_eval=ann_eval, fold=True, quantile=q)


if __name__ == "__main__":
    main()
