"""Decision-fidelity vs DEPTH at fixed T (fold, bit-exact per-layer), residual vs plain.

The central compounding axis: how deep LIF decision-fidelity degrades with depth at
a fixed rate resolution T, and how the residual skip changes that curve.
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import torch

import probe_lif_resnet_decision_fidelity as P
from probe_sweep_fold import cifar_tensors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="/home/yigit/repos/research_stuff/mimarsinan/probe_ckpt")
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--data", default="/home/yigit/data")
    ap.add_argument("--n-eval", type=int, default=2000)
    ap.add_argument("--Ts", default="16,32,64")
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    Ts = [int(t) for t in args.Ts.split(",")]
    cal_x, eval_x, eval_y = cifar_tensors(args.data, 256, args.n_eval, device)

    ckpts = sorted(glob.glob(os.path.join(args.ckpt_dir, "resnet_*_d*_w*.pt")))
    rows = []
    for cp in ckpts:
        ck = torch.load(cp, map_location="cpu", weights_only=False)
        model = P.ResNet(ck["depth"], width=ck["width"], residual=ck["residual"])
        model.load_state_dict(ck["state_dict"])
        model = model.eval().to(device)
        with torch.no_grad():
            ann = (model(eval_x).argmax(1) == eval_y).float().mean().item()
        tag = "res" if ck["residual"] else "plain"
        for T in Ts:
            flow, hybrid, teacher, nseg, _ = P.deploy_lif(model, cal_x, T, device=device, fold=True)
            df = P.decision_fidelity(flow, teacher, eval_x, T, device)
            nf = P.nf_logits(flow, eval_x, T, device)
            dep = (nf.argmax(1) == eval_y.cpu()).float().mean().item()
            print(f"{tag:5} depth={ck['depth']:>2} convs={2*ck['depth']+1:>2} "
                  f"ANN={ann:.3f} T={T:>3} nseg={nseg:>2} dec_fid={df:.4f} "
                  f"dep_top1={dep:.4f}", flush=True)


if __name__ == "__main__":
    main()
