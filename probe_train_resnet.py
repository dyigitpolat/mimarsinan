"""Train the strong-probe ResNet(s) on CIFAR10 and save checkpoints.

Trains BOTH a residual net and its plain (skip-ablated) twin at a given depth,
so the residual-vs-plain ablation reuses identically-trained backbones.
"""
from __future__ import annotations

import argparse
import os

import torch

import probe_lif_resnet_decision_fidelity as P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--data", default="/home/yigit/data")
    ap.add_argument("--out", default="/home/yigit/repos/research_stuff/mimarsinan/probe_ckpt")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"device={device} depth={args.depth} width={args.width} epochs={args.epochs}", flush=True)

    trl, tel = P.cifar_loaders(args.data, batch=128)

    for residual in (True, False):
        tag = "res" if residual else "plain"
        torch.manual_seed(0)
        model = P.ResNet(args.depth, width=args.width, residual=residual)
        print(f"=== training {tag} depth={args.depth} ===", flush=True)
        model = P.train(model, trl, tel, device, epochs=args.epochs)
        acc = P.evaluate(model, tel, device)
        path = os.path.join(args.out, f"resnet_{tag}_d{args.depth}_w{args.width}.pt")
        torch.save({"state_dict": model.state_dict(), "depth": args.depth,
                    "width": args.width, "residual": residual, "ann_top1": acc}, path)
        print(f"SAVED {path} ann_top1={acc:.4f}", flush=True)


if __name__ == "__main__":
    main()
