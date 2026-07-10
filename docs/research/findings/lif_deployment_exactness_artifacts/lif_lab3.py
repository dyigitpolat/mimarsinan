"""Round 3: signed per-hop drift law, Novena affine repair, noise-gain fit."""

from __future__ import annotations

import json
import os
import time

import torch

from lif_lab import (
    DEVICE, DTYPE, OUT_DIR,
    Converted, accuracy, deploy, tensor_split, train_model, to_uniform_spikes,
)
from lif_lab2 import deploy_custom


def signed_drift(conv, x, T, half_step):
    """Per-hop signed mean drift E[r_dep - r_float] (wire rate units)."""
    with torch.no_grad():
        acts = conv.model.activations(x.to(DEVICE))
    targets = [torch.clamp(acts[k + 1].to(DTYPE) / conv.thetas[k + 1], 0.0, 1.0)
               for k in range(conv.n_hops)]
    _, rates, _ = deploy(conv, x, T, half_step=half_step)
    return [float((rates[k + 1] - targets[k]).mean()) for k in range(conv.n_hops)]


def novena_affine(conv, calib_x, test_x, test_y, T):
    """Sequential per-channel LS affine repair for the Novena zero-reset,
    folded into consumers; readout host affine + membrane read."""
    with torch.no_grad():
        acts = conv.model.activations(calib_x.to(DEVICE))
    targets = [torch.clamp(acts[k + 1].to(DTYPE) / conv.thetas[k + 1], 0.0, 1.0)
               for k in range(conv.n_hops)]

    def run(x, gains, biases, memb_read=False):
        r0 = conv.encoder_rates(x)
        train = torch.stack([to_uniform_spikes(r0, t, T)
                             for t in range(T)]).to(DTYPE)
        rates = []
        counts = memb = theta = None
        for k in range(conv.n_hops):
            W = conv.W[k].clone()
            b = conv.b[k].clone()
            if biases[k] is not None:
                b = b + conv.W[k] @ biases[k]
            if gains[k] is not None:
                W = W * gains[k].unsqueeze(0)
            n_out = W.shape[0]
            theta = torch.ones(n_out, dtype=DTYPE, device=DEVICE)
            b = b + theta / (2.0 * T)
            B = r0.shape[0]
            memb = torch.zeros(B, n_out, dtype=DTYPE, device=DEVICE)
            counts = torch.zeros(B, n_out, dtype=DTYPE, device=DEVICE)
            out_train = torch.zeros(T, B, n_out, dtype=DTYPE, device=DEVICE)
            for t in range(T):
                memb = memb + torch.matmul(W, train[t].T).T + b
                fired = (memb > theta).to(DTYPE)
                memb = memb * (1.0 - fired)  # Novena zero-reset
                counts = counts + fired
                out_train[t] = fired
            train = out_train
            rates.append(counts / T)
        logits = conv.thetas[-1] * (counts + (memb if memb_read else 0)) / T
        return logits, rates

    gains = [None] * conv.n_hops
    biases = [None] * conv.n_hops
    host = None
    for k in range(conv.n_hops):
        _, rates = run(calib_x, gains, biases)
        r_dep, r_tgt = rates[k], targets[k]
        var = r_dep.var(dim=0, unbiased=False)
        cov = ((r_dep - r_dep.mean(0)) * (r_tgt - r_tgt.mean(0))).mean(0)
        a = torch.where(var > 1e-10, cov / torch.clamp(var, min=1e-10),
                        torch.ones_like(var)).clamp(0.25, 4.0)
        c = r_tgt.mean(0) - a * r_dep.mean(0)
        if k < conv.n_hops - 1:
            gains[k + 1] = a
            biases[k + 1] = c
        else:
            host = (a, c)
    logits, _ = run(test_x, gains, biases, memb_read=True)
    if host is not None:
        logits = conv.thetas[-1] * (host[0] * (logits / conv.thetas[-1]) + host[1])
    return accuracy(logits, test_y)


def main():
    t0 = time.time()
    (test_x, test_y), (calib_x, _) = tensor_split()
    out = {}
    for name, widths in {"mlp3": [784, 128, 128, 10],
                         "chain9": [784] + [128] * 8 + [10]}.items():
        model = train_model(widths)
        conv = Converted(model, calib_x)
        vres = {}
        for T in (4, 8, 16, 32):
            vres[f"T{T}"] = {
                "drift_floor": signed_drift(conv, test_x, T, half_step=False),
                "drift_nearest": signed_drift(conv, test_x, T, half_step=True),
                "pred_floor_drift": -1.0 / (2 * T),
                "novena_plain": accuracy(
                    deploy_custom(conv, test_x, T, firing="Novena"), test_y),
                "novena_affine_membread": novena_affine(
                    conv, calib_x, test_x, test_y, T),
            }
            d = vres[f"T{T}"]
            print(f"{name} T={T}: floor drift/hop "
                  f"{[f'{x:.4f}' for x in d['drift_floor']]} (pred {d['pred_floor_drift']:.4f})")
            print(f"         nearest drift/hop {[f'{x:.4f}' for x in d['drift_nearest']]}")
            print(f"         novena {d['novena_plain']:.4f} -> affine+membread "
                  f"{d['novena_affine_membread']:.4f}")
        # noise gain: mean abs row-sum gain of wire weights (depth amplification)
        vres["wire_gain_rms_row"] = [
            float((conv.W[k] ** 2).sum(dim=1).mean().sqrt()) for k in range(conv.n_hops)]
        out[name] = vres
    out["wall_seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(OUT_DIR, "results3.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("done", out["wall_seconds"], "s")


if __name__ == "__main__":
    main()
