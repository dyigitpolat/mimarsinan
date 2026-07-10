"""Round 2: comparator/reset/WQ/phase/V0 experiments on the same lab."""

from __future__ import annotations

import json
import os
import time

import torch

from lif_lab import (
    DEVICE, DTYPE, SEED, OUT_DIR, THRESHOLDING_MODE,
    Converted, Chain, accuracy, calibrate_affine, count_fn, deploy,
    run_hop, tensor_split, train_model, to_uniform_spikes,
)


def deploy_custom(conv, x, T, *, half_step=True, mode="<", firing="Default",
                  wq_bits=None, phase_stagger=False, v0_half=False,
                  memb_readout=False):
    """Deployment variant: Novena reset, per-hop 5-bit wire WQ, per-channel
    phase-staggered encode, or V0=theta/2 initial membrane instead of the
    per-cycle half-step ramp."""
    r0 = conv.encoder_rates(x)
    B = r0.shape[0]
    train = torch.stack([to_uniform_spikes(r0, t, T) for t in range(T)]).to(DTYPE)
    if phase_stagger:
        g = torch.Generator(device="cpu").manual_seed(123)
        offs = torch.randint(0, T, (train.shape[2],), generator=g)
        for c in range(train.shape[2]):
            train[:, :, c] = torch.roll(train[:, :, c], int(offs[c]), dims=0)
    counts = memb = theta_last = None
    for k in range(conv.n_hops):
        W = conv.W[k].clone()
        b = conv.b[k].clone()
        if wq_bits is not None:
            qmax = 2 ** (wq_bits - 1) - 1
            s = qmax / float(torch.maximum(W.abs().max(), b.abs().max()))
            W = torch.round(W * s).clamp(-qmax, qmax) / s
            b = torch.round(b * s).clamp(-qmax, qmax) / s
        n_out = W.shape[0]
        theta = torch.ones(n_out, dtype=DTYPE, device=DEVICE)
        v0 = torch.zeros(n_out, dtype=DTYPE, device=DEVICE)
        if half_step:
            if v0_half:
                v0 = theta / 2.0
            else:
                b = b + theta / (2.0 * T)
        # inline hop with configurable reset + V0
        memb_ = v0.expand(B, n_out).clone()
        counts_ = torch.zeros(B, n_out, dtype=DTYPE, device=DEVICE)
        out_train = torch.zeros(T, B, n_out, dtype=DTYPE, device=DEVICE)
        for t in range(T):
            contrib = torch.matmul(W, train[t].T).T + b
            memb_ = memb_ + contrib
            if mode == "<":
                fired = (memb_ > theta).to(DTYPE)
            else:
                fired = (memb_ >= theta).to(DTYPE)
            if firing == "Novena":
                memb_ = memb_ * (1.0 - fired)
            else:
                memb_ = memb_ - fired * theta
            counts_ = counts_ + fired
            out_train[t] = fired
        train = out_train
        counts, memb, theta_last = counts_, memb_, theta
    if memb_readout:
        Qhat = counts * theta_last + memb
        if half_step and not v0_half:
            Qhat = Qhat - theta_last / 2.0
        if half_step and v0_half:
            Qhat = Qhat - theta_last / 2.0
        logits = conv.thetas[-1] * Qhat / T
    else:
        logits = conv.thetas[-1] * counts / T
    return logits


def main():
    t0 = time.time()
    (test_x, test_y), (calib_x, _) = tensor_split()
    out = {}
    for name, widths in {
        "mlp3": [784, 128, 128, 10],
        "chain9": [784] + [128] * 8 + [10],
    }.items():
        model = train_model(widths)
        conv = Converted(model, calib_x)
        fl = conv.float_logits(test_x)
        vres = {"float_acc": accuracy(fl, test_y)}
        for T in (4, 8, 16, 32):
            tv = {}
            base = deploy_custom(conv, test_x, T, mode="<")
            tv["lt_strict"] = accuracy(base, test_y)
            tv["le_inclusive"] = accuracy(
                deploy_custom(conv, test_x, T, mode="<="), test_y)
            tv["novena"] = accuracy(
                deploy_custom(conv, test_x, T, firing="Novena"), test_y)
            tv["novena_membread"] = accuracy(
                deploy_custom(conv, test_x, T, firing="Novena",
                              memb_readout=True), test_y)
            tv["wq5_lt"] = accuracy(
                deploy_custom(conv, test_x, T, wq_bits=5, mode="<"), test_y)
            tv["wq5_le"] = accuracy(
                deploy_custom(conv, test_x, T, wq_bits=5, mode="<="), test_y)
            tv["phase_stagger"] = accuracy(
                deploy_custom(conv, test_x, T, phase_stagger=True), test_y)
            tv["v0_half"] = accuracy(
                deploy_custom(conv, test_x, T, v0_half=True), test_y)
            tv["v0_half_membread"] = accuracy(
                deploy_custom(conv, test_x, T, v0_half=True,
                              memb_readout=True), test_y)
            tv["ramp_membread"] = accuracy(
                deploy_custom(conv, test_x, T, memb_readout=True), test_y)
            vres[f"T{T}"] = tv
            print(f"{name} T={T}: " + " ".join(
                f"{k}={v:.4f}" for k, v in tv.items()))
        out[name] = vres
    out["wall_seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(OUT_DIR, "results2.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("done", out["wall_seconds"], "s")


if __name__ == "__main__":
    main()
