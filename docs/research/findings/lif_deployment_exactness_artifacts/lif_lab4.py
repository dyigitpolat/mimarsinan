"""Round 4: per-hop boundary re-encode (segmented, mixer-style) and
stochastic-vs-uniform encode contrast."""

from __future__ import annotations

import json
import os
import time

import torch

from lif_lab import (
    DEVICE, DTYPE, OUT_DIR,
    Converted, accuracy, run_hop, tensor_split, train_model, to_uniform_spikes,
)


def deploy_variant(conv, x, T, *, reencode_each_hop=False, encode="Uniform",
                   memb_readout=False, seed=0):
    r0 = conv.encoder_rates(x)
    B = r0.shape[0]

    def enc(rates):
        if encode == "Uniform":
            return torch.stack([to_uniform_spikes(rates, t, T)
                                for t in range(T)]).to(DTYPE)
        g = torch.Generator(device="cpu").manual_seed(seed)
        u = torch.rand(T, *rates.shape, generator=g).to(rates.device)
        return (u < rates.unsqueeze(0)).to(DTYPE)

    train = enc(r0)
    counts = memb = theta = None
    for k in range(conv.n_hops):
        theta = torch.ones(conv.W[k].shape[0], dtype=DTYPE, device=DEVICE)
        b = conv.b[k] + theta / (2.0 * T)
        train_out, counts, memb, _ = run_hop(train, conv.W[k], b, theta, "<")
        if reencode_each_hop and k < conv.n_hops - 1:
            # segment boundary: counts -> rate -> clamp -> Uniform re-encode
            # (segment_boundary.py:173-178 + uniform_rate_encode)
            train = enc(torch.clamp(counts / T, 0.0, 1.0))
        else:
            train = train_out
    if memb_readout:
        logits = conv.thetas[-1] * (counts + memb - theta / 2.0) / T
    else:
        logits = conv.thetas[-1] * counts / T
    return logits


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
            tv = {
                "direct": accuracy(
                    deploy_variant(conv, test_x, T), test_y),
                "reencode_each_hop": accuracy(
                    deploy_variant(conv, test_x, T, reencode_each_hop=True),
                    test_y),
                "reencode_membread": accuracy(
                    deploy_variant(conv, test_x, T, reencode_each_hop=True,
                                   memb_readout=True), test_y),
                "stochastic": accuracy(
                    deploy_variant(conv, test_x, T, encode="Stochastic",
                                   reencode_each_hop=True), test_y),
            }
            vres[f"T{T}"] = tv
            print(f"{name} T={T}: " + " ".join(
                f"{k}={v:.4f}" for k, v in tv.items()))
        out[name] = vres
    out["wall_seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(OUT_DIR, "results4.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("done", out["wall_seconds"], "s")


if __name__ == "__main__":
    main()
