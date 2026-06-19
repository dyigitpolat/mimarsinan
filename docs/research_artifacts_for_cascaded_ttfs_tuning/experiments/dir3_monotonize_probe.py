"""DIRECTION 3 probe: is the cascade collapse a CANCELLATION (mixed-sign partial-sum)
defect, and does a per-neuron NEGATIVE-WEIGHT monotonization fix it BY CONSTRUCTION?

Mechanism (single cascade neuron, one segment, window length T=S):
  - Input i arrives as a single spike at local cycle tau_i = round(T*(1 - v_i)),
    v_i in [0,1] the normalized presynaptic value (v_i=1 -> tau=0 earliest).
  - Ramp integrate: ramp_current[t] = sum_i W_i*[tau_i<=t]; membrane=cumsum(ramp).
    => spike i's running membrane contribution at cycle t is W_i*(t-tau_i+1).
  - The neuron fires ONCE at the first t with membrane[t] >= theta.

The staircase (lossless) uses the COMPLETE weighted sum z = sum_i W_i*v_i (+b),
out = clamp(relu(z)/theta, 0, 1).

CANCELLATION: with mixed-sign W_i, an early (high-v) positive input can drive
membrane over theta BEFORE a late (low-v) negative input arrives to cancel it
=> premature/wrong fire (decoded too-high) => death cascade with depth.

MONOTONIZATION (Stanojevic-style reference): rewrite each negative contribution
   -W_i^- * v_i  =  -W_i^-  +  W_i^- * (tau_i / T)        (since v_i = 1 - tau_i/T)
The constant -W_i^- is delivered at t=0 (folded into bias). The remaining
+W_i^- * (tau_i/T) GROWS with arrival time tau_i, is non-negative -> the running
partial sum is a MONOTONE NON-DECREASING lower bound of the complete sum, so the
first crossing reflects the COMPLETE sum. Exact in the value domain.
"""

from __future__ import annotations

import torch


def encode_tau(v, T):
    return torch.round(T * (1.0 - v.clamp(0, 1))).long()


def complete_decode(W, v, b, theta, T):
    z = (W * v).sum(-1) + b if W.dim() == 2 and v.dim() == 2 and W.shape[0] == 1 \
        else v @ W.t() + b
    return (torch.relu(z) / theta).clamp(0, 1)


def cascade_fire_value(W, v, b, theta, T, *, bias_const=0.0):
    """Genuine greedy single-spike cascade decode. W:(out,in) v:(B,in) b:(out,).
    bias_const: per-out constant pre-loaded each cycle (the monotonization offset)."""
    B = v.shape[0]
    out_n = W.shape[0]
    tau = encode_tau(v, T)
    fired_value = torch.zeros(B, out_n, dtype=v.dtype)
    has_fired = torch.zeros(B, out_n, dtype=torch.bool)
    ramp = torch.zeros(B, out_n, dtype=v.dtype)
    membrane = torch.zeros(B, out_n, dtype=v.dtype)
    bias_ramp = (b + bias_const) / theta
    for t in range(T):
        arrived = (tau == t).to(v.dtype)
        ramp = ramp + (arrived @ (W.t() / theta))
        membrane = membrane + ramp + bias_ramp
        fire = (~has_fired) & (membrane >= 1.0)
        fired_value = torch.where(fire, torch.full_like(fired_value, (T - t) / T), fired_value)
        has_fired = has_fired | fire
    return fired_value


def monotonized_cascade(W, v, b, theta, T):
    """Cascade decode after negative-weight monotonization (signed-channel + offset)."""
    Wp = W.clamp(min=0.0)
    Wn = (-W).clamp(min=0.0)
    bias_const = -Wn.sum(-1)          # constant -sum W^- pre-loaded at t=0
    vc = 1.0 - v                      # complement value carries tau/T
    Waug = torch.cat([Wp, Wn], dim=1)
    vaug = torch.cat([v, vc], dim=1)
    return cascade_fire_value(Waug, vaug, b, theta, T, bias_const=bias_const)


def run(seed=0, B=4000, n_in=64, n_out=64, T=8):
    torch.manual_seed(seed)
    torch.set_default_dtype(torch.float64)
    W = torch.randn(n_out, n_in) / (n_in ** 0.5)
    b = torch.randn(n_out) * 0.1
    v = torch.rand(B, n_in)
    theta = 1.0
    truth = complete_decode(W, v, b, theta, T)
    cold = cascade_fire_value(W, v, b, theta, T)
    mono = monotonized_cascade(W, v, b, theta, T)
    grid = torch.round(truth * T) / T

    def err(a):
        return float((a - truth).abs().mean())
    print(f"T={T:>3} | cold-truth={err(cold):.4f}  mono-truth={err(mono):.4f}  "
          f"grid-truth(1/T floor)={err(grid):.4f}")


if __name__ == "__main__":
    print("=== single-layer monotonization (value domain) ===")
    for T in (8, 16, 32):
        run(T=T)
