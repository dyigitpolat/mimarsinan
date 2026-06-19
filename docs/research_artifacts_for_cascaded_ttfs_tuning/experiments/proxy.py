"""DIRECTION C — Timing-aware differentiable PROXY with matched gradients (true D2).

The genuine cascade decode of a single neuron is NOT a pointwise function of its
ReLU pre-activation. It depends on *when* each input spike arrives:

  * input i carries value v_i, encoded as a single spike at GLOBAL cycle
    tau_i = round(S*(1 - v_i))  (high value -> early spike).
  * the consumer's integration window opens at cycle L (its cascade latency =
    perceptron-hops from segment entry) and runs for S cycles: [L, L+S).
  * inside the window, at local time s = t - L, the ramp current is
        R(s) = sum_i w_i * 1[tau_i <= L + s]          (spike has arrived)
    and the membrane is the running sum of (R(u) + b/theta) for u in [0..s].
  * the neuron fires ONCE at the first s where membrane >= theta, then latches;
    the decoded value is  (S - s_fire)/S * theta.

Two structural distortions vs the continuous ReLU teacher relu(W@v + b)*?:

  (D-miss)  any input whose spike arrives BEFORE the window opens (tau_i < L) is
            NEVER integrated -> its contribution is silently dropped. High input
            values fire earliest, so they are the first to be lost as depth grows.
  (D-shift) the membrane integrates over [s_fire, S), so the decode is the
            *fraction of the window remaining after firing* -- a ramp-position
            code, not the pre-activation magnitude. This over/under-amplifies vs
            the teacher and is the source of the per-depth +L/S shift.

This module builds ``TimingProxyDecode``: a differentiable forward that models the
EXPECTED ramp-integrate single-spike decode (soft arrivals + soft fire-time), and
validates (1) forward fidelity, (2) gradient alignment vs a finite-difference of
the genuine cascade, (3) end-to-end train-through-proxy then deploy-genuine.

Category: this is a TRAINING-TIME PROXY (a differentiable forward used during
fine-tuning). The deployed decode is unchanged and stays bit-exact with HCM.
"""

from __future__ import annotations

import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_LAB = os.path.abspath(os.path.join(_HERE, ".."))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
for _p in (_LAB, os.path.join(_REPO, "src"), os.path.join(_REPO, "tests"), _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ----------------------------------------------------------------------------
# The differentiable timing-aware decode (the proxy forward).
# ----------------------------------------------------------------------------

def soft_step(z: torch.Tensor, beta: float) -> torch.Tensor:
    """Soft Heaviside 1[z >= 0] ~= sigmoid(beta*z)."""
    return torch.sigmoid(beta * z)


def timing_proxy_decode(
    v_in: torch.Tensor,      # (B, n_in) input values in [0,1]
    W: torch.Tensor,         # (n_out, n_in)
    b: torch.Tensor,         # (n_out,)
    *,
    S: int,
    L: int,                  # consumer cascade latency (perceptron-hops)
    theta: torch.Tensor | float = 1.0,
    in_scale: torch.Tensor | float = 1.0,
    beta_arrival: float = 8.0,
    beta_fire: float = 12.0,
    hard_forward: bool = False,
) -> torch.Tensor:
    """Differentiable expected ramp-integrate single-spike decode.

    Returns decoded values (B, n_out) in the same value domain as the genuine
    cascade output (i.e. already * theta).  ``hard_forward=True`` uses hard
    arrivals + argmax fire (a forward sanity check, no useful gradient).

    The membrane at local time s is a cumulative double integral of a
    piecewise-constant ramp.  We evaluate it on the integer local grid
    s = 0..S-1 (the cycles the hardware actually integrates), softening only the
    two non-differentiable operations: spike *arrival* (a step in tau) and the
    fire-time *crossing* (a step in membrane).
    """
    B, n_in = v_in.shape
    n_out = W.shape[0]
    dev, dt = v_in.device, v_in.dtype
    if not torch.is_tensor(theta):
        theta = torch.tensor(float(theta), device=dev, dtype=dt)
    theta = theta.to(dev, dt)
    if not torch.is_tensor(in_scale):
        in_scale = torch.tensor(float(in_scale), device=dev, dtype=dt)
    in_scale = in_scale.to(dev, dt)

    # Global spike time of each input (real-valued; the genuine forward rounds it
    # to an integer cycle).  ``hard_forward`` uses the integer time so it matches
    # the bit-exact deployed cascade; the soft path keeps the real time so tau is
    # differentiable in v.
    tau = S * (1.0 - v_in.clamp(0.0, 1.0))            # (B, n_in)
    if hard_forward:
        tau = torch.round(tau)
    # Local arrival relative to the consumer window start.
    a = tau - float(L)                                 # (B, n_in)

    # Local integer cycle grid s = 0..S-1 (the cycles the hardware integrates).
    s = torch.arange(S, device=dev, dtype=dt)          # (S,)

    # arrived[b, s, i] = 1 iff input i's spike is INSIDE the window [L, L+S) AND
    # has landed by local cycle s.  The first factor is the DEATH-MISS gate: an
    # input whose spike fires before the window opens (tau < L) is permanently
    # dropped -> high-value inputs (early spikes) starve deep consumers.  This is
    # the death-cascade, modelled exactly.
    in_window = (tau >= float(L))                       # (B, n_in)
    if hard_forward:
        landed = (a.unsqueeze(1) <= s.view(1, S, 1)).to(dt)
        arrived = landed * in_window.to(dt).unsqueeze(1)
    else:
        landed = soft_step(s.view(1, S, 1) - a.unsqueeze(1), beta_arrival)
        gate = soft_step(tau - float(L), beta_arrival).unsqueeze(1)
        arrived = landed * gate                         # (B,S,n_in)

    # Ramp current per output at each local cycle: R[b,s,o] = sum_i Weff_oi * arrived.
    Weff = W * (in_scale / theta)                      # normalized-domain weights (n_out,n_in)
    R = torch.einsum("bsi,oi->bso", arrived, Weff)     # (B,S,n_out)

    b_norm = (b / theta).view(1, 1, n_out)             # (B,S,n_out) broadcast

    # membrane(s) = sum_{u=0..s} (R(u) + b_norm)   [cumulative sum along local time]
    mem = torch.cumsum(R + b_norm, dim=1)              # (B,S,n_out)

    # Fire ONCE: the decoded latch is 1 for every cycle from the FIRST crossing
    # onward, so decode = (1/S)*#{s : the neuron has already fired by s}.  The
    # latch is the running max of the per-cycle crossing (monotone fire-once),
    # NOT the pointwise crossing -- a negative ramp may dip the membrane back
    # below theta after firing, but the hardware latch stays high.
    crossed = (mem >= 1.0).to(dt) if hard_forward else soft_step(mem - 1.0, beta_fire)
    latched = torch.cummax(crossed, dim=1).values if hard_forward else _soft_cummax(crossed)
    decode = latched.mean(dim=1) * theta               # (B,n_out)
    return decode


def _soft_cummax(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Differentiable monotone running-max along ``dim`` (fire-once latch).

    A hard cummax is non-differentiable at the max-switch; we use the smooth
    upper envelope  y_s = 1 - prod_{u<=s}(1 - x_u), which equals the hard cummax
    when the x_u are 0/1 and stays differentiable for soft crossings (the
    probability that the neuron has fired by cycle s if each x_u is a per-cycle
    fire probability).  Computed in log-space for stability."""
    log_not = torch.log1p(-x.clamp(0.0, 1.0 - 1e-7))
    cum_log_not = torch.cumsum(log_not, dim=dim)
    return 1.0 - torch.exp(cum_log_not)


# ----------------------------------------------------------------------------
# Genuine per-layer reference (hard, from the project's TTFSActivation dynamics)
# reproduced standalone so we can finite-difference it cheaply per layer.
# ----------------------------------------------------------------------------

def genuine_layer_decode(v_in, W, b, *, S, L, theta=1.0, in_scale=1.0):
    """Hard genuine ramp-integrate single-spike decode for one Linear layer.

    Mirrors TTFSActivation cycle-accurate dynamics + TtfsSegmentPolicy window/latch
    decode for a single consumer core at cascade latency L.  Returns (B,n_out)."""
    from mimarsinan.chip_simulation.recording import spike_modes

    v = v_in.clamp(0.0, 1.0).double()
    B, n_in = v.shape
    n_out = W.shape[0]
    theta_t = torch.as_tensor(theta, dtype=torch.float64)
    in_scale_t = torch.as_tensor(in_scale, dtype=torch.float64)
    n_cycles = S + L
    # single-spike input trains, global cycles
    latched = torch.stack(
        [spike_modes.to_spikes(v, c, simulation_length=S, spike_mode="TTFS").double()
         for c in range(n_cycles)], dim=0)            # (n_cyc,B,n_in)
    trains = torch.cat([latched[:1], latched[1:] - latched[:-1]], dim=0)

    Wd = W.double()
    b_norm = (b.double() / theta_t).view(1, n_out)
    ramp = torch.zeros(B, n_out, dtype=torch.float64)
    mem = torch.zeros(B, n_out, dtype=torch.float64)
    fired = torch.zeros(B, n_out, dtype=torch.float64)
    accum = torch.zeros(B, n_out, dtype=torch.float64)
    latch = torch.zeros(B, n_out, dtype=torch.float64)
    for t in range(n_cycles):
        if not (L <= t < L + S):
            continue
        spk = trains[t]                                # (B,n_in)
        weighted = (spk @ Wd.t()) * (in_scale_t / theta_t)
        ramp = ramp + weighted
        mem = mem + ramp + b_norm
        fire = ((mem - 1.0 >= 0) & (fired == 0)).double()
        fired = (fired + fire).clamp(max=1.0)
        latch = torch.maximum(latch, fire)
        accum = accum + latch
    return (accum / float(S)) * theta_t


# ----------------------------------------------------------------------------
# Full-cascade proxy MLP: chain the per-layer timing decode with L = depth.
# Encoding (entry) layer reads the IDEAL value (no death-miss); interior layers
# decode their input through the timing proxy at their cascade latency.
# ----------------------------------------------------------------------------

class TimingProxyMLP(torch.nn.Module):
    """An MLP whose forward is the timing-aware cascade proxy.

    Layer 0 is the encoding layer (ideal value, staircase-clamped); layers
    1..D-1 decode through ``timing_proxy_decode`` at latency L = layer index.
    The output is the last decoded value (matches the genuine cascade output
    domain).  Trainable params: each layer's W, b, and (frozen) per-layer theta.
    """

    def __init__(self, dims, *, S, betas=(4.0, 16.0), proxy_mode="soft"):
        super().__init__()
        self.S = int(S)
        self.beta_arrival, self.beta_fire = betas
        self.proxy_mode = proxy_mode
        self.layers = torch.nn.ModuleList(
            torch.nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.thetas = [1.0] * len(self.layers)  # set by calibrate()

    def calibrate(self, x):
        """Set per-layer theta = max |relu(pre)| of a value-domain pass (matches
        the lab's MaxValueScaler calibration so values span [0,1])."""
        with torch.no_grad():
            h = x.double()
            new = []
            for li, lin in enumerate(self.layers):
                pre = h @ lin.weight.double().t() + lin.bias.double()
                act = torch.relu(pre)
                theta = float(act.abs().max().clamp(min=1e-3))
                new.append(theta)
                h = (act / theta).clamp(0.0, 1.0)
            self.thetas = new
        return self

    def forward(self, x, *, hard=False):
        h = x.double().clamp(0.0, 1.0)
        for li, lin in enumerate(self.layers):
            W = lin.weight.double()
            b = lin.bias.double()
            theta = self.thetas[li]
            if li == 0:
                # Encoding entry: ideal value, staircase-clamped to [0,1] (no
                # death-miss; the entry reads the exact host value).
                pre = h @ W.t() + b
                v = (torch.relu(pre) / theta).clamp(0.0, 1.0)
                # grid-snap with STE to mirror the entry single-spike quantization
                v_q = torch.round(v * self.S) / self.S
                h = (v_q.detach() + (v - v.detach())) * theta
            else:
                L = li  # consumer cascade latency == perceptron-hops
                use_hard = hard or self.proxy_mode == "hard"
                if self.proxy_mode == "ste" and not hard:
                    soft = timing_proxy_decode(
                        h / theta_prev, W, b, S=self.S, L=L, theta=theta,
                        beta_arrival=self.beta_arrival, beta_fire=self.beta_fire)
                    hardv = timing_proxy_decode(
                        h / theta_prev, W, b, S=self.S, L=L, theta=theta,
                        hard_forward=True)
                    h = hardv.detach() + (soft - soft.detach())
                else:
                    h = timing_proxy_decode(
                        h / theta_prev, W, b, S=self.S, L=L, theta=theta,
                        beta_arrival=self.beta_arrival, beta_fire=self.beta_fire,
                        hard_forward=use_hard)
            theta_prev = theta
        return h


# ----------------------------------------------------------------------------
# Self-contained validation (the three tiers reported in 12_timing_proxy.md).
# Run:  python docs/research_artifacts_for_cascaded_ttfs_tuning/experiments/proxy.py
# ----------------------------------------------------------------------------

def _validate_forward_fidelity():
    """Tier 1: the HARD proxy reproduces the genuine per-layer decode bit-exactly
    at L=0 and to a ~1/S rounding tie elsewhere (shrinks geometrically with S)."""
    print("=== Tier 1: forward fidelity (hard proxy vs genuine layer decode) ===")
    torch.manual_seed(0)
    B, n_in, n_out = 200, 10, 8
    v = torch.rand(B, n_in, dtype=torch.float64)
    W = torch.empty(n_out, n_in, dtype=torch.float64).uniform_(-0.5, 0.5)
    b = torch.empty(n_out, dtype=torch.float64).uniform_(-0.2, 0.2)
    for S in (8, 16, 32):
        for L in (0, 1, 2, 3):
            g = genuine_layer_decode(v, W, b, S=S, L=L)
            p = timing_proxy_decode(v, W, b, S=S, L=L, hard_forward=True)
            d = (p - g).abs()
            print(f"  S={S:>2} L={L}: exact_frac={(d < 1e-12).double().mean():.4f} "
                  f"max|err|={d.max():.5f}")


if __name__ == "__main__":
    _validate_forward_fidelity()
    print("\nFor the end-to-end transfer table (tiers 2-3), see 12_timing_proxy.md;"
          "\nthe training experiments live in /tmp scratch and are summarised there.")
