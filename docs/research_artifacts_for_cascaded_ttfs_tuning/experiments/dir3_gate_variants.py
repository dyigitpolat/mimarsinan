"""DIRECTION 3: DEPLOYABLE LOCAL monotonization gates on the real multi-layer net.

membrane_end/T == z EXACTLY (the ramp integral is value-weighted-correct); the only
cascade defect is PREMATURE firing on a non-monotone partial membrane (mixed-sign
cancellation). These variants add a purely LOCAL, no-extra-spike, no-extra-fan-in
firing GATE that suppresses a premature crossing while the running sum is still
changing — the cheapest possible monotonization. Cold genuine vs staircase, d in {6,9}.

Gates (replace the cascade fire rule of TTFSActivation.forward):
  baseline    : shipped greedy membrane>=theta.
  rising_gate : fire only if ramp_current>=0 at this cycle (membrane still rising) AND
                membrane>=theta — suppress a crossing that a later negative will undo.
  arrived_frac: fire only after a fraction `f` of this neuron's positive input mass has
                arrived (proxy: cycles>=f*T) — a per-neuron settle delay.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(__file__)
_ART = "/home/yigit/repos/research_stuff/mimarsinan/docs/research_artifacts_for_cascaded_ttfs_tuning/experiments"
for _p in (_HERE, _ART, os.path.join(_ART, ".."), "/home/yigit/repos/research_stuff/mimarsinan/tests"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

from recipe_harness import build, genuine_acc  # noqa: E402
from lif_vs_ttfs import ttfs_staircase_acc  # noqa: E402
import mimarsinan.models.nn.activations.ttfs_spiking as ts  # noqa: E402
from mimarsinan.models.nn.activations.ttfs_spiking import (  # noqa: E402
    TTFSActivation, _channel_broadcast_view, _heaviside_surrogate,
)

_ORIG = TTFSActivation.forward


def _pre(self, x):
    scale_v, in_scale_v = self._scale_values(x)
    bias = self._bias
    if bias is not None:
        bias = bias.to(x.device, x.dtype)
        bias_b = _channel_broadcast_view(bias, x)
        weighted = (x - bias_b) * (in_scale_v / scale_v)
        bias_norm = bias_b / scale_v
    else:
        weighted = x * (in_scale_v / scale_v)
        bias_norm = 0.0
    return weighted, bias_norm


def make_gate(kind, **kw):
    def forward(self, x):
        if not self._cycle_accurate_mode or self.encoding:
            return _ORIG(self, x)
        weighted, bias_norm = _pre(self, x)
        if self._membrane is None:
            self._ramp_current = torch.zeros_like(x)
            self._membrane = torch.zeros_like(x)
            self._has_fired = torch.zeros_like(x)
            object.__setattr__(self, "_cyc", 0)
        prev_ramp = self._ramp_current
        self._ramp_current = self._ramp_current + weighted
        self._membrane = self._membrane + self._ramp_current + bias_norm
        pre = self._membrane - 1.0
        spike_raw = _heaviside_surrogate(pre, self.thresholding_mode, alpha=self.surrogate_alpha)
        if kind == "rising":
            gate = (self._ramp_current >= 0).to(x.dtype)
        elif kind == "arrived":
            gate = torch.full_like(x, 1.0 if self._cyc >= kw["f"] * self.T else 0.0)
        else:
            gate = torch.ones_like(x)
        spike = spike_raw * (1.0 - self._has_fired) * gate
        self._has_fired = (self._has_fired + spike.detach()).clamp(max=1.0)
        object.__setattr__(self, "_cyc", self._cyc + 1)
        return spike
    return forward


def _patch(kind, **kw):
    if kind == "baseline":
        TTFSActivation.forward = _ORIG
    else:
        TTFSActivation.forward = make_gate(kind, **kw)
    if not hasattr(ts, "_reset_orig"):
        ts._reset_orig = TTFSActivation.reset_state
    def reset(self):
        ts._reset_orig(self)
        object.__setattr__(self, "_cyc", 0)
    TTFSActivation.reset_state = reset


def run():
    specs = [("baseline", {}), ("rising", {}), ("arrived", {"f": 0.5}), ("arrived", {"f": 0.75})]
    for depth in (6, 9):
        flow, xtr, ytr, xte, yte, cont, teacher, base = build(depth, 8, seed=0)
        stair = ttfs_staircase_acc(flow, xte, yte, 8)
        print(f"\n=== d={depth} S=8 cont={cont:.3f} stair={stair:.3f} ===")
        for kind, kw in specs:
            _patch(kind, **kw)
            acc = genuine_acc(flow, xte, yte, 8)
            tag = kind + (f":{kw}" if kw else "")
            print(f"  {tag:>16}: cold_genuine={acc:.3f}", flush=True)
        _patch("baseline")


if __name__ == "__main__":
    run()
