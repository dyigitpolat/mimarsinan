"""DIRECTION 3: A/B cascade-node decode variants on the REAL converted net.

Monkeypatches the TTFSActivation cascade forward to test, on the real harness
(d in {6,9}, mixed-sign trained nets), whether a value-domain monotonization of
the running partial sum recovers COLD genuine accuracy toward the staircase.

Variants (each replaces the cascade branch of TTFSActivation.forward):
  baseline   : the shipped membrane-integral greedy fire (control).
  rampcross  : fire when ramp_current (the PARTIAL COMPLETE weighted sum, no double
               integration) first crosses theta. Once all spikes arrive
               ramp_current == z; with monotone arrivals the first crossing
               reflects the complete sum.
  refractory : baseline membrane integral but suppress firing for the first
               `delay` cycles of the node's window (let cancelling inputs arrive).

We measure cold genuine vs staircase. No fine-tuning.
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

_ORIG_FORWARD = TTFSActivation.forward


def _cascade_pre(self, x):
    """Shared: compute normalized weighted increment and bias_norm for one cycle."""
    scale_v, in_scale_v = self._scale_values(x)
    bias = self._bias
    if bias is not None:
        bias = bias.to(x.device, x.dtype)
        bias_b = _channel_broadcast_view(bias, x)
        weighted_raw = x - bias_b
        bias_norm = bias_b / scale_v
    else:
        weighted_raw = x
        bias_norm = 0.0
    weighted = weighted_raw * (in_scale_v / scale_v)
    return weighted, bias_norm


def make_rampcross_forward():
    def forward(self, x):
        if not self._cycle_accurate_mode or self.encoding:
            return _ORIG_FORWARD(self, x)
        weighted, bias_norm = _cascade_pre(self, x)
        if self._ramp_current is None:
            self._ramp_current = torch.zeros_like(x)
            self._has_fired = torch.zeros_like(x)
        # ramp_current = running PARTIAL complete-sum (no second integration).
        self._ramp_current = self._ramp_current + weighted
        pre = self._ramp_current + bias_norm - 1.0
        spike_raw = _heaviside_surrogate(pre, self.thresholding_mode, alpha=self.surrogate_alpha)
        spike = spike_raw * (1.0 - self._has_fired)
        self._has_fired = (self._has_fired + spike.detach()).clamp(max=1.0)
        return spike
    return forward


def make_refractory_forward(delay_frac):
    def forward(self, x):
        if not self._cycle_accurate_mode or self.encoding:
            return _ORIG_FORWARD(self, x)
        weighted, bias_norm = _cascade_pre(self, x)
        if self._membrane is None:
            self._ramp_current = torch.zeros_like(x)
            self._membrane = torch.zeros_like(x)
            self._has_fired = torch.zeros_like(x)
            self._cyc = 0
        self._ramp_current = self._ramp_current + weighted
        self._membrane = self._membrane + self._ramp_current + bias_norm
        pre = self._membrane - 1.0
        spike_raw = _heaviside_surrogate(pre, self.thresholding_mode, alpha=self.surrogate_alpha)
        gate = 1.0 if self._cyc >= int(delay_frac * self.T) else 0.0
        spike = spike_raw * (1.0 - self._has_fired) * gate
        self._has_fired = (self._has_fired + spike.detach()).clamp(max=1.0)
        self._cyc += 1
        return spike
    return forward


def _orig_reset(self):
    ts.TTFSActivation_reset_orig(self)
    self._cyc = 0


def patch(variant):
    if variant == "baseline":
        TTFSActivation.forward = _ORIG_FORWARD
    elif variant == "rampcross":
        TTFSActivation.forward = make_rampcross_forward()
    elif variant.startswith("refractory"):
        frac = float(variant.split(":")[1])
        TTFSActivation.forward = make_refractory_forward(frac)
        if not hasattr(ts, "TTFSActivation_reset_orig"):
            ts.TTFSActivation_reset_orig = TTFSActivation.reset_state
        def reset(self):
            ts.TTFSActivation_reset_orig(self)
            object.__setattr__(self, "_cyc", 0)
        TTFSActivation.reset_state = reset
    else:
        raise ValueError(variant)


def run():
    variants = ["baseline", "rampcross", "refractory:0.5", "refractory:0.75"]
    for depth in (6, 9):
        flow, xtr, ytr, xte, yte, cont, teacher, base = build(depth, 8, seed=0)
        stair = ttfs_staircase_acc(flow, xte, yte, 8)
        print(f"\n=== d={depth} S=8 cont={cont:.3f} stair={stair:.3f} ===")
        for vname in variants:
            patch(vname)
            acc = genuine_acc(flow, xte, yte, 8)
            print(f"  {vname:>16}: cold_genuine={acc:.3f}", flush=True)
        patch("baseline")


if __name__ == "__main__":
    run()
