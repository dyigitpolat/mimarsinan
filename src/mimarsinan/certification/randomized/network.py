"""The integer network N of an X3a instance document, and what Proposition 2 claims of it per semantics.

``evaluate`` runs every layer's integer law on its sources' N values, with no timing and no floating point:

- a spiking layer (LIF counts, TTFS levels) takes the window charge ``Q = W c + T b`` of its sources' counts
  and emits ``clamp(floor(Q / theta), 0, T)`` (the inclusive comparator ``<=`` and the subtractive reset the
  code level fixes; LIF memo section 2, calculus section 16.3, TTFS closure section 3.1);
- a value layer emits the integer affine charge ``W x + b`` (the affine value core at theta 1); under the integer
  readout law (G2's E283, ``levels`` the activation grid's codes at step 1) every core reads its sources' codes
  clamped onto the grid (the entry snap; a core's own codes lie on it) and emits ``clamp(W x + b, -levels, levels)``;
- host operations act on decoded values: identity and ReLU pass a count or value through, the residual
  ``sum`` adds its two halves (under a spiking semantics the next segment entry clamps the rate, so the
  count is ``min(a + b, T)``).

The observable a run reports is a count under LIF and cascaded TTFS, a decoded level ``n / T`` under
synchronized TTFS and the integer under the value semantics (``observable``).
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import numpy as np

from mimarsinan.certification.randomized.instance import (
    CASCADED_TTFS,
    HOST,
    INPUT,
    STREAMED_LIF,
    SYNCHRONIZED_LIF,
    SYNCHRONIZED_TTFS,
    VALUE,
    WINDOWED_LIF,
    Instance,
)

#: Proposition 2 claims identity == N under the executor conditions and H8-staircase alone. Under synchronized
#: TTFS the composition memo's P1-P4 and P6 hold by construction on an integer identity program (theta per hop,
#: no QAT forward to match, no half-step fold, no readout staircase) and P5 (dtype tie stability) is H8-staircase.
CLAIMED = "claimed"
#: Claimed only where the data-dependent H2-H4 hold (window coverage, no overshoot, chase completion); the
#: code level does not observe them, so the relation is measured, never judged (the model level admits it).
CONDITIONAL = "conditional"
#: Outside Proposition 2 (cascaded TTFS: the greedy schedule is lossy by design; analytical TTFS: real-valued).
NOT_CLAIMED = "not_claimed"

CLAIMS: Mapping[str, str] = {
    STREAMED_LIF: CONDITIONAL, WINDOWED_LIF: CONDITIONAL, SYNCHRONIZED_LIF: CLAIMED,
    SYNCHRONIZED_TTFS: CLAIMED, VALUE: CLAIMED, CASCADED_TTFS: NOT_CLAIMED,
}


def claim(semantics: str) -> str:
    return CLAIMS.get(semantics, NOT_CLAIMED)


def evaluate(instance: Instance, *, levels: Optional[int] = None) -> Dict[str, np.ndarray]:
    """``{layer name: (B, width) int64}``: N on the document's inputs, every layer (neural and host); ``levels``
    is the integer readout law's grid under the value semantics (None: the affine value core)."""
    net, T = instance.network, int(instance.timesteps)
    spiking = instance.semantics != VALUE
    values: Dict[str, np.ndarray] = {INPUT: np.asarray(instance.inputs, dtype=np.int64)}
    for layer in net.layers:
        x = np.concatenate([values[s] for s in layer.sources], axis=1)
        if layer.kind == HOST:
            values[layer.name] = _host(layer.op, x, T if spiking else None)
            continue
        weights = np.asarray(layer.weights, dtype=np.int64).reshape(layer.width, -1)
        bias = np.asarray(layer.bias, dtype=np.int64)
        if spiking:
            charge = x @ weights.T + T * bias
            values[layer.name] = np.clip(np.floor_divide(charge, int(layer.theta)), 0, T)
        elif levels is None:
            values[layer.name] = x @ weights.T + bias
        else:
            values[layer.name] = np.clip(np.clip(x, -levels, levels) @ weights.T + bias, -levels, levels)
    return values


def _host(op: str, x: np.ndarray, window) -> np.ndarray:
    if op == "sum":
        half = x.shape[1] // 2
        total = x[:, :half] + x[:, half:]
        return np.minimum(total, window) if window is not None else total
    if op == "relu":
        return np.maximum(x, 0)
    return x


def observable(instance: Instance, values: np.ndarray) -> np.ndarray:
    """N's values in the unit a run reports them (float64): the level ``n / T`` under synchronized TTFS."""
    out = np.asarray(values, dtype=np.float64)
    return out / float(instance.timesteps) if instance.semantics == SYNCHRONIZED_TTFS else out
