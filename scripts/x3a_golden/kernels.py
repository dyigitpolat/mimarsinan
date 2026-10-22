"""Integer kernels of the box's semantics, shared by the reference executor and the packed machine.

Every function is exact integer arithmetic on int64 arrays (or Python ints):

- ``staircase(q, theta, cmp)``: thresholds passed by an integer charge, floor(q / theta) for the
  inclusive comparator (fire iff m >= theta) and floor((q - 1) / theta) for the strict one (fire iff
  m > theta) (LIF memo section 2, F(x); TTFS level formulas of the TTFS closure section 3.1);
- ``fire``/``reset``: the per-cycle fire predicate and the subtractive or zero reset;
- ``saturate``: a signed saturating membrane of ``bits`` bits (None: unbounded);
- ``uniform_trains``: the cycle-0-anchored even re-encode of a window count (spike at cycle k iff
  ceil((k+1) n / T) > ceil(k n / T)), used at every segment entry and by a count-collapsed carry;
  ``comb_trains`` is the LIF memo's Uniform comb (section 1, ``to_uniform_spikes``: spike at cycle k iff
  k n mod T < n); a declaration picks one by ``encoder`` (``trains``);
- ``value_law``: ReLU then half-away-from-zero requantization by 2**shift, clamped to the value grid
  (the integer value semantics, arithmetic toggle section 3.7); ``affine_law`` is the base tree's value
  core, the requantized affine charge with no activation and no clamp; ``value_out`` picks by ``value_law``;
- ``host_apply``: exact integer host operations on window values.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

FP32_EXACT = 1 << 24


def staircase(q, theta: int, cmp: str):
    return q // theta if cmp == "<=" else (q - 1) // theta


def fire(m, theta: int, cmp: str):
    return m >= theta if cmp == "<=" else m > theta


def reset(m, fired, theta: int, mode: str):
    if mode == "subtract":
        return np.where(fired, m - theta, m)
    return np.where(fired, 0, m)


def saturate(m, bits: Optional[int]):
    if bits is None:
        return m
    return np.clip(m, -(1 << (bits - 1)), (1 << (bits - 1)) - 1)


def out_of_range(m, bits: Optional[int]):
    if bits is None:
        return np.zeros(np.shape(m), bool)
    return (m < -(1 << (bits - 1))) | (m > (1 << (bits - 1)) - 1)


def uniform_trains(counts: np.ndarray, T: int) -> np.ndarray:
    """``[T, *counts.shape]`` 0/1 trains with exactly ``counts`` spikes, cycle 0 first."""
    c = np.clip(np.asarray(counts, np.int64), 0, T)
    k = np.arange(T, dtype=np.int64).reshape((T,) + (1,) * c.ndim)
    upper = ((k + 1) * c + T - 1) // T
    lower = (k * c + T - 1) // T
    return (upper - lower).astype(np.int64)


def comb_trains(counts: np.ndarray, T: int) -> np.ndarray:
    """``[T, *counts.shape]`` 0/1 trains of the LIF memo's Uniform comb: n spikes at the cycles k, k n mod T < n."""
    c = np.clip(np.asarray(counts, np.int64), 0, T)
    k = np.arange(T, dtype=np.int64).reshape((T,) + (1,) * c.ndim)
    return (((k * c) % T) < c).astype(np.int64)


ENCODERS = {"even": uniform_trains, "comb": comb_trains}


def trains(counts: np.ndarray, T: int, encoder: str = "even") -> np.ndarray:
    return ENCODERS[encoder](counts, T)


def round_half_away_shift(acc: np.ndarray, shift: int) -> np.ndarray:
    acc = np.asarray(acc, np.int64)
    if shift == 0:
        return acc
    half = 1 << (shift - 1)
    mag = (np.abs(acc) + half) >> shift
    return np.where(acc < 0, -mag, mag)


def value_law(acc: np.ndarray, shift: int, act_bits: int) -> np.ndarray:
    return np.clip(round_half_away_shift(acc, shift), 0, (1 << act_bits) - 1)


def affine_law(acc: np.ndarray, shift: int) -> np.ndarray:
    return round_half_away_shift(acc, shift)


def value_out(acc: np.ndarray, shift: int, decl) -> np.ndarray:
    if decl.value_law == "affine":
        return affine_law(acc, shift)
    return value_law(acc, shift, decl.act_bits)


def lif_law(q: np.ndarray, theta: int, cmp: str, T: int, reset_mode: str) -> np.ndarray:
    """The integer staircase of a window charge (synchronized emission of T cycles)."""
    count = np.clip(staircase(q, theta, cmp), 0, T)
    return np.minimum(count, 1) if reset_mode == "zero" else count


def ttfs_level(c: np.ndarray, theta: int, cmp: str, T: int) -> np.ndarray:
    return np.clip(staircase(c, theta, cmp), 0, T)


def clip_signed(x: np.ndarray, bits: int) -> np.ndarray:
    return np.clip(x, -(1 << (bits - 1)), (1 << (bits - 1)) - 1)


def host_apply(op: str, values: Sequence[np.ndarray], hi: int) -> np.ndarray:
    if op == "add":
        return np.clip(np.sum(np.stack(values), axis=0), 0, hi)
    if op == "max":
        return np.max(np.stack(values), axis=0)
    if op == "relay":
        return np.array(values[0], np.int64)
    if op == "relu":
        return np.maximum(np.array(values[0], np.int64), 0)
    if op == "sum":
        return np.sum(np.stack(values), axis=0)
    raise ValueError(f"unknown host op {op!r}")


def charge_bound(weights: np.ndarray, bias: np.ndarray, decl) -> np.ndarray:
    """Static reachable bound on every accumulator a neuron forms over one window (export-check form)."""
    absw = np.abs(weights).sum(axis=0)
    if decl.semantics == "value":
        return absw * decl.hi + np.abs(bias) * 1
    return decl.T * (absw + np.abs(bias))


def event_fold(m, x, w, bias, theta: int, decl, order, own_only: bool = True):
    """One per-event cycle: the bias event, then rows ``order`` of ``w`` [F, n] against spikes ``x`` [B, F].

    A neuron is updated only by its own synapses (a nonzero weight) that carry a spike; after each update
    it saturates, checks the comparator and resets; it emits at most one spike per cycle. ``own_only``
    False checks every neuron at every spiking axon of the core (a core-wide event loop): the finding
    witness ``witnesses.foreign_event_witness`` shows that it makes coalescing function-bearing.
    """
    fired = np.zeros(m.shape, bool)
    events = [(None, np.asarray(bias, np.int64))] + [(i, w[i]) for i in order]
    for i, wrow in events:
        if i is None:
            mask = np.broadcast_to(wrow != 0, m.shape)
            delta = np.broadcast_to(wrow, m.shape)
        else:
            mask = ((wrow != 0) | (not own_only))[None, :] & (x[:, i:i + 1] > 0)
            delta = x[:, i:i + 1] * wrow[None, :]
        m = np.where(mask, saturate(m + delta, decl.membrane_bits), m)
        hit = mask & ~fired & fire(m, theta, decl.comparator)
        fired = fired | hit
        m = reset(m, hit, theta, decl.reset)
    return m, fired
