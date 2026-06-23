"""Characterization phase (spec §10 / report V9): profile an axis before search.

Sweeps the paired drop on a grid of α and derives a ``Profile``: a monotonicity
verdict (A1 — a significant non-monotonicity downgrades the controller to dense-
grid safe mode), the maximum local slope (A3 — a near-vertical cliff calls for a
smaller ``epsilon``), and the highest α that stayed within budget. The profile
configures the controller per axis instead of trusting global assumptions.

The E4 keystone's pre-flight CONFIRM (``CascadeCharacterizer``) reuses this
``characterize`` monotonicity check, plus a ``CalibrationSource`` that threads the
trainer's calibration batches through the four forward-only cascade probes
(cold-cascade liveness / ramp monotonicity / staircase-vs-LIF ceiling / firing
gain). The probes are float64 toy and CPU-only; they confirm a proposed conversion
recipe on THIS model before it is trusted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mimarsinan.tuning.orchestration.conversion_policy import (
    CharacterizationResult,
    Characterizer,
)


@dataclass
class Profile:
    """Characterization output, archived with the run for reproducibility."""

    monotonic: bool
    max_slope: float
    epsilon_hint: float
    feasible_max: float
    drops: list


def characterize(drop_fn, grid, *, budget=0.02, epsilon_floor=2 ** -8, epsilon_cap=2 ** -4):
    """Profile ``drop_fn(alpha) -> drop`` over an increasing ``grid`` of α.

    ``monotonic`` is False if any later grid point's drop falls below an earlier
    one's by more than the noise budget (A1). ``epsilon_hint`` shrinks inversely
    with the steepest local slope so bisection never steps over a cliff (A3).
    ``feasible_max`` is the highest α whose drop stayed within ``budget``.
    """
    grid = [float(a) for a in grid]
    drops = [float(drop_fn(a)) for a in grid]

    monotonic = all(
        drops[i + 1] >= drops[i] - budget for i in range(len(drops) - 1)
    )

    slopes = [
        abs(drops[i + 1] - drops[i]) / max(grid[i + 1] - grid[i], 1e-12)
        for i in range(len(grid) - 1)
    ]
    max_slope = max(slopes) if slopes else 0.0

    # Steeper cliff → smaller admissible increment, clamped to [floor, cap].
    raw = budget / max_slope if max_slope > 0 else epsilon_cap
    epsilon_hint = min(max(raw, epsilon_floor), epsilon_cap)

    feasible = [a for a, d in zip(grid, drops) if d <= budget]
    feasible_max = max(feasible) if feasible else 0.0

    return Profile(
        monotonic=monotonic,
        max_slope=max_slope,
        epsilon_hint=epsilon_hint,
        feasible_max=feasible_max,
        drops=drops,
    )


class CalibrationSource:
    """The calibration-batch seam the cascade probes read inputs from.

    Decouples the probes from WHERE their forward inputs come from: a live tuner
    hands its trainer (``from_context``; batches pulled via
    ``iter_validation_batches(n)``), a toy test hands an explicit tensor
    (``from_inputs``). ``inputs()`` returns the concatenated float64 calibration
    batch, or ``None`` when no source is available (probes then short-circuit).
    """

    def __init__(self, *, context: Any = None, explicit=None, n_batches: int = 1):
        self._context = context
        self._explicit = explicit
        self._n_batches = max(1, int(n_batches))

    @classmethod
    def from_context(cls, context: Any, *, n_batches: int = 1) -> "CalibrationSource":
        return cls(context=context, n_batches=n_batches)

    @classmethod
    def from_inputs(cls, inputs) -> "CalibrationSource":
        return cls(explicit=inputs)

    def inputs(self):
        import torch

        if self._explicit is not None:
            return self._explicit
        if self._context is None:
            return None
        batches = []
        for x, _y in self._context.iter_validation_batches(self._n_batches):
            # Keep the NATIVE batch shape (e.g. N×C×H×W) — the model's own forward
            # owns the input rearrange; flattening would break its mapper graph.
            batches.append(x.double())
        if not batches:
            return None
        return torch.cat(batches, dim=0)


_LIVENESS_FLOOR = 1e-4
_LIVE_DEPTH_FRACTION = 0.75
_FIRING_GAIN_FLOOR = 0.1
_RAMP_BUDGET = 0.02


class CascadeCharacterizer(Characterizer):
    """The R1 forward-only pre-flight CONFIRM for the cascaded fire-once recipe.

    The four probes run the genuine single-spike cascade (and its analytical
    staircase twin) on a calibration batch and read per-perceptron decoded values
    through the segment forward's ``node_value_recorder`` side-channel (the SSOT
    the DFQ calibrators already use), so an IN-distribution cascade matches and an
    OFF-distribution model (a pruned / dead-at-depth cascade) escalates instead of
    silently shipping the fast recipe. Float64 toy, CPU-only.
    """

    def __init__(self, *, context: Any = None, calib_inputs=None,
                 n_batches: int = 1, S: int = 8):
        self._calib = (
            CalibrationSource.from_inputs(calib_inputs)
            if calib_inputs is not None
            else CalibrationSource.from_context(context, n_batches=n_batches)
        )
        self._S = int(S)
        self._context = context

    # ── decoded-value substrate (reuses the node_value_recorder side-channel) ──
    def _calibration_inputs(self):
        x = self._calib.inputs()
        if x is None:
            return None
        # NATIVE shape — the model's forward / mapper graph owns the input rearrange.
        return x.double()

    @staticmethod
    def _perceptron_decoded_means(model, x, S):
        """Per-depth genuine-cascade decoded channel-mean (keyed by perceptron index).

        Drives the production segment forward and reads every perceptron node's
        decoded value via ``forward_with_node_values`` — the exact NF-side mechanism
        the cascaded DFQ calibration consumes."""
        import torch

        from mimarsinan.models.spiking.training.ttfs_segment_forward import (
            TTFSSegmentForward,
        )
        from mimarsinan.spiking.segment_partition import perceptron_of

        try:
            drv = TTFSSegmentForward(model.get_mapper_repr(), S)
            with torch.no_grad():
                _out, rec = drv.forward_with_node_values(x)
        except Exception:
            # A model the genuine cascade cannot run (not a cascade / shape mismatch)
            # reads as fully dead ⇒ the OFF-distribution verdict (escalate), never a
            # crash inside the inert-by-default _configure path.
            return {}
        by_perc = {
            id(perceptron_of(n)): v
            for n, v in rec.items()
            if perceptron_of(n) is not None
        }
        out = {}
        for k, p in enumerate(model.get_perceptrons()):
            v = by_perc.get(id(p))
            if v is not None:
                out[k] = v.reshape(-1, v.shape[-1]).double().mean(0)
        return out

    @staticmethod
    def _staircase_means(model, x):
        """Per-depth analytical (cycle-accurate-OFF) staircase channel-mean.

        The staircase/LIF ceiling twin: the per-perceptron analytical forward (the
        pointwise composition the cascade approximates), captured by a forward hook
        with every TTFS node temporarily out of cascade mode then restored."""
        import torch

        from mimarsinan.models.nn.activations.ttfs_spiking import TTFSActivation

        means: dict = {}
        handles = []
        for k, p in enumerate(model.get_perceptrons()):
            def hook(_m, _i, out, k=k):
                means[k] = out.detach().reshape(-1, out.shape[-1]).double().mean(0)
            handles.append(p.activation.register_forward_hook(hook))

        nodes = [m for m in model.modules() if isinstance(m, TTFSActivation)]
        prev = [n._cycle_accurate_mode for n in nodes]
        for n in nodes:
            n.set_cycle_accurate(False)
        model.double().eval()
        try:
            with torch.no_grad():
                model(x.double())
        except Exception:
            # An unrunnable model reads as no-staircase ⇒ the OFF-distribution
            # verdict (firing-gain collapses), never a crash in _configure.
            means = {}
        finally:
            for n, pmode in zip(nodes, prev):
                n.set_cycle_accurate(pmode)
            for h in handles:
                h.remove()
        return means

    def _live_depth_fraction(self, model, x):
        decoded = self._perceptron_decoded_means(model, x, self._S)
        depth = len(model.get_perceptrons())
        if depth == 0:
            return 0.0, decoded, depth
        live = sum(
            1
            for k in range(depth)
            if decoded.get(k) is not None
            and float(decoded[k].abs().mean()) > _LIVENESS_FLOOR
        )
        return live / depth, decoded, depth

    # ── the four forward-only probes ──────────────────────────────────────────
    def cold_cascade_live(self, *, model: Any, context: Any = None) -> bool:
        """Is the cold genuine cascade alive through the full depth (no death cascade)?

        True iff at least ``_LIVE_DEPTH_FRACTION`` of the depths AND the output layer
        decode a non-trivial value (a pruned / dead-at-depth cascade collapses the
        deepest layers to zero)."""
        x = self._calibration_inputs()
        if x is None:
            return True
        frac, decoded, depth = self._live_depth_fraction(model, x)
        if depth == 0:
            return False
        out = decoded.get(depth - 1)
        out_live = out is not None and float(out.abs().mean()) > _LIVENESS_FLOOR
        return bool(frac >= _LIVE_DEPTH_FRACTION and out_live)

    def ramp_monotone(self, *, model: Any, context: Any = None) -> bool:
        """Does the output's decoded magnitude grow monotonically with S (no cliff)?

        Reuses ``characterize``'s A1 monotonicity verdict on the output-layer decoded
        mean swept over an S grid — a healthy cascade's decode rises with resolution."""
        x = self._calibration_inputs()
        if x is None:
            return True
        depth = len(model.get_perceptrons())
        if depth == 0:
            return False
        grid = [4, 8, 16, 32]

        def out_mean(S):
            decoded = self._perceptron_decoded_means(model, x, int(S))
            out = decoded.get(depth - 1)
            return float(out.abs().mean()) if out is not None else 0.0

        # ``characterize``'s A1 verdict is monotone-NON-DECREASING on its drop_fn;
        # a healthy cascade's output decode RISES with S, so feeding the decode
        # magnitude directly makes a clean monotone ramp pass and a cliff fail.
        profile = characterize(out_mean, grid, budget=_RAMP_BUDGET)
        return bool(profile.monotonic)

    def staircase_lif_ceiling(self, *, model: Any, context: Any = None) -> float:
        """The analytical staircase ceiling at the output (the achievable decode)."""
        x = self._calibration_inputs()
        if x is None:
            return 0.0
        means = self._staircase_means(model, x)
        depth = len(model.get_perceptrons())
        out = means.get(depth - 1)
        return float(out.clamp(min=0).mean()) if out is not None else 0.0

    def firing_gain(self, *, model: Any, context: Any = None) -> float:
        """The deepest-layer firing-gain ratio = genuine-decoded / staircase-decoded.

        Near 1 = faithful; collapsing toward 0 = the death-cascade firing deficit
        (the OFF-distribution signature). Read at the output (the deepest decode)."""
        x = self._calibration_inputs()
        if x is None:
            return 1.0
        decoded = self._perceptron_decoded_means(model, x, self._S)
        staircase = self._staircase_means(model, x)
        depth = len(model.get_perceptrons())
        gen = decoded.get(depth - 1)
        ceil = staircase.get(depth - 1)
        if gen is None or ceil is None:
            return 0.0
        gm = float(gen.abs().mean())
        cm = float(ceil.clamp(min=0).abs().mean())
        return gm / cm if cm > _LIVENESS_FLOOR else 0.0

    # ── the CONFIRM verdict ────────────────────────────────────────────────────
    def characterize(self, *, model: Any, recipe: Any, context: Any = None):
        ctx = context if context is not None else self._context
        live = self.cold_cascade_live(model=model, context=ctx)
        monotone = self.ramp_monotone(model=model, context=ctx)
        ceiling = self.staircase_lif_ceiling(model=model, context=ctx)
        gain = self.firing_gain(model=model, context=ctx)
        probes = {
            "cold_cascade_live": bool(live),
            "ramp_monotone": bool(monotone),
            "staircase_lif_ceiling": float(ceiling),
            "firing_gain": float(gain),
        }
        reasons = []
        if not live:
            reasons.append("cold cascade dead at depth")
        if not monotone:
            reasons.append("rate ramp non-monotone")
        if gain < _FIRING_GAIN_FLOOR:
            reasons.append(f"firing gain collapsed ({gain:.3f})")
        matches = not reasons
        return CharacterizationResult(
            matches=matches, probes=probes, reason="; ".join(reasons),
        )
