"""IV.7 perf / scaling gates — the cost contracts the refactor must hold.

These are the deterministic, CUDA-free gates: the predictor-corrector probe bound
(no linear rate scan) and the CheckpointGuard scope lever (skip the frozen backbone).
Wall-clock / peak-VRAM gates belong to the CUDA ViT-probe integration benchmark.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from mimarsinan.tuning.orchestration.rate_scheduler import RateScheduler
from mimarsinan.tuning.orchestration.checkpoint_guard import CheckpointGuard
from .mock_axis_zoo import cliff, smooth_monotone


# ── Probe-count: bisection is logarithmic, never a linear scan ────────────────

def _counting(attempt):
    calls = []

    def wrapped(target):
        calls.append(target)
        return attempt(target)

    return wrapped, calls


def test_cliff_probe_count_is_log_bounded():
    """Cornering a cliff costs O(log(gap/epsilon)) probes, not O(gap/epsilon)."""
    eps = 1e-3
    for alpha_star in (0.1, 0.37, 0.5, 0.83, 0.99):
        attempt, _ = cliff(alpha_star)
        wrapped, calls = _counting(attempt)
        sched = RateScheduler(epsilon=eps, policy="greedy_to_one", max_rounds=64)
        sched.run(0.0, wrapped)
        # one round near the cliff bisects the unit gap to epsilon; allow a small
        # constant of extra rounds for the climb above the cliff edge.
        bound = (math.ceil(math.log2(1.0 / eps)) + 1) * 4
        assert len(calls) <= bound, (
            f"alpha*={alpha_star}: {len(calls)} probes exceeds the log bound {bound}"
        )


def test_smooth_reaches_one_in_one_probe():
    """A fully feasible axis commits the whole jump in a single greedy probe."""
    attempt, _ = smooth_monotone()
    wrapped, calls = _counting(attempt)
    sched = RateScheduler(epsilon=1e-3, policy="greedy_to_one", max_rounds=64)
    out = sched.run(0.0, wrapped)
    assert out == 1.0
    assert calls == [1.0]


# ── CheckpointGuard scope: tunable skips the frozen backbone (the W6 lever) ───

class _BackboneHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(32, 32)   # frozen
        self.head = nn.Linear(32, 4)         # tunable
        for p in self.backbone.parameters():
            p.requires_grad_(False)


class _StubTrainer:
    def __init__(self, model):
        self.model = model
        self.device = "cpu"


def _handle_param_count(handle) -> int:
    state = getattr(handle, "state", handle)
    model_state = state[0] if isinstance(state, tuple) else state
    if isinstance(model_state, dict):
        return sum(1 for k in model_state if "backbone" in k or "head" in k)
    return 0


def test_tunable_scope_captures_fewer_tensors_than_full():
    """scope='tunable' must not snapshot the frozen backbone params."""
    trainer = _StubTrainer(_BackboneHead())
    full = CheckpointGuard(trainer, scope="full", location="device")
    tunable = CheckpointGuard(trainer, scope="tunable", location="device")

    full_h = full.snapshot()
    tunable_h = tunable.snapshot()

    full_keys = _collect_keys(full_h)
    tunable_keys = _collect_keys(tunable_h)
    assert full_keys, "full snapshot should capture the backbone + head params"
    # the frozen backbone weight must be absent from the tunable snapshot.
    assert any("backbone" in k for k in full_keys)
    assert not any("backbone" in k and k in tunable_keys for k in full_keys), (
        "tunable scope must skip frozen backbone params"
    )
    assert len(tunable_keys) < len(full_keys)


def _collect_keys(handle):
    state = getattr(handle, "state", handle)
    model_state = state[0] if isinstance(state, tuple) else state
    if isinstance(model_state, dict):
        return set(model_state.keys())
    return set()


def test_full_scope_round_trips_bitwise():
    """scope='full' restore is bit-exact (golden-safe)."""
    trainer = _StubTrainer(_BackboneHead())
    guard = CheckpointGuard(trainer, scope="full", location="device")
    before = {k: v.clone() for k, v in trainer.model.state_dict().items()}
    handle = guard.snapshot()
    with torch.no_grad():
        for p in trainer.model.parameters():
            p.add_(1.0)
    guard.restore(handle)
    for k, v in trainer.model.state_dict().items():
        assert torch.allclose(v, before[k], atol=0.0)
