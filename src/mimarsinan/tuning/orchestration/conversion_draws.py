"""[MBH-DRAWS] best-of-N conversion draws: independent RNG streams, D-hat-selected keep-best."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable

import torch

from mimarsinan.tuning.orchestration import run_ledger
from mimarsinan.tuning.orchestration.mbh_ledger import fp32_deployed_read


def configured_draws(pipeline) -> int:
    """The cell's ``conversion_draws`` knob (>= 1; default 1 = single draw)."""
    return max(1, int(pipeline.config.get("conversion_draws", 1)))


@dataclass
class _Draw:
    index: int
    tuner: Any
    model: Any
    adaptation_manager: Any
    dhat: float
    post_cache: dict


def _seed_draw(base_seed: int, index: int) -> None:
    torch.manual_seed(base_seed + index)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(base_seed + index)


def run_conversion_draws(
    pipeline,
    build: Callable[[Any, Any], Any],
    model,
    adaptation_manager,
    *,
    draws: int | None = None,
    target: float | None = None,
):
    """Run a conversion stage best-of-N; returns ``(tuner, model, manager)``.

    ``draws == 1`` (the default) is bit-identical to a plain
    ``build(model, adaptation_manager).run()``: no reseeding, no copies, no
    extra eval reads. For N > 1, every draw runs on its own deep copy of the
    (model, adaptation-manager) object graph and its own run-ledger scope,
    with the torch RNG seeded ``seed + k`` — the whole search is deterministic
    given the config seed. Each draw is independently keep-best/entry-floored
    inside the tuner, and selection takes the max full-transform fp32 D-hat,
    so the harness can only improve D-hat (monotone-safe). A raising draw is a
    measured outcome (logged, workers released); the harness fails loud when
    every draw raised.

    ``target``: best-of-N is a FALLBACK for the crater distribution, not a
    mandate — a draw whose full-transform D-hat reaches ``target`` (the
    stage's entry pipeline metric: conversion at no measured loss) ends the
    search immediately; healthy cells pay for exactly one draw.
    """
    n = configured_draws(pipeline) if draws is None else max(1, int(draws))
    if n == 1:
        tuner = build(model, adaptation_manager)
        tuner.run()
        return tuner, model, adaptation_manager

    base_seed = int(pipeline.config.get("seed", 0))
    # Each draw enters the run-scoped ledger scope fresh (and the kept draw's
    # exit scope is what persists) — otherwise draw k+1's targets/budgets would
    # ride draw k's trajectory and the draws would not be independent.
    entry_cache = run_ledger.snapshot(pipeline.cache)
    best: _Draw | None = None
    last_error: Exception | None = None
    for k in range(n):
        run_ledger.restore(pipeline.cache, entry_cache)
        _seed_draw(base_seed, k)
        draw_model, draw_manager = copy.deepcopy((model, adaptation_manager))
        tuner = build(draw_model, draw_manager)
        try:
            tuner.run()
            dhat = float(fp32_deployed_read(tuner))
        except Exception as error:
            tuner.close()
            last_error = error
            print(
                f"[MBH-DRAWS] k={k} full_acc=nan kept=False "
                f"error={type(error).__name__}: {error}",
                flush=True,
            )
            continue
        kept = best is None or dhat > best.dhat
        print(f"[MBH-DRAWS] k={k} full_acc={dhat:.6f} kept={kept}", flush=True)
        if kept:
            if best is not None:
                best.tuner.close()
            best = _Draw(
                k, tuner, draw_model, draw_manager, dhat,
                run_ledger.snapshot(pipeline.cache),
            )
        else:
            tuner.close()
        if target is not None and dhat >= float(target):
            print(
                f"[MBH-DRAWS] k={k} reached target={float(target):.6f}: "
                "skipping remaining draws",
                flush=True,
            )
            break
    if best is None:
        assert last_error is not None
        raise last_error
    run_ledger.restore(pipeline.cache, best.post_cache)
    print(
        f"[MBH-DRAWS] selected k={best.index} full_acc={best.dhat:.6f} n={n}",
        flush=True,
    )
    pipeline.reporter.report("conversion_draws_selected", {
        "k": best.index, "full_acc": round(best.dhat, 4), "n": n,
    })
    return best.tuner, best.model, best.adaptation_manager
