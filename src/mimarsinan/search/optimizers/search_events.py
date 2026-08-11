"""The live search-event channel: one vocabulary, every optimizer backend.

The GUI's live panel (``gui/static/js/search-live.js``) dispatches on
``event["type"]`` and silently drops anything it does not know, so the frame
shapes here — not the emitters — are the contract. Classical (NSGA-II) and LLM
backends both build their generation-level frames from these constructors so the
two can never drift into parallel vocabularies.

Emission is TELEMETRY: ``emit_search_event`` degrades through ``best_effort``
so a dead monitor can never take a search down with it.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Sequence

from mimarsinan.common.best_effort import best_effort
from mimarsinan.search.results import ObjectiveSpec

SEARCH_EVENT_METRIC = "search_event"

# The panel renders the leading rows of a front; the tail is a scroll nobody
# reads and a payload everybody pays for.
PARETO_FRONT_PREVIEW = 5


def emit_search_event(reporter: Any, event: Dict[str, Any]) -> None:
    """Emit a structured search event via the reporter."""
    if reporter is None:
        return
    with best_effort("emit search_event"):
        reporter(SEARCH_EVENT_METRIC, json.dumps(event, default=str))


def objective_declarations(
    objectives: Sequence[ObjectiveSpec],
) -> List[Dict[str, str]]:
    """The ``{name, goal}`` pairs the panel ranks and colours candidates by."""
    return [{"name": spec.name, "goal": spec.goal} for spec in objectives]


def generation_start_event(
    *,
    gen: int,
    total_gens: int,
    phase: str,
    objectives: Sequence[ObjectiveSpec],
    pop_size: int | None = None,
) -> Dict[str, Any]:
    """A generation opens: its ordinal, the budget it sits in, and the axes."""
    event: Dict[str, Any] = {
        "type": "generation_start",
        "gen": gen,
        "total_gens": total_gens,
        "phase": phase,
        "objectives": objective_declarations(objectives),
    }
    if pop_size is not None:
        event["pop_size"] = int(pop_size)
    return event


def candidates_generated_event(
    *, gen: int, count: int, reasoning: str = "",
) -> Dict[str, Any]:
    """A generation's candidate batch exists — how many, and (LLM) why."""
    return {
        "type": "candidates_generated",
        "gen": gen,
        "count": int(count),
        "reasoning": reasoning,
    }


def generation_complete_event(
    *,
    gen: int,
    valid_count: int,
    failed_count: int,
    pareto_objectives: Sequence[Mapping[str, float]],
) -> Dict[str, Any]:
    """A generation closes: its verdict counts and the front it leaves behind.

    ``pareto_objectives`` must already lead with the incumbent (the panel
    renders the head of the list): the preview is a truncation, not a ranking.
    """
    return {
        "type": "generation_complete",
        "gen": gen,
        "valid_count": int(valid_count),
        "failed_count": int(failed_count),
        "pareto_size": len(pareto_objectives),
        "pareto_front": [dict(row) for row in pareto_objectives[:PARETO_FRONT_PREVIEW]],
    }


def search_complete_event(
    *, total_valid: int, total_failed: int, final_pareto_size: int,
) -> Dict[str, Any]:
    """The search ends: run totals and the size of the front it hands over."""
    return {
        "type": "search_complete",
        "total_valid": int(total_valid),
        "total_failed": int(total_failed),
        "final_pareto_size": int(final_pareto_size),
    }
