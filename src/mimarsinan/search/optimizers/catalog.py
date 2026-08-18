"""[TS2] The optimizer catalogue: every backend a run may declare, named once.

Three surfaces have to agree on this list — the declared ``OptimizerType``, the
factory's builder table, and the wizard's choices — and they used to be three
hand-written copies, so a backend reached the pipeline without reaching the
GUI. The names live HERE, in a leaf that carries no backend imports, because
``gui`` and ``pipelining`` both read it and neither may pay for pymoo or an LLM
client just to render a list of buttons.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass(frozen=True)
class OptimizerChoice:
    """One backend: the id a config declares, and the label a user reads."""

    id: str
    label: str


OPTIMIZER_CHOICES: Tuple[OptimizerChoice, ...] = (
    OptimizerChoice("nsga2", "NSGA-II"),
    OptimizerChoice("agent_evolve", "Agentic Evolution (LLM-based)"),
    OptimizerChoice("compilagent", "Compilagent (LLM session)"),
    OptimizerChoice("random", "Random Sampling"),
    OptimizerChoice("sobol", "Sobol Sampling"),
    OptimizerChoice("exhaustive", "Exhaustive Grid"),
)

OPTIMIZER_IDS: Tuple[str, ...] = tuple(choice.id for choice in OPTIMIZER_CHOICES)


def optimizer_options() -> List[Dict[str, Any]]:
    """The catalogue as the wizard's ``{id, label}`` rows."""
    return [{"id": choice.id, "label": choice.label} for choice in OPTIMIZER_CHOICES]
