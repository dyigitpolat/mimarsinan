"""Broad exception catches are a tracked, shrinking allowlist (best_effort or documented fallbacks only)."""

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[3] / "src" / "mimarsinan"

_BROAD = re.compile(r"except (Exception|BaseException)?\s*(as \w+)?\s*:")
_NARROW = re.compile(
    r"except [A-Z][a-zA-Z]*Error|except \(|"
    r"except (Key|Value|Type|Attribute|Import|Index|OS|IO|Runtime|Module|File|Json|Stop)"
)

# file -> number of sanctioned broad catches (documented fallbacks, re-raising
# wrappers, and best_effort itself). Shrink these counts; never grow them.
ALLOWLIST = {
    # Cross-thread exception transport in the run_bounded watchdog: the callee's
    # exception is re-raised verbatim in the caller thread - nothing degrades.
    "chip_simulation/execution_bounds.py": 1,
    "chip_simulation/pareto.py": 2,
    "common/best_effort.py": 1,
    "common/diagnostics.py": 1,
    "config_schema/display_view_build.py": 1,
    "gui/handle.py": 1,
    "gui/server/routes_layout.py": 2,
    "gui/server/routes_wizard.py": 3,
    "mapping/model_representation.py": 1,
    "mapping/support/bias_compensation.py": 1,
    "mapping/verification/suggester/hw_config_suggester_scheduled.py": 1,
    "mapping/verification/verifier/mapping_verifier_soft.py": 2,
    "pipelining/core/engine/pipeline.py": 1,
    "pipelining/pipeline_steps/config/torch_mapping_step.py": 2,
    "search/optimizers/agent_evolve/batch_eval.py": 1,
    "search/optimizers/compilagent/backend/backend.py": 2,
    "search/optimizers/compilagent/backend/backend_layout.py": 1,
    "search/optimizers/compilagent/compilagent_optimizer.py": 1,
    "search/optimizers/llm/trace.py": 1,
    "search/optimizers/nsga2_optimizer.py": 1,
    "search/problems/joint/evaluate.py": 6,
    "search/problems/joint/validate.py": 5,
    # [MBH-DRAWS] re-raising boundary: a failed conversion draw is a measured
    # outcome (logged, workers released, independent redraw); the harness
    # re-raises when EVERY draw failed - nothing degrades silently.
    "tuning/orchestration/conversion_draws.py": 1,
    "torch_mapping/conversion_probe.py": 1,
    "torch_mapping/torch_graph_tracer.py": 2,
    "visualization/graphviz/common.py": 1,
    "visualization/graphviz/hybrid_combined_dot.py": 1,
    "visualization/graphviz/ir_summary.py": 1,
    "visualization/search_viz/html/sections_layout.py": 1,
    "visualization/search_viz/report_png.py": 1,
    "visualization/softcore_flowchart_dot.py": 1,
}


def _broad_counts():
    found = {}
    for path in SRC.rglob("*.py"):
        n = sum(
            1 for line in path.read_text().splitlines()
            if _BROAD.search(line) and not _NARROW.search(line)
        )
        if n:
            found[str(path.relative_to(SRC))] = n
    return found


def test_no_new_broad_excepts():
    grown = {
        f: (n, ALLOWLIST.get(f, 0))
        for f, n in _broad_counts().items()
        if n > ALLOWLIST.get(f, 0)
    }
    assert not grown, (
        "new broad 'except Exception' catches (use best_effort for telemetry, "
        f"narrow + propagate otherwise): {grown}"
    )


def test_allowlist_counts_are_current():
    stale = {
        f: (ALLOWLIST[f], current)
        for f in ALLOWLIST
        if (current := _broad_counts().get(f, 0)) < ALLOWLIST[f]
    }
    assert not stale, f"allowlist counts too high (ratchet them down): {stale}"
