from __future__ import annotations

import json
import os
from typing import Any, Dict

__all__ = ["write_final_population_json"]


def write_final_population_json(result_json: Dict[str, Any], out_path: str) -> None:
    """Write a compact list of candidate configs + objective values for quick inspection."""
    pop = []
    for c in result_json.get("pareto_front", []):
        row = {}
        row.update(c.get("configuration", {}).get("model_config", {}))
        row.update(c.get("configuration", {}).get("platform_constraints", {}))
        row.update(c.get("objectives", {}))
        pop.append(row)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(pop, f, indent=2)
