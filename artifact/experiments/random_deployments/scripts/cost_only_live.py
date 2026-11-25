""

from __future__ import annotations

import json

import mimarsinan
from mimarsinan.certification.randomized import cost_draws as cd


def main() -> None:
    parts = cd.partition()
    registry = sorted(cd.cost_only_set())
    rows = []
    for part, paths in parts.items():
        for p in paths:
            rows.append({"path": p, "part": part,
                         "screen": {"program_changing": "drawn by the X3a code-level screen",
                                    "derived_geometry": "drawn through the core-type fields",
                                    "drawn_by_fields": "drawn field by field (present or absent)",
                                    "deferred_to_cl1": "deferred by name: " + cd.DEFERRED_REASONS.get(p.split(".")[0], ""),
                                    "pricing_only": cd.PRESERVED_BY_CONSTRUCTION}[part]})
    print(json.dumps({"tree": mimarsinan.__file__, "cost_only_set_size": len(registry),
                      "partition_sizes": {k: len(v) for k, v in parts.items()}, "rows": rows,
                      "sign_granularity_field": cd.SIGN_GRANULARITY_FIELD}, indent=1))


if __name__ == "__main__":
    main()
