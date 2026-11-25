""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, Mapping


def _fmt(v: float) -> str:
    return f"{v:.3g}"


def reuse_reading(rows: Iterable[Mapping[str, Any]]) -> Dict[str, str]:
    ""
    values: Dict[str, Dict[float, set]] = defaultdict(lambda: defaultdict(set))
    seen = set()
    for row in rows:
        seen.add(row["network"])
        r = row.get("macs_per_weight")
        if r is not None:
            values[row["network"]][round(float(r), 9)].add(str(row.get("target")))
    out: Dict[str, str] = {}
    for network in sorted(seen):
        found = values.get(network) or {}
        if not found:
            out[network] = "-"
        elif len(found) == 1:
            out[network] = _fmt(next(iter(found)))
        else:
            out[network] = " / ".join(f"{_fmt(v)} ({', '.join(sorted(t))})" for v, t in sorted(found.items()))
    return out


def nevresim_cell(nev: Mapping[str, Any] | None) -> str:
    ""
    if not nev:
        return "-"
    if nev.get("not_run"):
        return f"not run: {nev['not_run']} ({nev.get('dataset')})"
    if nev.get("refusal"):
        return f"refused by name: {nev['refusal']['name']}"
    data = nev.get("data")
    source = (f"{data.get('dataset')} {data.get('split')} split" if isinstance(data, Mapping)
              else f"{data}: split not recorded")
    windows = nev.get("neuron_windows_compared")
    return (f"{'PASS' if nev.get('passed') else 'FAIL'} ({nev.get('gate')}), {'-' if windows is None else windows} "
            f"windows on {nev.get('inputs')} inputs ({source})")
