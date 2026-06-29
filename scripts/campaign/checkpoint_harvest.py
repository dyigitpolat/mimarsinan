"""Harvest checkpoint JSONL measurements into normalized ledger rows."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Any, Iterable

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SRC = os.path.join(_REPO, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from mimarsinan.chip_simulation.ledger_schema import normalize_ledger_record

_T_LABEL = re.compile(r"(?:^|_)T(\d+)(?:_|$)")


def _spiking_mode(record: dict[str, Any]) -> str:
    label = str(record.get("label") or record.get("cell") or "")
    return "lif" if label.startswith("lif_") else "ttfs_cycle_based"


def _schedule(record: dict[str, Any], spiking_mode: str) -> str | None:
    if spiking_mode != "ttfs_cycle_based":
        return None
    mode = record.get("mode")
    if mode in ("cascaded", "synchronized"):
        return str(mode)
    return "synchronized"


def _simulation_steps(record: dict[str, Any]) -> int:
    for key in ("S", "simulation_steps"):
        if record.get(key) is not None:
            return int(record[key])
    label = str(record.get("label") or "")
    match = _T_LABEL.search(label)
    if match:
        return int(match.group(1))
    return 4


def _validity(record: dict[str, Any]) -> str:
    rc = int(record.get("returncode", 0))
    if rc == 0:
        return "VALID_checkpoint_rc0"
    return "VALID_FLAGGED_checkpoint_rc1_predeploy_metric"


def _metric_context(record: dict[str, Any]) -> str:
    rc = int(record.get("returncode", 0))
    if rc == 0:
        return "clean_deployed_accuracy"
    if record.get("last_step") == "Activation Analysis":
        return "gate_rejected_predeploy_spiking_accuracy"
    return "gate_rejected_partial_pipeline_accuracy"


def _verdict(record: dict[str, Any]) -> str:
    rc = int(record.get("returncode", 0))
    if rc != 0:
        return "FAIL"
    ann = float(record.get("ann") or 0.0)
    deployed = float(record.get("deployed") or 0.0)
    if ann > 0.0 and deployed >= 0.85 * ann:
        return "MET"
    return "BOUNDED-GAP"


def checkpoint_record_to_ledger(
    record: dict[str, Any], *, source_file: str,
) -> dict[str, Any]:
    """Map one checkpoint JSONL record into the normalized campaign ledger shape."""
    firing = _spiking_mode(record)
    schedule = _schedule(record, firing)
    row: dict[str, Any] = {
        "model": "deep_cnn",
        "dataset": record.get("dataset", "cifar10"),
        "spiking_mode": firing,
        "deployment_validity": _validity(record),
        "deployed_acc": float(record["deployed"]),
        "ann_acc": float(record["ann"]),
        "retention_pp": float(record.get("retention_pp", 100.0 * (float(record["deployed"]) - float(record["ann"])))),
        "returncode": int(record.get("returncode", 0)),
        "metric_context": _metric_context(record),
        "verdict": _verdict(record),
        "S": _simulation_steps(record),
        "depth": int(record.get("depth", 4)),
        "source": "checkpoint_research",
        "source_file": source_file,
    }
    if schedule is not None:
        row["schedule"] = schedule
    for key in ("label", "cell", "wall_s", "gradual_s", "last_step", "config", "experiment"):
        if key in record and record[key] is not None:
            row[key] = record[key]
    if record.get("experiment"):
        row["run_id"] = record["experiment"]
        row["run_ids"] = [record["experiment"]]
    elif record.get("cell"):
        row["run_id"] = record["cell"]
    elif record.get("label"):
        row["run_id"] = record["label"]
    return normalize_ledger_record(row)


def read_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path) as fh:
        for line in fh:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def harvest_checkpoint_dir(data_dir: str) -> list[dict[str, Any]]:
    """Read every checkpoint JSONL file under ``data_dir`` into ledger rows."""
    rows: list[dict[str, Any]] = []
    for name in sorted(os.listdir(data_dir)):
        if not name.endswith(".jsonl"):
            continue
        path = os.path.join(data_dir, name)
        for record in read_jsonl(path):
            rows.append(checkpoint_record_to_ledger(record, source_file=path))
    return rows


def append_rows(path: str, rows: Iterable[dict[str, Any]]) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    count = 0
    with open(path, "a") as fh:
        for row in rows:
            out = dict(row)
            out.setdefault("ts", time.time())
            fh.write(json.dumps(out) + "\n")
            count += 1
    return count


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default=os.path.join(_REPO, "docs", "checkpoint_research", "data"),
    )
    parser.add_argument("--out", default=None, help="Optional JSONL path to append rows to")
    parser.add_argument("--print", action="store_true", help="Print normalized rows to stdout")
    args = parser.parse_args(argv)

    rows = harvest_checkpoint_dir(args.data_dir)
    if args.out:
        append_rows(args.out, rows)
    if args.print or not args.out:
        for row in rows:
            print(json.dumps(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
