"""The persistent research SCHEDULER — keeps the GPU queue full from a declarative
backlog, autonomously, so GPUs never idle waiting for a human to refill.

This is the structural fix for the stall: refill must NOT depend on an event-driven
operator. The scheduler runs forever, reloading the backlog each loop (so a research
workflow can APPEND new experiment batches), instantiating each batch's config-grid,
and enqueuing into the shared `GpuQueue` whenever pending drops below a high-watermark.
It dedupes against everything already enqueued/run (done/failed/pending/running), so a
batch is never re-run. Kill-gates: a batch with "enabled": false is skipped until a
workflow/operator flips it on.

Layers: scheduler FILLS the queue (this) · runner DRAINS it (campaign_runner) ·
research workflows DECIDE the science (append batches to the backlog + analyze the
ledger). GPUs stay busy as long as the backlog has enabled work.

Backlog JSON (runs/campaign/backlog.json): a list of batch specs:
  {"id","template","base":{dotted_path:val,...},"grid":{dotted_path:[vals],...},
   "id_template":"...{dotted_path}...","priority":int,"mode":"fit|free","need_mb":int,
   "timeout_s":int,"tags":{...},"enabled":true}

Run:  python scripts/campaign/scheduler.py [--hi 24] [--poll 20]
Stop: touch runs/campaign/q/SCHED_STOP  (or --stop)
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import sys
import time
from typing import Dict, Iterator, List, Tuple

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "scripts", "gpu"))
from gpu_queue import GpuQueue, campaign_dir  # noqa: E402

BACKLOG = os.path.join(campaign_dir(), "backlog.json")
SCHED_STATUS = os.path.join(campaign_dir(), "sched_status.json")
CFG_DIR = os.path.join(REPO, "experiments", "campaign")


def set_path(d: dict, dotted: str, val) -> None:
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = val


def get_path(d: dict, dotted: str):
    for k in dotted.split("."):
        d = d[k]
    return d


def _fmt_frac(value) -> str:
    return f"{value:.2%}" if isinstance(value, (int, float)) else str(value)


def existing_ids(q: GpuQueue) -> set:
    """Every job id already enqueued or finalized (pending/running/done/failed).

    The dedupe key for both the scheduler (don't re-enqueue) and the director
    (count un-enqueued work). One source of truth so the two daemons agree.
    """
    ids = set()
    for st in ("pending", "running", "done", "failed"):
        for name in os.listdir(q._dir(st)):
            if not name.endswith(".json"):
                continue
            try:
                ids.add(json.load(open(os.path.join(q._dir(st), name)))["id"])
            except (OSError, ValueError, KeyError):
                pass
    return ids


def _classify_cfg_validity(cfg: dict, *, floor: float, majority: float):
    """Build the job's model from its config and run the TIERED validity classifier.

    Pure-static (no GPU, no run): resolves model_type / model_config from
    ``deployment_parameters`` and input_shape / num_classes from the data provider,
    builds the native model via the registry, then runs the tiered both-metrics
    ``classify_validity``. Heavy torch imports are deferred to here so importing the
    scheduler stays light for the daemon. Patched in tests to avoid real model
    construction.
    """
    for sub in ("src", "spikingjelly"):
        path = os.path.join(REPO, sub)
        if path not in sys.path:
            sys.path.insert(0, path)
    from mimarsinan.data_handling.data_provider_factory import BasicDataProviderFactory
    from mimarsinan.mapping.verification.onchip_fraction import classify_validity
    from mimarsinan.pipelining.core.registry.model_registry import ModelRegistry

    dp = cfg.get("deployment_parameters", {}) or {}
    model_type = dp["model_type"]
    model_config = dp.get("model_config", {}) or {}
    placement = str(dp.get("encoding_layer_placement", "subsume"))

    meta = BasicDataProviderFactory.get_metadata(
        cfg["data_provider_name"],
        os.path.join(REPO, "datasets"),
    )
    input_shape = tuple(meta["input_shape"])
    num_classes = int(meta["num_classes"])

    builder_cls = ModelRegistry.get_builder_cls(model_type)
    builder = builder_cls("cpu", input_shape, num_classes, dp)
    model = builder.build(model_config)
    return classify_validity(
        model,
        input_shape,
        num_classes,
        encoding_placement=placement,
        floor=floor,
        majority=majority,
    )


def _estimate_cfg_capacity(cfg: dict):
    """Statically estimate the hard cores this job's IR needs vs. its core budget.

    Pure-static (no GPU, no run, no placement): builds the native model, converts
    it to an IR graph exactly as ``SoftCoreMappingStep`` would, then runs the SOUND
    ``estimate_cores_needed`` lower bound against the declared core budget. Heavy
    torch / mapping imports are deferred here so the daemon import stays light;
    patched in tests to avoid real IR construction.
    """
    for sub in ("src", "spikingjelly"):
        path = os.path.join(REPO, sub)
        if path not in sys.path:
            sys.path.insert(0, path)
    import numpy as np

    from mimarsinan.config_schema.runtime import build_flat_pipeline_config
    from mimarsinan.data_handling.data_provider_factory import BasicDataProviderFactory
    from mimarsinan.mapping.ir_mapping_class import IRMapping
    from mimarsinan.mapping.platform.platform_constraints import (
        resolve_platform_mapping_params,
    )
    from mimarsinan.mapping.support.per_source_scales import compute_per_source_scales
    from mimarsinan.mapping.verification.capacity import estimate_cores_needed
    from mimarsinan.pipelining.core.platform_constraints_resolver import (
        build_platform_constraints_resolved,
    )
    from mimarsinan.pipelining.core.registry.model_registry import ModelRegistry
    from mimarsinan.torch_mapping import convert_torch_model
    from mimarsinan.transformations.quantization_bounds import quantization_bounds

    dp = cfg.get("deployment_parameters", {}) or {}
    pc = cfg.get("platform_constraints", {}) or {}
    flat = build_flat_pipeline_config(
        dp, pc, pipeline_mode=cfg.get("pipeline_mode", "vanilla"),
    )
    model_type = dp["model_type"]
    model_config = dp.get("model_config", {}) or {}

    meta = BasicDataProviderFactory.get_metadata(
        cfg["data_provider_name"], os.path.join(REPO, "datasets"),
    )
    input_shape = tuple(meta["input_shape"])
    num_classes = int(meta["num_classes"])

    platform_constraints = build_platform_constraints_resolved(flat)
    params = resolve_platform_mapping_params(
        platform_constraints["cores"],
        allow_coalescing=bool(platform_constraints.get("allow_coalescing", False)),
    )

    builder_cls = ModelRegistry.get_builder_cls(model_type)
    builder = builder_cls("cpu", input_shape, num_classes, flat)
    flow = convert_torch_model(
        builder.build(model_config), input_shape, num_classes,
        device="cpu", strict=True,
    ).eval()
    _, q_max = quantization_bounds(flat["weight_bits"])
    mapper_repr = flow.get_mapper_repr()
    if hasattr(mapper_repr, "assign_perceptron_indices"):
        mapper_repr.assign_perceptron_indices()
    compute_per_source_scales(mapper_repr)
    ir_mapping = IRMapping(
        q_max=q_max, firing_mode=flat["firing_mode"],
        max_axons=params.effective_max_axons,
        max_neurons=params.effective_max_neurons,
        allow_coalescing=params.allow_coalescing, hardware_bias=params.hardware_bias,
    )
    ir_graph = ir_mapping.map(mapper_repr)
    return estimate_cores_needed(ir_graph, platform_constraints)


def capacity_precheck(cfg: dict) -> Tuple[bool, Dict]:
    """Decide whether a job may be enqueued under the STATIC placement-capacity gate.

    Returns ``(enqueue_ok, info)``. Rejects (``False`` — no GPU claimed) a config
    whose SOUND ``estimate_cores_needed`` lower bound EXCEEDS its declared core
    budget (the net cannot be placed by any packing strategy — VGG16@224 on a
    1000-core budget); admits a feasible one. Opt-out via
    ``deployment_parameters.capacity_gate=False``. Any model-build / IR / estimate
    failure is NON-FATAL — returns ``(True, {...})`` so a builder edge case never
    silently drops valid work.
    """
    dp = cfg.get("deployment_parameters", {}) or {}
    if not bool(dp.get("capacity_gate", True)):
        return True, {"reason": "gate_disabled"}
    try:
        est = _estimate_cfg_capacity(cfg)
    except Exception as exc:  # noqa: BLE001 - non-fatal by design
        return True, {"reason": "estimate_failed", "error": repr(exc)}
    info = {
        "cores_needed": int(est.cores_needed),
        "cores_available": int(est.cores_available),
        "feasible": bool(est.feasible),
        "overflowing_segment": est.overflowing_segment,
    }
    if not est.feasible:
        return False, dict(info, reason="capacity_exceeded")
    return True, dict(info, reason="ok")


def onchip_precheck(cfg: dict) -> Tuple[bool, Dict]:
    """Decide whether a job may be enqueued under the TIERED validity gate (v2).

    Returns ``(enqueue_ok, info)``. Rejects (``False`` — no GPU claimed) only an
    INVALID job: ``min(param_frac, mac_frac) < floor`` (the host does ~everything).
    Admits (``True``) VALID and VALID_FLAGGED; a flagged job carries its named
    ``research_gap_ops`` / ``placement_fixable_ops`` so the log records the
    transferable-tuning gap. The gate is opt-out via
    ``deployment_parameters.onchip_majority_gate=False``; the floor is
    ``deployment_parameters.onchip_min_fraction`` (default 0.20) and the majority is
    ``deployment_parameters.onchip_majority_fraction`` (default 0.50). Any
    model-build / classification failure is NON-FATAL — returns ``(True, {...})`` so
    a builder edge case never silently drops valid work.
    """
    dp = cfg.get("deployment_parameters", {}) or {}
    if not bool(dp.get("onchip_majority_gate", True)):
        return True, {"reason": "gate_disabled"}
    floor = float(dp.get("onchip_min_fraction", 0.20))
    majority = float(dp.get("onchip_majority_fraction", 0.50))
    try:
        verdict = _classify_cfg_validity(cfg, floor=floor, majority=majority)
    except Exception as exc:  # noqa: BLE001 - non-fatal by design
        return True, {"reason": "estimate_failed", "error": repr(exc)}
    info = {
        "tier": verdict.tier,
        "param_frac": float(verdict.param_frac),
        "mac_frac": float(verdict.mac_frac),
        "floor": floor,
        "majority": majority,
        "research_gap_ops": list(verdict.research_gap_ops),
        "placement_fixable_ops": list(verdict.placement_fixable_ops),
    }
    if verdict.tier == "INVALID":
        return False, dict(info, reason="invalid_host_majority")
    if verdict.tier == "VALID_FLAGGED":
        return True, dict(info, reason="valid_flagged")
    return True, dict(info, reason="ok")


def instantiate(batch: dict) -> Iterator[Tuple[str, dict]]:
    """Yield (job_id, config) for each point of the batch's grid."""
    template = json.load(open(os.path.join(REPO, batch["template"])))
    grid = batch.get("grid", {})
    keys = list(grid)
    combos = list(itertools.product(*[grid[k] for k in keys])) or [()]
    for combo in combos:
        cfg = copy.deepcopy(template)
        for path, val in (batch.get("base") or {}).items():
            set_path(cfg, path, val)
        point = dict(zip(keys, combo))
        for path, val in point.items():
            set_path(cfg, path, val)
        # id from id_template using the grid point's leaf values
        leaves = {p.split(".")[-1]: v for p, v in point.items()}
        jid = batch["id_template"].format(**leaves)
        cfg["experiment_name"] = jid
        yield jid, cfg


class Scheduler:
    def __init__(self, q: GpuQueue, *, hi: int = 24, poll: float = 20.0,
                 backlog_path: str = BACKLOG):
        self.q = q
        self.hi = hi
        self.poll = poll
        self.backlog_path = backlog_path
        self.stop_path = os.path.join(q.root, "SCHED_STOP")
        os.makedirs(CFG_DIR, exist_ok=True)

    def _existing_ids(self) -> set:
        return existing_ids(self.q)

    def _load_backlog(self) -> List[dict]:
        try:
            bl = json.load(open(self.backlog_path))
        except (OSError, ValueError):
            return []
        return sorted([b for b in bl if b.get("enabled", True)],
                      key=lambda b: b.get("priority", 50))

    def refill(self) -> int:
        pending = self.q.counts()["pending"]
        if pending >= self.hi:
            return 0
        existing = self._existing_ids()
        added = 0
        for batch in self._load_backlog():
            try:
                jobs = list(instantiate(batch))
            except Exception as exc:  # noqa: BLE001 - one bad batch must not kill the daemon
                print(f"scheduler: SKIP batch {batch.get('id')!r} — malformed "
                      f"(instantiate failed: {exc!r})")
                continue
            for jid, cfg in jobs:
                if pending >= self.hi:
                    return added
                if jid in existing:
                    continue
                ok, info = onchip_precheck(cfg)
                if not ok:
                    print(
                        f"scheduler: SKIP {jid} (INVALID): min(param,mac) on-chip "
                        f"({_fmt_frac(info.get('param_frac'))},"
                        f"{_fmt_frac(info.get('mac_frac'))}) < floor "
                        f"{_fmt_frac(info.get('floor'))} — not enqueued, no GPU claimed."
                    )
                    existing.add(jid)
                    continue
                cap_ok, cap_info = capacity_precheck(cfg)
                if not cap_ok:
                    print(
                        f"scheduler: SKIP {jid} (INFEASIBLE): needs "
                        f">= {cap_info.get('cores_needed')} hard cores but budget is "
                        f"{cap_info.get('cores_available')} (overflowing segment "
                        f"{cap_info.get('overflowing_segment')!r}) — not enqueued, "
                        "no GPU claimed."
                    )
                    existing.add(jid)
                    continue
                if info.get("reason") == "valid_flagged":
                    print(
                        f"scheduler: FLAGGED {jid}: param "
                        f"{_fmt_frac(info.get('param_frac'))} / mac "
                        f"{_fmt_frac(info.get('mac_frac'))} on-chip — VALID_FLAGGED "
                        f"(research_gap={info.get('research_gap_ops')}, "
                        f"placement_fixable={info.get('placement_fixable_ops')}); "
                        "enqueued as transferable-tuning evidence."
                    )
                path = os.path.join(CFG_DIR, jid + ".json")
                with open(path, "w") as fh:
                    json.dump(cfg, fh, indent=2)
                pmode = cfg.get("pipeline_mode", "phased")
                self.q.enqueue({
                    "id": jid, "mode": batch.get("mode", "fit"),
                    "need_mb": batch.get("need_mb", 8000),
                    "priority": batch.get("priority", 50),
                    "timeout_s": batch.get("timeout_s", 2400),
                    "cmd": ["python", "run.py", "--headless", os.path.relpath(path, REPO)],
                    "cwd": REPO,
                    "expect_artifact": f"generated/{jid}_{pmode}_deployment_run/__target_metric.json",
                    "tags": dict(batch.get("tags", {}), batch_id=batch["id"]),
                })
                existing.add(jid)
                pending += 1
                added += 1
        return added

    def _status(self, added: int) -> None:
        c = self.q.counts()
        bl = self._load_backlog()
        tmp = SCHED_STATUS + ".tmp"
        with open(tmp, "w") as fh:
            json.dump({"ts": time.time(), "queue": c, "last_refill_added": added,
                       "enabled_batches": [b["id"] for b in bl]}, fh, indent=2)
        os.replace(tmp, SCHED_STATUS)

    def run(self):
        print(f"scheduler: filling {self.q.root} to hi={self.hi} from {self.backlog_path}")
        while not os.path.exists(self.stop_path):
            added = self.refill()
            self._status(added)
            time.sleep(self.poll)
        print("scheduler: stopped.")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hi", type=int, default=24)
    p.add_argument("--poll", type=float, default=20.0)
    p.add_argument("--stop", action="store_true")
    args = p.parse_args(argv)
    q = GpuQueue()
    if args.stop:
        open(os.path.join(q.root, "SCHED_STOP"), "w").close()
        print("scheduler stop requested.")
        return 0
    # clear a stale stop sentinel
    sp = os.path.join(q.root, "SCHED_STOP")
    if os.path.exists(sp):
        os.remove(sp)
    Scheduler(q, hi=args.hi, poll=args.poll).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
