""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import sys
import time
import warnings
from pathlib import Path

W48DIR = Path("records/"
              "e2e/w48")
_spec = importlib.util.spec_from_file_location("w48_pipelined_ro_w0", W48DIR / "w48_pipelined.py")
WP = importlib.util.module_from_spec(_spec)
sys.modules["w48_pipelined_ro_w0"] = WP
_spec.loader.exec_module(WP)
W48 = WP.W48

CHIP = "study_compass"
MEMBER = "R18-IN224"
SOURCE_MEMBER = "C1"
WIDE = 1 << 17
SYNC = {"sync100": 100.0, "sync0": 0.0}


def arms_of(budget: int):
    one = f"B5m@{budget}"
    pipe = f"{one}+pipelined"
    return one, pipe, f"{pipe}+buffer1MiB"


class W0Toolchain(WP.PipelinedToolchain):
    count: int = 0

    def network(self, depth, width):
        import torch
        import torchvision
        from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
        from mimarsinan.mapping.verification.onchip_fraction import estimate_onchip_fractions
        from mimarsinan.torch_mapping.converter import convert_torch_model
        shape, classes = (3, 224, 224), 1000
        torch.manual_seed(0)
        model = torchvision.models.resnet18(num_classes=classes).eval()
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("ignore")
            flow = convert_torch_model(model, shape, classes, packaging=MVM_PACKAGING, strict=False).eval()
            params, macs, ops = estimate_onchip_fractions(flow, shape, classes, metrics=("params", "macs", "ops"))
        return self.LD.LadderNetwork(
            name="torchvision resnet18 224x224 1000", mapper_repr=flow.get_mapper_repr(), resolution=shape,
            total_params=float(sum(p.numel() for p in flow.parameters())),
            host_census={"host_macs": macs.host, "onchip_macs": macs.onchip, "compute_op_count": ops.host,
                         "host_params": params.host, "onchip_params": params.onchip},
            conversion=self.LD.certificate_record(getattr(flow, "conversion_certificate", None)))

    def platform(self, config_path, *, cores=None, **kw):
        if cores is None and self.count:
            shapes = self.__dict__.setdefault("_core_shape", {})
            if str(config_path) not in shapes:
                base = super().platform(config_path)["cores"]
                if len(base) != 1:
                    raise SystemExit(f"REFUSED: {config_path} declares {len(base)} core types, not one")
                shapes[str(config_path)] = {k: v for k, v in base[0].items() if k != "count"}
            cores = [{**shapes[str(config_path)], "count": int(self.count)}]
        return super().platform(config_path, cores=cores, **kw)

    def part_block(self, part, chip):
        if part in SYNC:
            v = SYNC[part]
            return "barrier_sensitivity", {"t_sync_barrier": {
                "low": v, "nominal": v, "high": v, "unit": "ns", "evidence_kind": "estimated",
                "note": f"cut:thru:w0 sensitivity: the barrier per pass stated as a point at {v:g} ns"}}
        return super().part_block(part, chip)


def members_for(group: str):
    src = WP.with_pipelined_arm(W48.route_members())[SOURCE_MEMBER]
    arms, declare = {}, {}
    row_b5m, row_b5u = src["arms"]["B5m"], src["arms"]["B5u"]
    if group == "A":
        arms["B5u"], declare["B5u"] = row_b5u, None
    for budget in (W48.ONE_COPY_BUDGET, WIDE):
        one, pipe, buf = arms_of(budget)
        if group == "A":
            arms[one], declare[one] = row_b5m, {"max_schedule_passes": budget}
        else:
            arms[pipe], declare[pipe] = row_b5m, {"max_schedule_passes": budget, "pipelined_multiplexing": True}
            arms[buf], declare[buf] = row_b5m, {**declare[pipe], **WP.buffer_declaration()[0]}
    return {MEMBER: {**{k: v for k, v in src.items() if k not in ("arms", "declare", "not_run")},
                     "arms": arms, "declare": declare}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worktree", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--cores", required=True, type=int)
    ap.add_argument("--group", required=True, choices=("A", "B"))
    ap.add_argument("--ks", default="1,2,4,8,16")
    ap.add_argument("--point-budget-s", type=float, default=3600.0)
    args = ap.parse_args(argv)
    start = time.time()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.out.with_suffix(".log")

    def log(line: str) -> None:
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(f"{time.time() - start:8.1f} {line}\n")
        print(line, flush=True)

    ks = tuple(int(k) for k in args.ks.split(","))
    variants = ("study:shipped",) if args.group == "A" else ("study:shipped", "study:sync100", "study:sync0")
    tc = W0Toolchain(args.worktree)
    tc.count = args.cores
    members = members_for(args.group)
    arms = tuple(members[MEMBER]["arms"])
    log(json.dumps({"start": True, "cores": args.cores, "group": args.group, "arms": arms, "ks": ks,
                    "variants": variants}))
    chips = {MEMBER: CHIP}
    priced = W48.price_members(tc, members, [MEMBER], arms, ks, ("candidate",), variants, W48.DEFAULT_RECORD_ROOT,
                               args.point_budget_s, log, chips=chips)
    rules = {}
    if args.group == "B":
        for budget in (W48.ONE_COPY_BUDGET, WIDE):
            _one, pipe, buf = arms_of(budget)
            try:
                rule = WP.buffer_size_rule(WP.live_peaks(tc, priced, members, pipe))
                WP.check_capacity(rule["capacity"], {CHIP: tc.carry_buffer_capacities(CHIP)})
            except SystemExit as refused:
                rules[str(budget)] = {"refused": str(refused)}
                log(json.dumps({"size_rule": budget, "refused": str(refused)[:300]}))
                continue
            label = WP.sized_arm(rule["capacity"]).split("+buffer")[-1]
            sized = f"{pipe}+buffer{label}"
            rule["arm"] = sized
            rules[str(budget)] = rule
            if sized == buf:
                continue
            m = members[MEMBER]
            m["arms"][sized] = m["arms"][pipe]
            m["declare"][sized] = {**m["declare"][pipe], "carry_buffer_bytes": rule["capacity"]}
            more = W48.price_members(tc, members, [MEMBER], (sized,), ks, ("candidate",), variants,
                                     W48.DEFAULT_RECORD_ROOT, args.point_budget_s, log, chips=chips)
            priced[MEMBER]["arms"].update(more[MEMBER]["arms"])
    result = {"format": "cut-thru-w0/1", "script_sha256": W48.sha256(Path(__file__).resolve()),
              "w48_pipelined_sha256": W48.sha256(W48DIR / "w48_pipelined.py"),
              "w48_pairs_sha256": W48.sha256(W48DIR / "w48_pairs.py"), "worktree": str(args.worktree),
              "chip": CHIP, "chip_profile_sha256": W48.sha256(args.worktree / WP.PROFILES / f"{CHIP}.json"),
              "cores": args.cores, "group": args.group, "ks": list(ks), "variants": list(variants),
              "source_member": SOURCE_MEMBER, "size_rules": rules, "members": priced,
              "wall_s": round(time.time() - start, 1)}
    args.out.write_text(json.dumps(result, indent=1, sort_keys=True, default=str) + "\n")
    log(json.dumps({"done": True, "wall_s": result["wall_s"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
