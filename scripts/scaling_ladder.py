#!/usr/bin/env python
"""The shape-only scaling ladder: roster x targets x core budgets x K through the toolchain's planes.

Each network is built by its registered builder, converted by ``convert_torch_model``
under the target's packaging contract (value probes run, ``strict`` off, so every row
prints the conversion certificate, certified or not), laid out once per core
geometry, packed and scheduled by the builder's planner at every core budget, and
priced by the absolute pricer; ``deployment_record.introspection.scaling_ladder``
turns each priced program into law rows (x_fluid, x_packed, the capacity and schedule
readings, replication, Pi, Pi/K, regime per band corner, M, K*). Budgets are
fractions of the network's packed tile count F on the target geometry, plus the
target's declared core count. The regime thresholds below are ESTIMATE-derived
reading constants of this script, never framework constants.

Rows are written as they complete (a cell stopped at its wall or memory cap keeps
every completed point), and each point runs under a declared per-point wall budget:
a point whose stage (convert, layout, pack, schedule or price) runs past it leaves
one row per K naming the stage and the elapsed wall, never a dropped point.

    python scripts/scaling_ladder.py --out RUN.jsonl [--smoke] [--networks resnet20,...]
        [--targets truenorth,...] [--budgets 1,0.5] [--ks 1,64] [--resolution 32]
        [--point-budget-s 3600] [--declare allow_scheduling=false]
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import signal
import sys
import time
import warnings
from typing import Any, Callable, Dict, Iterator, Optional, Sequence, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402

from mimarsinan.common.refusal import ToolchainRefusal  # noqa: E402
from mimarsinan.deployment_record.introspection.scaling_ladder import (  # noqa: E402
    LadderNetwork,
    certificate_record,
    ladder_row,
    packed_tiles,
    price_program,
    shape_census,
)
from mimarsinan.deployment_record.introspection.scaling_ladder_law import RegimeReading  # noqa: E402
from mimarsinan.mapping.platform.imc_platforms import get_imc_platform  # noqa: E402
from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING, SPIKING_PACKAGING  # noqa: E402
from mimarsinan.mapping.support.schedule.pass_cut import transfer_for  # noqa: E402
from mimarsinan.mapping.verification.onchip_fraction import estimate_onchip_fractions  # noqa: E402
from mimarsinan.models.builders import BUILDERS_REGISTRY  # noqa: E402
from mimarsinan.pipelining.core.platform_constraints_resolver import (  # noqa: E402
    build_platform_constraints_resolved,
)
from mimarsinan.search.problems.joint.candidate_fragments import (  # noqa: E402
    StageSemantics,
    candidate_program_facts,
    with_stage_placements,
)
from mimarsinan.torch_mapping.converter import convert_torch_model  # noqa: E402

PREACT_BOTTLENECK = {"block_kind": "bottleneck", "block_order": "pre_activation"}
#: name -> (registered builder, model config, classes); a member without a registered builder is listed absent.
ROSTER: Dict[str, Optional[tuple]] = {
    "resnet20": ("cifar_resnet20", {"blocks_per_stage": 3}, 10),
    "resnet56": ("cifar_resnet20", {"blocks_per_stage": 9}, 10),
    "resnet110": ("cifar_resnet20", {"blocks_per_stage": 18}, 10),
    "resnet164_preact": ("cifar_resnet_family", {"blocks_per_stage": 18, **PREACT_BOTTLENECK}, 10),
    "resnet1001_preact": ("cifar_resnet_family", {"blocks_per_stage": 111, **PREACT_BOTTLENECK}, 10),
    "vgg11": None,
    "vgg16_bn": ("torch_vgg16", {}, 10),
    "densenet_bc100": ("cifar_densenet_bc", {"depth": 100, "growth_rate": 12}, 10),
    "vit_leaf": ("cifar_vit_leaf", {}, 10),
}
#: G8's targets (CL5: G4's bundle names): physics profile, registered geometry, packaging contract.
TARGETS = {
    "truenorth": ("truenorth", "truenorth_like", "spiking"),
    "loihi": ("loihi", "loihi_dense_equiv_128x1024", "spiking"),
    "isaac_like": ("isaac_like", "isaac_like", "value"),
    "generic_22nm_on_imc_256x256": ("generic_estimated_22nm", "imc_256x256", "value"),
}
PACKAGINGS = {"spiking": SPIKING_PACKAGING, "value": MVM_PACKAGING}
BUDGET_FRACTIONS = (1.0, 2 ** -0.25, 2 ** -0.5, 2 ** -0.75, 0.5, 0.25, 0.125, 0.0625)
KS = (1, 16, 64)
RATE_HZ, LIFETIME_HOURS = 30.0, 43_800.0  # r = 30 inferences/s, T = 5 years: declared assumptions
READING = RegimeReading(
    capacity_free_max=0.099, resident_dominant_min=62.9,
    source="lens_arch.md:109-112 (law.md 3.5, v3 thresholds; ESTIMATE, law_check.py section E)",
)
#: The smoke subset LADDER-1/2 executed: 3 networks x 2 targets x 3 budgets x 2 K.
SMOKE = {"networks": ("vit_leaf", "resnet20", "vgg16_bn"), "targets": ("truenorth", "generic_22nm_on_imc_256x256"),
         "budgets": (1.0, 0.5, 0.25), "ks": (1, 64)}
#: The declared per-point wall budget (``--point-budget-s`` replaces it; 0 declares none).
POINT_BUDGET_S = 3600.0
#: The chip capabilities this script declares on every target (``--declare KEY=VALUE`` replaces one).
CAPABILITIES = {"allow_scheduling": True, "max_schedule_passes": 4096, "allow_coalescing": True,
                "allow_neuron_splitting": True}


class PointAboveBudget(BaseException):
    """A stage of one ladder point ran past the declared per-point wall budget.

    A ``BaseException``, so no ``except Exception`` on the toolchain's path (the mapper forward's re-wrap, the
    search plane's guards) can swallow the signal or turn it into a refusal."""

    def __init__(self, stage: str, elapsed_s: float, budget_s: float):
        super().__init__(f"{stage} ran {elapsed_s:.1f} s, past the {budget_s:.1f} s per-point budget")
        self.stage, self.elapsed_s, self.budget_s = stage, elapsed_s, budget_s


class StageClock:
    """The stage a point is in and when the point started."""

    def __init__(self) -> None:
        self.stage, self.start = "start", time.time()

    def enter(self, stage: str) -> None:
        self.stage = stage


@contextlib.contextmanager
def point_budget(seconds: Optional[float], clock: StageClock) -> Iterator[StageClock]:
    """Run the block under a wall budget (SIGALRM); the previous handler and any armed timer are restored."""
    if not seconds:
        yield clock
        return

    def expire(signum, frame):
        raise PointAboveBudget(clock.stage, time.time() - clock.start, float(seconds))

    previous = signal.signal(signal.SIGALRM, expire)
    outer, _ = signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield clock
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)
        if outer > 0.0:
            signal.setitimer(signal.ITIMER_REAL, max(outer - (time.time() - clock.start), 1e-3))


def attempt(seconds: Optional[float], work: Callable[[StageClock], Any]) -> Tuple[Any, Optional[Dict[str, Any]]]:
    """``work(clock)`` under the per-point budget: (its result, None), or (None, the mark that names the point's
    stage: above budget with the elapsed wall, or refused by name)."""
    clock = StageClock()
    try:
        with point_budget(seconds, clock):
            return work(clock), None
    except PointAboveBudget as over:
        return None, {"above_budget": {"stage": over.stage, "elapsed_s": round(over.elapsed_s, 3),
                                       "point_budget_s": over.budget_s}}
    except ToolchainRefusal as refused:
        reason = str(refused).splitlines()[0][:400] if str(refused) else ""
        return None, {"refusal": {"name": type(refused).__name__, "stage": clock.stage, "reason": reason}}


def marked_rows(name: str, target: str, budget, ks: Sequence[int], mark: Dict[str, Any],
                 known: Optional[Dict[str, Any]] = None) -> Iterator[Dict[str, Any]]:
    """One row per K for a point recorded by name; ``known`` carries what the point had resolved (cores, tiles)."""
    for k in ks:
        yield {"network": name, "target": target, "budget": budget, "inference_batch_K": int(k), **(known or {}),
               **mark}


def program_facts(softcores, noc, platform):
    """The candidate plane's census, exactly as the joint search's layout hook computes it."""
    semantics = StageSemantics()
    noc = with_stage_placements(softcores, noc, semantics)
    return noc, candidate_program_facts(
        softcores, noc, semantics, timesteps=platform.get("simulation_steps"),
        weight_bits=platform.get("weight_bits"), pass_transfer=transfer_for(streamed=False),
    )


def resolved_platform(target: str, fraction_or_declared, packed: Optional[int],
                      declared: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    profile, geometry, _ = TARGETS[target]
    chip = get_imc_platform(geometry)
    cores = []
    for ct in chip.cores:
        count = int(ct["count"]) if fraction_or_declared == "declared" else max(
            1, math.floor(float(fraction_or_declared) * int(packed or 0)))
        cores.append({**ct, "count": count})
    return build_platform_constraints_resolved({
        **CAPABILITIES, **(declared or {}), "cores": cores,
        "weight_bits": chip.weight_bits, "platform_physics_profile": profile,
    })


def build_network(name: str, resolution: int, packaging: str) -> LadderNetwork:
    builder_key, config, classes = ROSTER[name]  # type: ignore[misc]
    shape = (3, resolution, resolution)
    torch.manual_seed(0)
    model = BUILDERS_REGISTRY[builder_key]("cpu", shape, classes, {}).build(config).eval()
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        flow = convert_torch_model(model, shape, classes, packaging=PACKAGINGS[packaging], strict=False).eval()
        params, macs, ops = estimate_onchip_fractions(flow, shape, classes, metrics=("params", "macs", "ops"))
    return LadderNetwork(
        name=name, mapper_repr=flow.get_mapper_repr(), resolution=shape,
        total_params=float(sum(p.numel() for p in flow.parameters())),
        host_census={"host_macs": macs.host, "onchip_macs": macs.onchip, "compute_op_count": ops.host,
                     "host_params": params.host, "onchip_params": params.onchip},
        conversion=certificate_record(getattr(flow, "conversion_certificate", None)),
    )


def cell_rows(network: LadderNetwork, target: str, packaging: str, chip: Dict[str, Any],
              platform_at: Callable[[Any, int], Dict[str, Any]], *, budgets: Sequence, ks: Sequence[int],
              point_budget_s: Optional[float], wire_census: bool) -> Iterator[Dict[str, Any]]:
    """Rows of one (network, target) cell: one shape-only layout and pack, then every budget point."""

    def layout(clock: StageClock):
        clock.enter("layout")
        census = shape_census(network, chip, wire_census=wire_census)
        clock.enter("pack")
        return census, packed_tiles(census, chip)

    shared, mark = attempt(point_budget_s, layout)
    for budget in budgets:
        if shared is None:
            yield from marked_rows(network.name, target, budget, ks, mark or {})
            continue
        census, packed = shared
        platform = platform_at(budget, packed)
        priced, point_mark = attempt(point_budget_s, lambda clock: price_program(
            network, census, platform, program_facts, tiles=packed, on_stage=clock.enter))
        if priced is None:
            known = {"cores": sum(int(c.get("count", 1)) for c in platform["cores"]), "packed_tiles": packed}
            yield from marked_rows(network.name, target, budget, ks, point_mark or {}, known)
            continue
        for k in ks:
            point = {**platform, "inference_batch_K": int(k), "inference_rate_hz": RATE_HZ,
                     "min_lifetime_hours": LIFETIME_HOURS}
            row = ladder_row(network, census, priced, point, READING)
            row.update(target=target, packaging=packaging, budget=budget, point_budget_s=point_budget_s)
            yield row


def run(networks: Sequence[str], targets: Sequence[str], budgets: Sequence, ks: Sequence[int],
        resolution: int, wire_census: bool, declared: Optional[Dict[str, Any]] = None,
        point_budget_s: Optional[float] = POINT_BUDGET_S) -> Iterator[Dict[str, Any]]:
    for name in networks:
        if ROSTER[name] is None:
            yield {"network": name, "absent": "no registered builder"}
            continue
        built: Dict[str, Tuple[Optional[LadderNetwork], Optional[Dict[str, Any]]]] = {}
        for target in targets:
            packaging = TARGETS[target][2]

            def convert(clock: StageClock, name=name, packaging=packaging) -> LadderNetwork:
                clock.enter("convert")
                return build_network(name, resolution, packaging)

            if packaging not in built:
                built[packaging] = attempt(point_budget_s, convert)
            network, mark = built[packaging]
            if network is None:
                for budget in budgets:
                    yield from marked_rows(name, target, budget, ks, mark or {})
                continue
            chip = resolved_platform(target, "declared", None, declared)
            yield from cell_rows(
                network, target, packaging, chip,
                lambda budget, packed, target=target: resolved_platform(target, budget, packed, declared),
                budgets=budgets, ks=ks, point_budget_s=point_budget_s, wire_census=wire_census)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--networks")
    parser.add_argument("--targets")
    parser.add_argument("--budgets", help="comma-separated fractions of F, or 'declared'")
    parser.add_argument("--ks")
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--no-wire-census", action="store_true")
    parser.add_argument("--point-budget-s", type=float, default=POINT_BUDGET_S,
                        help="declared per-point wall budget in seconds (0: none)")
    parser.add_argument("--declare", action="append", default=[], metavar="KEY=VALUE",
                        help="replace one declared capability, e.g. allow_scheduling=false")
    args = parser.parse_args(argv)
    chosen = dict(SMOKE) if args.smoke else {"networks": tuple(ROSTER), "targets": tuple(TARGETS),
                                             "budgets": BUDGET_FRACTIONS + ("declared",), "ks": KS}
    if args.networks:
        chosen["networks"] = tuple(args.networks.split(","))
    if args.targets:
        chosen["targets"] = tuple(args.targets.split(","))
    if args.budgets:
        chosen["budgets"] = tuple(b if b == "declared" else float(b) for b in args.budgets.split(","))
    if args.ks:
        chosen["ks"] = tuple(int(k) for k in args.ks.split(","))
    declared = {k: (v.lower() == "true") if v.lower() in ("true", "false") else int(v)
                for k, v in (item.split("=", 1) for item in args.declare)}
    start = time.time()
    with open(os.path.abspath(__file__), "rb") as handle:
        script_sha = hashlib.sha256(handle.read()).hexdigest()[:16]
    header = {"header": True, "script_sha256_16": script_sha, "selection": {k: list(v) for k, v in chosen.items()},
              "resolution": args.resolution, "capabilities": {**CAPABILITIES, **declared},
              "wire_census": not args.no_wire_census, "assumptions": {"inference_rate_hz": RATE_HZ,
                                                             "min_lifetime_hours": LIFETIME_HOURS},
              "reading": READING.__dict__, "point_budget_s": args.point_budget_s or None}
    count = 0
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(header, sort_keys=True, default=str) + "\n")
        handle.flush()
        for row in run(chosen["networks"], chosen["targets"], chosen["budgets"], chosen["ks"], args.resolution,
                       wire_census=not args.no_wire_census, declared=declared,
                       point_budget_s=args.point_budget_s or None):
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
            handle.flush()
            count += 1
            print(json.dumps({k: row.get(k) for k in ("network", "target", "budget", "cores", "inference_batch_K",
                                                       "x_packed", "passes", "refusal", "above_budget")},
                             default=str), flush=True)
        handle.write(json.dumps({"trailer": True, "rows": count, "run_wall_s": time.time() - start}) + "\n")
    print(f"wrote {count} rows to {args.out} in {time.time() - start:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
