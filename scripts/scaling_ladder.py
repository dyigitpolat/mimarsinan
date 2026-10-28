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
target's declared core count; a budget c:<count> is an absolute core count (one chip for every
network, the chip space of a chip-first census). The regime thresholds below are ESTIMATE-derived
reading constants of this script, never framework constants.

Rows are written as they complete (a cell stopped at its wall or memory cap keeps
every completed point), and each point runs under a declared per-point wall budget:
a point whose stage (convert, layout, pack, schedule or price) runs past it leaves
one row per K naming the stage and the elapsed wall, never a dropped point.

Targets are the shipped target bundles by name (their geometry, keys and permissions; CL5),
plus the G8 point generic_22nm_on_imc_256x256. A geometry-only bundle (no physics profile)
gives the capacity columns only, its pricing refused by name. ``--variant`` prices one
schedule under several named override-block sets (a host block and a backing-store block,
merged by ``merge_override_blocks``); ``--arms`` schedules each point under U1 (key absent)
and under declared residency plans; ``--x-only`` stops after the layout and the pack on
unbounded cores (x_fluid, x_packed and replication at the declared core count).

    python scripts/scaling_ladder.py --out RUN.jsonl [--smoke] [--networks resnet20,...]
        [--targets truenorth,...] [--budgets 1,0.5] [--ks 1,64] [--resolution 32]
        [--point-budget-s 3600] [--declare allow_scheduling=false] [--core-field write_granularity=cell]
        [--variant LABEL=BLOCK:PATH[,BLOCK:PATH]] [--arms absent,floor,stateless] [--x-only]
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import resource
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
    price_scheduled,
    schedule_program,
    shape_census,
)
from mimarsinan.deployment_record.platform_physics.resolve import merge_override_blocks  # noqa: E402
from mimarsinan.deployment_record.introspection.scaling_ladder_law import RegimeReading  # noqa: E402
from mimarsinan.mapping.platform.imc_platforms import get_imc_platform  # noqa: E402
from mimarsinan.mapping.platform.target_bundles import declared_departures, target_bundles  # noqa: E402
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
CIFAR, MNIST, IMAGENET = (3, 32, 32), (1, 28, 28), (3, 224, 224)
#: name -> (builder, model config, classes[, input shape]); a member without a builder is listed absent. A builder
#: is a registered key, "torchvision:<constructor>" (random weights, no download) or "factory:<module>:<callable>"
#: (a module importable from the repository's src or tests). The shape defaults to CIFAR's; ``--resolution``
#: replaces the side of every image member.
ROSTER: Dict[str, Optional[tuple]] = {
    "lenet5": ("lenet5", {}, 10, MNIST),
    "resnet18_w1q": ("factory:unit.chip_simulation.value_run.integer_programs:resnet18_quarter", {}, 10),
    "resnet20": ("cifar_resnet20", {"blocks_per_stage": 3}, 10),
    "resnet56": ("cifar_resnet20", {"blocks_per_stage": 9}, 10),
    "resnet110": ("cifar_resnet20", {"blocks_per_stage": 18}, 10),
    "resnet164_preact": ("cifar_resnet_family", {"blocks_per_stage": 18, **PREACT_BOTTLENECK}, 10),
    "resnet1001_preact": ("cifar_resnet_family", {"blocks_per_stage": 111, **PREACT_BOTTLENECK}, 10),
    "vgg11": None,
    "vgg16_bn": ("torch_vgg16", {}, 10),
    "densenet_bc100": ("cifar_densenet_bc", {"depth": 100, "growth_rate": 12}, 10),
    "vit_leaf": ("cifar_vit_leaf", {}, 10),
    "vit_lite_cct": ("vit_lite_cct", {}, 10),
    "vit_b16": ("torch_vit", {}, 1000, IMAGENET),
    **{f"tv_{name}": (f"torchvision:{name}", kwargs, 1000, IMAGENET) for name, kwargs in (
        ("resnet18", {}), ("resnet50", {}), ("resnet152", {}), ("wide_resnet50_2", {}), ("vgg11_bn", {}),
        ("vgg16", {}), ("densenet121", {}), ("googlenet", {"aux_logits": False, "init_weights": True}),
        ("inception_v3", {"aux_logits": False, "init_weights": True}), ("squeezenet1_1", {}),
        ("mobilenet_v2", {}), ("efficientnet_b0", {}), ("regnet_x_400mf", {}), ("convnext_tiny", {}),
        ("swin_t", {}))},
}
#: The G8 targets that are not bundles: physics profile, registered geometry, packaging contract.
LEGACY_TARGETS = {"generic_22nm_on_imc_256x256": ("generic_estimated_22nm", "imc_256x256", "value")}
#: Every target: the shipped bundles by name (CL5) and the legacy points.
TARGETS: Dict[str, Any] = {**{name: name for name in target_bundles()}, **LEGACY_TARGETS}
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
#: The chip capabilities this script declares on every target (``--declare KEY=VALUE`` replaces one); a bundle's
#: own keys (its coalescing and neuron-splitting permissions among them) replace these.
CAPABILITIES = {"allow_scheduling": True, "max_schedule_passes": 4096, "allow_coalescing": True,
                "allow_neuron_splitting": True}
#: ``--arms``: the residency declaration of each arm (absent: U1's pass planner, the key undeclared).
ARMS = {"absent": None, "floor": {"step": "floor"}, "stateless": {"step": "stateless"}}


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
    except ValueError as refused:
        # [CL5] A toolchain error class that refuses by name without deriving from ToolchainRefusal is recorded as
        # that refusal, marked so; any other error propagates. ([W12] WideFanInUnsupportedError, this branch's first
        # case, is a ToolchainRefusal now and takes the branch above.)
        if not type(refused).__module__.startswith("mimarsinan."):
            raise
        reason = str(refused).splitlines()[0][:400] if str(refused) else ""
        return None, {"refusal": {"name": type(refused).__name__, "stage": clock.stage, "reason": reason,
                                  "toolchain_refusal": False}}


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


def declared_values(items: Sequence[str]) -> Dict[str, Any]:
    """``--declare KEY=VALUE`` items: true or false in any case a boolean, otherwise the JSON value (an integer, or an
    object such as ``residency_plan={"step": "floor"}``, the residency planner's key [W03]); refuses by name."""
    out: Dict[str, Any] = {}
    for item in items:
        key, value = item.split("=", 1)
        if value.lower() in ("true", "false"):
            out[key] = value.lower() == "true"
            continue
        try:
            out[key] = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"--declare {key}: {value!r} is neither true, false nor a JSON value") from exc
    return out


def target_spec(target: str) -> Dict[str, Any]:
    """A target's geometry and declared keys: {geometry, cores, weight_bits, keys, packaging, bundle}. A target
    "declared:<axons>x<neurons>x<count>:<packaging>[:<profile>]" is a declared geometry of one core type
    (8-bit weights, no on-core bias) under a packaging contract and an optional physics profile."""
    if target.startswith("declared:"):
        _, shape, packaging, *profile = target.split(":")
        axons, neurons, count = (int(v) for v in shape.split("x"))
        return {"geometry": shape, "cores": [{"max_axons": axons, "max_neurons": neurons, "count": count,
                                              "has_bias": False}], "weight_bits": 8,
                "keys": {"platform_physics_profile": profile[0]} if profile else {}, "packaging": packaging,
                "bundle": {"name": target, "claim_class": "declared", "faithfulness": "declared geometry",
                           "physics": bool(profile), "packaging_note": None}}
    if target in LEGACY_TARGETS:
        profile, geometry, packaging = LEGACY_TARGETS[target]
        chip = get_imc_platform(geometry)
        return {"geometry": geometry, "cores": [dict(c) for c in chip.cores], "weight_bits": chip.weight_bits,
                "keys": {"platform_physics_profile": profile}, "packaging": packaging, "bundle": None}
    bundle = target_bundles()[TARGETS[target]]
    values = bundle.values
    cores, bits = values.pop("cores"), values.pop("weight_bits")
    semantics = values.get("core_semantics")
    return {"geometry": bundle.geometry, "cores": cores, "weight_bits": bits, "keys": values,
            "packaging": "spiking" if semantics not in (None, "mvm") else "value",
            "bundle": {"name": bundle.name, "claim_class": bundle.claim_class,
                       "faithfulness": bundle.faithfulness.label(), "physics": "platform_physics_profile" in values,
                       "packaging_note": None if semantics else "no core_semantics declared: MVM packaging read"}}


def departure_labels(target: str, declared: Optional[Dict[str, Any]],
                     core_fields: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """[U-C C-5] The bundle keys and core fields ``--declare`` and ``--core-field`` set to other values than the target
    bundle's, printed by name on every row (``declared_departures``); none on a target that is no bundle."""
    if TARGETS.get(target) not in target_bundles():
        return {}
    departures = declared_departures(TARGETS[target], declared, core_fields)
    return {"declared_departures": list(departures)} if departures else {}


def resolved_platform(target: str, fraction_or_declared, packed: Optional[int],
                      declared: Optional[Dict[str, Any]] = None, core_fields: Optional[Dict[str, Any]] = None,
                      overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The resolved platform of one point: the target's cores at the budget (a fraction of the packed tiles F, or the
    declared count), the script's capabilities, the target's keys, ``--declare`` and the merged override blocks."""
    spec = target_spec(target)
    cores = []
    for ct in spec["cores"]:
        if fraction_or_declared == "declared":
            count = int(ct.get("count", 1))
        elif isinstance(fraction_or_declared, str) and fraction_or_declared.startswith("c:"):
            count = int(fraction_or_declared[2:])  # an absolute core count, the same chip for every network
        else:
            count = max(1, math.floor(float(fraction_or_declared) * int(packed or 0)))
        cores.append({**ct, **(core_fields or {}), "count": count})
    return build_platform_constraints_resolved({
        **CAPABILITIES, **spec["keys"], **(declared or {}), "cores": cores, "weight_bits": spec["weight_bits"],
        **({"platform_physics_overrides": overrides} if overrides else {}),
    })


def _builder(key: str) -> Callable[..., Any]:
    """The model factory of a roster builder: registered, torchvision constructor or an importable callable."""
    kind, _, rest = key.partition(":")
    if kind == "torchvision":
        import torchvision.models as tvm
        return lambda shape, classes, config: getattr(tvm, rest)(weights=None, num_classes=classes, **config)
    if kind == "factory":
        import importlib
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
        module, _, name = rest.partition(":")
        return lambda shape, classes, config: getattr(importlib.import_module(module), name)(**config)
    return lambda shape, classes, config: BUILDERS_REGISTRY[key]("cpu", shape, classes, {}).build(config)


def member_shape(name: str, resolution: Optional[int]) -> Tuple[int, ...]:
    entry = ROSTER[name]
    shape = tuple(entry[3]) if len(entry) > 3 else CIFAR  # type: ignore[arg-type,index]
    return shape if resolution is None or len(shape) != 3 else (shape[0], int(resolution), int(resolution))


def build_network(name: str, resolution: Optional[int], packaging: str) -> LadderNetwork:
    builder_key, config, classes = ROSTER[name][:3]  # type: ignore[index]
    shape = member_shape(name, resolution)
    torch.manual_seed(0)
    model = _builder(builder_key)(shape, classes, config).eval()
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


def x_rows(network: LadderNetwork, target: str, chip: Dict[str, Any], *, point_budget_s: Optional[float],
           wire_census: bool) -> Iterator[Dict[str, Any]]:
    """The capacity columns at the declared core count: one layout and one pack on unbounded cores. A pack the
    toolchain refuses keeps the layout's columns (x_fluid needs the logical cells only) beside the refusal."""

    def layout(clock: StageClock):
        clock.enter("layout")
        return shape_census(network, chip, wire_census=wire_census)

    census, mark = attempt(point_budget_s, layout)
    packed, pack_mark = (None, None)
    if census is not None:
        packed, pack_mark = attempt(point_budget_s, lambda clock: (clock.enter("pack"), packed_tiles(census, chip))[1])
    shared = None if census is None else (census, packed)
    cores = sum(int(c.get("count", 1)) for c in chip["cores"])
    capacity = sum(int(c.get("count", 1)) * int(c["max_axons"]) * int(c["max_neurons"]) for c in chip["cores"])
    row: Dict[str, Any] = {"network": network.name, "target": target, "budget": "declared", "x_only": True,
                           "resolution": list(network.resolution), "cores": cores, "capacity_cells": capacity,
                           "conversion": dict(network.conversion or certificate_record(None)),
                           "host_census": dict(network.host_census or {}), "point_budget_s": point_budget_s}
    if shared is None:
        yield {**row, **(mark or {})}
        return
    census, packed = shared
    onchip = (network.host_census or {}).get("onchip_macs")
    if pack_mark:
        row["pack_refusal"] = pack_mark.get("refusal") or pack_mark.get("above_budget")
    row.update(softcores=len(census.softcores), segments=census.segments, logical_cells=census.logical_cells,
               as_mapped_cells=census.as_mapped_cells, replication=census.as_mapped_cells / census.logical_cells,
               packed_tiles=packed, x_fluid=census.logical_cells / capacity,
               x_packed=None if packed is None else packed / cores,
               macs_per_weight=None if onchip is None else onchip / census.logical_cells,
               layout_wall_s=census.wall_s,
               peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    yield row


def cell_rows(network: LadderNetwork, target: str, packaging: str, chip: Dict[str, Any],
              platform_at: Callable[..., Dict[str, Any]], *, budgets: Sequence, ks: Sequence[int],
              point_budget_s: Optional[float], wire_census: bool,
              variants: Optional[Dict[str, Optional[Dict[str, Any]]]] = None,
              arms: Sequence[str] = ("absent",)) -> Iterator[Dict[str, Any]]:
    """Rows of one (network, target) cell: one shape-only layout and pack, then every budget point, each scheduled
    once per arm and priced once per override-block variant and batch K. ``platform_at(budget, packed)`` gives the
    point's platform; with variants or arms it is called ``platform_at(budget, packed, residency, overrides)``."""
    variants = variants or {"shipped": None}
    plain = list(variants) == ["shipped"] and variants["shipped"] is None and list(arms) == ["absent"]

    def layout(clock: StageClock):
        clock.enter("layout")
        census = shape_census(network, chip, wire_census=wire_census)
        clock.enter("pack")
        return census, packed_tiles(census, chip)

    def point(budget, packed, arm, overrides):
        return (platform_at(budget, packed) if plain else
                platform_at(budget, packed, ARMS[arm], overrides))

    shared, mark = attempt(point_budget_s, layout)
    for budget in budgets:
        for arm in arms:
            labels = {} if plain else {"arm": arm}
            if shared is None:
                for variant in variants:
                    yield from marked_rows(network.name, target, budget, ks, {**(mark or {}), **labels,
                                                                              **({} if plain else {"variant": variant})})
                continue
            census, packed = shared
            # a platform the toolchain refuses to declare at this budget (a floorplan too small for the cores, say)
            # is a point refused by name at the "declare" stage, like any later stage
            base, point_mark = attempt(None, lambda clock: (clock.enter("declare"),
                                                            point(budget, packed, arm, None))[1])
            scheduled = None
            if base is not None:
                scheduled, point_mark = attempt(point_budget_s, lambda clock: schedule_program(
                    network, census, base, program_facts, tiles=packed, on_stage=clock.enter))
            for variant, overrides in variants.items():
                tags = {**labels, **({} if plain else {"variant": variant})}
                if scheduled is None:
                    known = {"packed_tiles": packed, **({} if base is None else {
                        "cores": sum(int(c.get("count", 1)) for c in base["cores"])})}
                    yield from marked_rows(network.name, target, budget, ks, {**(point_mark or {}), **tags}, known)
                    continue
                for k in ks:
                    variant_platform, price_mark = attempt(None, lambda clock: (clock.enter("declare"), point(
                        budget, packed, arm, overrides))[1])
                    if variant_platform is None:
                        yield from marked_rows(network.name, target, budget, (k,), {**(price_mark or {}), **tags})
                        continue
                    platform = {**variant_platform, "inference_batch_K": int(k),
                                "inference_rate_hz": RATE_HZ, "min_lifetime_hours": LIFETIME_HOURS}
                    priced, price_mark = attempt(point_budget_s, lambda clock: price_scheduled(
                        network, census, scheduled, platform, on_stage=clock.enter))
                    if priced is None:
                        yield from marked_rows(network.name, target, budget, (k,), {**(price_mark or {}), **tags})
                        continue
                    row = ladder_row(network, census, priced, platform, READING)
                    row.update(target=target, packaging=packaging, budget=budget, point_budget_s=point_budget_s,
                               **tags)
                    yield row


def run(networks: Sequence[str], targets: Sequence[str], budgets: Sequence, ks: Sequence[int],
        resolution: Optional[int], wire_census: bool, declared: Optional[Dict[str, Any]] = None,
        point_budget_s: Optional[float] = POINT_BUDGET_S, core_fields: Optional[Dict[str, Any]] = None,
        variants: Optional[Dict[str, Optional[Dict[str, Any]]]] = None, arms: Sequence[str] = ("absent",),
        x_only: bool = False) -> Iterator[Dict[str, Any]]:
    for name in networks:
        if ROSTER[name] is None:
            yield {"network": name, "absent": "no registered builder"}
            continue
        built: Dict[str, Tuple[Optional[LadderNetwork], Optional[Dict[str, Any]]]] = {}
        for target in targets:
            spec = target_spec(target)
            packaging = spec["packaging"]

            def convert(clock: StageClock, name=name, packaging=packaging) -> LadderNetwork:
                clock.enter("convert")
                return build_network(name, resolution, packaging)

            if packaging not in built:
                built[packaging] = attempt(point_budget_s, convert)
            network, mark = built[packaging]
            labels = {"geometry": spec["geometry"], "bundle": spec["bundle"],
                      **departure_labels(target, declared, core_fields)}
            if network is None:
                for budget in (("declared",) if x_only else budgets):
                    for row in marked_rows(name, target, budget, ks, mark or {}):
                        yield {**row, **labels, "resolution": list(member_shape(name, resolution))}
                continue
            chip = resolved_platform(target, "declared", None, declared, core_fields)
            if x_only or (spec["bundle"] is not None and not spec["bundle"]["physics"]):
                for row in x_rows(network, target, chip, point_budget_s=point_budget_s, wire_census=wire_census):
                    yield {**row, **labels, "packaging": packaging}
                continue

            def platform_at(budget, packed, residency=None, overrides=None, target=target):
                plan = {} if residency is None else {"residency_plan": dict(residency)}
                return resolved_platform(target, budget, packed, {**(declared or {}), **plan}, core_fields,
                                         overrides)

            for row in cell_rows(network, target, packaging, chip, platform_at, budgets=budgets, ks=ks,
                                 point_budget_s=point_budget_s, wire_census=wire_census, variants=variants,
                                 arms=arms):
                yield {**row, **labels}


def variant_blocks(items: Sequence[str]) -> Optional[Dict[str, Optional[Dict[str, Any]]]]:
    """``--variant LABEL=BLOCK:PATH[,BLOCK:PATH]`` items: label -> the merged override value of its named blocks (each
    a JSON file of constant declarations); ``LABEL=`` declares none (the shipped physics)."""
    if not items:
        return None
    out: Dict[str, Optional[Dict[str, Any]]] = {}
    for item in items:
        label, _, spec = item.partition("=")
        blocks: Dict[str, Any] = {}
        for part in filter(None, spec.split(",")):
            block, _, path = part.partition(":")
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
            blocks[block] = data.get("platform_physics_overrides", data)
        out[label] = merge_override_blocks(blocks) if blocks else None
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--networks")
    parser.add_argument("--targets")
    parser.add_argument("--budgets", help="comma-separated fractions of F, 'declared', or c:<count> (absolute cores)")
    parser.add_argument("--ks")
    parser.add_argument("--resolution", type=int, default=None,
                        help="replace the side of every image member (default: each member's own shape)")
    parser.add_argument("--no-wire-census", action="store_true")
    parser.add_argument("--point-budget-s", type=float, default=POINT_BUDGET_S,
                        help="declared per-point wall budget in seconds (0: none)")
    parser.add_argument("--declare", action="append", default=[], metavar="KEY=VALUE",
                        help="replace one declared capability, e.g. allow_scheduling=false")
    parser.add_argument("--core-field", action="append", default=[], metavar="KEY=VALUE",
                        help="declare one field on every core type, e.g. write_granularity=cell")
    parser.add_argument("--variant", action="append", default=[], metavar="LABEL=BLOCK:PATH[,BLOCK:PATH]",
                        help="price each schedule under these merged override blocks too (LABEL= : shipped)")
    parser.add_argument("--arms", default="absent", help="comma-separated residency arms: absent, floor, stateless")
    parser.add_argument("--x-only", action="store_true", help="capacity columns only, at the declared core count")
    args = parser.parse_args(argv)
    chosen = dict(SMOKE) if args.smoke else {"networks": tuple(ROSTER), "targets": tuple(TARGETS),
                                             "budgets": BUDGET_FRACTIONS + ("declared",), "ks": KS}
    if args.networks:
        chosen["networks"] = tuple(args.networks.split(","))
    if args.targets:
        chosen["targets"] = tuple(args.targets.split(","))
    if args.budgets:
        chosen["budgets"] = tuple(b if b == "declared" or b.startswith("c:") else float(b)
                                  for b in args.budgets.split(","))
    if args.ks:
        chosen["ks"] = tuple(int(k) for k in args.ks.split(","))
    declared = declared_values(args.declare)
    core_fields = {}
    for item in args.core_field:  # a JSON value, else the text itself (an enum such as write_granularity=cell)
        key, value = item.split("=", 1)
        try:
            core_fields[key] = json.loads(value)
        except json.JSONDecodeError:
            core_fields[key] = value
    variants = variant_blocks(args.variant)
    arms = tuple(args.arms.split(","))
    start = time.time()
    with open(os.path.abspath(__file__), "rb") as handle:
        script_sha = hashlib.sha256(handle.read()).hexdigest()[:16]
    header = {"header": True, "script_sha256_16": script_sha, "selection": {k: list(v) for k, v in chosen.items()},
              "resolution": args.resolution, "capabilities": {**CAPABILITIES, **declared},
              "wire_census": not args.no_wire_census, "assumptions": {"inference_rate_hz": RATE_HZ,
                                                             "min_lifetime_hours": LIFETIME_HOURS},
              "reading": READING.__dict__, "point_budget_s": args.point_budget_s or None,
              "core_fields": core_fields, "variants": variants, "arms": list(arms), "x_only": args.x_only}
    count = 0
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(header, sort_keys=True, default=str) + "\n")
        handle.flush()
        for row in run(chosen["networks"], chosen["targets"], chosen["budgets"], chosen["ks"], args.resolution,
                       wire_census=not args.no_wire_census, declared=declared,
                       point_budget_s=args.point_budget_s or None, core_fields=core_fields, variants=variants,
                       arms=arms, x_only=args.x_only):
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
            handle.flush()
            count += 1
            print(json.dumps({k: row.get(k) for k in ("network", "target", "budget", "arm", "variant", "cores",
                                                       "inference_batch_K", "x_packed", "passes", "refusal",
                                                       "above_budget")},
                             default=str), flush=True)
        handle.write(json.dumps({"trailer": True, "rows": count, "run_wall_s": time.time() - start}) + "\n")
    print(f"wrote {count} rows to {args.out} in {time.time() - start:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
