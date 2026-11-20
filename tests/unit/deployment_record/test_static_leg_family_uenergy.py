"""[U-ENERGY V-M1] The one leg form over a generated family of descriptions: every energy figure over a wall holds its
static leg or is refused with the leg's reason, and no ratio of two energy costs is printed over unlike legs.

The family (``family``, seeded, over 1,000 descriptions): every shipped description that declares static power,
crossed with each wall refused in turn (the per-inference wall by its compute-time rows, the loads' walls by their
write-time rows, the measured model's whole latency by the record's compute time) and with a carry buffer declared
with and without its static row, each with its static rows scaled by a seeded factor. Every description is priced on
the candidate plane (a view with its load censuses), on the record plane (a sealed record's pricing and the measured
model), and, once per combination, on the scaling ladder (Pi, Pi with the host, M), where a per-event aggregate holding
the chip's static power joins the family. The expectations are computed here from the description's own rows."""

from __future__ import annotations

import random
from dataclasses import replace

import pytest

import mimarsinan.deployment_record.introspection.scaling_ladder as ladder
from mimarsinan.deployment_record.cost import DeploymentCostModel
from mimarsinan.deployment_record.cost.absolute import price_absolute
from mimarsinan.deployment_record.cost.absolute.report import absolute_pricing_for_record
from mimarsinan.deployment_record.platform_physics import (
    PlatformPhysics, available_profiles, get_platform_physics, profile_from_dict)
from mimarsinan.deployment_record.platform_physics.constants import resolve_supersessions
from mimarsinan.deployment_record.quantities.from_record import from_record
from mimarsinan.deployment_record.quantities.spec import Quantities, QuantityValue
from mimarsinan.deployment_record.schema import ComputePartitionRecord
from unit.deployment_record.introspection.test_scaling_ladder import READING, _facts, _network, _platform
from unit.deployment_record.load_census_fixtures import written_program

SEED = 20261112
J_TO_MJ = 1e3
STATIC_ROWS = ("p_static_per_core", "p_static_global")
BUFFER_ROW = "p_carry_buffer_static_per_byte"
BUFFER_ROWS = ("e_carry_buffer_per_byte", "area_carry_buffer_per_byte", BUFFER_ROW)
COMPUTE_TIME = ("t_cycle", "t_synaptic_op")
WRITE_TIME = ("t_program_per_byte", "t_write_per_unit", "t_core_program", "t_write_setup_per_event")
#: (wall, its energy figure, that figure's static leg) of every energy figure the absolute pricer prints over a wall.
FIGURES = (("e2e_latency_s", "energy_per_inference_mj", "energy_static_mj"),
           ("latency_load_s", "energy_load_mj", "energy_static_load_mj"),
           ("latency_programming_s", "energy_programming_mj", "energy_static_programming_mj"))
WALLS = ("none", "per_inference", "loads", "measured")
BUFFERS = ("none", "with_row", "without_row")
DRAWS_CANDIDATE, DRAWS_RECORD = 8, 5


def _declares_static(name):
    return any(get_platform_physics(name).has(r) for r in STATIC_ROWS)


SHIPPED = tuple(n for n in available_profiles() if _declares_static(n))


def description(name, *, wall, buffer, factor, buffer_nw, aggregate=False):
    """A shipped description with its static rows scaled by ``factor``, the refused wall's rows left out, and a flat
    carry buffer (its static row ``buffer_nw`` nW per byte where stated)."""
    data = get_platform_physics(name).to_dict()
    data.pop("carry_buffer", None)
    constants = {k: dict(v) for k, v in data["constants"].items() if k not in BUFFER_ROWS}
    for row in STATIC_ROWS:
        if row in constants:
            constants[row].update({c: constants[row][c] * factor for c in ("low", "nominal", "high")})
    dropped = COMPUTE_TIME if wall == "per_inference" else WRITE_TIME if wall == "loads" else ()
    constants = {k: v for k, v in constants.items() if k not in dropped}
    flat = {"e_carry_buffer_per_byte": 1.0, "area_carry_buffer_per_byte": 1.0}
    if buffer == "with_row":
        flat[BUFFER_ROW] = buffer_nw
    if aggregate:
        flat["e_synaptic_event_total"] = 3.0
    if buffer == "none":
        flat = {k: v for k, v in flat.items() if k == "e_synaptic_event_total"}
    constants.update(_declared(flat))
    data["constants"] = constants
    data["name"] = f"{name}_family"
    return PlatformPhysics.from_dict(data)


def _declared(values):
    """Test rows in the resolved form a description's constants take (each in its display unit)."""
    if not values:
        return {}
    return profile_from_dict({
        "format_version": 1, "name": "rows", "display_name": "rows", "description_file": "t.md",
        "validity": {"measurement_kind": "projection"},
        "constants": {k: {"nominal": v, "evidence_kind": "estimated", "note": "family"} for k, v in values.items()},
    }).to_dict()["constants"]


def family():
    rng = random.Random(SEED)
    for name in SHIPPED:
        for wall in WALLS:
            for buffer in BUFFERS:
                for plane, draws in (("candidate", DRAWS_CANDIDATE), ("record", DRAWS_RECORD)):
                    if wall == "measured" and plane == "candidate":
                        continue  # the candidate plane has no measured model: its draws go to the other walls
                    for _ in range(draws):
                        yield dict(name=name, wall=wall, buffer=buffer, plane=plane, factor=rng.uniform(0.1, 10.0),
                                   buffer_nw=rng.uniform(1.0, 100.0), capacity=rng.choice((256, 4096, 65536)),
                                   cores=rng.randint(1, 64))


def _ends(band):
    return [band.low, band.nominal, band.high]


def watts(physics, cores, capacity):
    """The description's static power (W, corner by corner) from its own rows, None where it states none."""
    priced = set(resolve_supersessions(physics.constants))
    parts = []
    if "p_static_per_core" in priced and cores is not None:
        parts.append([v * cores for v in _ends(physics.band("p_static_per_core"))])
    if "p_static_global" in priced:
        parts.append(_ends(physics.band("p_static_global")))
    if capacity and physics.has(BUFFER_ROW):
        parts.append([v * capacity for v in _ends(physics.band(BUFFER_ROW))])
    return None if not parts else [sum(p[i] for p in parts) for i in range(3)]


def failures_of(terms, refused, power, part_refused, figures=FIGURES):
    """Every energy figure over a wall held to the rule: (figure, why) for each that is not."""
    out = []
    for wall, energy, leg in figures:
        if part_refused:
            if energy in terms or leg in terms:
                out.append((energy, "a number while a declared part of its static power is refused"))
            elif leg in refused and BUFFER_ROW not in refused[leg]:
                out.append((leg, f"refused without the part's name: {refused[leg][:120]}"))
            continue
        if power is None:
            if leg in terms:
                out.append((leg, "a leg where no static power is declared"))
            continue
        if wall not in terms:
            if energy in terms or leg in terms:
                out.append((energy, "a number over a refused wall while static power is declared"))
            elif leg in refused and wall not in refused[leg]:
                out.append((leg, f"refused without the wall's name: {refused[leg][:120]}"))
            continue
        if energy in terms:
            if leg not in terms:
                out.append((energy, "a number without its leg"))
                continue
            expected = [w * t * J_TO_MJ for w, t in zip(power, _ends(terms[wall].band))]
            if _ends(terms[leg].band) != pytest.approx(expected, rel=1e-9, abs=1e-300):
                out.append((leg, f"{_ends(terms[leg].band)} is not {expected}"))
    return out


def _view(cores, capacity):
    values = {"synaptic_events": 2e5, "cores_physical": cores, "latency_steps": 40.0, "host_macs": 0.0,
              "compute_op_count": 0.0, **({"carry_buffer_capacity_bytes": capacity} if capacity else {})}
    return Quantities({k: QuantityValue(float(v), "static") for k, v in values.items()})


def _record(physics, capacity, measured_refused):
    from unit.deployment_record.record_fixtures import make_rewritten_record
    record = make_rewritten_record()
    platform = {**record.identity.platform, "platform_physics_resolved": physics.to_dict(),
                **({"carry_buffer_bytes": capacity} if capacity else {})}
    record = replace(record, identity=replace(record.identity, platform=platform),
                     energy=replace(record.energy, synaptic_events=1e5),  # a sealed event census and a
                     utilization=replace(record.utilization, partition=ComputePartitionRecord(  # partition census
                         onchip_params=900, host_params=0, total_params=900, onchip_macs=68096, host_macs=0,
                         total_macs=68096)))
    if measured_refused:
        latency = replace(record.timing.latency, compute_sim_time_s=None)
        record = replace(record, timing=replace(record.timing, latency=latency))
    return record


def _by_name(pricing):
    return {t.name: t for t in pricing.terms}, {r.name: r.reason for r in pricing.refusals}


def check(case):
    capacity = case["capacity"] if case["buffer"] != "none" else 0
    physics = description(case["name"], wall=case["wall"], buffer=case["buffer"], factor=case["factor"],
                          buffer_nw=case["buffer_nw"])
    part_refused = case["buffer"] == "without_row"
    if case["plane"] == "candidate":
        loads = written_program((48, 32), (40,), (24, 16), weight_bits=4, granularity="cell", axons=16, neurons=16)
        terms, refused = _by_name(price_absolute(_view(case["cores"], capacity), physics, loads=loads))
        return failures_of(terms, refused, watts(physics, case["cores"], capacity), part_refused)
    record = _record(physics, capacity, case["wall"] == "measured")
    terms, refused = _by_name(absolute_pricing_for_record(record))
    view = from_record(record)
    cores = view.get("cores_physical").value if view.has("cores_physical") else None
    power = watts(physics, cores, capacity)
    out = failures_of(terms, refused, power, part_refused)
    model = {t.name: t for t in DeploymentCostModel().energy(record)}
    if part_refused or (power is not None and case["wall"] == "measured"):
        if "total_mj" in model or "modeled_static_mj" in model:
            out.append(("total_mj", "the measured total without its refused leg"))
    elif power is not None:
        if "total_mj" not in model or "modeled_static_mj" not in model:
            out.append(("total_mj", "the measured total withheld while its leg prices"))
        else:
            total_s = [t for t in DeploymentCostModel().latency(record) if t.name == "total_s"][0].band
            expected = [w * t * J_TO_MJ for w, t in zip(power, _ends(total_s))]
            if _ends(model["modeled_static_mj"].band) != pytest.approx(expected, rel=1e-9):
                out.append(("modeled_static_mj", "not the static power over total_s"))
    return out


class TestTheFamily:
    def test_the_family_holds_over_a_thousand_descriptions(self):
        cases = list(family())
        assert len(cases) >= 1000 and len(SHIPPED) >= 8, (len(cases), SHIPPED)
        failed = [(c["name"], c["plane"], c["wall"], c["buffer"], f) for c in cases for f in check(c)]
        assert not failed, (len(failed), failed[:12])

    def test_the_rule_is_red_on_a_number_without_its_leg(self):
        terms = {"e2e_latency_s": _T([1.0, 1.0, 1.0]), "energy_per_inference_mj": _T([2.0, 2.0, 2.0])}
        assert failures_of(terms, {}, [1.0, 1.0, 1.0], False, FIGURES[:1]) == [
            ("energy_per_inference_mj", "a number without its leg")]


class _T:
    def __init__(self, ends):
        from mimarsinan.deployment_record.schema.provenance import Band
        self.band = Band(*ends, "test")


# ------------------------------------------------------------------------------------------------- the ratios
_SCHEDULED = {}


def ladder_row(physics, capacity):
    network = _network(4)
    point = {**_platform(8), **({"carry_buffer_bytes": capacity} if capacity else {})}
    if capacity not in _SCHEDULED:
        census = ladder.shape_census(network, point)
        _SCHEDULED[capacity] = (census, ladder.schedule_program(network, census, point, _facts))
    census, scheduled = _SCHEDULED[capacity]
    priced_point = {**point, "platform_physics_resolved": physics.to_dict()}
    priced = ladder.price_scheduled(network, census, scheduled, priced_point)
    return priced, ladder.ladder_row(network, census, priced, {**priced_point, "inference_batch_K": 16}, READING)


def ratio_failures(priced, row, *, due, aggregate):
    """Pi, Pi with the host and M printed only over like legs: (what, why) for each that is not."""
    axis = row["energy"]
    printed = [k for k in ("pi", "pi_host_included", "overhead") if axis.get(k) is not None]
    if not printed:
        return []
    program, compute = priced.terms["program"], priced.terms["compute_only"]
    out = []
    if aggregate:
        out.append((printed, "printed while the divisor's aggregate holds static power the numerator does not"))
    if due != ("energy_static_load_mj" in program) or due != ("energy_static_mj" in compute):
        out.append((printed, f"printed over unlike legs: once {'energy_static_load_mj' in program}, "
                             f"compute-only {'energy_static_mj' in compute}, static declared {due}"))
    legs = getattr(priced, "static_legs", None)
    if legs is None or legs.get("once") != legs.get("compute_only"):
        out.append((printed, f"the legs are not stated alike: {legs}"))
    return out


class TestTheRatios:
    def test_no_ratio_is_printed_over_unlike_legs(self):
        rng = random.Random(SEED + 1)
        failed, cases = [], 0
        combos = [(n, w, b, False) for n in SHIPPED for w in WALLS if w != "measured" for b in BUFFERS]
        combos += [(n, "none", "none", True) for n in SHIPPED]
        for name, wall, buffer, aggregate in combos:
            capacity = 4096 if buffer != "none" else 0
            physics = description(name, wall=wall, buffer=buffer, factor=rng.uniform(0.1, 10.0),
                                  buffer_nw=rng.uniform(1.0, 100.0), aggregate=aggregate)
            priced, row = ladder_row(physics, capacity)
            if priced.refusal is not None or "program" not in priced.terms:
                continue
            cases += 1
            due = watts(physics, 8, capacity) is not None and buffer != "without_row"
            failed += [(name, wall, buffer, aggregate, f) for f in ratio_failures(priced, row, due=due,
                                                                                  aggregate=aggregate)]
        assert cases >= 80 and not failed, (cases, len(failed), failed[:8])
