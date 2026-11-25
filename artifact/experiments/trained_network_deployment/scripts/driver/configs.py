""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import common
import declarations
import inventory
from e01_provider import PROVIDER_NAME


DISCOVERY_COUNT = 200_000
GEOMETRIES = {"256x256": (256, 256), "256x512": (256, 512)}
TARGETS = ("T-INT",)
ROUTES = {"b0": "qat", "b0-a4": "qat", "a-ptq": "float", "b": "qat", "b-op": "qat", "a": "float", "at": "float"}
V2_ROUTES = ("b0", "b0-a4", "a-ptq")


ACTIVATION_BITS = {"b0": 5, "b0-a4": 4, "a-ptq": 4}
BUDGETS = ("disc", "B0", "B1", "B2")

NOT_YET = {
    "T-LIF": "v2 routes are T-INT only; synchronized LIF rows are driver v3 route a (WI-8, WI-9)",
    "T-VAL8": "v2 routes are T-INT only; the ViT twin rows are driver v3 route b on T-VAL8 (WI-7)",
    "T-VAL8-OP": "v2 routes are T-INT only; the ViT operand arm is driver v3 route b-op (WI-7)",
    "B3": "v2 routes declare no residency plan; executed residency rows are driver v3 rows (WI-6)",
    "B4": "v2 routes declare no residency plan; executed residency rows are driver v3 rows (WI-6)",
    "B5": "v2 routes declare no residency plan; the exploration's designs are driver v3 rows (WI-6)",
}

DEFAULT_GEOMETRY = {"C1": "256x512", "C2": "256x512", "R10w100": "256x512"}


class RowRefusal(SystemExit):
    ""


@dataclass(frozen=True)
class Row:
    member: str
    target: str
    route: str
    budget: str
    geometry: str

    @property
    def key(self) -> str:
        return f"{self.member}.{self.target}.{self.route}.{self.budget}.{self.geometry}"

    @property
    def scheduled(self) -> bool:
        return self.budget in declarations.SCHEDULED

    @property
    def v3(self) -> bool:
        return self.route not in V2_ROUTES


def make_row(member: str, target: str, route: str, budget: str, geometry: str | None = None) -> Row:
    if route in declarations.V3_ROUTES:
        geometry = geometry or declarations.default_geometry(member, budget)
        try:
            declarations.check_row(member, target, route, budget, geometry)
        except declarations.DeclarationRefusal as exc:
            raise RowRefusal(str(exc)) from None
        return Row(member, target, route, budget, geometry)
    for part in (target, route, budget):
        if part in NOT_YET:
            raise RowRefusal(f"REFUSED: {part}: {NOT_YET[part]}")
    if member not in inventory.MEMBERS or inventory.MEMBERS[member].family != "cnn":
        raise RowRefusal(f"REFUSED: {member} is no CNN member of the inventory")
    if target not in TARGETS or route not in ROUTES or budget not in BUDGETS:
        raise RowRefusal(f"REFUSED: unknown target {target!r}, route {route!r} or budget {budget!r}")
    geometry = geometry or DEFAULT_GEOMETRY.get(member, "256x256")
    if geometry not in GEOMETRIES:
        raise RowRefusal(f"REFUSED: unknown geometry {geometry!r}")
    return Row(member, target, route, budget, geometry)


def budget_count(row: Row, footprint):
    ""
    if row.v3:
        try:
            return declarations.budget_count(row.budget, footprint, row.member)
        except declarations.DeclarationRefusal as exc:
            raise RowRefusal(str(exc)) from None
    if row.budget == "disc":
        return DISCOVERY_COUNT
    if footprint is None or footprint <= 0:
        raise RowRefusal(f"REFUSED: {row.key} needs the member's footprint F from its discovery row")
    return {"B0": footprint, "B1": math.ceil(footprint / 2), "B2": math.ceil(footprint / 8)}[row.budget]


def checkpoint_of(row: Row):
    if row.v3:
        return declarations.checkpoint(row.member, row.target, row.route)
    member = inventory.MEMBERS[row.member]
    return member.float_ckpt if ROUTES[row.route] == "float" else member.qat["qat"]


def make_config(row: Row, *, footprint, weight_source: Path, generated_root: Path,
                parity_samples: int | None = 512, census_batch: int | None = None, **v3) -> dict:
    ""
    if row.v3:
        if census_batch is not None and int(census_batch) != declarations.CENSUS_BATCH:
            raise RowRefusal(f"REFUSED: v3 rows declare census batch {declarations.CENSUS_BATCH}")
        return declarations.make_config(row, footprint=footprint, weight_source=weight_source,
                                        generated_root=generated_root, parity_samples=parity_samples, **v3)
    member = inventory.MEMBERS[row.member]
    axons, neurons = GEOMETRIES[row.geometry]
    count = budget_count(row, footprint)
    activation_bits = ACTIVATION_BITS[row.route]
    cores = [{"max_axons": axons, "max_neurons": neurons, "count": int(count), "has_bias": True}]
    if census_batch is not None and int(census_batch) < 1:
        raise RowRefusal(f"REFUSED: census_batch {census_batch} is below 1")
    document = {
        "seed": 0,
        "pipeline_mode": "phased",
        "experiment_name": row.key.replace(".", "__"),
        "generated_files_path": str(generated_root),
        "datasets_path": str(common.DATASETS_PATH),
        "data_provider_name": PROVIDER_NAME,
        "platform_constraints": {
            "target": "generic_estimated_22nm",
            "target_overrides": {
                "weight_bits": {"value": 4, "note": "the members' 4-bit per-tensor weights (b4444)"},
                "cores": {"value": cores, "note": f"{row.geometry} cores, the {row.budget} budget of design.md "
                                                  "section 5.3, on-chip bias for the two-scale projection"},
            },
            "activation_bits": activation_bits,
            "max_schedule_passes": 4096,
            "readout_law": {"adc_bits": 32, "range": "calibrated", "mult_bits": 24},
        },
        "deployment_parameters": {
            "model_type": member.model_type,
            "model_config": dict(member.model_config),
            "model_config_mode": "user",
            "hw_config_mode": "fixed",
            "preprocessing": inventory.preprocessing(member),
            "weight_source": str(weight_source),
            "finetune_epochs": 0,
            "training_epochs": 0,
            "tuning_budget_scale": 0.0,
            "wq_endpoint_recovery_steps": 0,
            "endpoint_floor_steps": 0,
            "weight_quantization": True,
            "wq_two_scale_projection": True,
            "pruning": False,
            "allow_scheduling": bool(row.scheduled),
            "value_parity_samples": int(parity_samples),
            "num_workers": 0,
            "pretrain_floor_chance_multiple": 0,
            "degradation_tolerance": 0.95,
            "lr": 0.001,
            "deployment_metric_full_eval": True,
        },
        "target_metric_override": None,
        "start_step": None,
        "stop_step": None,
    }
    if census_batch is not None:
        document["deployment_parameters"]["simulation_batch_size"] = int(census_batch)
    return document


def declared_inputs(config: dict, row: Row | None = None) -> dict:
    ""
    if row is not None and row.v3:
        return declarations.declared_inputs(config)
    params = config["deployment_parameters"]
    return {"weight_source": Path(params["weight_source"]), "generated_root": Path(config["generated_files_path"]),
            "parity_samples": int(params["value_parity_samples"]),
            "census_batch": params.get("simulation_batch_size")}
