""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

import common

_INVENTORY = json.loads((common.E2E / "checkpoints.json").read_text())
_BY_PATH = {row["path"]: row for row in _INVENTORY["checkpoints"]}
_SCREEN_RULES = json.loads((common.B / "screen_v42" / "screen_rules.json").read_text())
_HEADLINE_RULES = json.loads((common.B / "headline_v46" / "headline_rules.json").read_text())
_SCREEN_ARMS = {a["id"]: a for a in _SCREEN_RULES["arms"]}
_HEADLINE_ARCHS = {a["id"]: a for a in _HEADLINE_RULES["architectures"]}


CNN_NORMALIZATION = {"mean": [0.5071, 0.4865, 0.4409], "std": [0.2673, 0.2564, 0.2762]}
VIT_NORMALIZATION = {"mean": [0.5071, 0.4867, 0.4408], "std": [0.2675, 0.2565, 0.2761]}


@dataclass(frozen=True)
class Checkpoint:
    path: str
    sha256: str
    e01_correct: int
    kind: str
    record: str

    @property
    def file(self):
        return common.RA / self.path


def _checkpoint(rel: str) -> Checkpoint:
    row = _BY_PATH["06_build/" + rel]
    return Checkpoint(row["path"], row["sha256"], int(row["e01_correct_of_5000"]), row["kind"], row["record"])


@dataclass(frozen=True)
class Member:
    id: str
    label: str
    model_type: str
    model_config: dict
    params: int
    normalization: object
    float_ckpt: Checkpoint
    qat: dict = field(default_factory=dict)
    harness: Optional[dict] = None
    family: str = "cnn"


def _screen_member(arm: str) -> Member:
    spec = _SCREEN_ARMS[arm]
    if spec["builder"] == "resnet18_cifar":
        model_type, config = "e2e_x10ta_resnet", {"depth": "R18", "width": float(spec["config"]["width"])}
    else:
        model_type, config = spec["builder"], dict(spec["config"])
    return Member(arm, spec["member"], model_type, config, int(spec["params"]), CNN_NORMALIZATION,
                  _checkpoint(f"screen_v42/runs/{arm}/float_final.pt"),
                  {"qat": _checkpoint(f"screen_v42p/runs/{arm}/qat_final.pt")},
                  {"harness": "screen_v42p", "arm": arm})


def _headline_member(mid: str, arch_id: str, float_job: str, qat_job: str) -> Member:
    arch = _HEADLINE_ARCHS[arch_id]
    return Member(mid, f"{arch['depth']} w{arch['width_label']}", "e2e_x10ta_resnet",
                  {"depth": arch["depth"], "width": float(arch["width"])}, int(arch["params"]), CNN_NORMALIZATION,
                  _checkpoint(f"headline_v46/runs/{float_job}/float_final.pt"),
                  {"qat": _checkpoint(f"headline_v46/runs/{qat_job}/qat_final.pt")},
                  {"harness": "headline_v46", "job": qat_job})


MEMBERS: dict[str, Member] = {m.id: m for m in (
    *(_screen_member(a) for a in ("C1", "C2", "C3", "C4", "C5")),
    _headline_member("R10w100", "R10-w100", "F-R10-w100", "Q-R10-w100-b4444"),
    _headline_member("R10w050", "R10-w050", "F-R10-w050", "Q-R10-w050-b4444"),
    *(_screen_member(a) for a in ("S1", "S3", "S4")),
    Member("ViT", "ViT-Lite-7/4", "vit_lite_cct", {}, 3740004, VIT_NORMALIZATION,
           _checkpoint("vit_gate/runs/F/final.pt"),
           {"twin": _checkpoint("vit_gate/runs/QF-T/final.pt"), "operand": _checkpoint("vit_gate/runs/QF-O8/final.pt")},
           {"harness": "vit_gate", "predictions": {"twin": "vit_gate/runs/QF-T/pred_qat.npy",
                                                    "operand": "vit_gate/runs/QF-O8/pred_qat.npy",
                                                    "float": "vit_gate/runs/F/pred_float.npy"},


            "code_digest": "gate_common.code_digest", "rules": "gate_rules.json",
            "model_builder": "train_gate.build_model", "calibration_switch": "observe", "grid_observer": "observe",


            "code_list": "gate_common.CODE_FILES"},

           family="vit"),
)}

QAT_MEMBERS = tuple(m for m in MEMBERS.values() if m.family == "cnn")


TYPED_BITS = "b4442"
_HEADLINE_JOBS = {j["id"]: j for j in _HEADLINE_RULES["jobs"]}


def _typed_member(arch_id: str) -> Member:
    ""
    arch = _HEADLINE_ARCHS[arch_id]
    job = _HEADLINE_JOBS[f"Q-{arch_id}-{TYPED_BITS}"]
    return Member(f"{arch_id.replace('-', '')}{TYPED_BITS}", f"{arch['depth']} w{arch['width_label']} {TYPED_BITS}",
                  "e2e_x10ta_resnet", {"depth": arch["depth"], "width": float(arch["width"])}, int(arch["params"]),
                  CNN_NORMALIZATION, None, {"qat": _checkpoint(f"headline_v46/runs/{job['id']}/qat_final.pt")},
                  {"harness": "headline_v46", "job": job["id"], "bits": [int(b) for b in job["bits"]]}, family="cnn")


TYPED_MEMBERS: dict[str, Member] = {m.id: m for m in (
    _typed_member(a) for a in ("R10-w050", "R10-w071", "R10-w100", "R18-w050", "R18-w071", "R18-w100",
                               "R34-w050", "R34-w071", "R34-w100"))}


def member(mid: str) -> Member:
    ""
    if mid in MEMBERS:
        return MEMBERS[mid]
    if mid in TYPED_MEMBERS:
        return TYPED_MEMBERS[mid]
    raise KeyError(mid)


def stage_units(m: Member) -> list:
    ""
    job = _HEADLINE_JOBS[m.harness["job"]]
    return list(_HEADLINE_ARCHS[job["arch"]]["units"])


def preprocessing(member: Member) -> dict:
    return {"normalize": member.normalization}
