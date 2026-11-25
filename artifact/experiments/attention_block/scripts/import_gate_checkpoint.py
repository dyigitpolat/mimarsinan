""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

sys.dont_write_bytecode = True

import numpy as np
import torch

WT = "/work/worktrees/w2-n2"
GATE = Path("records/06_build/vit_gate")
sys.path.insert(0, str(GATE))

import gate_data as gd
import gate_models as gm

from mimarsinan.mapping.export.chip_quantize import quantize_ir_graph
from mimarsinan.mapping.export.law_solver import install_readout_laws
from mimarsinan.mapping.ir.dynamic_bank import validate_bindings
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.mappers.operand import operand_entry_perceptrons, operand_mappers, operand_perceptrons
from mimarsinan.mapping.packing.hybrid_build_pool import build_identity_hybrid_mapping
from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
from mimarsinan.mapping.support.boundary_grids import install_boundary_grids
from mimarsinan.mapping.support.operand_lowering import lower_contractions
from mimarsinan.models.families.vit_lite import ViTLiteCCT
from mimarsinan.models.nn.activations.integer_law import grid_levels
from mimarsinan.pipelining.pipeline_steps.quantization.boundary_quantization_step import (
    install_boundary_quantizers,
)
from mimarsinan.pipelining.pipeline_steps.quantization.readout_law_step import (
    calibrate_readout_quantizers,
    install_readout_quantizers,
)
from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.torch_mapping.encoding_layers import segment_entry_perceptrons
from mimarsinan.transformations.normalization_aware_perceptron_quantization import (
    NormalizationAwarePerceptronQuantization,
)
from mimarsinan.transformations.normalization_fusion import fuse_into_perceptron

ACT_BITS, WEIGHT_BITS = 8, 8
LAW = {"adc_bits": 40, "range": "calibrated", "mult_bits": 16}
CAL_IMAGES, CAL_BATCH, SEED = 2048, 1024, 0
NUM_CLASSES = 100


JOBS = {
    "softmax.O8": ("QF-O8", "softmax", "operand", 8),
    "softmax.O6": ("QF-O6", "softmax", "operand", 6),
    "softmax.O4": ("QF-O4", "softmax", "operand", 4),
    "softmax.T": ("QF-T", "softmax", "twin", None),
    "relu.O8": ("F-R", "relu", "operand", 8),
    "relu.O6": ("F-R", "relu", "operand", 6),
    "relu.O4": ("F-R", "relu", "operand", 4),
    "relu.T": ("F-R", "relu", "twin", None),
}

GATE_RECORD = {"softmax.O8": ("QF-O8", "qat"), "softmax.O6": ("QF-O6", "qat"), "softmax.O4": ("QF-O4", "qat"),
               "softmax.T": ("QF-T", "qat"), "relu.O8": ("A-R", "operand_8"), "relu.O6": ("A-R", "operand_6"),
               "relu.O4": ("A-R", "operand_4"), "relu.T": ("A-R", "twin")}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gate_record(job: str) -> dict:
    return json.loads((GATE / "runs" / job / "result.json").read_text())


@dataclass
class GateData:
    val: torch.Tensor
    labels: np.ndarray
    cal: torch.Tensor
    split_sha256: str
    calibration_sha256: str


def load_gate_data() -> GateData:
    ""
    images, labels = gd.load_train_batch()
    train_idx, val_idx = gd.load_split(labels)
    cal_idx = gd.calibration_indices(train_idx, CAL_IMAGES, SEED)
    val = torch.from_numpy(images[val_idx].transpose(0, 3, 1, 2).copy())
    cal = torch.from_numpy(images[cal_idx].transpose(0, 3, 1, 2).copy())
    return GateData(val, labels[val_idx], cal, gd.indices_sha256(val_idx), gd.indices_sha256(cal_idx))


def normalize64(x_uint8: torch.Tensor) -> torch.Tensor:
    ""
    mean = torch.tensor(gd.MEAN, dtype=torch.float64).view(1, 3, 1, 1)
    std = torch.tensor(gd.STD, dtype=torch.float64).view(1, 3, 1, 1)
    return (x_uint8.double() / 255.0 - mean) / std


def gate_model(form: str, payload_bits: int):
    return gm.vit_lite_7_4(NUM_CLASSES, block="cct", qkv_bias=False, attn_drop=0.1, drop=0.0, stochastic_depth=0.1,
                           act_bits=ACT_BITS, weight_bits=WEIGHT_BITS, payload_bits=payload_bits, attention=form)


def load_gate_network(job: str, form: str, arm: str, bits: Optional[int], data: GateData):
    ""
    record = gate_record(job)
    path = GATE / "runs" / job / "final.pt"
    sha = sha256_file(path)
    if sha != record["final_sha256"]:
        raise SystemExit(f"REFUSED: {path} sha256 {sha} != its record's {record['final_sha256']}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = gate_model(form, bits if bits is not None else 8)
    provenance = {"job": job, "checkpoint": str(path), "sha256": sha, "record_counts": record.get("counts")}
    if job.startswith("QF-"):
        model.load_state_dict(state, strict=True)
        model.set_arm(arm)
        provenance["steps"] = "frozen in the checkpoint (stage B, QAT fallback recipe)"
        return model, provenance

    net = {k: v for k, v in state.items() if not (k.endswith(".step") or k.endswith("_extra_state"))}
    missing, unexpected = model.load_state_dict(net, strict=False)
    if unexpected or any(not (k.endswith(".step") or k.endswith("_extra_state")) for k in missing):
        raise SystemExit(f"REFUSED: {job} keys: missing {missing[:4]}, unexpected {unexpected[:4]}")
    model.eval()
    model.observe(True)
    with torch.no_grad():
        for i in range(0, len(data.cal), CAL_BATCH):
            model(gd.normalize(data.cal[i:i + CAL_BATCH]))
    model.observe(False)
    model.configure(arm, payload_bits=bits, rule="toolchain")
    provenance["steps"] = "stage-A calibration recomputed on CPU by the gate's own observe/configure"
    return model, provenance


def gate_steps(model) -> Dict[str, List[float]]:
    return {name: [float(v) for v in g.step.detach().double().tolist()] for name, g in model.grids().items()}


def compare_with_stage_a_record(steps: Dict[str, List[float]], read_key: str) -> dict:
    ""
    recorded = gate_record("A-R")["steps"][read_key]
    worst, compared, names = 0.0, 0, 0
    for name, values in recorded.items():
        mine = steps.get(name)
        if mine is None:
            continue
        names += 1
        for a, b in zip(mine, values[:-2]):
            worst = max(worst, abs(round(a, 9) - b))
            compared += 1
    return {"read": read_key, "grids": names, "steps": compared, "max_abs_difference_at_9_decimals": worst}


def toolchain_network(gate, form: str) -> ViTLiteCCT:
    ""
    ours = ViTLiteCCT(num_classes=NUM_CLASSES, attention=form).eval()
    gate_state = gate.state_dict()
    state = ours.state_dict()
    missing = sorted(k for k in state if k not in gate_state)
    if missing:
        raise SystemExit(f"REFUSED: toolchain parameters absent in the gate: {missing[:5]}")
    ours.load_state_dict({k: gate_state[k] for k in state})
    return ours


def operand_platform_for(bits: Optional[int], cores: Optional[list] = None) -> dict:
    ""
    declared = cores or ([{"max_axons": 768, "max_neurons": 768, "count": 10 ** 6,
                           "operand": {"payload_bits": int(bits), "write_source": "host"}}] if bits is not None
                         else [{"max_axons": 768, "max_neurons": 768, "count": 10 ** 6}])
    return {"cores": declared, "activation_bits": ACT_BITS, "readout_law": dict(LAW),
            "readout_ranges": "per_consumer_group"}


_BLOCK = re.compile(r"^blocks_(\d+)_self_attn_qkv$")


def _product_roles(model_repr) -> Dict[str, tuple]:
    ""
    out = {}
    for mapper in operand_mappers(model_repr):
        roots = mapper.get_source_mappers()
        names = [getattr(getattr(roots[i], "perceptron", None), "name", None) for i in mapper.operand_roots]
        blocks = {int(m.group(1)) for n in names if n and (m := _BLOCK.match(n))}
        if len(blocks) != 1:
            raise SystemExit(f"REFUSED: product {mapper.name} reads no single block's qkv ({names})")
        out[mapper.name] = (blocks.pop(), "S" if all(n and _BLOCK.match(n) for n in names) else "O")
    return out


def _gate_crossbar(perceptron_name: str) -> str:
    ""
    if perceptron_name == "tokenizer_full":
        return "tokenizer.in_grid"
    if perceptron_name == "fc":
        return "fc.in_grid"
    m = re.match(r"^blocks_(\d+)_(self_attn_qkv|self_attn_proj|linear1|linear2)$", perceptron_name)
    if not m:
        raise SystemExit(f"REFUSED: no gate crossbar for perceptron {perceptron_name}")
    return f"blocks.{m.group(1)}.{m.group(2).replace('self_attn_', 'self_attn.')}.in_grid"


@dataclass
class GateProgram:
    key: str
    model: torch.nn.Module
    ir: object
    platform: dict
    report: object
    import_map: List[dict] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)

    def identity_mapping(self):
        return build_identity_hybrid_mapping(ir_graph=self.ir)


def to_fp64(flow, model_repr) -> None:
    ""
    flow.double()
    for op in operand_perceptrons(model_repr):
        op.double()


def _write(buffer: torch.Tensor, value) -> None:
    with torch.no_grad():
        buffer.copy_(torch.as_tensor(value, dtype=torch.float64))


def build_gate_program(key: str, data: GateData) -> GateProgram:
    job, form, arm, bits = JOBS[key]
    gate, provenance = load_gate_network(job, form, arm, bits, data)
    steps = gate_steps(gate)
    if job == "F-R":
        provenance["stage_a_record_check"] = compare_with_stage_a_record(
            steps, "twin" if arm == "twin" else f"operand_{bits}")
    platform = operand_platform_for(bits if arm == "operand" else None)
    with contextlib.redirect_stdout(io.StringIO()):
        flow = convert_torch_model(toolchain_network(gate, form), (3, 32, 32), NUM_CLASSES, device="cpu",
                                   packaging=MVM_PACKAGING).eval()
    repr_ = flow.get_mapper_repr()
    report = lower_contractions(repr_, platform)
    if arm == "operand" and (report.host or len(report.lowered) != 14):
        raise SystemExit(f"REFUSED: {key}: {len(report.lowered)} products lowered, {len(report.host)} on the host")
    for perceptron in flow.get_perceptrons():
        fuse_into_perceptron(perceptron, device="cpu")
    roles = _product_roles(repr_)
    imported: List[dict] = []
    q8 = grid_levels(ACT_BITS)


    entries = list(segment_entry_perceptrons(repr_))
    banks_in = list(operand_entry_perceptrons(repr_))
    quantizers = install_boundary_quantizers(entries + banks_in, ACT_BITS)
    to_fp64(flow, repr_)
    for perceptron, quantizer in zip(entries, quantizers[:len(entries)]):
        site = _gate_crossbar(perceptron.name)
        (step,) = steps[site]
        _write(quantizer.scale, step * q8)
        imported.append({"toolchain": f"{perceptron.name}.boundary", "gate": site, "step": [step], "levels": q8,
                         "realized_step": [float(quantizer.scale) / q8]})
    by_bank = {op.name: (mapper.name, op) for mapper in operand_mappers(repr_) for op in mapper.banks}
    for op, quantizer in zip(banks_in, quantizers[len(entries):]):
        mapper_name, _ = by_bank[op.name]
        block, role = roles[mapper_name]
        if role != "O" or form != "softmax":
            raise SystemExit(f"REFUSED: unexpected host-fed bank {op.name} ({role}, {form})")
        head = op.batch_index[0]
        step = steps[f"blocks.{block}.self_attn.pv.a_grid"][head]
        _write(quantizer.scale, step * q8)
        imported.append({"toolchain": f"{op.name}.boundary", "gate": f"blocks.{block}.self_attn.pv.a_grid[{head}]",
                         "step": [step], "levels": q8, "realized_step": [float(quantizer.scale) / q8]})


    perceptrons = list(flow.get_perceptrons())
    readout = perceptrons + list(operand_perceptrons(repr_))
    rq = install_readout_quantizers(readout, ACT_BITS)
    to_fp64(flow, repr_)
    cal = normalize64(data.cal)
    batches = [cal[i:i + CAL_BATCH] for i in range(0, len(cal), CAL_BATCH)]
    calibrate_readout_quantizers(flow, readout, rq, batches, platform=platform)
    to_fp64(flow, repr_)
    calibrated: List[dict] = []
    for perceptron, quantizer in zip(readout, rq):
        name = perceptron.name
        if quantizer.channel_groups is not None:
            block = int(_BLOCK.match(name).group(1))
            ranges, sources = [], []
            for g, levels in enumerate(quantizer.group_levels):
                cols = np.nonzero(np.asarray(quantizer.channel_groups) == g)[0]
                role, head = int(cols.min()) // 256, (int(cols.min()) % 256) // 64
                if int(cols.max()) // 256 != role or (int(cols.max()) % 256) // 64 != head or len(cols) != 64:
                    raise SystemExit(f"REFUSED: {name} group {g} spans roles or heads ({cols.min()}..{cols.max()})")
                site = {0: "qk.a_grid", 1: "qk.b_grid", 2: "pv.b_grid"}[role]
                step = steps[f"blocks.{block}.self_attn.{site}"][head]
                want = q8 if role == 0 else grid_levels(int(bits))
                if int(levels) != want:
                    raise SystemExit(f"REFUSED: {name} group {g} has {levels} levels, the gate's grid {want}")
                ranges.append(step * levels)
                sources.append(f"blocks.{block}.self_attn.{site}[{head}]")
                imported.append({"toolchain": f"{name}.readout.group{g}", "gate": sources[-1], "step": [step],
                                 "levels": int(levels)})
            _write(quantizer.group_ranges, ranges)
            _write(quantizer.range, max(ranges))
            for g, entry in enumerate(imported[-len(ranges):]):
                entry["realized_step"] = [float(quantizer.group_ranges[g]) / int(quantizer.group_levels[g])]
            continue
        if name in by_bank:
            block, role = roles[by_bank[name][0]]
            head = by_bank[name][1].batch_index[0]
            if role == "O":
                (step,) = steps[f"blocks.{block}.self_attn.proj.in_grid"]
                site = f"blocks.{block}.self_attn.proj.in_grid"
            elif form == "relu":
                step = steps[f"blocks.{block}.self_attn.pv.a_grid"][head]
                site = f"blocks.{block}.self_attn.pv.a_grid[{head}]"
            else:
                calibrated.append({"toolchain": f"{name}.readout", "range": float(quantizer.range),
                                   "step": float(quantizer.range) / q8, "what": f"S-core scores, block {block}, "
                                                                                 f"head {head}"})
                continue
            _write(quantizer.range, step * q8)
            imported.append({"toolchain": f"{name}.readout", "gate": site, "step": [step], "levels": q8,
                             "realized_step": [float(quantizer.range) / q8]})
            continue
        calibrated.append({"toolchain": f"{name}.readout", "range": float(quantizer.range),
                           "step": float(quantizer.range) / q8})


    projection = NormalizationAwarePerceptronQuantization(WEIGHT_BITS, "cpu", rate=1.0, two_scale=False)
    scales = {}
    for perceptron in perceptrons:
        projection.transform(perceptron)
        scales[perceptron.name] = float(perceptron.parameter_scale)
    repr_.assign_perceptron_indices()
    q_max = (1 << (WEIGHT_BITS - 1)) - 1
    with contextlib.redirect_stdout(io.StringIO()):
        ir = IRMapping(q_max=q_max, firing_mode="Default").map(repr_)
    install_boundary_grids(ir, flow)
    quantize_ir_graph(ir, WEIGHT_BITS, weight_quantization=True)
    install_readout_laws(ir, flow, platform, weight_quantization=True, weight_bits=WEIGHT_BITS)
    validate_bindings(ir, packaging=MVM_PACKAGING)
    worst = max(abs(e["realized_step"][0] / e["step"][0] - 1.0) for e in imported)
    provenance.update(calibrated=calibrated, weight_scales=scales, platform=platform,
                      gate_steps_installed=len(imported), import_worst_relative_step_error=worst)
    return GateProgram(key, flow, ir, platform, report, imported, provenance)


def main() -> int:
    ""
    data = load_gate_data()
    program = build_gate_program(sys.argv[1], data)
    out = {"key": program.key, "import_map": program.import_map, "provenance": program.provenance,
           "ir_cores": len(program.ir.get_neural_cores()), "ir_host_ops": len(program.ir.get_compute_ops()),
           "banks": len(program.ir.dynamic_bank_bindings)}
    Path(sys.argv[2]).write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: out[k] for k in ("key", "ir_cores", "ir_host_ops", "banks")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
