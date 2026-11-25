""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import torch

import read_common as rc

EVAL_BATCH = 500


CNN_FAMILIES = {"screen_v42p": ("rules_md", "train_qat"), "screen_v42": ("rules_md", "train_arm"),
                "headline": ("launcher", "train_headline")}
GPU_MEMORY_FRACTION = 0.25
CODE_SUFFIXES = (".py", ".sh")
CODE_DATA = ("sensitivity_rules.json", "sensitivity_rules.md", "sensitivity_lock.json", "headline_rules.json",
             "headline_rules.md", "headline_lock.json", "gate_rules.json", "gate_rules.md", "gate_lock.json",
             "harness_manifest.json", "SEED")


def instance_code_sha256(instance: Path) -> dict:
    ""
    instance = Path(instance)
    files = sorted(p for p in instance.iterdir() if p.is_file() and (p.suffix in CODE_SUFFIXES or p.name in CODE_DATA))
    return {p.name: rc.sha256_file(p) for p in files}


def _bind(instance: Path, first_module: str):
    ""
    if "mimarsinan" in sys.modules:
        raise rc.ReadRefusal("REFUSED: mimarsinan imported before the harness snapshot")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(instance))
    module = __import__(first_module)
    if Path(module.__file__).resolve().parent != Path(instance).resolve():
        raise rc.ReadRefusal(f"REFUSED: {first_module} imported from {module.__file__}, not {instance}")
    return module


def _gpu(device: str) -> torch.device:
    if device == "cpu":
        return torch.device("cpu")
    if device != "cuda" or not torch.cuda.is_available():
        raise rc.ReadRefusal(f"REFUSED: device {device!r} is not available to this job")
    torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION, 0)
    return torch.device("cuda", 0)


def _logits(model, images: np.ndarray, labels: np.ndarray, dtype, transforms_for, dataset_cls, make_loader,
            device: torch.device = torch.device("cpu")):
    ds = dataset_cls(images, labels, np.arange(len(labels), dtype=np.int64), transforms_for("val"))
    loader = make_loader(ds, EVAL_BATCH, shuffle=False, generator=None, workers=0, drop_last=False)
    out = []
    with torch.no_grad():
        for x, _y in loader:
            x = x.to(dtype)
            if device.type == "cuda":
                x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            out.append(model(x).double().cpu())
    return torch.cat(out).numpy()


def eval_cnn(spec: dict, images: np.ndarray, labels: np.ndarray, device: str = "cpu") -> dict:
    ""
    import harness_reference as hr

    if spec["harness"] not in CNN_FAMILIES:
        raise rc.ReadRefusal(f"REFUSED: {spec['id']}: unknown harness {spec['harness']!r}")
    lock_rule, main_module = CNN_FAMILIES[spec["harness"]]
    instance = rc.RA / spec["instance"]
    sc = _bind(instance, "screen_common")
    if lock_rule == "rules_md":
        sc.verify_lock(sc.sha256_file(sc.RULES_MD))
    else:
        sc.verify_lock(sc.launcher_rules_sha256())
    sc.verify_snapshot()
    sc.import_snapshot_toolchain()
    rules = sc.load_rules()
    from screen_data import CifarArrayDataset, make_loader, transforms_for

    ckpt = rc.RA / spec["checkpoint"]
    state = torch.load(ckpt, map_location="cpu", weights_only=True)

    def build():
        if spec["kind"] == "float" and spec["harness"] == "screen_v42":
            from screen_models import build_member

            torch.manual_seed(int(rules["recipe"]["seed"]))
            model = build_member(sc.arm_spec(rules, spec["job"]), rules["num_classes"])
            model.load_state_dict(state, strict=True)
            return model.eval()
        if spec["kind"] == "float":
            import train_headline

            job = sc.job_spec(rules, spec["job"])
            torch.manual_seed(int(rules["recipe"]["seed"]))
            model, _params, _census = train_headline.build_checked(sc.arch_spec(rules, job["arch"]), rules)
            model.load_state_dict(state, strict=True)
            return model.eval()
        from lane_quant import set_lane_mode

        member = types.SimpleNamespace(harness={"arm": spec["job"]} if spec["harness"] == "screen_v42p"
                                       else {"job": spec["job"]})
        model, _census, _ident = (hr.build_screen if spec["harness"] == "screen_v42p" else hr.build_headline)(
            sc, member, rules)
        model.load_state_dict(state, strict=True)
        set_lane_mode(model, "quantize")
        return model.eval()

    dev = _gpu(device)
    settings = {}
    if dev.type == "cuda":
        harness_main = __import__(main_module)
        settings = harness_main.set_determinism(int(rules["recipe"]["seed"]))
    preds = {}
    model = build()
    if dev.type == "cuda":
        model = model.to(dev).to(memory_format=torch.channels_last)
    logits32 = _logits(model, images, labels, torch.float32, transforms_for, CifarArrayDataset, make_loader, dev)
    preds[spec["reads"][0]] = logits32.argmax(1)
    if spec.get("fp64") and dev.type == "cpu":
        logits64 = _logits(build().double(), images, labels, torch.float64, transforms_for, CifarArrayDataset,
                           make_loader)
        preds[spec["reads"][0] + "_fp64"] = logits64.argmax(1)
    return {"predictions": preds, "harness_lock_verified": True, "device": str(dev),
            "device_name": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu", "device_settings": settings}


def eval_vit(spec: dict, images: np.ndarray, labels: np.ndarray, device: str = "cpu") -> dict:
    ""
    instance = rc.RA / spec["instance"]
    tg = _bind(instance, "train_gate")
    gc = tg.gc
    gc.verify_lock(gc.launcher_rules_sha256())
    rules = gc.load_rules()
    job = gc.job_spec(rules, spec["job"])
    dev = _gpu(device)
    settings = {}
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        settings = {"allow_tf32": False, "exact_device_check": tg.exact_device_check(dev)}
    ctx = types.SimpleNamespace(args=types.SimpleNamespace(smoke=False, cpu=dev.type == "cpu", smoke_eval_limit=None,
                                                           smoke_dir=None),
                                rules=rules, job=job, smoke=False, seed=int(rules["seed"]), device=dev, record={})
    tg.load_data(ctx)
    if spec.get("calibration_sha256") and ctx.record.get("calibration_sha256") != spec["calibration_sha256"]:
        raise rc.ReadRefusal(f"REFUSED: {spec['id']}: the calibration draw differs from the record's")
    ctx.val_u8 = torch.from_numpy(np.ascontiguousarray(images.transpose(0, 3, 1, 2)))
    ctx.val_labels = np.asarray(labels)
    state = torch.load(rc.RA / spec["checkpoint"], map_location="cpu", weights_only=True)
    preds, out = {}, {}
    if spec["kind"] == "float":
        model = tg.build_model(ctx)
        tg.load_network_state(model, state)
        model.set_arm("float")
        preds["float"] = tg.evaluate(ctx, model)[1]
    elif spec["kind"] == "qat":
        bits = int(job["payload_bits"]) if job.get("payload_bits") is not None else None
        model = tg.build_model(ctx, payload_bits=bits)
        model.load_state_dict(state, strict=True)
        model.set_arm(job["arm"])
        preds["qat"] = tg.evaluate(ctx, model)[1]
    elif spec["kind"] == "ptq":
        model = tg.build_model(ctx)
        tg.load_network_state(model, state)
        tg.calibrate(ctx, model, int(rules["data"]["calibration"]["batch"]))
        wanted = list(spec["reads"])
        for read in rules["stage_a"]["reads"]:
            if read["key"] not in wanted and read["key"] != "twin":
                continue
            model.configure(read["arm"], **tg.read_config(read))
            if read["key"] == "twin":
                out["twin_steps_sha256"] = tg.twin_steps_sha256(model)
                if spec.get("twin_steps_sha256") and out["twin_steps_sha256"] != spec["twin_steps_sha256"]:
                    raise rc.ReadRefusal(f"REFUSED: {spec['id']}: the twin steps differ from the record's")
            if read["key"] in wanted:
                preds[read["key"]] = tg.evaluate(ctx, model)[1]
        missing = sorted(set(wanted) - set(preds))
        if missing:
            raise rc.ReadRefusal(f"REFUSED: {spec['id']}: reads {missing} are not declared in stage_a.reads")
    else:
        raise rc.ReadRefusal(f"REFUSED: {spec['id']}: unknown kind {spec['kind']!r}")
    return {"predictions": {k: np.asarray(v, dtype=np.int64) for k, v in preds.items()}, "harness_lock_verified": True,
            "device": str(dev), "device_name": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
            "device_settings": settings, **out}


def evaluate(spec: dict, images: np.ndarray, labels: np.ndarray, device: str = "cpu") -> dict:
    torch.set_num_threads(1)
    if spec["harness"] in CNN_FAMILIES:
        return eval_cnn(spec, images, labels, device)
    if spec["harness"] == "vit_gate":
        return eval_vit(spec, images, labels, device)
    raise rc.ReadRefusal(f"REFUSED: unknown harness {spec['harness']!r}")


def record_predictions(spec: dict) -> dict:
    ""
    rec_path = rc.RA / spec["record"]
    if spec["harness"] != "vit_gate" or not rec_path.is_file():
        return {}
    rec = json.loads(rec_path.read_text())
    out = {}
    for key, entry in (rec.get("predictions") or {}).items():
        mapped = "qat" if spec["kind"] == "qat" else key
        out[mapped] = (rec_path.parent / entry["file"], entry["sha256"])
    return out
