""

from __future__ import annotations

import argparse
import contextlib
import gzip
import hashlib
import json
import pickle
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cl6_common as K


ROUTEB_PREFIX = "routeb:"


def member_name(name: str) -> str:
    ""
    if name in K.MEMBERS or str(name).startswith(ROUTEB_PREFIX):
        return name
    raise argparse.ArgumentTypeError(f"invalid choice: {name!r} (choose from "
                                     f"{', '.join(repr(m) for m in sorted(K.MEMBERS))} or {ROUTEB_PREFIX}<row key>)")


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
        return len(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def _model_digest(model) -> str:
    import torch
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode())
        if isinstance(value, torch.Tensor):
            t = value.detach().cpu().contiguous()
            h.update(str(t.dtype).encode())
            h.update(str(tuple(t.shape)).encode())
            h.update(t.numpy().tobytes() if t.dtype != torch.bfloat16 else t.float().numpy().tobytes())
    return h.hexdigest()[:16]


def is_toolchain_refusal(error: BaseException) -> bool:
    ""
    try:
        from mimarsinan.common.refusal import ToolchainRefusal
    except ImportError:
        return False
    return isinstance(error, ToolchainRefusal)


def hook_model_configuration(pipeline, point_cfg_flat):
    ""
    import torch
    from mimarsinan.pipelining.core.platform_constraints_resolver import build_platform_constraints_resolved
    from mimarsinan.pipelining.determinism import isolated_rng_stream

    step = dict(pipeline.steps)["Model Configuration"]
    cache = pipeline.cache
    old_config = cache.get("Model Configuration.model_config")
    old_builder = cache.get("Model Configuration.model_builder")
    old_platform = cache.get("Model Configuration.platform_constraints_resolved")
    step.process()
    new_config = cache.get("Model Configuration.model_config")
    new_builder = cache.get("Model Configuration.model_builder")
    new_platform = cache.get("Model Configuration.platform_constraints_resolved")
    assert new_config == old_config, f"hook: model_config moved {old_config!r} -> {new_config!r}"
    assert type(new_builder) is type(old_builder), f"hook: builder type {type(old_builder)} -> {type(new_builder)}"
    for attr in ("device", "input_shape", "num_classes"):
        a, b = getattr(old_builder, attr, None), getattr(new_builder, attr, None)
        assert str(a) == str(b), f"hook: builder {attr} {a!r} -> {b!r}"
    with isolated_rng_stream():
        import torch as T
        T.manual_seed(0)
        m_old = old_builder.build(old_config)
        T.manual_seed(0)
        m_new = new_builder.build(new_config)
    sd_old, sd_new = m_old.state_dict(), m_new.state_dict()
    assert list(sd_old) == list(sd_new), "hook: built parameter names differ"
    for name in sd_old:
        assert tuple(sd_old[name].shape) == tuple(sd_new[name].shape), f"hook: {name} shape differs"
        assert bool(T.equal(sd_old[name], sd_new[name])), f"hook: {name} values differ"
    expected = build_platform_constraints_resolved(point_cfg_flat)
    assert new_platform == expected, "hook: stored platform_constraints_resolved is not the point's resolution"
    changed = sorted(k for k in set(old_platform) | set(new_platform) if old_platform.get(k) != new_platform.get(k))
    pipeline.save_cache()
    return {"model_config_equal": True, "builder_type": type(new_builder).__name__,
            "builder_built_equal": True, "platform_keys_changed": changed}


def _sources_bytes(sources) -> bytes:
    ""
    import numpy as np
    if sources is None:
        return b""
    arr = np.asarray(sources)
    if arr.dtype == object:
        return "|".join(repr(x) for x in arr.reshape(-1)).encode()
    return str(arr.dtype).encode() + str(arr.shape).encode() + np.ascontiguousarray(arr).tobytes()


IR_DIGEST_VERSION = 2


def ir_readings(ir_path: Path) -> dict:
    ""
    import numpy as np
    from mimarsinan.mapping.ir.types import NeuralCore
    with open(ir_path, "rb") as f:
        ir = pickle.load(f)
    h = hashlib.sha256()
    widths, identity = [], 0
    cores = 0
    for node in ir.nodes:
        h.update(f"{type(node).__name__}|{node.id}|{node.name}|".encode())
        h.update(_sources_bytes(getattr(node, "input_sources", None)))
        if isinstance(node, NeuralCore):
            cores += 1
            if node.core_matrix is not None:
                h.update(np.ascontiguousarray(node.core_matrix).tobytes())
            h.update(f"{node.weight_bank_id}|{node.weight_row_slice}|{float(node.threshold).hex()}".encode())
            hb = getattr(node, "hardware_bias", None)
            if hb is not None:
                h.update(np.asarray(hb, dtype=np.float64).tobytes())
            law = getattr(node, "readout_law", None)
            if law is not None:
                h.update(law.digest().encode())
                width = int(int(np.max(law.adc_levels)).bit_length()) + 1
                widths.append(width)
                identity += int(not np.any(law.adc_shift))
        else:
            h.update(str(getattr(node, "op_type", "")).encode())
    banks = getattr(ir, "weight_banks", None) or {}
    for bank_id in sorted(banks):
        bank = banks[bank_id]
        mat = getattr(bank, "core_matrix", None)
        if mat is None:
            mat = getattr(bank, "matrix", None)
        if mat is not None:
            h.update(f"bank{bank_id}".encode())
            h.update(np.ascontiguousarray(mat).tobytes())
    return {"ir_function_digest": h.hexdigest()[:16], "ir_digest_version": IR_DIGEST_VERSION, "neural_cores": cores, "adc_width_max": max(widths, default=0),
            "adc_width_min": min(widths, default=0), "cores_with_shift_0": identity, "cores_with_law": len(widths)}


def footprint_of(record, declared: list) -> dict:
    ""
    out = {K.geometry_key(ct): 0 for ct in declared}
    by_neurons = {int(ct["max_neurons"]): ct for ct in declared}
    if len(by_neurons) != len(declared):
        raise ValueError("footprint: two declared core types share a neuron width; the match would be ambiguous")
    for stage in record.get("schedule", {}).get("stages", []):
        for core in stage.get("cores", []) or []:
            ct = by_neurons.get(int(core["neurons"]))
            if ct is None or int(core["axons"]) % int(ct["max_axons"]):
                raise ValueError(f"footprint: listed core {core['axons']}x{core['neurons']} matches no declared type")
            out[K.geometry_key(ct)] += int(core["axons"]) // int(ct["max_axons"])
    return out


def listed_of(record) -> dict:
    ""
    out: dict = {}
    for stage in record.get("schedule", {}).get("stages", []):
        for core in stage.get("cores", []) or []:
            key = f"{int(core['axons'])}x{int(core['neurons'])}"
            out[key] = out.get(key, 0) + 1
    return out


def tuning_after(run_dir: Path, start: str, steps: list) -> dict:
    ""
    after = steps[steps.index(start):]
    out = {}
    for name in after:
        p = run_dir / f"{name}.adaptation_ledger.json"
        if p.exists():
            ledger = json.loads(p.read_text())
            out[name] = {"total_steps": int(ledger.get("total_steps", 0)),
                         "endpoint_steps": int(ledger.get("endpoint_steps", 0)),
                         "recovery_steps": int(ledger.get("recovery_steps", 0))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("member", type=member_name)
    ap.add_argument("snapshot")
    ap.add_argument("out")
    ap.add_argument("--arm", required=True, choices=sorted(K.ARMS))
    ap.add_argument("--adc", type=int, required=True)
    ap.add_argument("--cores", required=True)
    ap.add_argument("--no-scheduling", action="store_true")
    ap.add_argument("--label", default="")
    ap.add_argument("--keep-run", action="store_true")
    ap.add_argument("--provider", default=None)
    ap.add_argument("--rows-root", default=None)
    args = ap.parse_args()
    routeb_member = args.member.startswith(ROUTEB_PREFIX)
    if args.provider and not routeb_member:
        ap.error("--provider applies to a route (b) member only")
    if args.rows_root and not routeb_member:
        ap.error("--rows-root applies to a route (b) member only")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "point.log"
    log_file = open(log_path, "w")
    real_stdout = sys.stdout
    sys.stdout = Tee(real_stdout, log_file)
    gpu_fraction = K.apply_gpu_cap()
    t0 = time.time()
    result = {"member": args.member, "arm": args.arm, "adc_bits": args.adc, "label": args.label,
              "cores": json.loads(args.cores), "allow_scheduling": not args.no_scheduling,
              "snapshot": str(Path(args.snapshot).resolve()), "src_digest": K.src_digest(), "wt": str(K.WT)}
    phase = "before run_from"
    try:
        from mimarsinan.deployment_record.schema.record import DEPLOYMENT_RECORD_FILENAME
        from mimarsinan.pipelining.core.pipelines.deployment_pipeline import merge_pipeline_config
        from mimarsinan.pipelining.session import PipelineSession

        snap_run = Path(args.snapshot) / "run"
        snap_meta = json.loads((Path(args.snapshot) / "snapshot.json").read_text())
        routeb = None
        if routeb_member:
            import cl6_routeb as R
            if snap_meta.get("member") != args.member:
                raise R.RouteBMemberError(f"the snapshot {args.snapshot} was cut for {snap_meta.get('member')!r}, "
                                          f"not {args.member!r}")
            routeb = R.member(args.member, start_step=snap_meta["start_step"],
                              provider=args.provider or snap_meta.get("provider"), rows_root=args.rows_root)
            result["routeb"] = routeb
            R.require_seal()
            cut_row = (snap_meta.get("routeb") or {}).get("row_dir")
            if cut_row is None or Path(cut_row).resolve() != Path(routeb["row_dir"]).resolve():
                raise R.RouteBMemberError(f"{R.SNAPSHOT_ROWS_ROOT}: the snapshot {args.snapshot} was cut from the row "
                                          f"at {cut_row}, not the point's {routeb['row_dir']} (rows root "
                                          f"{routeb['rows_root']})")
            result["row_of_record"] = R.row_counts(args.member, args.rows_root, require=True)
        run_dir = out / "run"
        if run_dir.exists():
            shutil.rmtree(run_dir)
        shutil.copytree(snap_run, run_dir)
        cfg = K.point_config(args.member, args.arm, args.adc, json.loads(args.cores),
                             allow_scheduling=not args.no_scheduling,
                             name=R.point_name(args.member, args.arm, args.adc) if routeb else None)
        cfg["_working_directory"] = str(run_dir)
        if routeb is None:
            session = PipelineSession.from_config(cfg)
        else:
            session = R.open_session(cfg)
            result["sealed_paths"] = list(session.parsed.sealed_paths)
        pipeline = session.pipeline
        start = K.start_step(args.member)
        steps = [name for name, _ in pipeline.steps]
        assert steps == snap_meta["steps"], f"point steps {steps} differ from the snapshot's {snap_meta['steps']}"
        pipeline.set_up_requirements()
        from mimarsinan.pipelining.core.pipelines.deployment_pipeline import DeploymentPipeline
        dp = dict(session.parsed.deployment_parameters)
        DeploymentPipeline.apply_preset(session.parsed.pipeline_mode, dp)
        merged = merge_pipeline_config(dp, dict(session.parsed.platform_constraints))
        result["hook"] = hook_model_configuration(pipeline, merged)
        resolved = pipeline.get_resolved_start_step(start)
        assert resolved == start, f"the pipeline resolves start step {start!r} to {resolved!r}"
        result["start_step"] = start


        scm_tol = pipeline.step_tolerances.get("Soft Core Mapping")
        if scm_tol is not None:
            pipeline.step_tolerances["Hard Core Mapping"] = scm_tol
        result["gate_mirror"] = {"Hard Core Mapping": scm_tol}
        print(f"[cl6_point] {args.member} {args.arm} adc {args.adc} cores {args.cores}: run_from {start!r}",
              flush=True)
        try:
            session.surface_config_advisories()
            phase = "run_from"
            pipeline.run_from(start)
        finally:
            session.finish()
        phase = "after run_from"
        record = json.loads((run_dir / DEPLOYMENT_RECORD_FILENAME).read_text())
        sys.stdout.flush()
        text = log_path.read_text()
        certs = K.certificates_from_log(text)
        sched = record["schedule"]
        result.update({
            "status": "ok",
            "identity_platform": record["identity"]["platform"],
            "pass_count": sched["pass_count"], "reprogram_passes": sched.get("reprogram_passes"),
            "reuse_passes": sched.get("reuse_passes"), "sync_count": sched.get("sync_count"),
            "neural_segment_count": record["utilization"]["layout"].get("neural_segment_count"),
            "total_hw_cores": record["utilization"]["layout"].get("total_hw_cores"),
            "footprint": footprint_of(record, json.loads(args.cores)),
            "listed_cores_by_geometry": listed_of(record),
            "deployed": record["accuracy"]["deployed"], "reads": record["accuracy"]["reads"],
            "certificates": certs, "atol0_certified": K.atol0_certified(certs),
            "step_sealed_ledgers_after_start": tuning_after(run_dir, start, steps),
            "retention_ledger": json.loads((run_dir / "retention_ledger.json").read_text())
            if (run_dir / "retention_ledger.json").exists() else None,
            "readout_law_report_present": (run_dir / "Readout Law.readout_law_report.json").exists(),
            "scm_read_derived": "rung-2 identity metric DERIVED via the twin certificate" in text,
        })
        if scm_tol is not None and not result["scm_read_derived"]:
            result["gate_mirror_unjustified"] = True
        if routeb is not None:
            result["e01"] = R.e01_count(record, routeb["provider"], datasets_path=routeb["datasets_path"])
        ir_path = run_dir / "Soft Core Mapping.ir_graph.pickle"
        if ir_path.exists():
            result.update(ir_readings(ir_path))
        from mimarsinan.pipelining.cache.pipeline_cache import PipelineCache
        meta = json.loads((run_dir / "metadata.json").read_text())
        if "Weight Quantization.model" in meta:
            strategy, fname = meta["Weight Quantization.model"]
            wq = PipelineCache.LOAD_STORE_STRATEGIES[strategy](fname).load(str(run_dir))
            result["wq_model_digest"] = _model_digest(wq)
        with gzip.open(out / "deployment_record.json.gz", "wt") as f:
            json.dump(record, f)
        if not args.keep_run:
            shutil.rmtree(run_dir)
    except Exception as error:
        import traceback
        result.update({"status": "error", "error_type": type(error).__name__, "error": str(error)[:4000],
                       "traceback": traceback.format_exc()[-6000:], "refused_in": phase,
                       "toolchain_refusal": is_toolchain_refusal(error)})
        print(f"[cl6_point] ERROR {type(error).__name__}: {error}", flush=True)
    finally:
        if not args.keep_run and (out / "run").exists():
            shutil.rmtree(out / "run", ignore_errors=True)
        result["wall_s"] = time.time() - t0
        result["gpu"] = K.gpu_peak(gpu_fraction)
        sys.stdout.flush()
        sys.stdout = real_stdout
        log_file.close()
        K.write_json(out / "point.json", result)
    sys.exit(0 if result.get("status") == "ok" else 3)


if __name__ == "__main__":
    main()
