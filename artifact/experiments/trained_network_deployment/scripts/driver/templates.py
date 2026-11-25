""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import declarations

FORMULA = {"disc": "200000 (discovery pool)", "B0": "F", "B1": "ceil(F / 2)", "B2": "ceil(F / 8)",
           "B3": "ceil(F / 2)", "B4": "ceil(F / 2)"}
DIV = {"B0": 1, "B1": 2, "B2": 8, "B3": 2, "B4": 2}
PLACEHOLDER = {"weight_source": "<run root>/<row>/weights_module_state.pt", "generated_root": "<run root>/<row>",
               "quantizer_state": {"file": "<rows root>/<row>/quantizer_state.json",
                                   "sha256": "<sha256 of the map the session writes>"},


               "datasets_path": "<datasets root>/cifar100_train_only"}


def counts(cfg: dict) -> list:
    ""
    plat = cfg["platform_constraints"]
    over = plat.get("target_overrides", {}).get("cores")
    return over["value"] if over else plat["cores"]


def to_template(cfg: dict, row) -> dict:
    ""
    t = json.loads(json.dumps(cfg))
    for core in counts(t):
        if row.budget not in declarations.DESIGN_BUDGETS:
            core["count"] = FORMULA[row.budget] + (f" of width {core['weight_bits']}" if "weight_bits" in core else "")
    t["deployment_parameters"]["weight_source"] = PLACEHOLDER["weight_source"]
    t["generated_files_path"] = PLACEHOLDER["generated_root"]
    if "quantizer_state" in t["platform_constraints"]:
        t["platform_constraints"]["quantizer_state"] = dict(PLACEHOLDER["quantizer_state"])
    if row.route in declarations.TUNED_ROUTES:
        t["datasets_path"] = PLACEHOLDER["datasets_path"]
    return t


def serialize(template: dict) -> bytes:
    ""
    return json.dumps(template, indent=1, sort_keys=True).encode()


def sha256_of(template: dict) -> str:
    return hashlib.sha256(serialize(template)).hexdigest()


def differences(a, b, prefix: str = "") -> list[str]:
    ""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            p = f"{prefix}.{k}" if prefix else str(k)
            if k not in a or k not in b:
                out.append(p)
            else:
                out += differences(a[k], b[k], p)
        return out
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += differences(x, y, f"{prefix}[{i}]")
        return out
    return [] if a == b else [prefix or "<root>"]


def declared_served_copy(path) -> bool:
    ""
    import os

    import common

    roots = json.loads(os.environ.get("E2E_DATASET_ROOTS") or "[]")
    return (isinstance(roots, list) and isinstance(path, str) and path in [str(r) for r in roots]
            and Path(path).name == Path(common.CANONICAL_DATASETS_PATH).name)


def _locked_row(plan: Path, key: str) -> dict | None:
    rows = json.loads((Path(plan) / "rows.json").read_text())["rows"]
    return next((r for r in rows if r["key"] == key), None)


def check_config(cfg: dict, row, rows_root: Path, plan: Path) -> dict:
    ""
    import configs

    rows_root, plan = Path(rows_root), Path(plan)
    problems: list[str] = []
    rec = {"row": row.key, "plan": str(plan)}
    locked = _locked_row(plan, row.key)
    tpl_path = plan / "configs" / f"{row.key}.json"
    if locked is None:
        problems.append(f"{row.key} is not a row of {plan / 'rows.json'}")
    if not tpl_path.is_file():
        problems.append(f"no locked template {tpl_path}")
    template = None
    if tpl_path.is_file():
        data = tpl_path.read_bytes()
        rec["template_sha256"] = hashlib.sha256(data).hexdigest()
        if locked is not None and rec["template_sha256"] != locked["config_template_sha256"]:
            problems.append("the template file's sha256 differs from rows.json's config_template_sha256")
        template = json.loads(data)
    if row.budget not in ("disc", *declarations.DESIGN_BUDGETS):
        import common

        disc = configs.make_row(row.member, row.target, row.route, "disc", row.geometry)
        disc_dir, rec["f_source"] = common.needed_row_dir(disc.key, rows_root)
        res_path = disc_dir / "result.json"
        try:
            res = json.loads(res_path.read_text())
            dep = res["deployment"]
            if res.get("status") != "complete":
                raise ValueError(f"{disc.key} is {res.get('status')}")
            cores = counts(cfg)
            if "physical_cores_by_width" in dep and any("weight_bits" in c for c in cores):
                for c in cores:
                    want = math.ceil(int(dep["physical_cores_by_width"][str(c["weight_bits"])]) / DIV[row.budget])
                    if int(c["count"]) != want:
                        problems.append(f"width {c['weight_bits']}: count {c['count']} != {want} from {disc.key}")
            else:
                want = math.ceil(int(dep["physical_cores"]) / DIV[row.budget])
                if int(cores[0]["count"]) != want:
                    problems.append(f"count {cores[0]['count']} != {want} from {disc.key}")
        except (OSError, KeyError, ValueError) as exc:
            problems.append(f"the F source {res_path} does not verify: {exc}")
    qs = cfg["platform_constraints"].get("quantizer_state")
    if qs is not None:
        path = Path(qs["file"])
        ok = path.resolve().parent == (rows_root / row.key).resolve() and path.is_file() and (
            hashlib.sha256(path.read_bytes()).hexdigest() == qs["sha256"])
        if not ok:
            problems.append(f"quantizer_state {qs['file']} is not the row's own map at its declared sha256")
    import common

    served, copy_name = str(common.DATASETS_PATH), Path(common.CANONICAL_DATASETS_PATH).name
    served_copy = cfg.get("datasets_path") == served and Path(served).name == copy_name

    served_copy = served_copy or declared_served_copy(cfg.get("datasets_path"))

    if not served_copy:
        problems.append(f"datasets_path {cfg.get('datasets_path')!r} is not the root the driver serves ({served}, "
                        f"a {copy_name} copy)")
    mapped = to_template(cfg, row)
    tpl_root = (template or {}).get("datasets_path")
    if (served_copy and row.route not in declarations.TUNED_ROUTES and isinstance(tpl_root, str)
            and Path(tpl_root).name == copy_name):
        mapped["datasets_path"] = tpl_root
    rec["mapped_sha256"] = sha256_of(mapped)
    if template is not None:
        rec["differences"] = differences(mapped, template)
        if rec["differences"]:
            problems.append(f"the document differs from the locked template at {rec['differences'][:8]}")
    rec.update(problems=problems, passed=not problems)
    return rec
