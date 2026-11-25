""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pairs_common as pc

EXPECTED_CODE = set(pc.COPIED_FILES) | {pc.PATCHED_FILE}


def _code_files_present(h: Path) -> set:
    return {p.name for p in h.iterdir() if p.is_file() and p.suffix in (".py", ".sh")}


def harness_digests(h: Path) -> dict:
    ""
    return pc.vsc().probe(Path(h), "digests")


def verify(label: str, h: Path | None = None) -> list:
    if label not in pc.NEW_SEEDS:
        return [f"{label} is not a new seed"]
    h = pc.harness_dir(label) if h is None else Path(h)
    if not h.is_dir():
        return [f"{h} does not exist (run build_harness.py build)"]
    m = pc.vsc()
    out = []
    for name in pc.COPIED_FILES:
        if not (h / name).exists():
            out.append(f"{name} missing")
        elif pc.sha256_file(h / name) != pc.sha256_file(pc.S0 / name):
            out.append(f"{name} differs from vit_gate/{name}")
    want = m.patched_make_rules((pc.S0 / pc.PATCHED_FILE).read_text())
    if not (h / pc.PATCHED_FILE).exists() or (h / pc.PATCHED_FILE).read_text() != want:
        out.append("make_rules.py is not vit_gate/make_rules.py plus the declared seed patch")
    extra = _code_files_present(h) - EXPECTED_CODE
    if extra:
        out.append(f"undeclared code files {sorted(extra)}")
    seed_file = (h / "SEED").read_text().strip() if (h / "SEED").exists() else None
    if seed_file != str(pc.SEEDS[label]):
        out.append(f"SEED holds {seed_file!r}, declared {pc.SEEDS[label]}")
    link = h.parent / "screen_v42"
    if not link.is_symlink() or link.resolve() != (pc.BUILD / "screen_v42").resolve():
        out.append(f"{link} does not link to 06_build/screen_v42")
    if out:
        return out
    s0_rules = json.loads((pc.S0 / "gate_rules.json").read_text())
    rules = json.loads((h / "gate_rules.json").read_text())
    diff = pc.deep_diff(s0_rules, rules)
    if diff != ["seed"] or rules.get("seed") != pc.SEEDS[label]:
        out.append(f"gate_rules.json differs from vit_gate/gate_rules.json at {diff[:6]} (only 'seed' may differ)")
    d = harness_digests(h)
    if d["generated_rules_problem"]:
        out.append(d["generated_rules_problem"])
    if d["seed"] != pc.SEEDS[label]:
        out.append(f"the harness's rules read seed {d['seed']}")
    if tuple(d["code_files"]) != pc.LOCKED_CODE_FILES:
        out.append("the harness's CODE_FILES differ from the locked list")
    if d["current_digests"]["code_sha256"] != pc.code_digest(h, pc.LOCKED_CODE_FILES):
        out.append("the harness's code digest differs from the recomputed one")
    caps = pc.read_json(h / "caps.json")
    want_caps = m.caps_from_s0(rules, d["current_digests"]["code_sha256"], d["path_fingerprint"])
    if caps != json.loads(json.dumps(want_caps)):
        out.append("caps.json differs from the caps the s0 records give (the locked seeds_common.caps_from_s0)")
    if d["lock_exists"]:
        lock = pc.read_json(h / "gate_lock.json") or {}
        if lock.get("digests") != d["current_digests"]:
            out.append("gate_lock.json digests differ from the harness's digests now")
    out += pc.locked_problems()
    return out


def build(label: str) -> dict:
    if label not in pc.NEW_SEEDS:
        raise SystemExit(f"REFUSED: {label} is not a new seed")
    pc.require_locked_sources()
    m = pc.vsc()
    h = pc.harness_dir(label)
    if (h / "gate_lock.json").exists():
        raise SystemExit(f"REFUSED: {label} is locked (gate_lock.json exists); a locked harness is never rebuilt")
    runs = h / "runs"
    if runs.exists() and any(p.is_dir() and not p.name.startswith("_") and (p / "result.json").exists()
                             for p in runs.iterdir()):
        raise SystemExit(f"REFUSED: {label} holds job records; never rebuilt over records")
    link = pc.HERE / "screen_v42"
    if not link.exists():
        link.symlink_to(Path("..") / "screen_v42")
    h.mkdir(parents=True, exist_ok=True)
    manifest = {"label": label, "seed": pc.SEEDS[label], "s0_gate_lock_sha256": pc.S0_LOCK_SHA256,
                "vit_gate_seeds_lock_sha256": pc.VS_LOCK_SHA256, "built_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
                "files": {}}
    for name in pc.COPIED_FILES:
        pc.copy_byte_identical(pc.S0 / name, h / name)
        manifest["files"][name] = {"sha256": pc.sha256_file(h / name), "origin": f"vit_gate/{name}, byte-identical"}
    (h / pc.PATCHED_FILE).write_text(m.patched_make_rules((pc.S0 / pc.PATCHED_FILE).read_text()))
    manifest["files"][pc.PATCHED_FILE] = {
        "sha256": pc.sha256_file(h / pc.PATCHED_FILE), "locked_sha256": pc.sha256_file(pc.S0 / pc.PATCHED_FILE),
        "origin": "vit_gate/make_rules.py plus the declared seed patch (vit_gate_seeds/seeds_common.patched_make_rules)"}
    (h / "SEED").write_text(f"{pc.SEEDS[label]}\n")
    manifest["files"]["SEED"] = {"sha256": pc.sha256_file(h / "SEED"), "origin": "the declared seed"}
    gen = subprocess.run([pc.PY, "-s", "-B", "make_rules.py"], cwd=h, capture_output=True, text=True,
                         env={"PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin:/bin"})
    if gen.returncode != 0:
        raise SystemExit(f"REFUSED: make_rules.py in {label} exited {gen.returncode}: {gen.stderr[-400:]}")
    manifest["files"]["gate_rules.json"] = {"sha256": pc.sha256_file(h / "gate_rules.json"),
                                            "origin": "generated by the patched make_rules.py"}
    d = harness_digests(h)
    rules = json.loads((h / "gate_rules.json").read_text())
    caps = m.caps_from_s0(rules, d["current_digests"]["code_sha256"], d["path_fingerprint"])
    (h / "caps.json").write_text(json.dumps(caps, indent=2) + "\n")
    manifest["files"]["caps.json"] = {"sha256": pc.sha256_file(h / "caps.json"),
                                      "origin": "vit_gate_seeds/seeds_common.caps_from_s0 on the s0 records"}
    manifest["code_sha256"] = d["current_digests"]["code_sha256"]
    manifest["path_fingerprint"] = d["path_fingerprint"]
    (h / "harness_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    problems = verify(label)
    if problems:
        raise SystemExit(f"REFUSED: {label} built but does not verify: " + "; ".join(problems))
    return manifest


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("build", "verify"))
    p.add_argument("--seeds", default=",".join(pc.NEW_SEEDS))
    args = p.parse_args(argv)
    rc = 0
    for label in [s for s in args.seeds.split(",") if s]:
        if args.mode == "build":
            m = build(label)
            print(f"{label}: built, seed {m['seed']}, code {m['code_sha256'][:16]}, path {m['path_fingerprint'][:16]}")
        problems = verify(label)
        if problems:
            rc = 1
            print(f"{label}: does not verify:\n  " + "\n  ".join(problems))
        else:
            print(f"{label}: verifies (copies byte-identical to vit_gate, one declared patch, rules differ in 'seed' "
                  "only, caps from the s0 records)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
