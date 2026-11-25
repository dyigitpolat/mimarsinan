""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
CL5 = HERE.parent
HOST = CL5.parent / "cl0" / "host_calibration.json"
STORES = ("onchip_sram_1mb", "hbm2", "ddr4")
PHYSICS = ("truenorth", "loihi", "odin", "generic_estimated_22nm", "isaac_like", "hermes_64", "neurram",
           "sram_cim_macro")
GEOMETRY_ONLY = ("admm_crossbar_128x64", "dynap_se_64x256x4", "hermes_core_like", "ibm_analog_ai_512x512x34",
                 "meng_subarray_144x32", "prime_mat_256x256", "puma_node_like", "recom_64x64_256", "xformer_pe_like")
SPIKING = ("truenorth", "loihi", "odin")

ODIN_GENERATED = "declared:1024x256x1:spiking"


WRITE_GRANULARITY = {"isaac_like": "cell", "hermes_64": "core", "neurram": "cell", "sram_cim_macro": "cell"}

NO_COALESCING = ("truenorth", "loihi", "odin")

PRICED = (("lenet5", 2, 7200), ("resnet18_w1q", 3, 10800), ("resnet20", 3, 10800), ("vit_leaf", 3, 10800),
          ("vit_lite_cct", 3, 10800), ("vgg16_bn", 3, 10800), ("resnet56", 3, 14400),
          ("densenet_bc100", 4, 14400), ("resnet110", 4, 14400), ("resnet164_preact", 6, 14400))

CAPACITY = (("lenet5", 2), ("resnet18_w1q", 2), ("resnet20", 2), ("resnet56", 2), ("resnet110", 2),
            ("resnet164_preact", 3), ("vgg16_bn", 2), ("densenet_bc100", 2), ("vit_leaf", 2), ("vit_lite_cct", 2),
            ("tv_resnet18", 6), ("tv_squeezenet1_1", 6), ("tv_mobilenet_v2", 6), ("tv_efficientnet_b0", 6),
            ("tv_regnet_x_400mf", 6), ("tv_googlenet", 6), ("tv_swin_t", 6), ("tv_resnet50", 8),
            ("tv_densenet121", 8), ("tv_inception_v3", 8), ("tv_convnext_tiny", 8), ("tv_wide_resnet50_2", 8),
            ("tv_resnet152", 10), ("tv_vgg11_bn", 10), ("tv_vgg16", 10), ("vit_b16", 10), ("resnet1001_preact", 18))


PRICED_POINT_BUDGET_S, CAPACITY_POINT_BUDGET_S = 600, 5400


LARGE = ("vit_leaf", "vit_lite_cct", "vgg16_bn", "resnet56", "densenet_bc100", "resnet110", "resnet164_preact")
LARGE_ARM_BUDGETS = "1.0,0.5,0.125"

IMAGENET_PRICED = ("tv_resnet18", "tv_resnet50", "vit_b16")
IMAGENET_POINT_BUDGET_S = 1800

CENSUS = ("resnet18_w1q", "resnet20", "resnet56", "vgg16_bn", "densenet_bc100", "resnet110", "resnet164_preact")
FULL_ARMS = ("lenet5", "resnet18_w1q", "resnet20")


CONSTS_PATCH = CL5.parent.parent / "consts" / "consts.patch"
CONSTS_PATCH_SHA256 = "72a8296fcc6786aa7a095b418fc0262d1d748da3c84769f561031c87dbf0e700"
CONSTS_COMMIT = "eeed146e13add4c025c6aaadfa91e3898c7f67cd"

PIN_EXCLUDES = (":(exclude)nevresim", ":(exclude)datasets")

BUNDLE_PROBE = ("import json, sys\nsys.path.insert(0, sys.argv[1])\n"
                "from mimarsinan.mapping.platform.target_bundles import target_bundles\n"
                "from mimarsinan.deployment_record.platform_physics import get_platform_physics\n"
                "out = []\n"
                "for name, bundle in sorted(target_bundles().items()):\n"
                "    v = bundle.values\n"
                "    profile = v.get('platform_physics_profile')\n"
                "    out.append({'name': name, 'physics': profile is not None,\n"
                "                'spiking': v.get('core_semantics') not in (None, 'mvm'),\n"
                "                'allow_coalescing': bool(v.get('allow_coalescing', False)),\n"
                "                'write_time': profile is not None and get_platform_physics(profile).has('t_write_per_unit')})\n"
                "print(json.dumps(out))\n")


ACTIVITY = CL5 / "activity" / "declarations.json"
ACTIVITY_FORMAT = "u-s-activity-declarations/1"
ACTIVITY_FIELDS = ("activity_factor", "simulation_steps", "evidence", "source", "source_sha256", "inputs", "semantics")
ACTIVITY_EVIDENCE = ("MEASURED", "DERIVED")

ROSTER = tuple(dict.fromkeys([n for n, _, _ in PRICED] + [n for n, _ in CAPACITY]))


MEMBERS_FILE = CL5 / "members.json"
MEMBERS_FORMAT = "cl5-members/1"
WRAPPER = HERE / "cl5_ladder.py"
MEMBER_EXCLUDE = ("target", "target_overrides", "quantizer_state")

MEMBER_CORE_EXCLUDE = ("count", "max_axons", "max_neurons")
RA_ROOT = CL5.parents[2]

CONVERTERS_FILE = CL5 / "converters.json"
CONVERTERS_FORMAT = "cl5-converters/1"

LADDER_BUDGETS, LADDER_KS = 9, 3


BLOCKS = CL5.parent.parent / "consts" / "blocks"


def host_block() -> Path:
    ""
    if not os.environ.get("CL5_HEAD_SET"):
        return HOST
    named = os.environ.get("CL5_HOST_BLOCK")
    if not named:
        raise SystemExit("REFUSED: HOST_BLOCK_UNDECLARED: the paper version admits a host row only from an identified "
                         "host description: declare CL5_HOST_BLOCK (cl5_host_block.py's output)")
    try:
        doc = json.loads(Path(named).read_text())
    except (OSError, ValueError) as exc:
        raise SystemExit(f"REFUSED: HOST_BLOCK_UNREADABLE: {named}: {type(exc).__name__}") from None
    if not isinstance(doc, dict) or doc.get("format") != "cl5-host-block/1" or not doc.get("platform_physics_overrides"):
        raise SystemExit(f"REFUSED: HOST_BLOCK_UNREADABLE: {named} is not a cl5-host-block/1 file with its block")
    return Path(named)


def host_block_pin() -> Optional[Dict[str, Any]]:
    ""
    if not os.environ.get("CL5_HEAD_SET"):
        return None
    path = host_block()
    return {"path": str(path), "sha256": _sha256(path), "description": json.loads(path.read_text()).get("description")}


def variants(target: str):
    ""
    host = host_block()
    out = [f"host=host:{host}"]
    out += [f"host+{s}=host:{host},backing_store:{BLOCKS / (s + '.json')}" for s in STORES]
    return out


TREE_PROBE = ("import sys; sys.path.insert(0, sys.argv[1]); "
              "from mimarsinan.deployment_record.platform_physics.loader import COMBINE_RULES, OVERRIDE_FIELDS; "
              "sys.exit(0 if 'slower' in COMBINE_RULES and 'combine' in OVERRIDE_FIELDS else 3)")


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_git(wt: Path, *args: str) -> subprocess.CompletedProcess:
    ""
    return subprocess.run(["git", "-C", str(wt), *args], capture_output=True, text=True, timeout=600)


def tree_probe(wt: Path) -> None:
    ""
    done = subprocess.run([sys.executable, "-B", "-I", "-c", TREE_PROBE, str(wt / "src")], capture_output=True,
                          text=True, timeout=600)
    if done.returncode != 0:
        last = (done.stderr.strip().splitlines() or ["exit %d" % done.returncode])[-1]
        raise SystemExit(f"{wt}: the tree does not compose the consts blocks (combine: slower, the closed "
                         f"declaration fields): {last}")


def require_export(wt: Path, head: Optional[str]) -> Dict[str, Any]:
    ""
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import cl5_head
    names = cl5_head.names()
    if head is not None and head != names["head"]:
        raise SystemExit(f"REFUSED: HEAD_NOT_THE_FUNCTIONS: --head {head} is not the head the declared set's function "
                         f"names ({names['head']})")
    if not CONSTS_PATCH.exists() or _sha256(CONSTS_PATCH) != CONSTS_PATCH_SHA256:
        raise SystemExit(f"{CONSTS_PATCH}: consts.patch is not the pinned 72a8296f (the unit the tree pin names)")
    cs = cl5_head.code_state(wt)
    tree_probe(wt)
    return {"head": names["head"], "base_commit": names["base_commit"],
            "export_record_sha256": cs["export_record_sha256"], "tree_sha256_16": cs["tree_sha256_16"],
            "tree_sha256": cs["tree_sha256"],
            "set": cs["set"], "consts_commit": CONSTS_COMMIT, "consts_patch": str(CONSTS_PATCH),
            "consts_patch_sha256": CONSTS_PATCH_SHA256,
            "consts_basis": "the base commit carries the consts unit (X/cl5/r5/run_record.md: tree pin head e5f6bb37, "
                            "consts commit eeed146e, clean)", "clean": cs["clean"], "excludes": []}


def require_tree(wt: Path, head: Optional[str], *, git: Callable[..., subprocess.CompletedProcess] = run_git
                 ) -> Dict[str, Any]:
    ""
    if os.environ.get("CL5_HEAD_SET"):
        return require_export(wt, head)
    if not head:
        raise SystemExit(f"{wt}: no head of record named (pass --head <the release head R> or set CL5_HEAD): the "
                         "exploration of record runs on the release head only")
    if not CONSTS_PATCH.exists() or _sha256(CONSTS_PATCH) != CONSTS_PATCH_SHA256:
        raise SystemExit(f"{CONSTS_PATCH}: consts.patch is not the pinned 72a8296f (the unit the tree pin names)")
    named = git(wt, "rev-parse", "--verify", f"{head}^{{commit}}")
    if named.returncode != 0:
        raise SystemExit(f"{wt}: --head {head} does not name a commit of the tree's repository")
    want = named.stdout.strip()
    at = git(wt, "rev-parse", "--verify", "HEAD^{commit}").stdout.strip()
    if at != want:
        raise SystemExit(f"{wt}: the tree is at {at[:12]}, which is not at the head of record {want[:12]}")
    if git(wt, "diff", "--quiet", "--ignore-submodules=all", "HEAD", "--", ".", *PIN_EXCLUDES).returncode != 0:
        raise SystemExit(f"{wt}: the tree has uncommitted changes against its head {want[:12]} (outside nevresim "
                         "and datasets): the exploration of record runs on the committed release tree")
    untracked = git(wt, "status", "--porcelain", "--ignore-submodules=all", "--untracked-files=all", "--", "src",
                    "scripts").stdout
    stray = [line for line in untracked.splitlines() if line.startswith("??")]
    if stray:
        raise SystemExit(f"{wt}: untracked code under src or scripts ({len(stray)} files, first {stray[0][3:]!r})")
    if git(wt, "merge-base", "--is-ancestor", CONSTS_COMMIT, want).returncode != 0:
        raise SystemExit(f"{wt}: the head {want[:12]} does not carry the consts unit (commit {CONSTS_COMMIT[:12]}, "
                         "consts.patch 72a8296f, is not its ancestor)")
    tree_probe(wt)
    return {"head": want, "consts_commit": CONSTS_COMMIT, "consts_patch": str(CONSTS_PATCH),
            "consts_patch_sha256": CONSTS_PATCH_SHA256, "clean": True, "excludes": list(PIN_EXCLUDES)}


def tree_bundles(wt: Path) -> List[Dict[str, Any]]:
    ""
    done = subprocess.run([sys.executable, "-B", "-I", "-c", BUNDLE_PROBE, str(wt / "src")], capture_output=True,
                          text=True, timeout=600)
    if done.returncode != 0:
        last = (done.stderr.strip().splitlines() or ["exit %d" % done.returncode])[-1]
        raise SystemExit(f"{wt}: the tree's bundles do not read: {last}")
    return json.loads(done.stdout.strip().splitlines()[-1])


def bundle_sets(bundles: Sequence[Mapping[str, Any]]) -> Dict[str, Tuple[str, ...]]:
    ""
    names = {b["name"]: b for b in bundles}

    def ordered(declared: Sequence[str], keep) -> Tuple[str, ...]:
        first = [n for n in declared if n in names and keep(names[n])]
        return tuple(first + sorted(n for n, b in names.items() if keep(b) and n not in first))

    physics = ordered(PHYSICS, lambda b: b["physics"])
    geometry = ordered(GEOMETRY_ONLY, lambda b: not b["physics"])
    spiking = tuple(n for n in physics if names[n]["spiking"])
    no_coal = tuple(n for n in physics if not names[n]["allow_coalescing"])
    write_time = tuple(n for n in physics if names[n].get("write_time"))
    known = set(PHYSICS) | set(GEOMETRY_ONLY)
    return {"physics": physics, "geometry_only": geometry, "spiking": spiking, "no_coalescing": no_coal,
            "write_time": write_time,
            "added": tuple(sorted(set(names) - known)), "dropped": tuple(sorted(known - set(names)))}


def _ladder_items(block: Mapping[str, Any]) -> List[str]:
    ""
    return [f"activity_factor={json.dumps(float(block['activity_factor']))}",
            f"simulation_steps={int(block['simulation_steps'])}"]


def activity_declarations(path: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    ""
    path = Path(path)
    if not path.exists():
        return {}, {"state": "absent", "name": "ActivityDeclarationsAbsent", "path": str(path),
                    "reason": "no per-network activity declarations on disk (U-S S-4 not frozen): every spiking "
                              "bundle's energy axis keeps its refusal by name (no synaptic_events census)"}
    data = json.loads(path.read_text())
    if data.get("format") != ACTIVITY_FORMAT:
        raise SystemExit(f"{path}: format {data.get('format')!r} is not {ACTIVITY_FORMAT} (U-S's writer)")
    blocks: Dict[str, Dict[str, Any]] = {}
    evidence: Dict[str, Dict[str, Any]] = {}
    for net, block in sorted((data.get("networks") or {}).items()):
        where = f"{path}: {net}"
        if net not in ROSTER:
            raise SystemExit(f"{where}: a network outside the CL5 roster")
        missing = [f for f in ACTIVITY_FIELDS if f not in block]
        if missing:
            raise SystemExit(f"{where}: the block lacks {missing}")
        a, steps = block["activity_factor"], block["simulation_steps"]
        if not isinstance(a, (int, float)) or isinstance(a, bool) or not 0.0 < float(a) <= 1.0:
            raise SystemExit(f"{where}: activity_factor {a!r} lies outside (0, 1]")
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
            raise SystemExit(f"{where}: simulation_steps {steps!r} is not a positive window")
        split = str((block["inputs"] or {}).get("split", "")) if isinstance(block["inputs"], Mapping) else ""
        if not split or "test" in split.lower():
            raise SystemExit(f"{where}: inputs split {split!r} names the test split or no split; activity is "
                             "measured on validation inputs only")
        if block["evidence"] not in ACTIVITY_EVIDENCE:
            raise SystemExit(f"{where}: evidence {block['evidence']!r} is not one of {ACTIVITY_EVIDENCE}")
        if "ladder_declare" in block and list(block["ladder_declare"]) != _ladder_items(block):
            raise SystemExit(f"{where}: ladder_declare {block['ladder_declare']} is not the block's own "
                             f"activity_factor and simulation_steps {_ladder_items(block)}")
        source = Path(block["source"])
        if not source.exists() or _sha256(source) != block["source_sha256"]:
            raise SystemExit(f"{where}: the source record {source} is absent or its sha256 differs from the block's")
        blocks[net] = dict(block)
        check = block.get("executed_check")
        evidence[net] = {"evidence": block["evidence"],
                         "executed_check": None if check is None else str((check or {}).get("status"))}
    return blocks, {"state": "declared", "path": str(path), "sha256": _sha256(path), "networks": sorted(blocks),
                    "evidence": evidence, "refused": dict(data.get("refused") or {})}


def activity_args(name: str, target: str, spiking: Sequence[str], blocks: Mapping[str, Mapping[str, Any]]) -> List[str]:
    ""
    block = blocks.get(name)
    if block is None or target not in spiking:
        return []
    return [arg for item in _ladder_items(block) for arg in ("--declare", item)]


def converter_declarations(path: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    ""
    path = Path(path)
    data = json.loads(path.read_text())
    if data.get("format") != CONVERTERS_FORMAT:
        raise SystemExit(f"{path}: format {data.get('format')!r} is not {CONVERTERS_FORMAT}")
    laws: Dict[str, Dict[str, Any]] = {}
    for chip, entry in sorted((data.get("chips") or {}).items()):
        law = dict(entry.get("readout_law") or {})
        bits = law.get("adc_bits")
        if (not isinstance(bits, int) or isinstance(bits, bool) or bits < 2 or law.get("range") not in ("calibrated",
                                                                                                         "fixed")
                or not isinstance(law.get("mult_bits"), int) or not entry.get("source")):
            raise SystemExit(f"{path}: {chip}'s readout law {law} is not of the registry's form (adc_bits >= 2, range "
                             "calibrated or fixed, mult_bits) or names no source")
        laws[chip] = law
    return laws, {"path": str(path), "sha256": _sha256(path), "chips": laws,
                  "boundaries": dict(data.get("boundaries") or {})}


def converter_args(target: str, converters: Optional[Mapping[str, Mapping[str, Any]]]) -> List[str]:
    law = (converters or {}).get(target)
    return [] if law is None else ["--declare", f"readout_law={json.dumps(law, sort_keys=True)}"]


def member_declared(pc: Mapping[str, Any]) -> Dict[str, Any]:
    ""
    out = {k: v for k, v in pc.items() if k not in MEMBER_EXCLUDE}
    for k, v in (pc.get("target_overrides") or {}).items():
        if k != "cores":
            out[k] = v["value"] if isinstance(v, Mapping) and "value" in v else v
    return dict(sorted(out.items()))


def member_core_fields(pc: Mapping[str, Any]) -> Dict[str, Any]:
    ""
    cores = ((pc.get("target_overrides") or {}).get("cores") or {})
    types = cores.get("value") if isinstance(cores, Mapping) else None
    if not types:
        return {}
    common = [{k: v for k, v in ct.items() if k not in MEMBER_CORE_EXCLUDE} for ct in types]
    keys = set(common[0]).intersection(*common[1:]) if len(common) > 1 else set(common[0])
    return {k: common[0][k] for k in sorted(keys) if all(c.get(k) == common[0][k] for c in common)}


def member_specs(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    ""
    path = Path(path)
    data = json.loads(path.read_text())
    if data.get("format") != MEMBERS_FORMAT:
        raise SystemExit(f"{path}: format {data.get('format')!r} is not {MEMBERS_FORMAT}")
    root = RA_ROOT / data["rows_root"]
    specs = []
    for m in data.get("members") or []:
        row = root / m["row"]
        where = f"{path}: member row {m['row']}"
        if not (row / "config.json").exists() or not (row / "result.json").exists():
            raise SystemExit(f"{where}: absent from {root} (config.json and result.json)")
        result = json.loads((row / "result.json").read_text())
        if result.get("status") != "complete":
            raise SystemExit(f"{where}: its result is {result.get('status')!r}, not complete")
        if m.get("form") not in [n for n, _, _ in PRICED]:
            raise SystemExit(f"{where}: form {m.get('form')!r} is no priced roster network")
        pc = json.loads((row / "config.json").read_text())["platform_constraints"]
        specs.append({"key": ".".join(m["row"].split(".")[:3]), "row": m["row"], "row_dir": str(row),
                      "member": result.get("member"), "target": pc["target"], "form": m["form"],
                      "declared": member_declared(pc), "core_fields": member_core_fields(pc),
                      "mem_gib": int(m["mem_gib"]),
                      "timeout_s": int(m["timeout_s"])})
    return specs, {"path": str(path), "sha256": _sha256(path), "members": [s["key"] for s in specs]}


def _declare_items(declared: Mapping[str, Any]) -> List[str]:
    return [arg for k, v in declared.items() for arg in ("--declare", f"{k}={json.dumps(v, sort_keys=True)}")]


def expected_points(job: Mapping[str, Any]) -> int:
    ""
    args = job["args"]
    budgets = len(_opt(args, "--budgets", ",".join(["b"] * LADDER_BUDGETS)).split(","))
    arms = len(_opt(args, "--arms", "absent").split(","))
    return budgets * arms * max(1, args.count("--variant")) * LADDER_KS


def _opt(args: Sequence[str], name: str, default: str) -> str:
    return args[args.index(name) + 1] if name in args else default


def member_jobs(wt: Path, run_dir: Path, specs: Sequence[Mapping[str, Any]], granularity: Mapping[str, str],
                grids: Optional[Mapping[str, List[int]]] = None) -> List[Dict[str, Any]]:
    ""
    jobs = []
    for spec in specs:
        target, form = spec["target"], spec["form"]
        head = [str(WRAPPER), str(wt), "--member", f"{spec['key']}={spec['row_dir']}", "--networks", spec["key"],
                "--targets", target]
        tail = [a for v in variants(target) for a in ("--variant", v)]
        if target in granularity:
            tail += ["--core-field", f"write_granularity={granularity[target]}"]
        tail += [a for k, v in (spec.get("core_fields") or {}).items()
                 for a in ("--core-field", f"{k}={json.dumps(v, sort_keys=True)}")]
        tail += _declare_items(spec["declared"])
        parts = ((("", "absent,floor,stateless", None),) if form not in LARGE else
                 (("", "absent", None), (":arms", "floor,stateless", LARGE_ARM_BUDGETS)))
        for part, arms, budgets in parts:
            args = head + ["--arms", arms, "--point-budget-s", str(PRICED_POINT_BUDGET_S)]
            if budgets:
                args += ["--budgets", budgets]
            jobs.append({"id": f"m:{spec['key']}:{target}{part}", "member": spec["key"], "row": spec["row"],
                         "out": str(run_dir / "member" / f"{spec['key']}__{target}{part.replace(':', '_')}.jsonl"),
                         "args": args + tail, "mem_gib": spec["mem_gib"], "timeout_s": spec["timeout_s"]})
        grid = (grids or {}).get(target)
        if grid and form in CENSUS:
            budgets = ",".join([f"c:{c}" for c in grid] + ["declared"])
            jobs.append({"id": f"mc:{spec['key']}:{target}", "member": spec["key"], "row": spec["row"],
                         "out": str(run_dir / "member" / f"{spec['key']}__{target}__census.jsonl"),
                         "args": head + ["--arms", "absent", "--budgets", budgets, "--point-budget-s",
                                         str(PRICED_POINT_BUDGET_S)] + tail,
                         "mem_gib": spec["mem_gib"], "timeout_s": spec["timeout_s"], "grid": grid})
    return jobs


def build_jobs(wt: Path, run_dir: Path, sets: Mapping[str, Tuple[str, ...]],
               blocks: Mapping[str, Mapping[str, Any]],
               members: Sequence[Mapping[str, Any]] = (),
               converters: Optional[Mapping[str, Mapping[str, Any]]] = None
               ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    ""
    physics, spiking, no_coal, geometry = (sets["physics"], sets["spiking"], sets["no_coalescing"],
                                           sets["geometry_only"])

    unstated = [t for t in sets.get("write_time", ()) if t not in WRITE_GRANULARITY]
    if unstated:
        raise SystemExit(f"bundles {unstated} state a per-unit write time (t_write_per_unit) and CL5 declares no "
                         "write granularity for their cores (WRITE_GRANULARITY): the write law would refuse every "
                         "write time of theirs")
    granularity = {t: WRITE_GRANULARITY[t] for t in sets.get("write_time", ())}
    script = str(wt / "scripts" / "scaling_ladder.py")
    jobs = []
    departures = []
    for name, mem, timeout in PRICED:
        big = name in LARGE
        for target in physics:
            for departure in ((), ("allow_coalescing=true",)) if target in no_coal else ((),):
                tag = "+coalescing" if departure else ""

                for part, arms, budgets in ((("", "absent,floor,stateless", None),) if not big else
                                            (("", "absent", None), (":arms", "floor,stateless", LARGE_ARM_BUDGETS))):
                    args = [script, "--networks", name, "--targets", target, "--arms", arms,
                            "--point-budget-s", str(PRICED_POINT_BUDGET_S)]
                    if budgets:
                        args += ["--budgets", budgets]
                    for v in variants(target):
                        args += ["--variant", v]
                    if target in granularity:
                        args += ["--core-field", f"write_granularity={granularity[target]}"]
                    for d in departure:
                        args += ["--declare", d]
                    args += converter_args(target, converters)
                    args += activity_args(name, target, spiking, blocks)
                    job = {"id": f"p:{name}:{target}{tag}{part}",
                           "out": str(run_dir / "priced" / f"{name}__{target}{tag}{part.replace(':', '_')}.jsonl"),
                           "args": args, "mem_gib": mem, "timeout_s": timeout}
                    (departures if departure else jobs).append(job)
    jobs += departures
    for name, mem in CAPACITY:
        for packaging, targets in (("spiking", spiking + (ODIN_GENERATED,)),
                                   ("value", tuple(t for t in physics + geometry if t not in spiking))):
            args = [script, "--networks", name, "--targets", ",".join(targets), "--x-only",
                    "--point-budget-s", str(CAPACITY_POINT_BUDGET_S)]
            if packaging == "spiking":
                args += ["--declare", "allow_coalescing=true"]
            jobs.append({"id": f"x:{name}:{packaging}", "out": str(run_dir / "capacity" / f"{name}__{packaging}.jsonl"),
                         "args": args, "mem_gib": mem, "timeout_s": 4 * CAPACITY_POINT_BUDGET_S})


    for name in IMAGENET_PRICED:
        for target in ("generic_estimated_22nm", "neurram", "hermes_64"):
            for side in (None, 112):
                args = [script, "--networks", name, "--targets", target, "--arms", "absent", "--budgets",
                        "1.0,0.125,declared", "--point-budget-s", str(IMAGENET_POINT_BUDGET_S)]
                if side:
                    args += ["--resolution", str(side)]
                for v in variants(target)[:1] + [v for v in variants(target) if "hbm2" in v]:
                    args += ["--variant", v]
                if target in granularity:
                    args += ["--core-field", f"write_granularity={granularity[target]}"]
                args += converter_args(target, converters)
                args += activity_args(name, target, spiking, blocks)
                tag = "" if side is None else f"@{side}"
                jobs.append({"id": f"i:{name}:{target}{tag}",
                             "out": str(run_dir / "imagenet" / f"{name}__{target}{tag}.jsonl"),
                             "args": args, "mem_gib": 6, "timeout_s": 4 * IMAGENET_POINT_BUDGET_S})


    census_jobs = []
    grids: Dict[str, List[int]] = {}
    for target in physics:
        fs = []
        for name in CENSUS:
            path = run_dir / "capacity" / f"{name}__{'spiking' if target in spiking else 'value'}.jsonl"
            if path.exists():
                for line in path.read_text().splitlines():
                    row = json.loads(line)
                    if row.get("target") == target and row.get("packed_tiles"):
                        fs.append(int(row["packed_tiles"]))
        if not fs:
            continue
        lo, hi = max(1.0, min(fs) / 16.0), float(max(fs))
        grid = sorted({max(1, round(lo * (hi / lo) ** (i / 7.0))) for i in range(8)})
        grids[target] = grid
        budgets = ",".join([f"c:{c}" for c in grid] + ["declared"])
        for name in CENSUS:
            args = [script, "--networks", name, "--targets", target, "--arms", "absent", "--budgets", budgets,
                    "--point-budget-s", str(PRICED_POINT_BUDGET_S)]
            for v in variants(target):
                args += ["--variant", v]
            if target in granularity:
                args += ["--core-field", f"write_granularity={granularity[target]}"]
            args += converter_args(target, converters)
            args += activity_args(name, target, spiking, blocks)
            census_jobs.append({"id": f"c:{name}:{target}", "out": str(run_dir / "census" / f"{name}__{target}.jsonl"),
                                "args": args, "mem_gib": 3, "timeout_s": 4 * 3600, "grid": grid})


    first = [j for j in jobs if j["id"].split(":")[1] in FULL_ARMS and j["id"].startswith("p:")]
    rest = [j for j in jobs if j not in first]
    mjobs = member_jobs(wt, run_dir, members, granularity, grids)
    jobs[:] = first + mjobs + census_jobs + rest
    counts: Dict[str, int] = {}
    for j in jobs:
        counts[j["id"].split(":")[0]] = counts.get(j["id"].split(":")[0], 0) + 1
    info: Dict[str, Any] = {"counts": counts}
    if members:
        info["member_points"] = {s["key"]: {"row": s["row"], "target": s["target"], "form": s["form"],
                                            "jobs": {j["id"]: expected_points(j) for j in mjobs
                                                     if j["member"] == s["key"]}} for s in members}
        for v in info["member_points"].values():
            v["expected_points"] = sum(v["jobs"].values())
    return jobs, info


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("worktree", type=Path)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--head", default=os.environ.get("CL5_HEAD"))
    ap.add_argument("--activity", type=Path, default=ACTIVITY)
    ap.add_argument("--members", default=str(MEMBERS_FILE),
                    help="[U-XID] the members' declaration file (cl5-members/1); 'none' writes no member job")
    ap.add_argument("--converters", default=str(CONVERTERS_FILE),
                    help="[U-XID] the declared converters (cl5-converters/1); 'none' declares none")
    args = ap.parse_args(argv)
    wt, run_dir, out = args.worktree, args.run_dir, args.out
    pin = require_tree(wt, args.head)
    bundles = tree_bundles(wt)
    sets = bundle_sets(bundles)
    blocks, activity = activity_declarations(args.activity)
    specs, members_meta = (([], {"state": "none declared (--members none)"}) if args.members == "none"
                           else member_specs(Path(args.members)))
    laws, converters_meta = (({}, {"state": "none declared (--converters none)"}) if args.converters == "none"
                             else converter_declarations(Path(args.converters)))
    jobs, info = build_jobs(wt, run_dir, sets, blocks, specs, laws)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(jobs, indent=1))
    meta = {"format": "cl5-jobs-meta/1", "jobs": str(out), "jobs_sha256": _sha256(out), "script_sha256":
            _sha256(Path(__file__)), "worktree": str(wt), "tree_pin": pin, "bundles": bundles,
            "sets": {k: list(v) for k, v in sets.items()}, "activity": activity,
            "members": {**members_meta, "wrapper": str(WRAPPER), "wrapper_sha256": _sha256(WRAPPER)},
            "converters": converters_meta, **info,
            **({"host_block": host_block_pin()} if os.environ.get("CL5_HEAD_SET") else {})}
    Path(str(out)[:-len(".json")] + ".meta.json" if str(out).endswith(".json") else str(out) + ".meta.json"
         ).write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    print(f"{len(jobs)} jobs -> {out} (head {pin['head'][:12]}, bundles added {list(sets['added'])} dropped "
          f"{list(sets['dropped'])}, activity {activity['state']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
