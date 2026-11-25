""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys

sys.dont_write_bytecode = True
P = "/work/research/tcad_paper"
RA = P + "/research_artifacts"
B06 = RA + "/06_build"
CD = B06 + "/codesign"
NEW7 = CD + "/v47_composed_r7"
NEW6 = CD + "/v47_composed_r6"
HV = B06 + "/headline_v46"
HP = B06 + "/headline_v46_pairs"
H34 = B06 + "/headline_v46_r34"
TR = RA + "/12_tcad_readiness"
BR = TR + "/tiers/paper_v6/fix5/study_build"
RR = BR + "/reread"
OUT = RR + "/out"
SR3 = BR + "/sources_r3"
SR4 = BR + "/sources_r4"
SCR = "/scratch/tcad/study_r7/sb11_reread"
CACHE = SCR + "/cache"
EVC = SCR + "/evc"
GENERIC = ("static_rule", "sysmem_rule", "store_state_rule", "period_rule", "state_rule", "noc_rule", "noc_census",
           "closure_rule", "plan_layout")


def sha(path, n=64):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:n]


def load_tree(tree=NEW7, name="cd47_reread7"):
    ""
    if name in sys.modules:
        return sys.modules[name]
    saved = list(sys.path)
    for gen in GENERIC:
        sys.modules.pop(gen, None)
    sys.path.insert(0, tree)
    spec = importlib.util.spec_from_file_location(name, os.path.join(tree, "cd47_common.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    sys.path[:] = saved
    if os.path.dirname(os.path.abspath(mod.__file__)) != os.path.abspath(tree):
        raise SystemExit(f"loaded {mod.__file__}, not from {tree}")
    return mod


def _np(o):
    import numpy as np
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, tuple):
        return list(o)
    if isinstance(o, set):
        return sorted(o)
    return str(o)


def dump(obj, path):
    tmp = path + ".part"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, default=_np)
        fh.write("\n")
    os.replace(tmp, path)
    return sha(path, 8)
