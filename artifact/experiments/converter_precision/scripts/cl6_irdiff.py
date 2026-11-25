""

from __future__ import annotations

import collections
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cl6_common as K

FUNCTION_BEARING = {"core_matrix", "threshold", "hardware_bias", "readout_law", "input_sources", "weight_bank_id",
                    "weight_row_slice", "activation_scale", "parameter_scale", "input_activation_scale",
                    "boundary_grid"}


def equal(a, b, depth: int = 0) -> bool:
    ""
    import numpy as np
    import torch
    if depth > 8:
        return pickle.dumps(a) == pickle.dumps(b)
    if isinstance(a, torch.nn.Module) or isinstance(b, torch.nn.Module):
        if not (isinstance(a, torch.nn.Module) and isinstance(b, torch.nn.Module)) or type(a) is not type(b):
            return False
        sa, sb = a.state_dict(), b.state_dict()
        return repr(a) == repr(b) and list(sa) == list(sb) and all(equal(sa[k], sb[k], depth + 1) for k in sa)
    if isinstance(a, dict) and isinstance(b, dict):
        return list(a) == list(b) and all(equal(a[k], b[k], depth + 1) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(equal(x, y, depth + 1) for x, y in zip(a, b))
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.shape == b.shape and bool(
            torch.equal(a.cpu(), b.cpu()))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.shape == b.shape and bool(
            np.array_equal(a, b))
    if hasattr(a, "digest") and hasattr(b, "digest"):
        return a.digest() == b.digest()
    if (hasattr(a, "__dict__") and hasattr(b, "__dict__") and type(a) is type(b) and not isinstance(a, type)
            and not isinstance(a, (int, float, str, bytes))):
        return equal(vars(a), vars(b), depth + 1)
    try:
        r = a == b
        return bool(r) if isinstance(r, bool) else pickle.dumps(a) == pickle.dumps(b)
    except Exception:
        return pickle.dumps(a) == pickle.dumps(b)


def main() -> None:
    a_dir, b_dir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    ir = [pickle.load(open(d / "Soft Core Mapping.ir_graph.pickle", "rb")) for d in (a_dir, b_dir)]
    na, nb = ir[0].nodes, ir[1].nodes
    census = collections.Counter()
    examples = {}
    if len(na) != len(nb):
        census["node_count"] += 1
    for x, y in zip(na, nb):
        va, vb = vars(x), vars(y)
        for k in sorted(set(va) | set(vb)):
            if k not in va or k not in vb or not equal(va[k], vb[k]):
                census[k] += 1
                examples.setdefault(k, f"{x.name}: {str(va.get(k))[:120]!r} vs {str(vb.get(k))[:120]!r}")
    graph_attrs = {k for k in set(vars(ir[0])) | set(vars(ir[1])) if k != "nodes"}
    graph_diff = sorted(k for k in graph_attrs if k not in vars(ir[0]) or k not in vars(ir[1])
                        or not equal(vars(ir[0])[k], vars(ir[1])[k]))
    res = {"nodes": [len(na), len(nb)], "differing_node_attributes": dict(census), "examples": examples,
           "function_bearing_differences": sorted(set(census) & FUNCTION_BEARING),
           "differing_graph_attributes": graph_diff}
    K.write_json(out, res)
    print(json.dumps(res, indent=1)[:3000])


if __name__ == "__main__":
    main()
