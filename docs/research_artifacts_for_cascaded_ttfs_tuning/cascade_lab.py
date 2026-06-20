"""Cascade research lab — a fast, isolated, GPU-free harness for studying the
single-spike TTFS cascade's depth-attenuation (the representation limit).

Everything here is float64, deterministic, and runs in <1s for a deep toy cascade,
so candidate fixes can be iterated on without the full pipeline. Two substrates:

* ``attenuation_profile`` — the FIDELITY metric: per-depth ratio of the genuine
  cascade's decoded value to the continuous teacher's activation (mean over a
  batch). A flat ratio of 1.0 = no attenuation = faithful; a ratio that decays
  with depth IS the §2.2 representation limit. No training needed.
* ``toy_task_accuracy`` — the TASK metric: train a small cascade on a synthetic
  linearly-separable-ish task and report continuous-teacher vs genuine-cascade
  test accuracy. Slower (trains), but measures end behavior.

Import path note: pulls the tested builders from ``tests/cascade_fixtures.py``.
"""

from __future__ import annotations

import os
import sys

import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in (
    os.path.join(_REPO, "src"),
    os.path.join(_REPO, "spikingjelly"),
    os.path.join(_REPO, "tests"),
    _REPO,
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cascade_fixtures import (  # noqa: E402
    _SingleSegmentMLP,
    _calibrate_scales,
    cascade_forward,
    install_ttfs_nodes,
)
from mimarsinan.torch_mapping.converter import convert_torch_model  # noqa: E402


def build_cascade(*, depth=6, width=12, in_dim=12, out_dim=6, seed=0, calib_n=128):
    """A converted ``depth``-layer cascade flow + a frozen continuous-teacher
    snapshot of per-perceptron activations. Returns ``(flow, calib_x, teacher_means)``
    where ``teacher_means[k]`` is layer k's continuous activation channel-mean."""
    torch.manual_seed(seed)
    base = _SingleSegmentMLP(depth, width, in_dim, out_dim)
    for m in base.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.uniform_(m.weight, -0.4, 0.4)
            torch.nn.init.uniform_(m.bias, -0.05, 0.05)

    flow = convert_torch_model(base, (in_dim,), out_dim, device="cpu")
    calib_x = torch.rand(calib_n, in_dim, dtype=torch.float64)
    _calibrate_scales(flow, calib_x)

    # Teacher = continuous per-perceptron activation (still ReLU, pre-TTFS-install).
    teacher = _capture_activation_means(flow, calib_x)
    install_ttfs_nodes(flow, 0)  # placeholder; real S set per measurement below
    return base, flow, calib_x, teacher


def _capture_activation_means(flow, x):
    means = {}
    handles = []
    for k, p in enumerate(flow.get_perceptrons()):
        def hook(_m, _i, out, k=k):
            means[k] = out.detach().reshape(-1, out.shape[-1]).double().mean(0)
        handles.append(p.activation.register_forward_hook(hook))
    flow.double().eval()
    with torch.no_grad():
        flow(x.double())
    for h in handles:
        h.remove()
    return means


def _cascade_decoded_means(flow, x, S):
    """Per-perceptron genuine-cascade decoded channel-mean, keyed by depth index."""
    from mimarsinan.models.spiking.training.ttfs_segment_forward import TTFSSegmentForward
    from mimarsinan.spiking.segment_partition import perceptron_of

    rec = {}
    drv = TTFSSegmentForward(flow.get_mapper_repr(), S)
    drv._driver.policy.node_value_recorder = rec
    with torch.no_grad():
        drv(x.double())
    drv._driver.policy.node_value_recorder = None
    by_perc = {id(perceptron_of(n)): v for n, v in rec.items() if perceptron_of(n) is not None}
    out = {}
    for k, p in enumerate(flow.get_perceptrons()):
        v = by_perc.get(id(p))
        if v is not None:
            out[k] = v.reshape(-1, v.shape[-1]).double().mean(0)
    return out


def attenuation_profile(*, depth=6, width=12, S=8, seed=0):
    """Per-depth attenuation ratio = mean(cascade_decoded) / mean(teacher_activation).

    Returns a list of dicts (one per depth) with teacher mean, cascade mean, ratio,
    and absolute gap. Ratio < 1 and decaying with depth IS the representation limit.
    """
    base, flow, x, teacher = build_cascade(depth=depth, width=width, seed=seed)
    install_ttfs_nodes(flow, S)
    cascade = _cascade_decoded_means(flow, x, S)
    rows = []
    for k in range(depth):
        t = teacher.get(k)
        c = cascade.get(k)
        if t is None or c is None:
            continue
        n = min(t.numel(), c.numel())
        tm = float(t[:n].clamp(min=0).mean())
        cm = float(c[:n].mean())
        rows.append({
            "depth": k,
            "teacher_mean": round(tm, 5),
            "cascade_mean": round(cm, 5),
            "ratio": round(cm / tm, 4) if tm > 1e-9 else None,
            "abs_gap": round(abs(cm - tm), 5),
        })
    return rows


def digits_task(*, seed=1, test_frac=0.3):
    """sklearn 8x8 digits (10 classes, ~1797 samples), normalized to [0,1] — a
    real, reliably-learnable small task so the continuous cascade trains high and
    the genuine-cascade gap is the conversion gap, not a training failure.
    Returns (x_train, y_train, x_test, y_test) as float64 / long tensors. in_dim=64."""
    from sklearn.datasets import load_digits

    d = load_digits()
    x = torch.tensor(d.data, dtype=torch.float64) / 16.0   # pixels 0..16 -> [0,1]
    y = torch.tensor(d.target, dtype=torch.long)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(x.shape[0], generator=g)
    x, y = x[perm], y[perm]
    n_test = int(test_frac * x.shape[0])
    return x[n_test:], y[n_test:], x[:n_test], y[:n_test]


# Back-compat alias for callers expecting a generic task name.
def synthetic_task(**_kw):
    return digits_task(seed=_kw.get("seed", 1))


def train_continuous(base, x, y, *, epochs=40, lr=3e-3):
    """Standard backprop on the continuous (ReLU) cascade module (float32)."""
    x = x.float()
    opt = torch.optim.Adam(base.parameters(), lr=lr)
    lossf = torch.nn.CrossEntropyLoss()
    base.train()
    for _ in range(epochs):
        opt.zero_grad()
        lossf(base(x), y).backward()
        opt.step()
    base.eval()
    return base


def _accuracy(logits, y):
    return float((logits.argmax(-1) == y).double().mean())


def conversion_gap(*, depth=6, width=64, in_dim=64, n_classes=10, S=8, seed=0,
                   epochs=120):
    """Train a continuous cascade on digits, convert, and report continuous vs
    genuine-cascade test accuracy + the per-depth attenuation on the TRAINED
    weights. This is the real representation gap the research targets."""
    torch.manual_seed(seed)
    xtr, ytr, xte, yte = digits_task(seed=seed + 1)
    base = _SingleSegmentMLP(depth, width, in_dim, n_classes)
    train_continuous(base, xtr, ytr, epochs=epochs)

    with torch.no_grad():
        cont_acc = _accuracy(base(xte.float()), yte)

    flow = convert_torch_model(base, (in_dim,), n_classes, device="cpu")
    _calibrate_scales(flow, xtr[:256])
    teacher = _capture_activation_means(flow, xte)
    install_ttfs_nodes(flow, S)
    gen_logits = cascade_forward(flow, xte, S)
    gen_acc = _accuracy(gen_logits, yte)

    cascade = _cascade_decoded_means(flow, xte, S)
    profile = []
    for k in range(depth):
        t, c = teacher.get(k), cascade.get(k)
        if t is None or c is None:
            continue
        n = min(t.numel(), c.numel())
        tm = float(t[:n].clamp(min=0).mean())
        cm = float(c[:n].mean())
        profile.append(round(cm / tm, 3) if tm > 1e-9 else None)
    return {
        "depth": depth, "S": S, "cont_acc": round(cont_acc, 4),
        "gen_acc": round(gen_acc, 4), "gap": round(cont_acc - gen_acc, 4),
        "atten_ratio_by_depth": profile,
    }


# PRIMARY BENCHMARK: depth=3 digits cascade. Continuous ~0.944; the genuine
# single-spike cascade COLLAPSES to chance (~0.074) at S<=8 (death cascade) and
# recovers (~0.861) at S=32. A candidate that revives the dead deep layers
# (atten -> 1) at S=8 and lifts genuine toward continuous is a win — measured in
# seconds, no full pipeline.
PRIMARY = dict(depth=3, width=64, in_dim=64, n_classes=10)


if __name__ == "__main__":
    print("=== PRIMARY BENCHMARK: depth=3 digits, continuous vs genuine single-spike ===")
    for S in (4, 8, 16, 32):
        r = conversion_gap(S=S, seed=0, **PRIMARY)
        print(f"S={S:>2}: cont={r['cont_acc']} gen={r['gen_acc']} gap={r['gap']:+.4f} "
              f"atten={r['atten_ratio_by_depth']}")
    print("\n=== full depth x S conversion gap ===")
    for depth in (2, 3, 4, 6):
        for S in (4, 8, 16, 32):
            r = conversion_gap(depth=depth, S=S, seed=0)
            print(f"depth={depth:>2} S={S:>2}: cont={r['cont_acc']} gen={r['gen_acc']} "
                  f"gap={r['gap']:+.4f}  atten={r['atten_ratio_by_depth']}")
    print("\n=== random-init attenuation profile (depth x S) ===")
    for S in (4, 8, 16, 32):
        for depth in (3, 6, 10):
            rows = attenuation_profile(depth=depth, S=S, seed=0)
            ratios = [r["ratio"] for r in rows if r["ratio"] is not None]
            last = ratios[-1] if ratios else None
            head = ratios[0] if ratios else None
            print(f"S={S:>2} depth={depth:>2}: ratio head={head} tail={last} "
                  f"full={[r['ratio'] for r in rows]}")
