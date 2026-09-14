"""Phase-D fix prototype: recalibrate the LIF output currency (activation_scale)
to a QUANTILE of the branch output, the sigma-in-the-op analog at the LIF seam.

Measures genuine LIF accuracy at three settings of activation_scale:
  (a) as-is (stream-referenced, the 0.011 entry),
  (b) q99 of the branch output (the fix),
  (c) analytic (non-spiking) accuracy, the target band.

If (b) >> (a) and approaches (c), the collapse is an output-currency mismatch
and the e2e closes without a long adaptation run.
"""
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path[:0] = ["./src"]

import torch

RUN = sys.argv[1] if len(sys.argv) > 1 else (
    "generated/t2_04_lif_vit_wq_s32_sched_offload_pruned_phased_deployment_run"
)
DEV = sys.argv[2] if len(sys.argv) > 2 else "cuda:0"
QUANTILE = float(sys.argv[3]) if len(sys.argv) > 3 else 0.99
T = 32
N_BATCHES = 4
BS = 64

from mimarsinan.pipelining.session import apply_determinism

apply_determinism(0)

model, _ = torch.load(
    f"{RUN}/Activation Quantization.model.pt", map_location="cpu", weights_only=False
)
model = model.to(DEV).eval()

import mimarsinan.data_handling.data_providers  # noqa: F401
from mimarsinan.data_handling.data_provider_factory import BasicDataProviderFactory

provider = BasicDataProviderFactory(
    "CIFAR100_DataProvider", "./datasets", seed=0, batch_size=BS,
    preprocessing={"interpolation": "bicubic", "resize_to": 224, "normalize": "imagenet"},
).create()
import torch.utils.data as D

loader = list(
    D.DataLoader(provider._get_validation_dataset(), batch_size=BS, shuffle=False)
)[:N_BATCHES]

repr_ = model.get_mapper_repr()
perceptrons = list(repr_.get_perceptrons())

from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.spiking.lif_utils import unwrap_lif_activation
from mimarsinan.spiking.chip_aligned_nf import chip_aligned_segment_forward


def _preact_quantiles(q):
    """Per-perceptron q-quantile of |pre-activation| (the LIF INPUT the scale
    must cover) via a forward_pre_hook on each perceptron.activation."""
    caps = {}
    handles = [
        p.activation.register_forward_pre_hook(
            lambda m, inp, _p=p: caps.setdefault(id(_p), []).append(
                inp[0].detach().float().abs().reshape(-1)
            )
        )
        for p in perceptrons
    ]
    with torch.no_grad():
        for x, _ in loader:
            model(x.to(DEV))
    for h in handles:
        h.remove()
    out = {}
    for p in perceptrons:
        chunks = caps.get(id(p))
        if not chunks:
            continue
        a = torch.cat(chunks)
        if a.numel() > 1_000_000:
            a = a[:: a.numel() // 1_000_000]
        out[id(p)] = float(a.quantile(q)) if q < 1.0 else float(a.max())
    return out


def _install_lif(scale_fn):
    """Replace each perceptron.activation with an entry LIF (rate==1.0) whose
    activation_scale is scale_fn(perceptron)."""
    for p in perceptrons:
        p.activation = LIFActivation(
            T=T,
            activation_scale=torch.tensor(float(scale_fn(p))),
            thresholding_mode="<",
            firing_mode="Default",
            bias_mode="on_chip",
        ).to(DEV)


def _genuine_acc():
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEV), y.to(DEV)
            logits = chip_aligned_segment_forward(model, x, T)
            correct += int((logits.argmax(1) == y).sum())
            total += int(y.numel())
    return correct / total


def _analytic_acc():
    for p in perceptrons:
        lif = unwrap_lif_activation(getattr(p, "activation", None))
        if lif is not None:
            lif.set_cycle_accurate(False)
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEV), y.to(DEV)
            logits = model(x)
            correct += int((logits.argmax(1) == y).sum())
            total += int(y.numel())
    return correct / total


orig_scale = {id(p): float(torch.as_tensor(p.activation_scale).detach().float().mean())
              for p in perceptrons}
analytic_acc = _analytic_acc()          # base activation, before any LIF install
FLOOR = 1e-4

print(f"\n=== LIF activation_scale sweep  (T={T}, n={N_BATCHES*BS}) ===")
print(f"analytic (base activation) acc : {analytic_acc:.4f}\n")
print(f"{'setting':>28} {'genuine acc':>12}")

results = []
# (a) multiples of the current stream-referenced scale
for k in (0.5, 1.0, 2.0, 4.0):
    _install_lif(lambda p, _k=k: max(_k * orig_scale[id(p)], FLOOR))
    acc = _genuine_acc()
    results.append((f"{k:g}x orig(stream)", acc))
    print(f"{f'{k:g}x orig(stream)':>28} {acc:>12.4f}")
# (b) pre-activation (LIF-input) quantiles — the correct coverage reference
for q in (0.99, 0.999, 1.0):
    qq = _preact_quantiles(q)
    label = f"preact q{q:g}" if q < 1.0 else "preact max"
    _install_lif(lambda p, _qq=qq: max(_qq.get(id(p), orig_scale[id(p)]), FLOOR))
    acc = _genuine_acc()
    results.append((label, acc))
    print(f"{label:>28} {acc:>12.4f}")

# ---- input-currency sweep: reproduce the pipeline entry, then recalibrate it ----
import torch.nn as nn
from mimarsinan.models.nn.activations.autograd import ChipInputQuantizer

orig_input_act = {id(p): p.input_activation for p in perceptrons}
orig_in_scale = {
    id(p): float(torch.as_tensor(p.input_activation_scale).detach().float().mean())
    for p in perceptrons
}


def _install_input_quant(scale_fn):
    for p in perceptrons:
        base = orig_input_act[id(p)]
        base = base if not isinstance(base, nn.Identity) else None
        q = ChipInputQuantizer(T=T, activation_scale=torch.tensor(float(scale_fn(p)))).to(DEV)
        p.input_activation = nn.Sequential(base, q) if base is not None else q


_install_lif(lambda p: max(orig_scale[id(p)], FLOOR))  # fix output at best (1x)
print(f"\n--- input-currency sweep (output fixed 1x; reproduce+recalibrate entry) ---")
print(f"{'input setting':>28} {'genuine acc':>12}")
in_pre = _preact_quantiles  # reuse: but pre-act of the LINEAR is the stream; capture below
for k in (1.0, 2.0, 4.0, 8.0):
    _install_input_quant(lambda p, _k=k: max(_k * orig_in_scale[id(p)], FLOOR))
    acc = _genuine_acc()
    results.append((f"inq {k:g}x", acc))
    print(f"{f'inq {k:g}x orig':>28} {acc:>12.4f}")
for p in perceptrons:  # restore
    p.input_activation = orig_input_act[id(p)]

best = max(results, key=lambda r: r[1])
print("\n--- verdict ---")
print(f"best setting: {best[0]} -> genuine {best[1]:.4f}  (analytic {analytic_acc:.4f})")
if best[1] > 0.55:
    print("a static scale recovers the genuine forward toward analytic -> the "
          "collapse is an activation_scale calibration; build the seam at that "
          "reference. NO long adaptation run needed.")
elif best[1] > 0.30 + 0.05:
    print("scale helps but plateaus below analytic -> scale is PART of it; the "
          "residual gap needs adaptation or an attention-seam fix. Partial lever.")
else:
    print("no static scale recovers it -> the collapse is NOT primarily "
          "activation_scale; inspect the attention host seam and the pipeline's "
          "input-quantizer / exact-QAT install (my bare install already reads "
          "0.30 vs the pipeline's 0.011 -> the extra install steps may be the "
          "culprit, not resolution).")
