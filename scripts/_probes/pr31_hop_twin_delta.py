"""PR31 (calculus §15.13): identical-input hop twin-delta on the FIXED composition.

For hop k: pin hops 0..k-1 to their genuine-walk records in the analytic
forward, capture hop k's analytic output on the genuine-matched input, and
compare with the walk's genuine output g_k. d_k isolates hop-INTERNAL
temporal defect; hops where matched d is small but the k-cut curve dropped
indict the UPSTREAM seam instead.
"""
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path[:0] = ["./src"]

import torch

RUN = "generated/t2_04_stemheal_ab3_phased_deployment_run"
DEV = os.environ.get("PR31_DEV", "cuda:1")
T = 32
V0 = -0.25
HOPS = (0, 1, 2, 3, 6, 9)
N_BATCHES = int(os.environ.get("PR31_BATCHES", "8"))

from mimarsinan.pipelining.session import apply_determinism

apply_determinism(0)

model, _ = torch.load(
    f"{RUN}/Activation Quantization.model.pt", map_location="cpu", weights_only=False
)
model = model.to(DEV)

from mimarsinan.models.nn.activations import LIFActivation

for p in model.get_perceptrons():
    p.activation = LIFActivation(
        T=T, activation_scale=p.activation_scale,
        thresholding_mode="<", firing_mode="Default", bias_mode="on_chip",
        membrane_init=V0,
    ).to(DEV)
model.eval()

import mimarsinan.data_handling.data_providers  # noqa: F401
from mimarsinan.data_handling.data_provider_factory import BasicDataProviderFactory

provider = BasicDataProviderFactory(
    "CIFAR100_DataProvider", "./datasets", seed=0, batch_size=64,
    preprocessing={"interpolation": "bicubic", "resize_to": 224, "normalize": "imagenet"},
).create()
import torch.utils.data as D

val_loader = list(D.DataLoader(provider._get_validation_dataset(), batch_size=64))[:N_BATCHES]

from mimarsinan.spiking.segment_forward import LifSegmentPolicy, SegmentForwardDriver

driver = SegmentForwardDriver(
    model.get_mapper_repr(), T, LifSegmentPolicy(retime=True, phase_dither=True)
)
perceptrons = list(model.get_perceptrons())
thetas = [
    max(float(torch.as_tensor(p.activation_scale).detach().float().mean()), 1e-12)
    for p in perceptrons
]


def _sub(t: torch.Tensor, cap: int = 100_000) -> torch.Tensor:
    flat = t.detach().float().reshape(-1)
    return flat[:: max(1, flat.numel() // cap)].cpu()


stats = {k: [] for k in HOPS}
with torch.no_grad():
    for bi, (x, y) in enumerate(val_loader):
        x = x.to(DEV)
        rec = {}
        driver(x, node_value_recorder=rec)
        pins = {k: rec[id(p)] for k, p in enumerate(perceptrons)}

        for k in HOPS:
            handles = []
            for j in range(k):
                def _pin(_m, _inp, _out, _j=j):
                    return pins[_j]
                handles.append(perceptrons[j].activation.register_forward_hook(_pin))
            captured = {}

            def _cap(_m, _inp, out):
                captured["a"] = out
            handles.append(perceptrons[k].activation.register_forward_hook(_cap))
            model(x)
            for h in handles:
                h.remove()
            g = _sub(pins[k])
            a = _sub(captured["a"])
            n = min(g.numel(), a.numel())
            d = (g[:n] - a[:n]) / thetas[k]
            stats[k].append((float(d.mean()), float(d.abs().mean())))
        print(f"[PR31] batch {bi + 1}/{N_BATCHES}", flush=True)

print(f"{'hop':>4} {'theta':>7} {'d_mean/th':>10} {'d_abs/th':>9}  (matched inputs)")
for k in HOPS:
    dm = sum(s[0] for s in stats[k]) / len(stats[k])
    da = sum(s[1] for s in stats[k]) / len(stats[k])
    print(f"[PR31] {k:>2} {thetas[k]:>7.3f} {dm:>+10.4f} {da:>9.4f}", flush=True)
print("PR31-DONE", flush=True)
