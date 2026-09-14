"""PR15a (calculus §15.10): the k-cut prefix-hybrid tax decomposition.

For cut k: perceptrons 0..k output their GENUINE-walk decoded values (recorded
per batch via node_value_recorder), the suffix runs analytically. acc(k) vs k
localizes the temporal tax per hop; acc(all) must reproduce the genuine census
(endpoint self-check of the instrument itself).
"""
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path[:0] = ["./src"]

import torch

RUN = os.environ.get(
    "PR15_RUN", "generated/t2_04_stemheal_ab3_phased_deployment_run"
)
STATE = os.environ.get("PR15_STATE", "")
N_BATCHES = int(os.environ.get("PR15_BATCHES", "4"))
STRIDE = int(os.environ.get("PR15_STRIDE", "1"))
DITHER = bool(int(os.environ.get("PR15_DITHER", "0")))
V0 = float(os.environ.get("PR15_V0", "0.0"))
DEV = "cuda:0"
T = 32

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
if STATE:
    model.load_state_dict(torch.load(STATE, map_location=DEV))
    print(f"[PR15A] loaded state {STATE}", flush=True)
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
    model.get_mapper_repr(), T, LifSegmentPolicy(retime=True, phase_dither=DITHER)
)
perceptrons = list(model.get_perceptrons())
N = len(perceptrons)
cuts = list(range(0, N, STRIDE))
if cuts[-1] != N - 1:
    cuts.append(N - 1)
print(f"[PR15A] run={RUN} hops={N} cuts={len(cuts)} n={64 * N_BATCHES}", flush=True)
for k, p in enumerate(perceptrons):
    print(f"[PR15A] hop {k}: {type(p.layer).__name__} "
          f"theta={float(torch.as_tensor(p.activation_scale).float().mean()):.3f}",
          flush=True)

correct = {k: 0 for k in cuts}
correct_analytic = correct_genuine = total = 0
with torch.no_grad():
    for bi, (x, y) in enumerate(val_loader):
        x, y = x.to(DEV), y.to(DEV)
        rec = {}
        logits_g = driver(x, node_value_recorder=rec)
        correct_genuine += int((logits_g.argmax(1) == y).sum())
        pins = {}
        for k, p in enumerate(perceptrons):
            v = rec.get(id(p))
            assert v is not None, f"hop {k} missing from the walk recorder"
            pins[k] = v
        logits_a = model(x)
        correct_analytic += int((logits_a.argmax(1) == y).sum())
        total += int(y.numel())

        for cut in cuts:
            handles = []
            for j in range(cut + 1):
                def _pin(_m, _inp, _out, _j=j):
                    return pins[_j]
                handles.append(perceptrons[j].activation.register_forward_hook(_pin))
            logits_h = model(x)
            for h in handles:
                h.remove()
            correct[cut] += int((logits_h.argmax(1) == y).sum())
        del rec, pins
        print(f"[PR15A] batch {bi + 1}/{N_BATCHES} done", flush=True)

print(f"[PR15A] analytic={correct_analytic / total:.4f} "
      f"genuine={correct_genuine / total:.4f}", flush=True)
prev = correct_analytic / total
for cut in cuts:
    acc = correct[cut] / total
    print(f"[PR15A] cut<= {cut:>3}: acc={acc:.4f} (delta {acc - prev:+.4f})", flush=True)
    prev = acc
print(f"[PR15A] endpoint check: cut=all {correct[N - 1] / total:.4f} "
      f"vs genuine {correct_genuine / total:.4f}", flush=True)
print("PR15A-DONE", flush=True)
