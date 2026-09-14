"""Phase-D probe: the LIF spike-capacity audit on the offloaded ViT.

Read-only, analytic-only, one small batch. The genuine LIF forward at scale=1.0
reads chance BY DESIGN; the question is whether LIF adaptation CAN recover it.
The structural limit is grid-vs-steps: each perceptron's analytic output is
quantized to grid step ``activation_scale``, so it uses

    levels = R_out / activation_scale        (R_out = 99th-pct |output|)

distinct levels. A signed LIF neuron's spike count over S steps spans ~2S+1
levels, so it can represent the analytic output only if levels <= 2S. When
levels > 2S the spiking forward CANNOT match the analytic at any scale
(structural: need larger S or a coarser analytic) — no amount of adaptation
closes it. levels <= 2S means the 0.011 is a curable calibration gap.

  m = 2S / levels = 2S * activation_scale / R_out   (m>=1 curable; m<1 structural)

Ordered by execution, the first m<1 seam is the collapse origin.

Usage: python scripts/_probes/lif_seam_capacity_audit.py [RUN_DIR] [DEVICE]
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
S = 32
N_SAMPLES = 64

from mimarsinan.pipelining.session import apply_determinism

apply_determinism(0)

model, _ = torch.load(
    f"{RUN}/Activation Quantization.model.pt", map_location="cpu", weights_only=False
)
model = model.to(DEV).eval()

import mimarsinan.data_handling.data_providers  # noqa: F401
from mimarsinan.data_handling.data_provider_factory import BasicDataProviderFactory

PREPROCESSING = {"interpolation": "bicubic", "resize_to": 224, "normalize": "imagenet"}
provider = BasicDataProviderFactory(
    "CIFAR100_DataProvider", "./datasets", seed=0, batch_size=N_SAMPLES,
    preprocessing=PREPROCESSING,
).create()
loader = torch.utils.data.DataLoader(
    provider._get_validation_dataset(), batch_size=N_SAMPLES, shuffle=False
)
x, _ = next(iter(loader))
x = x.to(DEV)

repr_ = model.get_mapper_repr()
perceptrons = list(repr_.get_perceptrons())

# Per-perceptron analytic activation OUTPUT (the value the spike count must encode).
outputs = {}
handles = [
    p.register_forward_hook(
        lambda m, i, o, _p=p: outputs.__setitem__(
            id(_p), o.detach().float().reshape(-1)
        )
    )
    for p in perceptrons
]
with torch.no_grad():
    _ = model(x)
for h in handles:
    h.remove()

# output = rate*scale, rate = spikes/T in [-1,1]; scale = max encodable value,
# grid step = scale/T, so the signal occupies  eff = T*R_out/scale  spike-levels
# (the per-neuron signal-to-quantization-noise ratio). eff >> 1 healthy;
# eff <~ 1 = the residual-branch signal drowns in one-spike quantization noise.
DEAD = 1e-4     # R_out below this: the branch is ~silent (spiking 0 matches analytic)
COLLAPSE = 2.0  # eff spike-levels below this: resolution collapse
print(f"\n=== LIF resolution audit  (RUN={RUN.split('/')[-1]}, S={S}) ===")
print(f"{'idx':>3} {'seam':>26} {'R_out(99pct)':>12} {'act_scale':>10} "
      f"{'eff_levels':>10}  note")
collapsed = []
rows = []
for i, p in enumerate(perceptrons):
    o = outputs.get(id(p))
    if o is None or o.numel() == 0:
        continue
    a = o.abs()
    if a.numel() > 1_000_000:  # torch.quantile caps ~16M; strided subsample
        a = a[:: a.numel() // 1_000_000]
    R = float(a.quantile(0.99))
    sc_t = getattr(p, "activation_scale", None)
    sc = float(torch.as_tensor(sc_t).detach().float().mean()) if sc_t is not None else float("nan")
    eff = (S * R / sc) if (sc and sc > 0) else float("nan")
    name = getattr(p, "name", type(p).__name__)[-26:]
    if R < DEAD:
        note = "silent (ok)"
    elif eff == eff and eff < COLLAPSE:
        note = f"RESOLUTION COLLAPSE (<{COLLAPSE:g})"
        collapsed.append((i, name, eff, sc, R))
    else:
        note = ""
    rows.append((i, name, R, sc, eff, note))
    print(f"{i:>3} {name:>26} {R:>12.4f} {sc:>10.4f} {eff:>10.2f}  {note}")

print("\n--- verdict ---")
live = [r for r in rows if r[2] >= DEAD]
if live:
    decay = ", ".join(f"{r[4]:.0f}" for r in live)
    print(f"eff-levels by depth (live seams): {decay}")
if not collapsed:
    print(f"no live seam below {COLLAPSE:g} spike-levels: S={S} resolves every "
          f"active seam -> the 0.011 is a CURABLE calibration gap; long run justified.")
else:
    print(f"{len(collapsed)} live seam(s) in resolution collapse; first = "
          f"#{collapsed[0][0]} {collapsed[0][1]} (eff={collapsed[0][2]:.2f}, "
          f"scale={collapsed[0][3]:.2f} vs R={collapsed[0][4]:.3f}). The deep "
          f"activation_scale dwarfs the residual-branch signal, so the spike "
          f"count quantizes it toward zero (death cascade). CURABLE iff LIF "
          f"adaptation drives these scales DOWN by ~scale/R x; the concrete "
          f"lever = seed/allow much smaller deep activation_scales, else raise S.")
