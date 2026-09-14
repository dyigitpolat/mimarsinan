"""Run the seam-certificate auditor on the cached sigma-armed t2_04 AQ model
(PR4 pre-read + Phase-I DoD smoke): value-domain certificates for every armed
ViT seam on one CIFAR-100 batch.
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
BS = 32
T = 32

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

x, _y = next(iter(D.DataLoader(provider._get_validation_dataset(), batch_size=BS)))
x = x.to(DEV)

from mimarsinan.spiking.seam_audit import audit_model

with torch.no_grad():
    ledger = audit_model(model, T, x)

counts: dict = {}
for c in ledger.certificates:
    counts[(c.kind, c.classification)] = counts.get((c.kind, c.classification), 0) + 1
print(f"\n=== seam audit: {len(ledger.certificates)} certificates ===")
for (kind, cls), n in sorted(counts.items()):
    print(f"  {kind:>10} {cls:>2}: {n}")

print("\n--- Type-B (convention defects) ---")
for c in ledger.type_b:
    print(f"  {c.site:>40} {c.kind:>10} d_mean={c.delta_mean:.4g} {c.note}")
if not ledger.type_b:
    print("  none")

print("\n--- Type-C (capacity) ---")
for c in ledger.certificates:
    if c.classification == "C":
        print(
            f"  {c.site:>40} {c.kind:>10} oob={c.oob_fraction:.3f} "
            f"kappa={c.kappa:.3g} sigma={c.sigma:.3g} {c.note}"
        )

print("\n--- host_twin (armed SNW) certificates ---")
for c in ledger.by_kind("host_twin"):
    print(f"  {c.site:>40} {c.classification:>2} rel={c.delta_mean:.3g} kappa={c.kappa:.3g}")

print("\n--- boundary sigma/oob per entry seam ---")
for c in ledger.by_kind("boundary"):
    print(
        f"  {c.site:>44} {c.classification:>2} oob={c.oob_fraction:.3f} "
        f"bias={c.delta_mean:.4g} kappa={c.kappa:.3g}"
    )
