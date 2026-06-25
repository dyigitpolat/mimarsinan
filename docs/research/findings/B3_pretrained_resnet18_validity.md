# B3 — the pretrained-CNN validity mechanism (ResNet-18, measured)

**Unit:** `wave5/b3-pretrained-bridge` (merged). **Module:** `mimarsinan.models.pretrained_bridge`
(loads a real torchvision model, runs it through the framework's own `classify_validity` +
`estimate_cores_needed` — no training, no GPU deployment). This **adds the `regime=pretrained`
hypervolume region** (vehicle = a stock ImageNet model) and produces an honest measured descriptor.

## The measured result (real ImageNet ResNet-18)

`torchvision` **ResNet-18 (`ResNet18_Weights.IMAGENET1K_V1`)** through the real instruments:

| metric | value | meaning |
|---|---|---|
| tier | **VALID_FLAGGED** | flagged, but NOT for the reason ViT is |
| `param_frac` | **0.4232** | param-MINORITY on-chip (< 0.50) |
| `mac_frac` | **0.9989** | MAC-MAJORITY on-chip (≫ 0.50) |
| `research_gap_ops` | **[]** | **no unsupported op** — conv/bn/relu/pool/linear/residual-add are all mappable; no MultiheadAttention / LayerNorm / GroupNorm; all `Conv2d.groups==1` |
| cores (3×32×32) | 651 | (204 at 3×16×16 — responds to input shape) |

## The insight — a SECOND class of VALID_FLAGGED

The validity gate-v2 splits `VALID_FLAGGED` into two structurally distinct causes, and the pretrained
CNN reveals the second one cleanly:

- **research-gap flag** (e.g. **ViT-B**, 0.33/0.33): flagged because of **unsupported host ops**
  (attention × 12 + LayerNorm × 25). The MAC metric does *not* rescue it (attention is MAC-heavy too).
  This is the genuine research frontier (needs D5 on-chip attention/LN).
- **placement/structural flag** (e.g. **ResNet-18**, 0.42/**0.999**): flagged ONLY because the
  **residual-add segment boundaries push segment-start conv encoders host-side**, dropping the *param*
  fraction below 0.50 while the *MAC* fraction stays at 0.999. There is **no unsupported op**. This is
  the SAME family as the deep_mlp `subsume`→`offload` placement flag — a structural artifact of where
  the encoder is placed, **not** a capability gap.

**Hypothesis (placement-fixable):** because `research_gap_ops==[]` and the deficit is host-side conv
encoders at residual boundaries, an `encoding_layer_placement=offload` (or a residual-boundary encoder
re-placement) is a candidate to lift ResNet-18's `param_frac` above 0.50 → VALID — exactly as it did
for deep_mlp d8 (0.36→0.99). This is a concrete, testable placement question, not a research gap.

## Why this matters for the genericity claim

The pretrained-CNN region is now **classifiable + capacity-estimable by the same instruments** used for
the from-scratch vehicles — evidence the toolchain's measurement is generic across the `regime` axis.
The honest caveat: this lands the **capability** (a pretrained model is mappable + validity-classified);
the actual pretrained **deployment / accuracy** is a GPU follow-up (regime cross-screen → F3, ImageNet →
F4). The `regime` axis therefore remains `ASSERTED_UNSCREENED` in the coverage denominator until a real
from-scratch↔pretrained equivalence run exists — the bridge makes that run *possible*, it does not
substitute for it.
