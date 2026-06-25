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

---

## Wave-6 follow-up — the offload hypothesis RESOLVED + the ResNet-50 region

**Unit:** `wave6/pretrained-validity-sweep`. **Module:** `mimarsinan.models.pretrained_bridge`
(extended with `load_pretrained_resnet50`). All numbers below come straight from the SAME live
instruments (`classify_validity` + `estimate_cores_needed`) over both `encoding_placement` values,
at `3×32×32`, 10 classes (random-weight build, offline-safe — the verdict is structural).

### 1. The offload hypothesis: **REFUTED — RESOLVED-STILL-FLAGGED**

Wave-5 hypothesised that `encoding_layer_placement=offload` would lift ResNet-18's `param_frac`
above 0.50 → VALID (as it did for deep_mlp d8, 0.36→0.99). **It does NOT.** MEASURED:

| ResNet-18 | tier | `param_frac` | `mac_frac` | `research_gap_ops` | `placement_fixable_ops` |
|---|---|---|---|---|---|
| **subsume** | VALID_FLAGGED | **0.422340** | 0.996931 | `[]` | `['Linear']` |
| **offload** | VALID_FLAGGED | **0.423193** | 0.998918 | `[]` | `[]` |

Offload moves `param_frac` by only **~0.0008** (0.4223 → 0.4232) — it stays a param-MINORITY, so the
tier stays **VALID_FLAGGED** under both placements.

**Why offload can't fix it (the precise mechanism).** A host-param breakdown under offload shows the
host param majority is **57.63 % in 11 residual-boundary `Sequential` ComputeOps** (the
shortcut/downsample blocks at each `add` boundary), which the instrument classifies as
**`supported_host`**, NOT as `placement`-fixable encoders. Offload's *only* effect is relocating the
single `placement` Linear encoder (9 541 params ≈ **0.09 %** of the model) on-chip — hence
`placement_fixable_ops` drops `['Linear']`→`[]` while `param_frac` barely moves. The deficit is a
`supported_host` residual structure, not an offloadable encoder, so the wave-5 deep_mlp analogy does
not transfer. The verdict still RESPONDS to placement (the `fixable_ops` change + the small frac lift
prove the number is live, not a constant) — it just doesn't cross the majority line.

Capacity also responds to placement (default 1000-core SUM budget): ResNet-18 needs **504 cores
(subsume) / 651 cores (offload)** — both feasible.

### 2. The ResNet-50 region — bottleneck blocks stay param-MAJORITY → **VALID**

Added `load_pretrained_resnet50` (ImageNet1K_V1, same mappable op set — all `Conv2d.groups==1`, no
attention/LayerNorm/GroupNorm). Its measured descriptor is the instructive CONTRAST to ResNet-18:

| ResNet-50 | tier | `param_frac` | `mac_frac` | `research_gap_ops` | SUM cores @1000 | SCHEDULED |
|---|---|---|---|---|---|---|
| **subsume** | **VALID** | **0.665655** | 0.998093 | `[]` | 1460 — **infeasible** | feasible, **16 phases**, peak **208** |
| **offload** | **VALID** | **0.666060** | 0.998694 | `[]` | 1607 — **infeasible** | feasible, **17 phases**, peak **208** |

ResNet-50's bottleneck blocks (1×1→3×3→1×1 trunk) hold the param **majority** on-chip, so
`param_frac ≈ 0.666 ≥ 0.50` → **VALID (not flagged)** under BOTH placements. **The param-minority
verdict is architecture-dependent, not an intrinsic residual-net property:** BasicBlock (R18) tips
param-minority; Bottleneck (R50) stays param-majority. Same instrument, same shape — only the block
type differs.

Capacity is an honest two-part verdict: ResNet-50 is VALID in PLACEMENT terms but exceeds the default
1000-core single-pool SUM budget (1460/1607 cores). The **SCHEDULED** (fresh-pool-per-phase) path
time-multiplexes it — peak phase **208 cores** fits the budget across **16–17 reprogramming passes** —
so it is deployable on a scheduling-capable chip.

### Verdict

The wave-5 offload hypothesis is **REFUTED** for ResNet-18 (RESOLVED-STILL-FLAGGED): offload does not
lift it to VALID because the host deficit is a `supported_host` residual structure, not an offloadable
encoder. The ResNet-50 region resolves the deeper question — a stock residual net **can** be VALID
(param-majority) without any placement trick; whether it is depends on the block type, and ResNet-50's
bottleneck design keeps the param majority on-chip natively.
