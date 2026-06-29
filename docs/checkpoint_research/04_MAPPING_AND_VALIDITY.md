# Mapping & Validity Engineering

Engineering summary of mapping and validity changes since `9383d1bc`. Science detail lives in linked finding docs.

---

## 1. Validity tiers (gate-v2)

**SSOT:** [`mapping/verification/onchip_majority.py`](../../src/mimarsinan/mapping/verification/onchip_majority.py), [`mapping/verification/onchip_fraction.py`](../../src/mimarsinan/mapping/verification/onchip_fraction.py)

Three tiers on **both** parameter and MAC fractions:

| Tier | Condition | Enqueue behavior |
|---|---|---|
| **INVALID** | below 20% on either metric | Rejected at scheduler |
| **VALID_FLAGGED** | 20% ≤ min < 50% | Admitted; records research-gap ops |
| **VALID** | ≥50% both | Admitted |

Wired at:

- **Enqueue:** `scripts/campaign/scheduler.py` → `onchip_precheck()` → `classify_validity()`
- **Pipeline defense-in-depth:** `SoftCoreMappingStep` after IR pruning

**Doc:** [`docs/research/VALIDITY_AUDIT.md`](../research/VALIDITY_AUDIT.md), v2 [`PROGRAM_CHECKPOINT_v2.md`](../research/PROGRAM_CHECKPOINT_v2.md) §2.5

### Per-family on-chip fractions (valid vehicles)

| Vehicle | On-chip (typical) |
|---|---|
| `lenet5` | ~99.1% |
| `deep_cnn` | ~98.5–99.5% |
| `mlp_mixer_core` | ~90.1% |
| `deep_mlp` w64 d4 | ~19.7% (INVALID) |
| ResNet-50 Bottleneck | ~66.6% (VALID) |

---

## 2. Capacity gates (E4)

Two-layer static placement gate prevents doomed GPU jobs.

### Layer 1 — Lower bound estimate

| Path | Role |
|---|---|
| [`mapping/verification/capacity/estimate.py`](../../src/mimarsinan/mapping/verification/capacity/estimate.py) | `estimate_cores_needed` — scheduling-aware |

When `allow_scheduling=true`, admits "feasible-via-scheduling" if peak reprogram phase fits core budget.

### Layer 2 — Dry-run real packer

| Path | Role |
|---|---|
| [`mapping/verification/capacity/dryrun.py`](../../src/mimarsinan/mapping/verification/capacity/dryrun.py) | `dryrun_pack_feasible` — runs actual `build_hybrid_hard_core_mapping` on untrained IR (~1s CPU) |

**Why needed:** lower bound ignores per-perceptron threshold-group fragmentation. 126/1267 campaign runs (~10%, ~40 GPU-h) crashed at Hard Core Mapping with "No more hard cores available".

**Measured:** 60/60 crash configs REJECT, 80/80 done configs ADMIT, zero false rejections.

**Doc:** [capacity_dryrun_gate.md](../research/findings/capacity_dryrun_gate.md)

### Config keys

| Key | Default | Purpose |
|---|---|---|
| `capacity_gate` | on | Master E4 switch |
| `capacity_dryrun_gate` | on | Real packer after estimate admits |
| `allow_scheduling` | per-template | Enables phase-based scheduling |

**Tests:** `tests/unit/mapping/test_capacity_dryrun.py`, `tests/unit/gpu/test_scheduler_capacity_gate.py`

---

## 3. Residual mapping

### Tier-0 — host-side add (default, zero code change)

`y = x + F(x)` deploys as multi-input host-side `ComputeAdapter(operator.add)`. Bit-exact with historical behavior.

**Commit:** `206dc2a8`

### Tier-1 — on-chip merge (default-off)

| Path | Role |
|---|---|
| [`mapping/support/residual_merge.py`](../../src/mimarsinan/mapping/support/residual_merge.py) | `lower_residual_adds_to_onchip_merge` |
| [`mapping/layout/layout_ir_mapping.py`](../../src/mimarsinan/mapping/layout/layout_ir_mapping.py) | Invokes when `onchip_residual_merge=True` |

**Config:** `IRMapping(onchip_residual_merge=True)` — default off → byte-identical.

**Measured characterization:**

- Bounded ~**1/T** re-quant at merge boundary (max Δcount = 1 spike)
- **NOT a collapse fix:** sync retention 75.0% → 45.5% (host-add better)

**Docs:** [D2_tier1_deployable.md](../research/findings/D2_tier1_deployable.md), [residual_tier1_intrinsic_limit.md](../research/findings/residual_tier1_intrinsic_limit.md)

**HCM-fill fix:** raw-input skip read aligned to consuming core latency window (`41f53206`).

---

## 4. D5 — On-chip attention / LayerNorm

| Path | Content |
|---|---|
| [`mapping/onchip_attention/attention_mappability.py`](../../src/mimarsinan/mapping/onchip_attention/attention_mappability.py) | Per-sub-op verdicts |
| [`mapping/onchip_attention/onchip_layernorm.py`](../../src/mimarsinan/mapping/onchip_attention/onchip_layernorm.py) | On-chip mean-centering (`x − μ`) |

**Result:** Only **`layernorm_centering`** is crossbar-realizable. QK score, softmax, attention-value matmul, variance/÷σ remain **host-only** research frontier.

**Doc:** [D5_onchip_attention.md](../research/findings/D5_onchip_attention.md)

---

## 5. Scheduling & weight reuse (D3/E4)

| Path | Role |
|---|---|
| [`mapping/weight_reuse.py`](../../src/mimarsinan/mapping/weight_reuse.py) | Reprogram vs reuse phase classifier |
| D3 scheduled-build probe | `314a8aea` — bit-exact overflow→scheduling |

Enables ImageNet-scale conv at costed phase budget (~16 reprogram + ~142 reuse for ResNet-50).

**Docs:** [D3_scheduled_build_probe.md](../research/findings/D3_scheduled_build_probe.md), [WEIGHT_REUSE_SCHEDULING_DESIGN.md](../research/WEIGHT_REUSE_SCHEDULING_DESIGN.md)

---

## 6. D4 — Structured pruning (default-off)

Opt-in `prune_sparsity` shrinks cores pre-mapping. Measured cost demo: 2.54× scheduling cost reduction at configured sparsity.

**Doc:** [D4_pruning_scheduling_cost.md](../research/findings/D4_pruning_scheduling_cost.md)

---

## 7. GAP-1 attribution (C3)

Joint `(perceptron_output_slice, ir_id)` keying → per-neuron attribution bit-exact under coalescing+output-tiling. Value domain remains exact; production gate is identity-mapping-only.

**Commit:** `1112068b`

---

## 8. Chip lowering invariant (all vehicles)

Across every valid deployment measured to date:

- **NF↔SCM per-neuron mismatch:** 0.0000%
- **torch↔deployed-sim parity:** 1.0

All accuracy loss is in the **conversion fine-tune**, not mapping or simulation. CIFAR confirms this ([05_MEASURED_RESULTS.md](05_MEASURED_RESULTS.md) §A).

---

## 9. A2 semantic-axis screen

Instrument for equivalence-screening semantic config axes before collapse claims.

**Path:** `scripts/campaign/` (A2 screen module)  
**Commit:** `44b6f897`

Used to justify hypervolume axis collapses on fidelity artifacts only (faithfulness vs semantic-knob distinction in ROADMAP).
