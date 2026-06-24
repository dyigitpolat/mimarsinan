# HYPERVOLUME — the versioned, self-auditing coverage hypervolume (P1)

**Version 1** · keystone for the E1 coverage instrument · SSOT:
`src/mimarsinan/chip_simulation/coverage_ledger.py` (`AXES`) — this doc is the
human-readable witness; the code is authoritative. Re-generate the axis table from
`AXES` whenever an axis changes.

## Why this exists (the reviewers' keystone)

Genericity was *asserted, not measured*: the old coverage ledger let an axis **collapse
on a hunch** — an unscreened breadth axis (backend, mapping, pruning, regime) silently
took a single default that auto-matched the tested rows, inflating the coverage
fraction. The fix makes the coverage **DENOMINATOR a function of each axis's SCREENING
STATUS**, so collapse-on-a-hunch is structurally impossible:

- a `SCREENED_COLLAPSED` axis collapses to one representative **only with a linked
  artifact** (constructing one without an artifact RAISES);
- an `ENUMERATED_INTERACTING` axis (proven to interact) is enumerated;
- an `ASSERTED_UNSCREENED` axis (no screen yet) is **also enumerated** — it is counted
  interacting until a P3 screen earns the collapse.

A bigger denominator means a **lower, honest** coverage fraction. No axis can inflate
coverage without a linked artifact.

## The three screening statuses

| status | meaning | denominator effect | artifact required? |
|---|---|---|---|
| `SCREENED_COLLAPSED` | a cheap screen PROVED the values equivalent | collapses to one representative (NOT a cell coordinate) | **YES** — non-empty `screening_artifact` (construction RAISES otherwise) |
| `ENUMERATED_INTERACTING` | PROVEN to interact (the death-cascade firing×sync law; the dual-axis depth×dataset law) | enumerated over its full domain | the proving result is linked where it exists |
| `ASSERTED_UNSCREENED` | no screen run yet — we make NO collapse claim | enumerated over its full domain (counted interacting until P3) | none — it is honestly labelled "ASSERTED-UNSCREENED" |

## The axes (Version 1)

| axis | domain | screening_status | linked artifact (or "ASSERTED-UNSCREENED — counted interacting") |
|---|---|---|---|
| `firing` | {lif, rate, ttfs, ttfs_quantized, ttfs_cycle_based} | `ENUMERATED_INTERACTING` | enumerated interacting (firing×sync death-cascade law; firing×S d_max(S) budget) |
| `sync` | {none, cascaded, synchronized} | `ENUMERATED_INTERACTING` | enumerated interacting (the cascaded↔synchronized gap is a firing×sync joint result) |
| `encoding_placement` | {subsume, offload} | **`SCREENED_COLLAPSED` → `subsume`** | `docs/research/PROGRAM_CHECKPOINT.md` + `E3_SCALE_PROBE.md` — offload==subsume to **~1e-6** under signed integrate-and-fire (value-preserving encode/decode). FIDELITY-ONLY: NOT collapsed for cost/utilization (PROGRAM_PLAN_v2.md §E5 caveat). |
| `quantization` | {none, wq, aq, wq_aq} | `ENUMERATED_INTERACTING` | enumerated interacting (DFQ-for-LIF hurts; the cascade ramp is quantization-sensitive) |
| `pruning` | {dense, pruned} | `ASSERTED_UNSCREENED` | ASSERTED-UNSCREENED — counted interacting (no screen yet → P3) |
| `backend` | {nevresim, sanafe, hcm, lava} | `ASSERTED_UNSCREENED` | ASSERTED-UNSCREENED — parity-locked only in the VALIDATED CORNER; no cross-cell screen → counted interacting until P3 |
| `mapping_strategy` | {packed, identity, neuron_split, coalesced} | `ASSERTED_UNSCREENED` | ASSERTED-UNSCREENED — GAP-1 (coalescing+neuron_split at VGG scale) is KNOWN-CRACKED for attribution → counted interacting until P3 |
| `S` | {4, 8, 16, 32} | `ENUMERATED_INTERACTING` | enumerated interacting (d_max(S)≈0.56√S firing-gain budget). Honest denominator keeps S a wildcard ("any S"), not an enumerated harsher claim. |
| `depth` | {4, 6, 8, 12, 16} | `ENUMERATED_INTERACTING` | enumerated interacting + a cell COORDINATE (a shallow INVALID and a deeper VALID_FLAGGED at one recipe must NOT collapse and mutually demote). Honest denominator keeps depth a wildcard. |
| `vehicle` | {deep_mlp, deep_cnn, lenet5, mlp_mixer_core, vit_b} | `ENUMERATED_INTERACTING` | `docs/research/findings/WS3_depth_firing_gain.md` + `WS1_WS6_breadth_rigor.md` — the dual-axis **depth×dataset law with an ARCHITECTURE-DEPENDENT onset** PROVES the vehicle interacts |
| `dataset` | {mnist, fmnist, kmnist, svhn, cifar10} | `ENUMERATED_INTERACTING` | `docs/research/findings/WS3_depth_firing_gain.md` + `PROGRAM_CHECKPOINT_v2.md` — the dual-axis depth×dataset law (deep_cnn KMNIST d4→d10 gap shrink; dataset-dominant death-cascade) PROVES the dataset interacts with depth |
| `regime` | {from_scratch, pretrained} | `ASSERTED_UNSCREENED` | ASSERTED-UNSCREENED — counted interacting (no dual-regime cross-screen yet → P3) |

Only `encoding_placement` collapses. Every other axis is counted interacting (enumerated)
in the honest denominator.

## Attribution fidelity (per-region, known-cracked NOW)

The instrument distinguishes **value-domain** bit-exactness from **per-neuron
attribution** bit-exactness. The KNOWN-CRACKED regions are marked `VALUE_DOMAIN_ONLY`
(deployed accuracy is bit-exact; the per-neuron reassembly is not):

- **GAP-1**: coalescing + neuron_split at VGG scale (~2% per-neuron attribution scramble;
  value-exact + spike-conserved).
- **residual Tier-1 merge** in the fused-mapping reassembler.

All other regions are full `ATTRIBUTION`.

## Flag owner + aging

Every `VALID_FLAGGED` cell carries flag metadata: `owner` (None ⇒ UNOWNED), `flag_ts`,
and a computed `age_days` (vs the report's `now_ts`; the live ledger writes `ts` as a
Unix-epoch float, which the parser handles). The CI guard fails when an **UNOWNED** flag
ages past the threshold (default 90 days) — a flag must not rot without an owner.

## The honest deep_cnn coverage (measured, Version 1)

Run: `python scripts/campaign/coverage_report.py --vehicle deep_cnn --dataset mnist
fmnist kmnist svhn cifar10 --sync cascaded synchronized [--firing ttfs_cycle_based]`.

| denominator | claimed size | covered | fraction |
|---|---|---|---|
| LEGACY (collapse-on-a-hunch, single default) | 10 | 6 | **0.60** |
| HONEST (firing pinned ttfs_cycle_based; backend/mapping/pruning/regime/quantization enumerated) | 2560 | 6 | **0.0023** |
| HONEST FULL (firing also enumerated over all 5 modes) | 12800 | 8 | **0.000625** |

The legacy 0.60 was a hunch: it collapsed the four unscreened axes
(backend×4·mapping×4·pruning×2·regime×2) plus quantization×4 to single defaults that
auto-matched the only-tested corner. The honest denominator enumerates them (256× / 1280×
bigger), and the fraction drops to **0.0023 over 2560** (firing-pinned) / **0.000625 over
12800** (full). This is the keystone working: the deep_cnn claim is honestly tiny against
the real hypervolume until those axes are actually screened.

## CI guards (`coverage_ci.py`)

- `assert_axes_screening_sound` — every `SCREENED_COLLAPSED` axis has an artifact.
- `assert_no_merged_valid_tiers` — no report headline fuses VALID + VALID_FLAGGED.
- `assert_no_aged_unowned_flags` — no flag aged past threshold without an owner.

`python scripts/campaign/coverage_report.py --ci` runs all three and exits non-zero on
any violation. Enforced by `tests/unit/chip_simulation/test_coverage_self_audit.py`.

## Follow-up (P3)

The four `ASSERTED_UNSCREENED` axes (pruning, backend, mapping_strategy, regime) are the
P3 screening targets: each cheap screen that proves marginal-equivalence flips the axis to
`SCREENED_COLLAPSED` (with the screen as its artifact) and legitimately shrinks the
denominator. Until then the honest fraction counts them interacting.
