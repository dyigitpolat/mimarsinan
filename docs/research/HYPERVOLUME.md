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
| `pruning` | {dense, pruned} | `ASSERTED_UNSCREENED` | ASSERTED-UNSCREENED — SEMANTIC knob (changes the trained result); cannot collapse on fidelity → counted interacting until a GPU equivalence screen (P3) |
| `backend` | {nevresim, sanafe, hcm, lava} | **`SCREENED_COLLAPSED` → `sanafe`** | `docs/research/findings/backend_cross_sim_screen.md` (+ `.json`) — **FAITHFULNESS axis**: a LIVE cross_sim_parity screen records nevresim/HCM/SCM **AGREE to max_abs_diff=0** on lif + ttfs_cycle_based × identity + neuron_split; the standing multi-backend parity LOCKS (`test_scm_hcm_sim_parity.py`, `test_nf_hcm_per_node_spike_parity_mmixcore.py`, `test_nf_scm_parity_gate.py`, env-gated nevresim/SANA-FE/Lava) establish nevresim/sanafe/lava≡HCM in the corner; lava INAPPLICABLE for TTFS (LIF-only). **FIDELITY-ONLY** (deployed value): faithful sims of one contract must agree — a disagreement is a BUG, not an interaction. Capability + cost NOT collapsed. |
| `mapping_strategy` | {packed, identity, neuron_split, coalesced} | **`SCREENED_COLLAPSED` → `packed`** | `docs/research/findings/mapping_strategy_fidelity_screen.md` — **FAITHFULNESS axis**: `test_torch_sim_fidelity.py` PROVES torch-NF == deployed HCM sim **BIT-EXACT** (float64 atol=0; LIF per-neuron k==k) for identity / neuron_split / axon_fuse across every bit-exact mode × model. Equivalent packings of one contract compute the same deployed value (a mismatch is a packing BUG). **FIDELITY-ONLY**: `coalescing` carries the GAP-1 caveat → per-neuron attribution `VALUE_DOMAIN_ONLY`; cost NOT collapsed. |
| `S` | {4, 8, 16, 32} | `ENUMERATED_INTERACTING` | enumerated interacting (d_max(S)≈0.56√S firing-gain budget). Honest denominator keeps S a wildcard ("any S"), not an enumerated harsher claim. |
| `depth` | {4, 6, 8, 12, 16} | `ENUMERATED_INTERACTING` | enumerated interacting + a cell COORDINATE (a shallow INVALID and a deeper VALID_FLAGGED at one recipe must NOT collapse and mutually demote). Honest denominator keeps depth a wildcard. |
| `vehicle` | {deep_mlp, deep_cnn, lenet5, mlp_mixer_core, vit_b} | `ENUMERATED_INTERACTING` | `docs/research/findings/WS3_depth_firing_gain.md` + `WS1_WS6_breadth_rigor.md` — the dual-axis **depth×dataset law with an ARCHITECTURE-DEPENDENT onset** PROVES the vehicle interacts |
| `dataset` | {mnist, fmnist, kmnist, svhn, cifar10} | `ENUMERATED_INTERACTING` | `docs/research/findings/WS3_depth_firing_gain.md` + `PROGRAM_CHECKPOINT_v2.md` — the dual-axis depth×dataset law (deep_cnn KMNIST d4→d10 gap shrink; dataset-dominant death-cascade) PROVES the dataset interacts with depth |
| `regime` | {from_scratch, pretrained} | `ASSERTED_UNSCREENED` | ASSERTED-UNSCREENED — SEMANTIC knob (changes the trained result); cannot collapse on fidelity → counted interacting until a dual-regime cross-screen (P3) |

**Faithfulness vs semantic — the collapse rule.** `encoding_placement`, `backend` and
`mapping_strategy` collapse because they are **FAITHFULNESS axes**: different
simulators / packings of the **same deployment contract** that must agree on the
**deployed value** (a disagreement is a BUG, not an interaction). Each collapses on a
measured **PARITY/FIDELITY** artifact and is **FIDELITY-ONLY** — capability and
cost/utilization stay frontiers. `pruning` and `regime` are **SEMANTIC knobs** — they
change the *trained result*, so they **cannot** collapse on a fidelity artifact; they
stay `ASSERTED_UNSCREENED` until a real GPU equivalence screen earns the collapse. So the
collapsed set is exactly **{`encoding_placement`, `backend`, `mapping_strategy`}**; every
other axis is counted interacting (enumerated) in the honest denominator.

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
a computed `age_days` (vs the report's `now_ts`; the live ledger writes `ts` as a
Unix-epoch float, which the parser handles), and a `fix_path` (the named resolution step
when the flag has a KNOWN fix). The CI guard fails when an **UNOWNED** flag ages past the
threshold (default 90 days) — a flag must not rot without an owner.

**Placement-fixable flags auto-own.** A `VALID_FLAGGED_placement` cell (the 38 live
flagged cells) is **NOT drift**: it has a KNOWN fix — flip the encoding-layer placement
to `offload`, mapping the host-placed encoder on-chip and un-flagging the cell. The
instrument therefore **auto-assigns** such a flag the standing owner
`program:placement-offload` + `fix_path = "set encoding_layer_placement=offload"`
(`PLACEMENT_FIXABLE_DEFAULT_OWNER` / `PLACEMENT_FIXABLE_FIX_PATH`) when the row names no
explicit owner — so it is never UNOWNED and `assert_no_aged_unowned_flags` stays green. An
explicit row owner wins; a genuine **research-gap** flag (an unsupported host op with no
on-chip SNN mapping) is **not** auto-owned (it is a real open target). A flag that owes
both a placement fix AND a research gap is not auto-resolvable, so it is not auto-owned
either.

## The honest deep_cnn coverage (measured, Version 1)

Run: `python scripts/campaign/coverage_report.py --vehicle deep_cnn --dataset mnist
fmnist kmnist svhn cifar10 --sync cascaded synchronized [--firing ttfs_cycle_based]`.

| denominator | claimed size | covered | fraction |
|---|---|---|---|
| LEGACY (collapse-on-a-hunch, single default) | 10 | 6 | **0.60** |
| HONEST — Wave-2 (firing pinned ttfs_cycle_based; **backend + mapping_strategy COLLAPSED**; quantization/pruning/regime enumerated) | **160** | 6 | **0.0375** |
| HONEST FULL — Wave-2 (firing also enumerated over all 5 modes) | **800** | 8 | **0.01** |

*(All numbers MEASURED by `scripts/campaign/coverage_report.py` against the live
`runs/campaign/ledger.jsonl`.)*

The legacy 0.60 was a hunch: it collapsed the four unscreened axes
(backend×4·mapping×4·pruning×2·regime×2) plus quantization×4 to single defaults that
auto-matched the only-tested corner. Before Wave-2 the honest denominator enumerated ALL
of them at **2560** (firing-pinned) / 12800 (full), for **0.0023 / 0.000625**.

**Wave-2 re-pricing.** Two of those axes — `backend` and `mapping_strategy` — are
**FAITHFULNESS** axes and now legitimately collapse on a measured fidelity artifact
(backend cross-sim screen; the torch↔sim bit-exact lock). Collapsing **backend 4→1** and
**mapping_strategy 4→1** shrinks the firing-pinned denominator by exactly **16×**: **2560
→ 160**, so the honest fraction rises **0.0023 → 0.0375** (= 6/160); the full denominator
goes **12800 → 800** (= 8/800 = **0.01**). The covered-cell tally is **unchanged** — the
collapse re-prices the *claim* (the artifact-backed denominator), not the coverage.
`quantization × pruning × regime` (4·2·2 = 16) remain enumerated: quantization interacts
with the firing law, and pruning/regime are SEMANTIC knobs awaiting a real GPU
equivalence screen. The deep_cnn claim is still honestly small against the real
hypervolume until those remaining axes are screened.

## CI guards (`coverage_ci.py`)

- `assert_axes_screening_sound` — every `SCREENED_COLLAPSED` axis has an artifact.
- `assert_no_merged_valid_tiers` — no report headline fuses VALID + VALID_FLAGGED.
- `assert_no_aged_unowned_flags` — no flag aged past threshold without an owner.

`python scripts/campaign/coverage_report.py --ci` runs all three and exits non-zero on
any violation. Enforced by `tests/unit/chip_simulation/test_coverage_self_audit.py`.

## Follow-up (P3)

Two faithfulness axes (`backend`, `mapping_strategy`) collapsed in **Wave-2** on measured
fidelity artifacts (above). The remaining `ASSERTED_UNSCREENED` axes are `pruning` and
`regime` — both **SEMANTIC knobs** that change the trained result, so they cannot collapse
on a fidelity artifact. They are the P3 screening targets: each requires a real **GPU
equivalence screen** (pruned≡dense, from_scratch≡pretrained-bridge on the deployed metric)
to flip to `SCREENED_COLLAPSED` and legitimately shrink the denominator. Until then the
honest fraction counts them interacting.
