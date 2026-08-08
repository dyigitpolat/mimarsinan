# Activation semantics reconceptualization + streamed LIF deployment — engineering plan

**Date:** 2026-08-07 · **Status:** EXECUTED 2026-08-07 (P0 `a2ae7a33`,
P2 `4d9da585`, P3+P1 `f53f0167`, P4+P5 in the closing landing; per-phase
gates recorded in the commit messages). Approved axes (user, 2026-08-07):
two-key family+variant taxonomy · strict interior-host-op ban for streamed
modes · streamed lif is the default and the starter re-bases on a streamable
vehicle · a new spiking-native CNN vehicle joins tier-0.

**Trigger:** the lenet5_baseline two-stage split after `features_5` and the
audit `docs/how_networks_reach_the_chip.md`. The user's verdict: mimarsinan
must be able to deploy **binary-spike-only, streamed LIF SNNs**; the
configuration surface must present activation semantics as a first-class,
comprehensible choice (mvm ↔ spiking at the top; spiking options only when
spiking); and the current "spike-count re-encoding" deployment must become one
honest mode among six, not the silent meaning of "lif".

This plan subsumes `docs/lif_hop_fused_mapping_design.md` (unimplemented,
docs-only commit `32a8e92c`) as its Phase 2.

---

## 0. Verified findings this plan is built on

All verified against source + the `lenet5_baseline_20260805_220253` artifact
(four investigation reports, 2026-08-07):

- **F1 — the taxonomy already exists, scattered.** `spiking_mode ∈ {lif, ttfs,
  ttfs_quantized, ttfs_cycle_based}` (`chip_simulation/spiking_semantics.py:15`)
  × `ttfs_cycle_schedule ∈ {cascaded, synchronized}` (`:115-116`) ×
  `lif_execution_discipline ∈ {streaming, synchronized}` (`:68-72`, ADVANCED,
  derives `"streaming"`) × `lif_per_hop_retiming` (mapping_strategy, ADVANCED).
  The user's six modes are exactly the legal points of these axes — minus the
  one that matters: **end-to-end streamed LIF is not a reachable point.**
- **F2 — the split is retiming, not capacity.** The lenet5 hop0/hop1 stages come
  from `partition_ir_graph(per_hop=True)` (`mapping/layout/segmentation.py:66-96`),
  armed via lif recipe → `lif_exact_qat` → `recipe_fold._pair_lif_exact_qat_retiming`.
  57/120 cores used; `schedule_pass_index=None` everywhere; the scheduler never ran.
- **F3 — the streamed executor already exists *within* a segment.**
  `models/spiking/hybrid/lif_step.py:145-231` streams binary spikes hop-to-hop
  with latency gating (`active_by_cycle`), one cycle per hop, per-core window
  realignment `local_cycle = cycle − latency`. The packed fast twin
  (`executors/packed_cycle.py`) is bit-locked to it. The spike-count
  certificate (`pipelining/core/spike_count_gate.py`) already certifies
  identity-vs-packed **under `discipline="streaming"` at atol=0**.
- **F4 — timing dies at every stage boundary, by construction.**
  `rate_forward._on_neural_rate` writes only `counts/T` to the state buffer
  (`models/spiking/hybrid/rate_forward.py:116-128`); the next stage re-encodes
  a fresh uniform comb (`spiking/segment_boundary.py:248-296`). No queue/event
  machinery exists anywhere. Segment boundaries are created by host ComputeOps
  (`segmentation.py:51-55`) and by the per-hop split (F2).
- **F5 — the NF twin has a cycle-accurate streaming branch.**
  `spiking/segment_policies.py::LifSegmentPolicy` streaming branch (`:229-250`)
  streams trains hop-to-hop with **no latency offsets** (`dt[t]`). The SCM's
  realignment makes consumer-local cycle t consume producer-local cycle t —
  the two are cycle-isomorphic **iff every consumer sits exactly one latency
  step above its producers** (the invariant `_enforce_core_latency_invariant`
  guarantees ≥, not ==; depth-balancing relays close the gap — see §3.5).
- **F6 — the mvm 4-error bug is an authoring-model hole, not a validation bug.**
  `config_schema/registry/domain_rules.py:12-29` is a hardcoded key list +
  prefix tuple, independent of the registry (no per-key domain field). The
  starter pins `encoding_layer_placement`, `simulation_steps`, `target_tq`
  (`gui/wizard/starter_baseline.json:12,26,27`); clicking the mode control pins
  `spiking_mode` (`fields.js:420`). Nothing prunes on `core_semantics` switch
  (`main.js:381-415` has no branch for it); `core_semantics` appears in **zero**
  relevance predicates; emission is verbatim (`wizard/emit.py:43-46`). Every
  error mis-keys to `core_semantics` itself (`resolve.py:50-55` takes the first
  registry token) and carries no remedies. `_vehicle_rows` never consults
  `core_semantics` (`schema_api.py:162-202`).
- **F7 — "Core semantics" is currently the first field cell of the
  "Deployment target" group card**, while "Spiking semantics" is a sibling
  card (`workbench.js:8-19`, `groups.py:16-18,41-43`) — the inversion the user
  called out.
- **F8 — streamable vehicles today**: `simplemlp` fully neural (zero
  ComputeOps); `deepmlp` (host readout Linear only); `mmixcore` (host readout
  mean+Linear suffix only). `lenet5`/`deepcnn` blocked by interior MaxPools;
  `mmix` by interior bare-fc2 seams. Final bare-Linear classifiers are host ops
  under spiking packaging (`packaging_contract.py:63-72`).
- **F9 — measured stakes of the windowed↔streamed gap**: deploying mismatched
  semantics costs ~2.5 pp / train↔deploy agreement 0.8438
  (`docs/how_networks_reach_the_chip.md:93-101`) — which is precisely why
  streamed must be a *trained-for* mode (recipe), not a deploy-time toggle.

---

## 1. The semantic model (SSOT taxonomy)

### 1.1 Authored keys (two-key family + variant; user decision)

```
core_semantics   ∈ {spiking, mvm}          (unchanged key; becomes the top-level switch)
spiking_family   ∈ {lif, ttfs}             (NEW; exists only under core_semantics=spiking)
spiking_variant  ∈ family-scoped set       (NEW; legal set depends on family — legal-value-set law)
    family=lif : {streamed, synchronized}
    family=ttfs: {analytical, quantized, synchronized, cascaded}
```

(Key names `spiking_family`/`spiking_variant` are working names, finalized at
P0 landing; values are final.)

Six legal spiking points, with canonical derived mode ids used for display,
policy tables, backend caps, template tags, and logs:

| canonical id | (family, variant) | semantics | today's encoding |
|---|---|---|---|
| `lif` | (lif, streamed) | **NEW deployable**: end-to-end event streaming, binary spikes on every wire, timing normalized ONLY at encode/readout | unreachable (F1) |
| `lif_sync` | (lif, synchronized) | windowed LIF: count re-encode at boundaries (per-hop when exact-QAT pairing arms, as today); tick loop ≡ staircase, bit-gated | `spiking_mode=lif` (+ discipline knob's staircase path) |
| `ttfs` | (ttfs, analytical) | closed-form TTFS, continuous activation domain | `spiking_mode=ttfs` |
| `ttfs_quantized` | (ttfs, quantized) | closed-form TTFS, quantized activations | `spiking_mode=ttfs_quantized` |
| `ttfs_sync` | (ttfs, synchronized) | per-spike-cycle synchronized schedule, bit-identical to ttfs_quantized | `ttfs_cycle_based × synchronized` |
| `ttfs_cascaded` | (ttfs, cascaded) | streamed greedy TTFS (lossy; kept deliberately) | `ttfs_cycle_based × cascaded` |

Derived predicates (all in `chip_simulation/spiking_semantics.py`; no literal
ladders anywhere else — the existing D2/domain-dispatch ratchet extends to the
new predicates):

```
family(config)                    → lif | ttfs
is_streamed(config)               → id ∈ {lif, ttfs_cascaded}     # the discipline axis
is_windowed(config)               → id ∈ {lif_sync, ttfs_sync}
is_analytical(config)             → id ∈ {ttfs, ttfs_quantized}
activation_domain_quantized(cfg)  → id != ttfs                    # replaces forces_activation_quantization
is_cycle_executed(config)         → id ∈ {lif, lif_sync, ttfs_sync*, ttfs_cascaded}  (*sanafe only, as today)
canonical_mode_id(config)         → the six ids above; INERT under mvm (as today)
```

A frozen `ActivationSemantics(family, variant)` value is resolved once by
`DeploymentPlan.resolve` and threaded to steps/policies; policies re-key:
`spiking_mode_policy.policy_for_*`, `_BACKEND_CAPS`, `ConversionPolicy.derive`,
firing/spike-gen/thresholding legality, `certification_observable`.

### 1.2 Retired keys and migration (loud, remediable, never silent)

Retired: `spiking_mode`, `ttfs_cycle_schedule`, `lif_execution_discipline`,
`lif_per_hop_retiming`. Following the round-6/7 precedent, a document pinning a
retired key gets a **keyed `retired_key` error with a one-click remedy** that
rewrites it (wizard) / a precise message (headless):

| old | remedy |
|---|---|
| `spiking_mode: lif` | `spiking_family: lif, spiking_variant: synchronized` (**preserves semantics** — old lif was windowed) |
| `spiking_mode: ttfs` | `(ttfs, analytical)` |
| `spiking_mode: ttfs_quantized` | `(ttfs, quantized)` |
| `spiking_mode: ttfs_cycle_based` + `ttfs_cycle_schedule: synchronized` | `(ttfs, synchronized)` |
| `spiking_mode: ttfs_cycle_based` + (`cascaded` or absent) | `(ttfs, cascaded)` |
| `lif_execution_discipline: *` | remove key (executor choice is internal to lif_sync; §4) |
| `lif_per_hop_retiming: *` | remove key (folded into lif_sync's recipe pairing; §4) |

No value of any old key maps to the NEW `(lif, streamed)` — reaching streamed
is always an explicit post-migration choice, so no config changes meaning
silently.

Defaults: `spiking_family` derives `lif`; `spiking_variant` derives per family
(`lif → streamed` — the new default per user decision; `ttfs → analytical`).
`_REMOVED_SPIKING_MODES`-style tombstones carry the table above.

### 1.3 What stays

`simulation_steps` (T) and `target_tq` keep their meaning, home (hardware
group), and the divides-invariant. `firing_mode`/`spike_generation_mode`/
`thresholding_mode` keep the legality machinery, re-keyed on the six ids
(streamed lif legal sets = lif_sync's: {Default, Novena} × rate encoders).
`s_allocation` remains ttfs_cascaded-scoped. `spike_phase_dither`,
`lif_membrane_init`, `comparator_half_step`, `lif_membrane_readout` stay
ADVANCED with unchanged semantics.

---

## 2. Streamed LIF — the deployment contract

> **SUPERSEDED in part by §9 (P6, 2026-08-08).** The definition below made
> streaming a NETWORK-level topology requirement. §9 redefines `streamed` as
> the per-Neural-Segment execution discipline; the one-span form survives as
> the reported `end_to_end` special case (`segments == 1`).

**Definition.** One chip program; one continuous run of `C = T + L` cycles
(`L = ChipLatency(mapping).calculate()`); every inter-core message is a binary
spike event; values are transcoded exactly twice — encode (value→uniform comb,
optional dither) before the first neural core, decode (window counts→values)
after the last. No interior count-collapse, no re-emission, no passes, no
reprogramming. `ttfs_cascaded` shares the structural contract (single-spike
discipline, as today's executor).

### 2.1 Structural streamability (strict interior ban; user decision — REVERSED by §9)

A model is **streamable** iff, in IR topological order, host ComputeOps form
only a *prefix* (encode side: subsumed encoder chain) and a *suffix* (readout
head: e.g. deepmlp's bare classifier; mmixcore's mean+classifier), with ONE
contiguous all-`NeuralCore` span between them.

- New `mapping/layout/streamability.py` (small, ≤150 LOC):
  `streamable_span(ir_graph) → NeuralSpan | StreamabilityViolation(ops, positions)`.
  `assert_streamable_ir` raises `NotStreamableError` naming each interior host
  op and its neighbors, with remedies ("replace MaxPool2d with strided conv",
  "use lif_sync", "re-architect the head as suffix").
- Enforced twice, same SSOT: statically at `ModelBuildingStep` (spec-level
  probe, before pretraining — the existing static-gate pattern,
  `model_building_step.py:44`) and at `build_hybrid_mapping_for_pipeline`
  before packing. An `advisories` rule mirrors it as a wizard warning row.
- Segmentation output for a streamed run is therefore
  `[Host prefix]* + ONE NeuralSegment + [Host suffix]*` — no `per_hop`, ever,
  for streamed modes (predicate-gated at `simulation_factory.py:46-66`).

### 2.2 Residency and scheduling

Streamed requires the whole span resident simultaneously:

- `allow_scheduling` legal set under streamed = `{False}` → the field LOCKS
  (legal-value-set law; the |legal|==1 rendering already exists). Bank-clustered
  and capacity pass composition are structurally unreachable.
- Capacity: the existing gate (`verification/capacity/estimate.py`, unscheduled
  SUM bound) applies; overflow → `CapacityExceededError` + the existing
  "Suggest hardware" affordance. No silent fallback to passes.
- Weight sharing stays legal only as *spatial* residency (banks + references,
  as in single-pool today); temporal multiplexing of instances is not.

### 2.3 Executor (SCM)

`_run_neural_segment_rate` + `packed_cycle` already implement streamed
execution for one segment (F3). Changes are confined to:

- **selection**: the `sync_counts` staircase fast path engages only for
  `lif_sync` (today keyed off the retired discipline knob, `lif_step.py:100-105`);
  streamed always takes the cycle loop (dense or packed twin).
- **five-step boundary chain** (`rate_forward.py:76-128`): untouched — with a
  single interior-free segment it runs exactly at encode and decode, which is
  the contract. No new streaming machinery in `rate_forward`.
- **readout**: counts × T → logits as today; membrane-charge readout stays a
  diagnostic excluded from chip claims. (On-chip integrator readout of the
  suffix Linear is explicitly out of scope this round; the suffix is honest
  host post-processing, reported as such.)
- **nevresim**: one neural stage → one compiled chip binary streaming
  end-to-end (`simulation_runner/hybrid.py` unchanged in structure; fewer,
  larger segments). **SANA-FE**: single-segment run, `T_eff = T + L`.
  **Lava/Loihi**: v1 = unavailable-with-reason for streamed (muted-line rule);
  enabling it is a follow-up capability audit (`_BACKEND_CAPS` row).

### 2.4 NF twin, training recipe, and the parity-by-construction claim

- **NF forward for streamed lif** = `LifSegmentPolicy(retime=False)` streaming
  branch over the (single) segment — already cycle-accurate (F5). The recipe
  (`ConversionPolicy` row for `(lif, streamed)`) arms exact-QAT with
  **retime=False** — the training forward IS the deployed forward, so the F9
  −2.5 pp mismatch cannot arise by construction. `cycle_accurate_lif_forward`
  stays derived-ON.
- **Latency isomorphism** (F5): SCM streamed mapping REQUIRES
  `consumer.latency == max(producer.latency) + 1` on the streamed span.
  Where alignment padding would violate it, **depth-balancing relays**
  (existing machinery, `lif_depth_balancing_relays`, relay-liveness gate C5/V9)
  are armed by the streamed recipe to equalize path depths. A structural check
  in `ChipLatency` asserts the invariant for streamed programs.
- **Gates** (extending, not weakening — parity is fixed, never budgeted):
  1. `nf_scm_per_neuron_parity`, streamed branch: per-neuron **window-count
     equality, atol=0** (integer counts; stronger than the analytic-staircase
     branch's float tolerance), NF streaming twin vs identity SCM.
  2. the existing FATAL spike-count certificate (identity vs packed,
     streaming, atol=0) — unchanged, now load-bearing for the flagship mode.
  3. **binary-traffic certificate** (new, certification module): the streamed
     program's inter-core traffic is {0,1} events and the program contains no
     interior value stage — a structural certificate stamped on the run.
  4. a per-cycle **train-equality probe** (diagnostic, few samples): NF hop
     trains vs SCM buffers, cycle-exact — catches any residual alignment bug
     the count gate could mask.
  5. `torch_vs_deployed_sim_parity` as today (argmax agreement).

### 2.5 Honest reporting

Stage lists show `[encode host ops] → stream:neural_span (C = T+L cycles) →
[readout host ops]`. `passes` reporting means chip programming passes only
(zero for streamed). The GUI honest-assembly rail and
`docs/how_networks_reach_the_chip.md` Parts 2–4 are updated: the honesty
ledger gains the streamed row ("timing normalized only at encode/readout;
binary spikes on wire; NF↔SCM window counts exact by gate"), and Part 4's open
decision is recorded as resolved (both arms carried: `lif` streamed default,
`lif_sync` windowed).

---

## 3. lif_sync consolidation (subsumes the fused-mapping design)

- `lif_sync` = today's windowed LIF semantics, unchanged numerically.
- **Fused mapping** (Phase 2 = `docs/lif_hop_fused_mapping_design.md` Option A):
  `per_hop_neural_segments=False` everywhere; when the exact-QAT pairing arms
  re-timing, `_run_neural_segment_rate` executes level-by-level (levels = the
  existing per-core latency order), applying the IDENTICAL five-step chain
  between levels. Bit-exactness gated: differential fused-vs-split per-hop
  counts + final logits as float hex on every lif tier-0 vehicle; goldens
  byte-identical.
- The exact-QAT ↔ re-timing pairing is preserved EXACTLY as today, including
  the Novena downgrade (t0_02 keeps its current no-retime semantics and its
  number). The pairing becomes an internal derivation of the `lif_sync` recipe
  — not a user knob (`lif_per_hop_retiming` retired, §1.2).
- The staircase evaluation (`sync_counts`) and the analytical NF staircase
  remain the fast/training surfaces, bit-gated to the tick loop as today; the
  retired `lif_execution_discipline` knob's role collapses into this internal
  choice.
- Stage labels: hop stages die; `_capN` capacity sub-segments keep labels;
  stage-count-bearing views change shape (intended, not a regression).

---

## 4. Config schema + wizard re-architecture

### 4.1 Registry: a real domain axis (kills the F6 root cause)

- `ConfigKeySchema` gains `domain: Domain = EVENT | VALUE | UNIVERSAL`
  (build-time required — every key declares or defaults UNIVERSAL; a
  build-time error mirrors the provenance rules).
- `domain_rules.py` is rewritten to DERIVE its forbidden sets from the
  registry (error-text contract preserved); the hardcoded list + prefix tuple
  die. `simulation_steps`/`target_tq` are EVENT-domain platform keys;
  `activation_bits` is VALUE-domain (the reverse rule, same mechanism).
- **Generic relevance injection at build time**: every EVENT key's relevance
  is AND-ed with `core_semantics == spiking` (VALUE keys with `== mvm`). No
  per-key edits; relevance controls *existence* (the round-2 rule), so under
  mvm no spiking field renders anywhere — including `spiking_family`,
  `spiking_variant`, T, Tq, `encoding_layer_placement`.

### 4.2 The authoring model on semantics switch (the 4-error bug, fixed)

- **Dormant keys, not errors**: the wizard draft keeps domain-mismatched keys
  in memory but excludes them from resolve and emission; a quiet notice chip
  ("3 spiking keys dormant under MVM — restored on switch back") replaces the
  error cards. Switching `core_semantics` is always zero-error from any green
  state.
- **Emission strips by domain** (`wizard/emit.py`): an emitted document can
  never contain a key its own `core_semantics` forbids. Hand-authored files
  keep strict fail-loud validation (unchanged contract for programmatic
  callers).
- **Error keying + remedies**: domain violations (from loaded files) become
  structured per-key errors `{key, message, remedy: remove_key}` — attached to
  the OFFENDING key (fixing `attach_error_key` first-token mis-keying), with
  one-click remedies, badges landing on the section that actually holds the
  field.
- `_vehicle_rows` consults `core_semantics`: under mvm the spiking simulators
  render structurally-off rows with the reason (recipe already forces them
  off; the display now tells the truth).

### 4.3 Sections and prominence (F7 inversion, fixed)

- The `spiking` group is retitled **"Core semantics"**; `core_semantics` moves
  INTO it as its first, `important` field (segmented MVM | Spiking).
  `spiking_family` and `spiking_variant` render as BASIC segmented controls
  directly under it (variant offers only the family's legal values — existing
  |legal|>1 machinery). The "Deployment target" card keeps vehicles, samples,
  gates — its title finally matches its content.
- The variant picker IS the streaming choice — primary by construction (the
  buried `lif_execution_discipline` ADVANCED knob is retired). Mode-defining
  knobs are never advanced (round-2 rule 1 generalization).
- `spike_phase_dither` stays ADVANCED (it already is); its label/doc are
  rewritten in plain language ("rotates each channel's evenly-spaced spike comb
  by a fixed offset; count-exact; decorrelates simultaneous arrivals").
- Starter contract test extends to **seven single-switches** from the starter:
  {lif, lif_sync, ttfs, ttfs_quantized, ttfs_sync, mvm} (+ ttfs_cascaded
  resolve-only, since casc is descoped from tier-0 runs) — each resolves
  error-free and maps feasibly.

---

## 5. Tier-0, starter, and the new vehicle

### 5.1 Starter re-base (user decision)

The starter re-bases on a **streamable vehicle** with `spiking_family=lif,
spiking_variant=streamed` as its pinned mode-shaped defaults. Selection by a
short robustness study across the seven switches (candidates ranked:
`simplemlp` — fully neural, fastest walls, green in 4 modes today; `deepmlp`
+ offload — 0.99 on-chip; `mmixcore` — flagship-shaped but t0_01 history).
The J1 rule stands: the starter must be green TODAY under every single-switch.

### 5.2 New vehicle: `stream_cnn` (user decision)

Spiking-native MNIST CNN: stride-2 convs replace pools; every feature block
activated (absorbable BN+ReLU); bare-Linear readout suffix; registered
builder + `ModelWorkloadProfile` (provider-side facts; the workload-literal
ratchet stays green). Target: streamable by construction, ≤120-core class
platforms, tier-0 walls.

### 5.3 New tier-0 cells (regenerated via `templates/generate.py` only)

- Existing lif rows are re-tagged `lifsync` (numbers keep: t0_01→
  `t0_01_lifsync_mmixcore_wq_s32`, etc.); ttfs/ttfsq/sync/mvm rows re-key to
  family+variant with unchanged numbers; COVERAGE_NOTES records the rename.
- NEW streamed rows (fresh numbers, t0_45+), covering topology × quant × S:
  `simplemlp` (wq), `deepmlp` (wq, offload), `mmixcore` (wq), `stream_cnn`
  (wq + one fp) — ≥5 streamed cells, at least one `offload` (fully on-chip
  including encoder) and one `subsume`.
- `ttfs_cascaded` stays out of tier-0 (2026-07-12 directive unchanged);
  tier-1 casc rows re-key to (ttfs, cascaded).
- Floors unchanged: campaign PASS = acc ≥ max(0.97, 0.98·pretrain), wall ≤
  300 s. Streamed QAT cost ≈ today's lif cells (exact-QAT already
  cycle-accurate; retime=False is cheaper). Template tests: TestModeCoverage
  gains streamed; representability + assembly-contract tests regenerate.

---

## 6. Verification & no-regression matrix

| Surface | Gate | Phase |
|---|---|---|
| Taxonomy rename is numerically inert | golden resolution snapshot regenerated deliberately; diff audited = key renames only; recipe/policy numeric surfaces identical | P0 |
| Old configs never silently change meaning | retired-key errors + remedy tests for every row of §1.2; representability suite over regenerated tier configs | P0 |
| mvm switch is zero-error | starter×7-switch contract test; emission-strip test (emitted doc + `/api/run?validate=1` clean); dormant-restore round-trip test | P1 |
| lif_sync fused ≡ split | differential per-hop counts + logits as float hex, every lif tier-0 vehicle; goldens byte-identical; lenet5_baseline re-run shows one fused stage after features_5 | P2 |
| streamed NF ≡ SCM | per-neuron window counts atol=0 (real vehicles + constructed unequal-depth branch fixtures); per-cycle train probe; latency invariant test | P3 |
| streamed packed ≡ dense | `test_packed_cycle_equivalence` extended to multi-level streamed fixtures | P3 |
| binary-spike-only | binary-traffic certificate + structural no-interior-stage assertion | P3 |
| deployment quality | tier-0 full sweep green pre-merge of each phase; suite ≤2 min; typecheck 0; ratchets only tighten (module budget honored by the named new small files); ARCHITECTURE.md drift guard | all |

Explicit non-regressions: every existing tier-0 cell keeps its deployed number
(rename-only through P0–P1; P2 bit-gated; P3–P4 add cells, touch no existing
semantics). mvm family untouched except display/validation honesty.

---

## 7. Phasing

- **P0 — taxonomy SSOT + migration** (config_schema registry/derivation/
  validation, spiking_semantics, spiking_mode_policy, ConversionPolicy,
  DeploymentPlan, templates regen, goldens, fixtures). Pure rename, zero
  numeric change. *Acceptance:* §6 rows 1–2.
- **P1 — domain axis + wizard re-architecture** (registry domain field +
  injection, domain_rules rewrite, emit strip, dormant keys, group re-home,
  error keying/remedies, vehicle rows, 7-switch contract). *Acceptance:* §6
  row 3 + screenshot review vs this spec (GUI pixel rule).
- **P2 — lif_sync fused mapping** (segmentation stops splitting; level-by-level
  chain in executor; labels). *Acceptance:* §6 row 4.
- **P3 — streamed deployment** (streamability SSOT + static gate + advisory;
  scheduling legality; recipe row; relay-armed latency invariant; parity gates
  1–5 of §2.4; nevresim/sanafe runs; caps matrix). *Acceptance:* §6 rows 5–7 +
  one full streamed run per candidate vehicle with all gates green.
- **P4 — vehicles + cells + starter** (stream_cnn builder/profile; new t0
  cells; starter re-base + study; campaign integration). *Acceptance:* new
  cells PASS floors; starter 7-switch green; tier-0 sweep green.
- **P5 — docs** (how_networks_reach_the_chip Parts 2–4, root + module
  ARCHITECTURE.md, advisories text). *Acceptance:* drift guards green.

Order: P0 → (P1 ∥ P2) → P3 → P4 → P5. Each phase lands independently green
(suite + typecheck + tier-0 spot check per repo rule 11).

---

## 8. Risks / open items

- **Streamed QAT accuracy on the floors** — the real research risk. Evidence
  for: t0_02 (Novena, unretimed, fp) at 0.9798; streamed targets are small
  vehicles first. If a wq streamed cell misses retention, the cell ships only
  when fixed — floors are not relaxed; the honest fallback is fp cells plus a
  named finding.
- **Latency alignment corner cases** (multi-branch spans, coalesced groups):
  covered by constructed fixtures + the invariant check; relays are the
  designed mechanism, and their liveness gate already exists.
- **nevresim scale**: one fused binary replaces N small ones — expected
  faster; verify compile wall on deepmlp d8 early in P3.
- **Backend caps**: streamed on lava deferred with a stated reason; sanafe
  single-segment path needs a timing audit (`T_eff`).
- **Naming**: final key names (`spiking_family`/`spiking_variant`) and the
  streamed template tag (`lifs` vs `stream`) fixed at P0 review.

## 9. P6 amendment (2026-08-08): `streamed` is the per-Neural-Segment discipline

**User decision (2026-08-08, supersedes §2/§2.1).** Streaming — like every
other core semantics — is a property of *each Neural Segment*, not of the
network: within a segment (one or more latency groups) spikes flow
cycle-by-cycle with no interior transcode; at segment boundaries, host
ComputeOps (pooling, encode, readout) operate on window counts and the next
segment re-encodes — transcodes happen exactly where host ops force them and
nowhere else. A hybrid program `[host]* (NeuralSegment [host]+)* NeuralSegment
[host]*` is therefore fully deployable under `spiking_variant='streamed'`.
The former network-level guarantee is preserved as a *reported property*:
`end_to_end = (neural segment count == 1)` — all five tier-0 streamed cells
keep it. Demanding `end_to_end` (host-free targets) becomes an optional,
later platform constraint, not part of the variant's meaning.

Rationale recorded from the discussion: the strict ban conflated the
execution discipline with span topology; the mapper, the NF, and the SCM were
already per-segment machines (F10–F13), so the ban rejected valid hybrid
mappings (lenet5's pools) without buying correctness. What it *did* buy —
no silent semantic dilution — is preserved by loud span reporting plus the
train-forward == deploy-forward requirement below.

### 9.1 Verified findings the amendment stands on

- **F10 — the NF is already per-segment streamed.** `LifSegmentPolicy`
  (default branch: `retime=False, synchronized=False`) runs each segment as a
  per-cycle raw cascade (`segment_policies.py:229-250`); the
  `SegmentForwardDriver` composes segments through host ComputeOps on
  count-decoded values and re-encodes at entry via the boundary SSOT
  (`normalize_boundary_value` + `uniform_spike_train`). The native model
  forward mirrors this through `compute_boundary`.
- **F11 — the SCM is already per-segment streamed.** `SpikingHybridCoreFlow`
  executes each neural stage's cycle loop (`_run_neural_segment_rate` →
  `run_neural_segment_packed`) and routes ComputeOps through the state
  buffer; multi-segment streaming is exactly the pre-P2 shipped execution.
- **F12 — relays are per-segment by construction.** Depth-balancing operates
  on *intra-segment* gaps (`mapping/latency/depth_balancing.py`), so the
  `+1`-latency invariant holds within every span with no change.
- **F13 — the exactness gate is span-agnostic.** `assert_streamed_nf_scm_
  exact_or_raise` compares per-perceptron window counts over the whole
  network; boundary transcodes are deterministic on counts, so atol=0
  composes across spans unchanged.

**Consequence:** P6 is mostly *removal* (the ban) plus *proof* (a multi-span
exactness fixture) plus *presentation* (span report, mini-view). The recipe
(`streamed_raw_cascade`) and legality (scheduling locked False) are
unchanged; train forward IS deploy forward, now per segment.

### 9.2 Work items

- **P6.1 Gate → report.** `streamability.py` gains
  `streamed_span_report(...)` → `{segments, interior_host_ops, end_to_end}`;
  `NotStreamableError` survives ONLY for the degenerate no-neural-span case.
  `ModelBuildingStep` prints the report (no raise); the two
  `assert_streamable_ir` calls in `simulation_factory` become report/log.
  The advisory rule text flips from "will fail in seconds" to the span
  topology note.
- **P6.2 Multi-span exactness proof.** CPU fixture (tiny conv→pool→fc model,
  interior host op) driven through the real identity mapping; the existing
  streamed gate must hold at atol=0 across ≥2 spans. Whatever breaks gets
  fixed at the SSOT, never budgeted.
- **P6.3 Pipeline wiring.** Nothing attaches `retimed_level_stages` under
  streamed (already per-hop-gated); nevresim/SANA-FE run per-stage cycle
  loops as today; Loihi stays deferred (stated reason unchanged).
- **P6.4 Mini-view honesty.** Layout preview groups latency-group boxes
  inside Neural Segment envelopes (interior levels must be visually distinct
  from separate segments — the 2026-08-08 misreading), host boxes carry op
  *names* (maxpool/encode/readout), and an `end-to-end` badge appears when
  `segments == 1`. Pixel screenshots for user review.
- **P6.5 Acceptance.** lenet5_baseline deploys `streamed` by default again
  (multi-span: conv spans + classifier span, pools as ComputeOps): streamed
  NF↔SCM EXACT atol=0, floors honored. New tier-0 cell `t0_50` (lenet5,
  lifs, multi-span coverage) via `templates/generate.py`. Docs: this section,
  `how_networks_reach_the_chip.md`, module ARCHITECTUREs, error/advisory
  texts. Suite ≤2 min green, typecheck 0, budgets/ratchets clean.

### 9.3 Gates

| Gate | Criterion |
|---|---|
| Multi-span exactness | fixture: NF↔SCM per-neuron window counts, atol=0, ≥2 spans |
| lenet5 streamed run | pipeline completes; streamed EXACT gate green; floor met |
| No windowed regression | windowed lenet5/t0 numbers byte-identical (no shared-path drift) |
| End-to-end property | tier-0 streamed cells report `end_to_end=True` unchanged |
| Presentation | mini-view: segment envelopes, named host ops, badge — pixels reviewed |
