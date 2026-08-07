# Fused-segment per-hop LIF re-timing — dependent cores in one pass

**Date:** 2026-08-05 · **Trigger:** lenet5_baseline hop0/hop1 split after features_5
(user: dependent soft cores must schedule together in a single pass).

**STATUS: IMPLEMENTED 2026-08-07** (activation-semantics branch), with one
refinement over §2: instead of a level loop inside `_run_neural_segment_rate`,
the fused stage carries `retimed_level_stages` — per-depth `HybridStage`s built
by the SAME flush path the split used (`mapping/packing/retimed_levels.py`) —
and the shared stage loop (`hybrid_run/hybrid_stage_runner.py`) runs them
through every backend's EXISTING per-stage path (torch, nevresim, SANA-FE,
Lava alike), so the five-step chain is reused verbatim and all spike-count
certificates stay integer-exact. Verified: split-vs-fused differential
(counts + logits bit-equal, `tests/unit/mapping/test_per_hop_segmentation.py`),
t0_05 all-backend certificates exact=1.0, lenet5_baseline re-run = ONE fused
stage after features_5 at HCM 0.9906.

## 1. What the barrier actually is (traced, all layers)

Arming chain: `spiking_mode` defaults to lif → lif recipe arms `lif_exact_qat`
(conversion_policy) → `recipe_fold._pair_lif_exact_qat_retiming` arms
`lif_per_hop_retiming` (pair is inseparable: unpaired deploys measured −2.5 pp /
parity 0.8438) → `_per_hop_retiming_enabled` → `partition_ir_graph(per_hop=True)`
splits dependent chains into one segment per hop (A/B on the run's IR:
`per_hop=False` yields `[classifier_4_col0: 2 cores]`, one segment).

**The twins are asymmetric — this is the finding:**

- **NF twin** (`LifSegmentPolicy(retime=True)`, segment_policies.py:239-249):
  re-times **per perceptron inside one segment walk** — uniform re-encode of each
  hop's window count, STE-wrapped. It never needed mapping-level splits.
- **Deployed executor** (`SpikingHybridCoreFlow`): within a segment, cores stream
  raw per-cycle trains through latency gating (`active_by_cycle`,
  lif_step.py:145-231) — chip-faithful streaming, NO hop re-encode. The uniform
  re-encode exists ONLY at segment boundaries, as the five-step chain in
  `rate_forward._on_neural_rate`:
  1. `decode_segment_output_torch(counts, T)` (counts→rates)
  2. store / `_assemble_segment_input` (state buffer round-trip)
  3. `normalize_boundary_slices_torch` (wire divisors)
  4. `_apply_input_shifts` → `warn_once_lossy_negative_clamp` → `clamp(0,1)`
  5. `_encode_segment_input` (uniform train)

  Per-hop mapping splits exist **solely to route every hop through that chain**.
  The barrier is an implementation seam, not chip semantics.

**Structural truth at the same time:** the single-pool build creates ONE shared
pool before the segment loop (`_build_single_pool`); hop segments place on
disjoint physical cores of one chip program. `schedule_pass_index=None` on the
lenet5 artifact — no scheduler ran, no reprogramming between hops. Today's cost
is two sequential sim stages + misleading labels, not wasted capacity. (Bad
interaction that DOES waste passes: lif per-hop + a scheduled/weight-reuse
platform would multiply capacity passes per hop; the fix removes it.)

## 2. The fix — Option A, fused mapping with intra-segment hop re-encode

Mapping: lif stops splitting (`per_hop_neural_segments=False` everywhere);
dependent cores pack into one segment/pass. Deployed executor: when
`lif_per_hop_retiming` is armed, `_run_neural_segment_rate` executes the segment
**level-by-level** (levels = existing per-core `latency` order): run level ℓ's
cores over their T-window on the uniform train, collect counts, then apply the
IDENTICAL five-step boundary chain (same functions, same order, same tensors —
levels wired by intra-segment identity maps) to produce level ℓ+1's train.
Work per level = T cycles, exactly what each split segment costs today.

**Bit-exactness argument:** the fused path composes the same functions in the
same order on the same values as the split path; equality is therefore
mechanical, and it is GATED, not assumed:

- differential: fused vs split on every lif tier-0 vehicle — per-hop counts and
  final logits compared as float hex; any mismatch kills the change (O3 rule);
- NF twin untouched (already per-perceptron) → twin parity gates unchanged;
- non-lif modes: predicate never arms → zero change (mvm/ViT pipeline untouched).

What changes shape (intended, not a regression): stage lists (one fused stage
replaces N hop stages), stage-count-bearing reports/GUI views. Deployed
outputs, metrics, elimination records, parity gates: byte-identical, gated.
Labels: hop stages die; capacity sub-segments keep `_capN`; `passes` in
reporting means chip programming passes only.

## 3. Non-goals

- No change to the exact-QAT ↔ re-timing pairing or its recipe defaults (the
  −2.5 pp hole stays closed; only WHERE re-timing executes moves).
- No touch of `synchronized` ([§16] count-domain, already single-eval per hop).
- No mvm/ttfs/lava/sanafe behavior change.

## 4. Verification plan

1. Unit: level-schedule equivalence on constructed 2-3-hop vehicles, split vs
   fused, hex-equal counts per hop (including shifts, wire divisors, clamp
   warns, membrane readout, recording path refusal).
2. Tier-0 lif cells (t0_01/03/05 family): deployed metric + parity gates
   byte-identical vs pre-change goldens.
3. lenet5_baseline re-run: single fused stage after features_5; identical
   accuracy; stage list shows one segment.
