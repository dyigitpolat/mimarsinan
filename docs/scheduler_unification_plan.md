# Scheduler Unification (U-series) — one reprogram-minimizing scheduler

## Status (2026-08-18)

| Stage | State |
|---|---|
| U1 core: unified planner + chooser + builder-consumes-planner + enum removal | pending |
| U2 surface: record/introspection echoes, templates (incl. t1_02 fix), docs | pending |
| U3 ViT relaunch: feasibility probe, eval-cost measurement, seeded search + per-offspring report | pending |

## Owner directive (verbatim, 2026-08-18)

> "why do we have separate schedule policies? we need to have a single policy
> that must minimize weight reprogramming anyway, and it should still be able
> to schedule in other ways too."

Sequencing decision: **unify first**, then the seeded ViT hardware-shape
search runs under the final semantics. Companion decisions: the H4 pack
consolidation is **measure-first** (consolidate only if per-candidate eval at
ViT scale is prohibitive), and the t1_02 template cell is **fixed** to declare
`allow_scheduling` (its 69+69-core chip is only coherent under per-segment
residency; the fixed-mode run as shipped dies at packing).

## Why the enum exists today (and why it dies)

`pool` is the older pure-feasibility mechanism (capacity split, every pass
fully reprogrammed). `bank_clustered` landed in W5.2 as the
reprogramming-minimizing composition for the verified shared-bank structure,
shipped as an opt-in enum to keep every non-declaring platform byte-identical,
then C3 made the enum searchable because it existed. The enum is rollout scar
tissue: `allow_scheduling` is a genuine hardware capability (mid-inference
reprogramming), but the *policy* is toolchain strategy — the mapper's job.
Every template/preset that declares a policy declares `bank_clustered`; `pool`
survives only as the silent default.

## The unified contract

One planner, no policy parameter. Per segment:

1. **Residency composition** via the shared law (`compose_bank_clustered_passes`)
   whenever applicable — every softcore carries a `bank_id`, one latency
   level — and every chunk packs. Adopted UNCONDITIONALLY when it composes:
   the owner directive makes weight programming the primary objective, and
   the token vehicle's own numbers show why pass-count dominance would be
   wrong — a fitting flat pack programs 7 bank copies in 1 pass while the
   residency composition programs 2 copies over 4 passes. Fewer programmed
   copies always (law expansion caps allocation at the instance count);
   pass inflation is bounded by `max_schedule_passes` (the law's floor) and
   its sync/latency cost is priced, so the search sees the trade.
2. **Capacity split** (`split_softcores_by_capacity`, validated) as the
   fallback wherever residency is inapplicable or fails to pack — the
   "schedule in other ways too" half of the directive.

`policy_applied` becomes `residency_applied` — an **outcome** the planner
reports, not a config the operator declares. The E2 residency law
(`resident_passes`) keeps its semantics unchanged.

Named follow-up — priced composition choice: comparing compositions by
priced cost (programming bytes vs sync/latency) would need the physics
pricer, which `mapping/` must not import (AST import-direction wall).
Reprogramming-first is deterministic, layer-clean, identical on both
planes, and is the directive's own priority order; revisit only if a real
vehicle shows the pass-inflation cost dominating.

Applicability note (one behavior delta, conservative direction): the builder
previously checked intra-segment dependencies exactly (core-level source
scan) while the spec plane uses latency-tag equality, which declines a
same-segment multi-depth structure with no actual edges. Unification puts
BOTH planes on the spec check — the planes now agree exactly, at the cost of
declining residency on that edge case (falls back to capacity, correct
program either way).

Standing conservatism (unchanged, stated in-code): shape-only specs are sized
pre-elimination (>= the builder's post-compaction extents), so the candidate
plane can only decline/over-count — never claim residency the hardware cannot
hold. The builder plans over `spec_at_compacted_extent` specs; the two planes
coincide exactly there.

## Builder consumes the planner (the twin-wrapper consolidation)

`_flush_scheduled_subsegments` stops re-deciding: it builds the segment's
specs (the existing `_split_segment_by_capacity` spec machinery, compacted
extents when `ir_graph.layout_softcores` is present), calls THE planner, maps
spec chunks back to `NeuralCore` chunks, and flushes each chunk as a pass.
`mark_bank_residency` + `dedup_resident_stage_matrices` remain as the
builder's core-by-core geometry verification and storage dedup when residency
was adopted. `try_bank_clustered_passes`' decision role ends (the planner is
the decision); `partition_segment_into_passes` is dead code (zero callers)
and is deleted. This completes for scheduling exactly the move the packer
already made: one engine (`run_placement`), one planner, two materializers.

## Consumer inventory (recon 2026-08-18)

- **mapping**: `mapping_structure.py` (ChipCapabilities field, layout_kwargs,
  MappingStrategy), `schedule_policy.py` (planner), `schedule_partitioner.py`
  (dead fn), `hybrid_build_scheduled.py` + `hybrid_build_pool.py` (builder),
  `schedule_bank_clustered.py`, `layout_verification_scheduling.py`,
  `mapping_verifier_hw.py` (param + a `== BANK_CLUSTERED` branch),
  `noc/fragments.py`, `layout_plan.py`, `imc_platforms{,_literature}.py`
  (8 presets declare the enum).
- **config_schema**: `registry/entries_execution.py` (the key),
  `defaults.py:103` (merge list), `retired_keys.py:102` (a retirement message
  that POINTS at the enum — text updated to point at the unified scheduler).
- **pipelining**: `platform_constraints_resolver.py:40` (writes the key into
  pcfg), `deployment_record_assembly.py:143` (identity echo).
- **search**: C3 option axis (`encoding.py`, option-axes tests), `layout_hook`
  docstring, `compilagent_optimizer.py` lever text.
- **deployment_record**: `introspection/payloads.py` (field),
  `introspection/catalog.py` (reads + doc text), `quantities/from_candidate.py`
  (comment only).
- **templates**: `generate.py` (5 `extra_dp` declarations, all
  `bank_clustered`) + regenerated JSONs; t1_02 gains `allow_scheduling: true`.
- **tests**: ~30 files (the parity family, option axes, resolver, retired
  keys, record fixtures, compilagent).

## Migration

`schedule_policy` becomes a scoped retired key (W1.1 mechanism): presence
fails loud with "the scheduler composes residency-first automatically
(capacity split where residency is inapplicable or would inflate passes);
delete the key." Record readers tolerate absence (`.get`); sealed records
keep their historical echo. New records carry the pass structure + residency
flags they already seal — the outcome, not a declared enum.

## Stages

- **U1 (core, one tested commit)**: planner signature drops the param +
  chooser rule; builder consumes planner; mapping-layer param removals;
  ChipCapabilities/MappingStrategy field removal; registry entry -> retired
  key; resolver stops writing; presets drop the enum; C3 axis removed;
  tests-first across the parity family + new chooser pins (incl. a
  stubbed-law inflation case pinning the dominance rule itself);
  mutation-check the chooser guard and the retired-key guard.
- **U2 (surface)**: record/introspection echoes (field -> outcome or absent),
  catalog text, `templates/generate.py` (drop enum declarations; t1_02
  `allow_scheduling: true`) + regeneration, ARCHITECTURE.md per touched
  module, this doc's status table.
- **U3 (relaunch)**: offline feasibility probe of the declared t1_02 chip
  under the unified scheduler (the R6 seed must validate feasible); measure
  per-candidate eval wall at ViT scale (H4 decision point: consolidate only
  if prohibitive); relaunch the seeded NSGA hardware-shape search;
  per-generation, per-offspring report.

## Verification protocol (every stage)

Tests first; suite <=2 min green; `./scripts/typecheck.sh` zero;
ratchets/budgets clean; load-bearing guards mutation-checked; generic-only;
ARCHITECTURE.md per touched module; per-stage commits, no AI-attribution
trailers.
