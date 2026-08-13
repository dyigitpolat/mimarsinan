# Co-Optimization CAD Program — from a deployment recorder to a multi-objective EDA tool

2026-08-13 · status: **approved plan of record** · baseline `c2909f76` (physics P0–P1
landed). Closes the verified gap between the current state and a true multi-objective
co-optimization CAD/EDA tool and consolidates every remaining open item into one
program. Supersedes the staging table (§8) of `target_platform_physics_plan.md`:
P2–P6 land here as stages C0–C7. Owner decisions were asked and answered before
approval (§"Owner decisions").

## Context — the verified gap

The W0–W5 program built the measurement half of a CAD tool: a typed, sealed
`DeploymentRecord` (the join of schedule/placement/utilization/traffic/timing/energy/
accuracy), an objectives registry v2 over two completenesses (`CandidateStaticView` |
`DeploymentRecordView`), a search surface whose decode routes through the same
`build_platform_constraints_resolved` SSOT deployment uses, and plug-and-play optimizers
(NSGA-II, compilagent) behind `EncodedProblem`. Physics P0–P1 added the vendor's half:
evidence-carrying per-unit constants (`platform_physics/`), resolved once and carried
verbatim in the record identity.

Three gaps, verified in-code on 2026-08-13:

1. **Physics is inert.** `deployment_record/cost/*.py` has zero references to
   `platform_physics`; the cost model prices with global default bands. A run declaring
   `platform_physics_profile: truenorth` produces cost output byte-identical to one
   declaring nothing. No mm²/J/s/samples-s axis exists anywhere.
2. **Search sees none of the chip-designer metrics.** All search modes expose only the
   legacy-8 static/proxy axes. Every record axis declares `requires = the sealed
   record's … fragment` — structurally unreachable from a candidate, which has no record.
3. **The searched space is a slice of the deployment-configuration hypervolume.**
   `encoding_placement` (moves ~75 % of parameters across the NeuralOps/ComputeOps
   boundary — measured 25.5 % vs 100 % on-chip), `schedule_policy`, `weight_bits`,
   `pruning_fraction` are fixed problem inputs, not decision variables. The on-chip
   floor is discovered as a pipeline crash, not seen as a constraint.

Thesis tie: this program IS the machinery of the thesis' co-optimization chapters
(hardware-agnostic IMC deployment workflow; deployment = schedule of passes over
NeuralOps/ComputeOps; targets bring cost parameters). Every stage produces evidence the
thesis can cite.

## Design thesis — one pricing engine, two completenesses, EDA discipline

The EDA analogy is load-bearing, not decorative. Map each missing piece onto the
mechanism EDA settled on decades ago:

| EDA mechanism | This program's realization |
|---|---|
| Liberty file (vendor cell characterization) | `PlatformPhysics` profile (exists, P0–P1) |
| Static timing/power analysis — one engine at synthesis (estimates) and signoff (extracted) | **One analytical pricer** over a `Quantities` surface produced by BOTH a candidate layout (static, partial) and a sealed record (measured, complete) |
| Switching-activity assumption for pre-simulation power | Declared `activity_factor` constant for candidate-time spike-dependent energy; measured spikes at record time |
| Wireload model before placement | (later stage) hop-count estimator for candidate NoC terms |
| SDC constraints; infeasible ≠ crash | Typed feasibility constraints (`onchip_min_fraction`, capacity) surfaced to the optimizer |
| Estimate↔signoff correlation reports | **Fidelity contract**: per sealed run, re-evaluate the candidate view of its own config and compare axis-by-axis |
| QoR / cross-corner reports | Cross-platform Pareto comparison artifacts disclosing per-constant evidence kinds |

The one new load-bearing abstraction is the **quantity catalog**: the named multiplicands
(cells, MACs, spikes, hops, bytes, passes, timesteps, host walls…) as a typed surface
with availability and provenance, extracted from either completeness. The physics
vocabulary already declares WHICH quantity each constant multiplies (the `multiplicand`
column) — this program makes that column an executable, test-pinned contract instead of
documentation. Pricing becomes a pure function:

```
price(quantities: Quantities, physics: PlatformPhysics | None) -> absolute CostTerms
```

Every absolute `CostTerm` = constant band × quantity, with provenance = join(quantity
provenance, constant evidence), the SUPERSEDES rule resolved at pricing time, and
refusals that name exactly the missing constant or quantity. Objectives attach through
the EXISTING `cost_term(group, name)` backing — no parallel plumbing.

## Owner decisions (asked and answered before this plan was finalized)

1. **C3 axes**: all four cost axes — `encoding_layer_placement`, `schedule_policy`,
   `weight_bits`, `pruning_fraction`.
2. **Band policy**: optimizer sees the **nominal corner**; full bands ride into results,
   Pareto artifacts, and fidelity reports; corner selectable per run config later.
3. **Host pricing**: **declared host rates** — the vocabulary's `host` group gains
   host-rate constants (operator-declared, like `p_host`); candidate host MACs price
   host statically; when placement is searched with host undeclared, host-inclusive
   axes are refused by name.
4. **Fidelity**: **report-first** — only structural equalities gate (pass counts);
   numeric in-band gates decided later on accumulated evidence.

## The verified search seams this plan builds on

- **Objective flow**: `deployment_parameters.arch_search.objectives` →
  `architecture_search_step.py:150-152` `resolve_active_objectives(mode, names)` →
  `problem.active_objective_names` → `evaluate` returns
  `{spec.key: spec.value(view) for spec in active_specs}` — one contract, no per-axis
  code. The wizard writes the same key (`search_objectives.js`); compilagent re-reads
  `problem.objectives` live. **New axes are user-selectable with zero new config keys**;
  per-mode DEFAULT objective sets stay unchanged (opt-in by name).
- **Candidate availability is registry-gated**: an axis is searchable iff its `Backing`
  answers on `candidate_capability_probe(mode)`. Adding a candidate-time axis =
  view field + `CANDIDATE_FRAGMENTS` + probes + one Backing over both views + catalog
  row — then `ALL_OBJECTIVES`, the wizard catalog, and compilagent's `list_objectives`/
  `hw_objectives` pick it up automatically. (One hand-kept table doesn't: `backend_eval.
  unit_for` — C2 boy-scouts it to read registry units.)
- **Encoding**: one homogeneous real box `[xl,xu]`; integrality/categoricality live in
  `decode` (`_decode_arch` index pattern, `clip_int`, `_snap_core_dim`) — the stated
  ARCHITECTURE contract. Categorical option axes = index-coded dims through option
  lists; **no pymoo mixed-variable machinery needed**.
- **Constraint seam already exists**: `nsga2_optimizer` declares `n_ieq_constr=1` and
  reads `problem.constraint_violation(cfg)` into `out["G"]`. The on-chip floor slots
  into that seam; `estimate_onchip_fraction(model, input_shape, num_classes,
  encoding_placement=…)` is callable per candidate — all inputs already live on the
  layout mixin, the model is materialized (warmup forward), and the flow fast-path
  avoids re-conversion.
- **The winner stamp** (`architecture_search_step.py:232-259`) writes
  `platform_constraints_resolved` from `resolve_candidate_platform(overlay)`; today's
  overlay is exactly `{"cores", "target_tq"}`. Deployment-parameter options
  (placement, pruning) are NOT in the resolved dict — the winner stamp for C3 must
  write searched DP options back into the run config alongside it.

Deltas the exploration surfaced (C3 absorbs all four):

1. **`pruning_fraction` on the problem is dead wiring** — declared at `problem.py:85`,
   read by nothing (not on `JointHostContract`). The candidate census ignores pruning
   entirely today. Promoting it means first making it MOVE the candidate: a banded
   pruning discount on the cost quantities (capacity feasibility stays conservative,
   preserving the shape-only twin's one-way bound).
2. **`weight_bits` never reaches `SearchSpaceDescription`** — `create_optimizer` omits
   it, so LLM prompts always claim "8 (fixed)" and `plan_codec.CodecDefaults` writes a
   literal 8 into candidate platforms. C3 fixes this drift as part of the promotion.
3. **`schedule_policy` already flows into the packing census** through
   `ChipCapabilities.from_platform_constraints(pcfg).layout_kwargs()` — promotion is an
   overlay key + one encoding dim, nothing to teach the packer.
4. **Placement is already per-hook-parameterized** (`layout_hook.py:47,80` read
   `self.encoding_placement`) — per-candidate placement threads the candidate's value
   instead of the problem-fixed field; the source-grep pin
   `test_the_search_step_passes_the_configured_placement` is updated to assert the new
   path.

## Stages

### C0 — Quantity catalog + the missing multiplicands (pure addition, S–M)

New `deployment_record/quantities/` (4 siblings, `schema`-only imports — the package
never touches `objectives.*` or `mapping`, duck-typing layout stats like
`LayoutStatsView` does):
- `spec.py` — `QuantitySpec{key, unit, dimension, doc}`, the closed `QUANTITY_SPECS`
  catalog (~29 keys: cells/cores/macs/neurons/axons used+physical, pass/sync counts,
  reprogrammed bytes, connectivity entries, latency steps, host_ops_s, spikes,
  boundary events, NoC packets+hops, tiles, weight_bits, timesteps, synaptic events,
  onchip/host params+MACs, …), `QuantityValue{value, provenance ∈
  measured|static|modeled}`, the frozen `Quantities` mapping.
- `from_record.py` — full extraction from a sealed record. `noc_total_hops` is
  **derive-only** (`Σ link_loads[*].packet_count` — already sealed; no redundant schema
  field, zero serialization drift).
- `from_candidate.py` — `CandidateQuantityContext{timesteps, activity_factor,
  weight_bits, cores_per_tile, host_macs, onchip_macs}` + partial extraction from
  (layout stats, capacity, params, host census, context).
- `probe.py` — the all-keys probe for capability questions.

Record gains ONE additive-optional field (the `invocations` precedent: `= None` default,
last position, no format-version bump — old JSON loads, mirrors untouched):
`UtilizationRecord.partition: Optional[ComputePartitionRecord]` with
`{onchip,host,total}×{params,macs}` (§8b item 4) — a NEW sub-record, because the mirror
pins forbid adding keys to `CrossbarUtilizationRecord`/`LayoutStatsRecord`. Producer:
`deployment_record_emission.py` via the existing `compute_onchip_fraction` /
ops-fraction readers; quantities read it when present, stay absent otherwise (decouples
C1/C2 from this wiring).

Physics vocabulary `multiplicand` strings revised to quantity keys, with a pin test
admitting three token kinds: quantity keys, physics-constant keys (e.g.
`neurons_physical x membrane_bits`), and the `declared` marker. Constants whose
quantities have no producer yet (`adc_conversions`, `adc_count`) stay honestly
absent-with-reason until C6's conversion models produce them.

Hygiene (small, prevents landmines this program would otherwise trip):
- relocate `CONFIG_KEYS_SET` out of `config_schema/defaults.py` (file at exactly 300/300
  LOC; `config_schema/` at 10/10 siblings — `registry/` is allowlisted, so the key set
  moves there)
- the load-sensitive `test_synthetic_imagenet_conv_segment_overflows_1000_budget`
  (52.7 s vs 60 s timeout) gets a budget that survives concurrent load

### C1 — The analytical pricer: physics × quantities (behavior change gated on declared profile, M)

`deployment_record/cost/` gains the absolute engine as three new siblings (9/10 after;
`model.py` at 287/300 stays untouched as the default-band engine):
- `pricing_formulas.py` — the constant → (quantity keys × report group) table AS DATA,
  cross-checked against the vocabulary multiplicands by test.
- `quantity_pricing.py` — `price_absolute(quantities, physics)`: every term =
  constant band × quantity, basis = the constant's evidence string, missing
  constant/quantity ⇒ term absent WITH a written reason (never a silent default).
- `extended_report.py` — `candidate_cost_report(...)` and
  `report_with_absolute_terms(record_report, ...)`; exports `ABSOLUTE_TERM_NAMES` for
  the fidelity zip.

The formulas:
- **area** `chip_area_mm2`: cells×`area_per_cell`(+per-bit) + neurons×logic + state
  bits + rows×drivers + ADCs×`area_per_adc` + tiles×(router+fixed) + `area_global_fixed`
- **energy** `energy_per_inference_mj`: decomposed groups × counts OR the
  `e_synaptic_event_total` aggregate (`resolve_supersessions` decides — never both) +
  programming + sync + static power × priced latency + **host**
  (`host_ops_s` × `p_host` / `host_compute_rate`; §8b item 1)
- **latency** `e2e_latency_s`: `t_cycle`×steps + per-byte programming (monotone) +
  core init + sync + host wall — `t_hop` deliberately NOT added (timestep-synchronous
  execution already covers hop time; the same no-double-count discipline as
  `sim_time_s`, stated in the term's basis). Steady-state per-sample headline;
  programming amortization its own term.
- **throughput** `throughput_inferences_s`: inverse e2e (v1; pipelining refinement
  deferred, stated).

The absolute terms merge into the EXISTING `DeploymentCostReport` groups
(area/energy/latency/throughput — verified no name collisions), so the same term names
appear on candidate and record reports by construction and the existing `cost_term`
backing reaches them. One enabling fix in `terms.py`: `find_term_or_none`, and the
`cost_term` backing returns `None` on a missing term — availability==extraction must
never throw on a partial (candidate) report. Priced terms are never summed into the
measured `total_mj`/`total_s` (a report note states the parallel-prediction
discipline); `_require_no_double_count_note` is untouched.

Discipline: a run with no profile is byte-identical (A/B pinned; the step writes the
extended report only inside the physics-declared branch); measured `mj_per_sample`
(SANA-FE) and modeled `energy_per_inference_mj` coexist as separate axes with distinct
provenance — their agreement is exactly what C4 measures.

### C2 — Absolute objectives at both completenesses (behavior change: new axes, M)

Register `chip_area_mm2`, `energy_per_inference_mj`, `e2e_latency_s`,
`throughput_inferences_s` (+ decomposition axes) via the existing `cost_term` backings —
availability and extraction stay one reader, no per-axis special cases.

The candidate side follows the registry's own established chain — with the red-teamed
shape: `CandidateStaticView` stores two new FRAGMENTS, `physics` and
`quantity_context`, and **derives** `quantities` from them plus the layout (a stored
quantities field would break `candidate_probe_without` semantics: `validate.py` gates
the mapping on `_requires_fragment("layout")`, and pre-baked quantities would let an
energy-only active set skip the mapping and die at extraction). `cost_report()` prices
the derived quantities when physics is present (memoized the same way
`DeploymentRecordView` does); `DeploymentRecordView.cost_report()` merges
`price_absolute(from_record(record), declared_physics)` into the measured report. The
probe machinery (grown by `run_capability_probe`) moves to a new `objectives/probes.py`
so `views.py` stays under its 300-LOC cap. Probe answers stay honest:
`probe_without("physics")` loses all four axes; `probe_without("layout")` keeps
`chip_area_mm2` (capacity+physics suffice) — so an area-only hardware search stays
layoutless and the `test_joint_error_contract` pin survives unchanged.

The run-level gate is an EXPLICIT kwarg, not a default: `resolve_active(..., probe=)` /
`resolve_active_specs(..., physics=)`, passed at the two run call sites
(`architecture_search_step`, `problem.active_specs`) from the run's own
`platform_physics_resolved` — the parameterless form keeps today's capability-level
behavior, so the wizard round-trip pin (`…resolves_without_a_word`) stays green. With
no profile declared, requesting an absolute axis refuses by name at resolution time.

After the chain, `ALL_OBJECTIVES`, the wizard catalog, and compilagent's
`list_objectives`/`hw_objectives` pick the axes up with no further plumbing (verified
propagation path); the one hand-kept table that would not — `backend_eval.unit_for` —
is boy-scouted to read units from the registry. New axes register AFTER `_RECORD_AXES`
(catalog order is contract; every optimizer's vector prefix survives), and per-mode
DEFAULT objective sets stay byte-identical — the axes are opt-in by name in
`arch_search.objectives`.

Candidate-time spike-dependent energy uses the declared `activity_factor` (the EDA
switching-activity discipline; provenance `estimated`, fidelity-tracked);
`connectivity_entries` and NoC hops are stated-unavailable at candidate time in v1.
Per owner decision 2, the optimizer sees the **nominal corner**; bands ride into
results and reports. Per owner decision 3, the `host` group gains declared host-rate
constants; with placement searched and host undeclared, host-inclusive axes are refused
by name. Probe physics declares every constant at strictly positive placeholder values
(evidence validation requires the note; zero would make availability lie).

Pinned tests updated with intent: the "record-only axis is not searchable" pins split
into "the OLD record axes stay record-only" + "the NEW absolute axes are
searchable-iff-physics-declared"; `test_objectives_drift_pins` gains a `PHYSICS_AXES`
pin dict; `test_objectives_projection.LEGACY_TUPLE` grows by four; the legacy-8 order
stays byte-equal.

### C3 — The deployment-option decision space + typed constraints (behavior change, M–L)

Promote the four owner-chosen options to decision variables: `encoding_layer_placement`
(ENUM subsume/offload), `schedule_policy` (ENUM pool/bank_clustered), `weight_bits`
(INT), `pruning_fraction` (FLOAT). All four already exist as registry keys — no new
config surface; the searched-axis declaration rides the existing `arch_search`/
`search_space` JSON (`hw_search_space_fields` schema extended; the wizard structured
editor follows for free).

- **Space declaration**: `SearchSpaceDescription` gains an `option_axes` table
  (key, kind, choices/bounds) mirroring `arch_options`' shape; `create_optimizer` plumbs
  it (and fixes the verified `weight_bits` drift: today it never reaches the
  description, so LLM prompts claim "8 (fixed)" and `plan_codec.CodecDefaults` writes a
  literal 8 into candidate platforms). Compilagent levers render one lever per option
  axis (`EnumChoice`/`IntFreeform` — the factory pattern exists).
- **Encoding**: option axes append index-coded/real dims after the hw block
  (`_decode_arch` pattern); `decode` overlays PC-section options (`weight_bits`,
  `schedule_policy`) into `resolve_candidate_platform(overlay)` and DP-section options
  (`encoding_layer_placement`, `pruning_fraction`) into the candidate configuration
  dict — both through the SSOT resolver path, preserving
  `test_searched_chip_is_a_deployable_chip` (the law: resolution takes no flag that
  could drop a permission).
- **The candidate census must MOVE with every axis** (else the objective vector is
  flat along it): placement threads per-candidate into the layout hook (already
  parameterized); schedule_policy flows via `layout_kwargs()` (verified live);
  weight_bits changes `params_bytes` quantities; pruning applies as a **banded discount
  on cost quantities only** — capacity feasibility stays conservative (unpruned
  sizing), preserving the shape-only twin's one-way bound. `pruning_fraction`'s dead
  problem field is deleted in favor of the live axis (and added to `JointHostContract`).
- **Typed constraints**: `onchip_min_fraction` computed per candidate via
  `estimate_onchip_fraction` (inputs all on the mixin; flow fast-path; cheap) feeds the
  EXISTING `constraint_violation` → `out["G"]` pymoo seam. Violations become a typed
  census (constraint id, measured fraction, floor) on the search result and the
  `search_event` stream, instead of an undifferentiated penalty (§8b item 3). The
  downstream pipeline gate stays (defense in depth) but a search can no longer discover
  it as a crash.
- **Winner stamp**: searched DP options are written back into the run config alongside
  `platform_constraints_resolved` (today's stamp only carries the PC dict) — the
  deployed run must run under the options the winner was scored with.
- The search e2e template cell gains one option axis so drift is impossible silently.

Test pins (from the seam map): MUST-update — `test_problem_surface_decode`
(decode==SSOT equality construction), `test_problem_surface_evaluation`,
`test_objectives_projection` (LEGACY_TUPLE grows), `test_objectives_candidate_view`,
`test_joint_layout_honors_encoding_placement` (source-grep pin → per-candidate path),
`test_search_space_description` (lever counts), `test_host_contracts` (new fields),
compilagent `test_lever_factory`/`test_plan_codec`. MUST-preserve —
`test_searched_chip_is_a_deployable_chip`, `test_joint_error_contract`,
`test_optimizer_error_contracts`, `test_nsga2_search_events`,
`test_introspection_import_direction` (AST wall).

### C4 — The fidelity contract (pure addition + closes the §9.4 gap, M)

Per sealed run: rebuild the candidate view from the run's own config, evaluate the SAME
objective vector both ways, write `fidelity.json` (per axis: predicted band, measured
value, in-band?, relative error) + an aggregation script over run dirs. Report-first
per owner decision 4 — numeric gates are a later decision made on evidence. Structural
agreements gate immediately: candidate `pass_count` == deployed pass census, which means
closing the known pool search/deploy divergence (`test_schedule_pool_residual_gap` is
built to fail the moment it's closed — flip it as designed).

### C5 — Wizard: physics panel + honest objective picker (GUI, pixel-reviewed, M–L)

Co-design tab: profile selector (STR key + dynamic-options endpoint precedent, like
`model_type`) and a configurator panel (grouped by vocabulary groups; per row: band,
unit, evidence badge published/datasheet/derived/estimated, citation on hover, override
field with delete-on-empty, overridden rows visibly marked — the `search_space`
sparse-map precedent). Completeness readout driven by the SAME availability predicates
the registry uses ("area: available · energy: needs e_adc_conversion").

The objective picker gains a physics dimension (closes the C2 red-team finding: after
C2 the catalog is maximal, so a physics-less draft could select `chip_area_mm2` and hit
a loud-but-late refusal at the step) — the served catalog rows carry a
`requires_physics` bit and the JS chip filter greys physics-gated axes with the reason
until the draft declares a profile, exactly mirroring the `resolve_active` refusal.
Written UX spec; owner screenshot review (standing GUI rule).

### C6 — Profile library + conversion models (data + research, M) — AS LANDED

`loihi` (simulation — Davies 2018 Table 2 is explicitly pre-silicon; no `t_cycle`,
because Loihi is asynchronous, so it honestly cannot back latency or throughput),
`generic_estimated_22nm` (29 constants, every one `estimated`, each anchored on
something published and each stating its expected error), and `isaac_like` — the
first analog target, and the reason `platform_physics/conversion.py` exists.

**PRIME is REFUSED, and that is the result.** The research pass established that
Chi et al. (ISCA 2016) publishes **no absolute per-unit area, energy, power or
compute-mode latency at all** — there is no PRIME analogue of ISAAC's Table I, only
structure, area *percentages*, and CPU-normalised *ratios*. A `prime_like` profile
could only be built by borrowing ISAAC's energies or a generic node's, which is
precisely the "no proxy presented as a measurement" line this program exists to
hold. The structure it *does* publish is recorded
(`docs/research/physics/prime_constants_research.md`), including its conversion
model (`macs x 3..4 / 256`, ~64-85x fewer conversions per MAC than ISAAC), so the
profile becomes writable the moment absolute numbers appear. `neurram_like` is not
attempted for the same reason: unresearched, therefore undeclared.

The conversion-model defect the ISAAC pass found is worth recording: the first
formula omitted the WEIGHT-side column slicing, under-counting ISAAC's conversions
— the paper's own dominant cost, 58% of tile power — by 8x, and over-counting
PRIME's by 1.5-2x. Erring in opposite directions on its two motivating targets is
what made it visible. Fixed and pinned by `TestTheIsaacReferenceCase`.

### C6 — original plan



- `loihi` from the completed research pass (validity `simulation` — Davies Table 2 is
  pre-silicon; band = published extrema, never an invented nominal).
- `generic_estimated_22nm` — the labelled all-`estimated` exemplar.
- Analog IMC (`isaac_like`, `prime_like`, `neurram_like`) gated on the ADC conversion
  model: `adc_conversions` quantity produced by a per-profile declared `ConversionModel`
  (named models in the physics package; digital profiles declare zero — already the
  truenorth precedent).

### C7 — The cross-platform co-optimization study (research deliverable, M–L)

CLI over profiles × search modes on a fixed workload family: per-profile Pareto fronts,
hypervolume indicator, constraint census, and a comparison artifact that discloses
per-constant evidence kinds in every row (a comparison where one side is `estimated`
must say so — standing discipline). Feeds the thesis co-optimization chapter directly.

## Consolidation of remaining items

| Open item | Lands in |
|---|---|
| TPP P2 (missing multiplicands) | C0 |
| TPP P3 (cost model consumes physics; absolute objectives) | C1 + C2 |
| TPP P4 (wizard selector/configurator) | C5 |
| TPP P5 (remaining profiles) | C6 |
| TPP P6 (first cross-platform study) | C7 |
| §8b.1 host ComputeOps priced or comparison rigged | C1 + C2 (declared host rates, owner decision 3) |
| §8b.2 placement as first-class search axis | C3 |
| §8b.3 on-chip floor as constraint, not crash | C3 |
| §8b.4 param split in the record | C0 |
| Known limitation §9.4 (pool pass-count divergence) | C4 |
| `config_schema` LOC/sibling caps at limit | C0 hygiene |
| Load-sensitive capacity test (52.7 s / 60 s) | C0 hygiene |
| W3 stages 2–3 (adaptation rebalance) | **out of scope** — research-adjacent tuning, orthogonal to CAD infrastructure; stays its own tracked task |
| ViT/streamed host-prefix entry gauge memo | **out of scope** — correctness research item, tracked in its findings memo |

## Sequencing

```
C0 ──► C1 ──► C2 ──┬──► C3 ──┐
                   ├──► C4 ──┼──► C7
                   └──► C5   │
        C6 (after C1) ───────┘
```

C0–C2 strictly ordered (each consumes the previous). C3/C4/C5 independent after C2.
C6 needs only C1. C7 needs C3 + C4 + C6.

## New test surface (C0–C2, tests-first)

`tests/unit/deployment_record/quantities/{test_quantity_catalog, test_quantities_from_
record, test_quantities_from_candidate}.py`; `test_cost_absolute_pricing.py` (bands
from `PhysicsConstantValue.band` with evidence basis; supersession; absence-with-
reason; m²→mm² display; throughput inversion; candidate report lacks segment terms);
`test_extended_cost_report.py`; `platform_physics/test_vocabulary_multiplicands_are_
quantities.py`; a `DeploymentRecordStep` test pinning "extended report written iff
physics declared, no-physics output byte-identical". Every load-bearing guard
mutation-checked.

## Verification protocol (every stage)

1. Tests first; `python -m pytest` ≤2 min green; `./scripts/typecheck.sh` zero;
   ratchets/budgets clean; goldens only via regen scripts; templates only via
   `templates/generate.py`; ARCHITECTURE.md updated per touched module.
2. Load-bearing guards mutation-checked (cp backup/restore, real exit codes) — the
   established practice of this program.
3. Behavior-change stages carry a fresh-run A/B: C1 pins no-profile byte-identity; C3
   pins that a config with fixed options reproduces pre-stage results.
4. GUI (C5) verified in pixels against a written UX spec, reviewed by the owner.
5. No AI-attribution trailers in commits.

## Program-level acceptance ("CAD tool" defined)

1. A run declaring profile P emits absolute Area/Energy/E2E-Latency/Throughput with
   bands and evidence, host side included; a no-profile run is unchanged and every
   refused axis names what it is missing.
2. A search in any mode optimizes those axes at candidate time over model × hardware ×
   deployment options; the winner deploys to a sealed record whose measured/priced
   values are compared axis-by-axis in `fidelity.json`; pass counts agree exactly.
3. One model family across ≥3 profiles yields a comparative Pareto artifact disclosing
   evidence kinds.
4. The wizard can select, inspect, and override a profile, and shows which objectives
   each profile can back BEFORE launch.
