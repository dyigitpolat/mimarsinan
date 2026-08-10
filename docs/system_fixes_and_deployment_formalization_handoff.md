# System Fixes & Deployment-Formalization Program — Handoff

2026-08-10 · prepared for the incoming engineering team · repo state: `main @ d4e4ab4e`
(post activation-semantics merge: per-segment streamed LIF, nevresim parity closure §10–§13,
measurement-plane lattice, contained experimental walk recovery)

This document has three parts. **Part I** restates the eleven requested items in full,
with the intent and the missing context filled in. **Part II** records what the
investigation of the current codebase found for each item (root causes where
established, engineering pointers everywhere). **Part III** is the roadmap: workstreams,
sequencing, acceptance criteria, and the standing engineering rules that bound all work.

Nothing in Part I may be dropped or renegotiated silently; where an item interacts with
a measured constraint, the interaction is stated explicitly in Part II/III.

---

## Part 0 — Standing engineering rules (bounding every workstream)

These are the owner's standing directives; they are not optional style:

1. **No e2e regression, ever.** Any change must not regress end-to-end deployment
   accuracy or runtime on any covered configuration; only improve. Accuracy A/Bs MUST
   be fresh-run vs fresh-run (see Part II §3a: the MBH endpoint-steps ledger persists
   in run dirs and silently starves resumed reruns — resume-based comparisons are
   invalid evidence).
2. **Fix parity, never budget it.** Twin/certificate mismatches are root-caused at the
   SSOT; no tolerances, no gate exclusions.
3. **SSOT discipline.** One owner per behavior; generic abstractions over per-case
   handling; no workload constants framework-side (registry contracts only); the wizard
   is the configurability SSOT with the representability guarantee.
4. **Tests first; suite ≤2 min green; typecheck 0; ratchets only tighten; 300-LOC
   module budgets; templates only via `templates/generate.py`.**
5. **GUI quality bar is HIGH and verified in pixels, not APIs**: fresh/default state
   must resolve to a runnable pipeline; UI acceptance = browser screenshots against a
   written UX spec, reviewed by the owner.
6. Never add AI-attribution trailers to commits.

---

## Part I — The eleven items, in full

### I.1 Weight reuse must stop being a knob (and be ON)

`allow_weight_reuse` currently exists as a platform-constraints key (the GUI baseline
emits `"allow_weight_reuse": true`). The owner's position: weight reuse is a net
positive for mapping performance — it should not be user-visible configuration at all,
and the behavior must be the default-on, always-on path. Work items: remove the key
from the config surface through the existing retired-key migration machinery (loud,
keyed remedies — the P0 pattern), collapse the OFF branches in the mapper, and keep
exactly one escape hatch only if a real platform capability makes banks unrepresentable
(in which case it is a *platform capability*, not a user knob). Templates, fixtures,
goldens, and the wizard schema all migrate in the same change. Part II §1 has the
current consumer map.

### I.2 Deployment advisories move to the "Review & Launch" panel

Advisories (config-time findings with actionable remedies) currently render in the
wizard's right-side bar. Requirement: display them on the **Review & Launch** panel —
the pre-launch checklist is their natural home; the operator should confront advisories
exactly where the launch decision is made. Design latitude: the right-side bar may keep
a compact count/badge, but the full advisory list with remedies belongs to Review &
Launch. GUI acceptance per Part 0 rule 5 (pixel review).

### I.3 REGRESSION: lenet5 baseline fails NF↔SCM parity with a SHAPE mismatch

Run `simplemlp_baseline_20260810_091219_phased_deployment_run_20260810_091740` (the
name is stale; the config is **lenet5**) fails at Soft Core Mapping:

```
NfScmParityError: NF↔SCM parity: perceptron 2 neuron-count mismatch (2, 120) vs (2, 97)
```

This is not a spike-count mismatch — it is a **neuron-index-space mismatch**: the NF
side reports 120 neurons for perceptron 2 (lenet5 fc1) while the SCM/mapping side
reports 97. Root cause established in Part II §3: the config has `pruning: true,
pruning_fraction: 0.2`, and the bare config derives the **(lif, streamed)** default
(the P6 flip) — so the streamed parity gate runs, for the first time anywhere, on a
**pruned** model; the mapping compacts pruned neurons (120 → 97 kept) while the NF
capture stays full-width. The combination streamed × pruning has zero tier coverage.
Required: fix at the SSOT (shared neuron-index map between the twins), plus a
regression-pinning tier cell for pruned+streamed.

### I.4 REGRESSION-CLASS: LIF Adaptation drops accuracy that WQ then recovers

Run `mmixcore_baseline_20260810_091759_phased_deployment_run_20260810_091914`
(mlp_mixer_core, bare config → derived (lif, streamed), T=4, 2-epoch budget):

```
Activation Adaptation 0.9546 → LIF Adaptation 0.9191 → Weight Quantization 0.9408
→ deployed 0.9435
```

The owner's critique, verbatim in intent: *a −3.6 pp collapse at LIF Adaptation that
Weight Quantization then recovers (+2.2 pp) is incoherent — if the accuracy was
recoverable under the additional constraint of quantized weights, it was necessarily
recoverable at the LIF Adaptation step itself, which faces strictly fewer
constraints.* The demand is architectural, not cosmetic: **recovery capacity must sit
at the step that introduces the loss**, and each step must hand off inside its
retention envelope (this is the monotone-adaptation principle the MBH program
formalizes). Part II §4 connects this to the measured curriculum/ledger findings
(memo §13) and lays out the rebalance program.

### I.5 Monitor UI: failed runs present as live

When a pipeline run fails: (a) the run-level **Stop** button is still shown, and
(b) the failed step's tab still reads **Running**. Requirement: a correct terminal
state machine — on failure the run transitions to a terminal `failed` presentation
(no Stop affordance; a Retry/Inspect affordance is welcome), and the failing step is
marked `failed` with the error surfaced (the recorded traceback already lands in
`_GUI_STATE/console.jsonl`; `run_info.json` already says `"status": "failed"` — the
presentation layer is not consuming it). Also cover the other terminal states
(crashed process, externally killed) so silence is never rendered as progress.

### I.6 Hard Core Mapping tab: heatmaps rendered wrong and far too slowly

Two independent defects: (a) the heatmaps are rendered incorrectly (visual defects in
the mapping/utilization heatmaps on the Hard Core Mapping monitor tab), and (b) they
are generated so inefficiently that they take a long time to load, where at
UI-required resolution they should be effectively instant. Requirement: fix the
renderer, and re-architect generation so the UI-resolution artifact is produced
directly (downsampled at source, cached, generated once per mapping — not
re-rasterized per request at full core-matrix resolution). Target: perceptually
instant loads. Part II §6 identifies the current renderer and the cost drivers.

### I.7 Pruning Adaptation tab: maps invisible, dimensions missing

The pruning maps (per-layer masks) do not display at all, and the tab does not show
pre-pruning vs post-pruning dimensions per layer. Requirement: the tab must show, per
layer: the pruning map (mask visualization), pre → post neuron/channel counts, and the
achieved sparsity vs the configured `pruning_fraction`. Whatever artifact is missing
(mask not persisted, wrong path, renderer bug) gets fixed at the artifact SSOT so the
data exists for every pruned run.

### I.8 SANA-FE NoC: tile grouping and floorplan must be explicit configuration

Definition (owner's): a **tile** is the group of hard cores whose spike communication
is already wired — traffic inside a tile does not enter the NoC. Observed: (a) the
cores-per-tile grouping differs across runs with no user input (16 cores/tile in some
runs, 6 in others), and (b) the tile floorplan geometry differs (a single row of tiles
vs 2 rows × 3 columns), which changes XY-routing distances and therefore materially
changes the SANA-FE latency/energy/NoC statistics. Requirement: both become explicit,
deterministic configuration under **Hardware platform / capabilities**: cores-per-tile
and tile-grid geometry (rows × columns, or an explicit floorplan). Derived defaults are
acceptable only if they are deterministic functions of the platform description alone
(never of run-to-run mapping accidents), and the derivation must be visible in the
resolved platform constraints. Part II §8 has the current derivation logic.

### I.9 The deployment formalization and formal cost model (thesis §2, made concrete)

The thesis proposal (§2, "A Generic Model of IMC Deployment") defines: a
**target-system contract** promising NeuralOps (in-memory compute core types — MVM and
spiking cores of heterogeneous geometries, capabilities, counts, firing regimes) and
ComputeOps (general-purpose units for what cannot lower to MVMs). A **deployment is a
schedule**: a sequence of **passes** over the alternating NeuralOps/ComputeOps
structure; multiple passes arise when blocking non-MVM operations interleave with the
network or when the model exceeds chip capacity; each pass either **reprograms or
reuses** core-resident weights, with a **banded cost model** for the difference. Pass
count, accelerator area (core counts and occupancy), energy, latency, and throughput
all emerge from workload × topology × hardware constraints × deployment options
(mapping strategy, losslessness tolerances → attainable quantization).

The requirement is to reflect this model **elegantly in the software design** — not as
a bag of ad-hoc metrics, but as the concrete formalization:

- **A deployment model object** — the schedule: ordered passes; per pass, the neural
  segments placed on cores and the interleaved ComputeOps; per segment, its cores,
  parameters, connectivity, latencies, spans.
- **A parameterized cost model over that object.** Examples the owner gave, to be
  treated as the seed of an exhaustive and extensible set: neural-segment
  initialization = constant per-core reset/init overhead + a reprogramming term
  proportional to what must be sent to the chip (number of parameters + number of
  connectivity entries for that segment), with the reuse-vs-reprogram band from the
  proposal; chip utilization metrics (cores used, occupancy, waste) as area
  contributors; spike traffic (per boundary, per tile, per NoC hop) as energy and
  congestion contributors; end-to-end deployment latency decomposition (programming +
  compute passes + host ComputeOps + synchronization barriers); throughput.
- **An exhaustive, extensible objectives registry**: every optimization
  target/objective the system can score — accuracy (deployed, certified), area/cores,
  energy, latency, throughput, pass count, reprogramming volume, NoC traffic — one
  registry, typed, with provenance (which stage/simulator produced each number), so
  new objectives are additions, not surgeries.
- The model must be populated from **actual** deployment artifacts — configuration,
  mapping, scheduling, and simulation (nevresim cycles, SANA-FE energy/latency/NoC),
  never from proxies — matching the framework's "deployed accuracy is read from the
  chip" discipline.

### I.10 Optimizer integration revision (after I.9)

Two requirements on top of the formalization:

1. **A generic multi-objective problem surface.** The deployment problem (decision
   variables from the wizard-representable space; objectives from the I.9 registry;
   constraints/feasibility from mapping) must present a clean adapter surface that
   plugs into generic optimizer libraries (NSGA-II/pymoo-class evolutionary baselines
   and anything else) without bespoke glue per optimizer.
2. **Fine-grain introspection injection for custom optimizers.** Beyond scalar
   objectives (axon/neuron utilization…), customized optimizers (Compilagent-class
   LLM agents) must be able to request structured, fine-grain deployment detail —
   e.g., how many softcores each layer mapped to, shared-weight-bank composition,
   per-segment programming payloads, per-tile traffic — through a designed
   introspection channel that keeps the software design clean (no reach-ins), with
   the rigor of typed, versioned payloads. The existing Compilagent integration is the
   prototype to generalize, not the end state.

### I.11 The Search button is disconnected

Pressing **Search** under Hardware platform / capabilities with the baseline config
and default NSGA-II options fails outright. This is the symptom of a broader drift:
the search/optimization side has not tracked the config-schema and pipeline evolution
(retired keys, derived axes, new gates). Requirement: reconnect end-to-end — the
baseline config's Search must run NSGA-II defaults to completion and return results to
the GUI — and add coverage so the search path can never silently drift again. The
full reconnect lands on the I.9/I.10 surfaces; a minimal repair may precede them if
the team needs Search operational earlier (Part III sequencing).

---

## Part II — Investigation findings

### II.3 (I.3) The pruned-streamed parity shape mismatch — root cause

Established from the run artifacts (`_RUN_CONFIG/config.json`, `_GUI_STATE/*`):

- Config: `model_type: lenet5`, `pruning: true, pruning_fraction: 0.2`,
  `spiking_family/variant: null` → the post-P6 derivation defaults to
  **(lif, streamed)**; `encoding_layer_placement: offload`; T=4, wb5.
- Failure at Soft Core Mapping's streamed gate
  (`src/mimarsinan/pipelining/core/nf_scm_parity.py::assert_streamed_nf_scm_exact_or_raise`):
  perceptron 2 (lenet5 fc1, width 120) — NF capture `(2, 120)` vs SCM grouped record
  `(2, 97)`; 120 × (1 − 0.2) ≈ 96–97: **the mapping compacts pruned neurons; the NF
  activation-hook capture does not**.
- Why now: the gate itself landed 2026-08-07 (P3) and the streamed default for bare
  configs landed 2026-08-08 (P6). Tier-0 has streamed cells (t0_45–50, none pruned)
  and pruned cells (t0_09/15/21/27, none streamed) — the product cell does not exist,
  so no ratchet ever exercised streamed × pruning. The P6 default flip put every bare
  pruned config onto the uncovered path.
- Fix direction (SSOT): the twins must compare in ONE neuron index space. The mapping
  side already owns the kept-neuron index map (pruning masks → IR → core neuron
  lists); expose that map through the parity module's grouping
  (`_group_record_by_perceptron`) or project the NF capture through the same kept-index
  selection before comparison. Do NOT widen the SCM side back to 120 (the chip
  genuinely has 97 neurons; parity must certify what deploys).
- Off-by-one note for the implementer: 0.2 × 120 = 24 pruned ⇒ 96 kept; the record
  says 97 — account for the extra row (bias/always-on or rounding convention) while
  building the shared index map; the certificate must reconcile exactly, not modulo 1.
- Regression pin: add a tier-0 cell `lifs + pruned` (lenet5 or simplemlp shape) to the
  template matrix via `templates/generate.py`.

### II.4 (I.4) The LIF-drop/WQ-recovery asymmetry — analysis

The mmixcore trail confirms the class measured throughout the N7/N8 arc (memo
`docs/research/findings/lif_deployment_exactness.md` §13):

- The streamed LIF Adaptation trains the raw cascade with a small blend/recovery
  budget and exits −3.6 pp under its entry; WQ then climbs +2.2 pp because the **WQ
  endpoint recovery holds the dominant share of the run-total step ledger**
  (`endpoint_floor_steps`, drawn through
  `tuning/orchestration/frontier/endpoint_recovery.py`) and trains the deployed
  composition with it.
- The owner's inference is correct and is the monotone-adaptation thesis: the WQ-step
  recovery optimizes a strictly harder problem (quantized weights) from a worse
  init — that the same accuracy was reachable there proves LIF Adaptation
  under-recovered, i.e., the budget/geometry allocation is mis-shaped, not the loss
  intrinsic.
- Measured constraints the rebalance must respect (all fresh-run evidence, §13):
  naive re-arming of the LIF-step recovery alone measurably REGRESSED healthy cells
  (t0_30 −0.41 pp; t0_01 −0.05 pp via the too-broad host-op gate), and the honest
  end-to-end chain under current budgets loses to the incumbent curriculum
  everywhere. The fix is therefore a *designed reallocation* inside the MBH
  monotone-adaptation program: per-step retention envelopes as the arbiter (the
  machinery exists: `__mbh_retention_envelope.json`, retention_armed gauges), the
  ledger split derived from where the envelope is violated, convergence geometry
  (patience/check cadence) that can actually climb at the violating step, and the
  no-regression contract enforced by fresh-run A/B on the affected cell set.
- Note the interaction with I.9: "recovery spent where the loss occurs" is also the
  correct *cost-model* accounting — the same per-step envelope data belongs in the
  formal deployment record.

### II.9 (I.9) Current metrics landscape vs the target model

The structural finding: **three cost surfaces exist and are not connected.**

1. **Measured**: `chip_simulation/cost_extraction.py` — `CostRecord` v3
   (`cost_record.json`), produced ONLY by `SanafeSimulationStep._emit_cost_record`
   (`verification/sanafe_simulation_step.py:222`). Live fields: `acc_deploy,
   mj_per_sample, spikes, latency_steps, cores, s_global, depth`. The
   reuse/reprogramming fields (`reprogram_passes, reuse_passes, params_reloaded,
   activation_bytes_moved`) exist in the schema and are **dead-by-default** — the
   step never passes them (verified on a scheduled run: all zeros).
2. **Static-mapping**: `LayoutVerificationStats` (34 fields:
   `mapping/verification/layout_verification_types.py:5` — waste %, coalescing,
   splits, `schedule_pass_count`, `schedule_sync_count`, segment latencies,
   fragmentation) reaching `steps.json → mapping_performance`; and
   `CrossbarUtilizationReport` (`mapping/crossbar_utilization.py`, incl.
   `programming_bits = cells_used × weight_bits`) written to
   `crossbar_utilization.json` every run — **which nothing in src/ ever reads
   back**. `weight_programming_report` (`mapping/weight_programming.py:42` —
   `programming_events, params_programmed, params_unique, reuse_factor`) and
   `weight_reuse_plan_from_graph` (`mapping/weight_reuse.py:140` — reprogram/reuse
   phases, `params_reloaded`) are **print/event-only**.
3. **Physical coefficients**: `chip_simulation/weight_reuse_cost_model.py`
   (pJ/byte, bytes/param, µJ/barrier, `DEFAULT_COEFFICIENT_BAND`) + the proxy band
   `chip_simulation/pareto.py::CostProxyBand` (whose `COST_BAND_DISCLAIMER`
   explicitly documents the instrumentation gap) — exercised today only on
   hard-coded VGG16 constants, never on real-run quantities.

Rich data measured and then dropped:
- **Spike traffic tap ready-made**: `certification/count_alignment.py:160`
  (`flow_node_counts`) yields per-node, per-neuron window counts on the deployed
  program at every stage boundary — currently collapsed to 3 certificate scalars.
  This IS the per-boundary/per-segment traffic a cost model needs, for free.
- SANA-FE per-tile energy, NoC link loads/hops, `inter/intra_tile_packets`, cycle
  energy waterfalls (`sanafe/stats.py`, `analysis/noc.py`, `analysis/energy.py`) —
  GUI-render-only; only 6 scalars reach `CostRecord`.
- nevresim: cycle counts never extracted; `total_spikes` computed then discarded
  (`nevresim_driver.py:86`).
- Depth-balancing relay count: returned by `insert_depth_balancing_relays`
  (`mapping/latency/depth_balancing.py:255`) and **discarded at every call site**.
- `ft_pass_walls.json`: read by `cost_extraction.py:314`, written by nothing; the
  tuner has the producer shape (`smooth_adaptation_cycle.py:148`) but only the max
  is reported.

Objectives today: `search/results.py::ALL_OBJECTIVES` — 8 objectives, ALL derived
from static mapping stats (params, capacity, sync barriers, utilization/wastage/
fragmentation). **No energy, latency, spikes, pass, or reprogramming objective
exists** — the entire measured-cost axis set is invisible to search.
`DeploymentPlan` (`pipelining/core/deployment_plan.py`) centralizes configuration
resolution only — it carries zero cost fields; cost consumers touch it just for
cell keying.

Thesis terms with NO producer at all: reprogramming BYTES (params counted, never
sized; connectivity payload never sized — spans counted only in the profiling-only
`nevresim/profiling/mapping_metrics.py`), per-core reset/init constant overhead,
NoC traffic as a *cost term* (measured, never costed, absent for
nevresim/Loihi backends), end-to-end latency decomposition (single scalar today;
compute vs reprogram vs sync vs host-ComputeOp vs NoC unjoined; host ComputeOp
wall time not timed at all).

The formalization (Part III W4) is therefore largely a **join and a registry**, not
green-field measurement: `weight_reuse_plan + schedule_pass_count +
programming_bits → CostRecord reprogramming fields; flow_node_counts + SANA-FE
noc_links → traffic terms; ChipLatency + _SegmentTiming + sync counts + timed host
ops → the latency decomposition` — under one typed deployment-record schema with
per-field provenance.

### II.1 (I.1) Weight reuse — the knob is declared-but-inert; the behavior is already on

The investigation changes the shape of this item. `allow_weight_reuse`
(`config_schema/registry/entries_execution.py:81`, platform_constraints group,
default frozen False) has exactly **one functional consumer**: it gates the
"Weight-reuse schedule: N reprogram + M reuse phases" report in
`pipelining/pipeline_steps/mapping/soft_core_mapping_step.py:275`. The weight banks
themselves (`IRGraph.weight_banks`, `NeuralCore.weight_bank_id`) are built
**unconditionally** by the conv/fc mappers, and the real behavioral lever is
`schedule_policy="bank_clustered"` (`mapping/packing/schedule_bank_clustered.py`),
which all 12 literature IMC presets already declare — *without* declaring the
capability flag. `MappingStrategy.allow_weight_reuse` is documented as a RESERVED
gate no mapping decision consults.

So "make it default-on and not a knob" decomposes into:

1. **Remove the key** from the config surface: registry entry, both
   `config_schema/defaults.py` membership lists (three ratchet tests tie these
   together — `test_namespaced_schema.py:93/101`, `test_ssot_flag_collapse.py:119`),
   the resolver pass-through (`platform_constraints_resolver.py:35`), the
   `ChipCapabilities`/`MappingStrategy` field+accessor
   (`mapping/platform/mapping_structure.py:57,67,82,141`), 5 template sites in
   `templates/generate.py` (+ regenerate), golden snapshot regen
   (`scripts/regen_golden_resolution_snapshot.py`), and the two test files pinning
   the old default. Update `mapping/ARCHITECTURE.md:25` ("default-off" wording).
2. **Migration**: the retired-key machinery (`config_schema/registry/retired_keys.py`)
   currently inspects ONLY `deployment_parameters` — it must be generalized to
   `platform_constraints` scope so saved user configs get a keyed remedy instead of a
   bare unknown-key error. This generalization is the right investment regardless
   (first platform-scope retirement of more to come).
3. **Make the reuse-phase report unconditional** (the ON behavior the owner wants
   globally; it is a pure IR read with a total fallback — no banks ⇒ all-reprogram).
4. **Re-home the capability semantics**: the registry doc says "the bank-aware
   schedule policy consumes it next" — that future consumer must read a **platform
   capability declaration** (the `imc_platforms` preset surface, where
   `bank_clustered` already lives), never a user knob. Audit the 12 presets for
   whether bank residency actually holds per platform when that lands; also note
   `chip_simulation/cost_extraction.py` keeps DMA/sync reuse coefficients at 0.0
   (byte-identical) — the I.9 cost model is where reuse-vs-reprogram becomes a real
   priced term (the proposal's banded cost), closing the loop.

Safety: today's flip is byte-identical except one stdout line; the no-regression
contract is trivially satisfiable and should still be demonstrated with one fresh
tier spot-run.

### II.8 (I.8) SANA-FE NoC — root cause and the ready-made fix path

Both observed symptoms have one root cause in
`chip_simulation/sanafe/arch_synth/spec.py:163-172` (`derive_arch_spec`):

- `cores_per_tile = ceil(sqrt(total_hard_cores))` when unspecified — a **step
  function of the packed core count** (16/tile ⟺ N∈[226,256]; 6/tile ⟺ N∈[26,36]).
  The call site (`verification/sanafe_simulation_step.py:138`) never passes
  `cores_per_tile`, so the auto branch always runs.
- The tile grid is `_mesh_dims(n_tiles)` (`spec.py:13-25`): the most-square **exact**
  divisor pair — so with 6 cores/tile, N∈[26,30] ⇒ 5 tiles ⇒ a 5×1 row while
  N∈[31,36] ⇒ 6 tiles ⇒ 3×2. One extra packed core flips the floorplan, and every
  NoC metric (XY route walk `sanafe/analysis/noc.py:99-163`, `inter_tile_packets`,
  `cross_tile_connectivity_edges`, tile-hop energy `analysis/energy.py:47-61`) is a
  function of that mesh shape. Cross-run NoC numbers are currently
  **non-comparable by construction**. The last tile also takes the remainder
  (N=31 → tiles [6,6,6,6,6,1]) — intra-run non-uniformity.
- `sanafe_arch_preset` fixes **only** per-event energy/latency scalars
  (`sanafe/presets.py`) — the "loihi" preset does NOT pin Loihi's real 8×4 tiles ×
  4 cores floorplan (`sana_fe/arch/loihi.yaml:9-19`). Latent bug found on the way:
  the registry advertises `sanafe_arch_preset="custom"` as an enum option
  (`entries_platform.py:174`) but both `SanafeRunner.__init__` (`runner/core.py:68`)
  and `derive_arch_spec` (`spec.py:114`) validate against `PRESETS` (loihi/truenorth
  only) and raise — fix alongside.

The fix is mostly plumbing that already half-exists:

1. New platform keys in `config_schema/registry/entries_platform.py`
   (`section="platform_constraints"`, `group="hardware"` — the "Hardware platform /
   capabilities" card, `registry/groups.py:19`): `cores_per_tile` (int, 0 = derived)
   and the tile-grid geometry (`tile_grid_rows`/`tile_grid_cols`, 0 = derived) —
   plus defaults in `DEFAULT_PLATFORM_CONSTRAINTS` (`config_schema/defaults.py:68`)
   and the key set; golden snapshot regen; the registry tests to satisfy are listed
   in `test_registry.py:28,38,54,157`.
2. Pass-through in `SanafeSimulationStep` → `SanafeRunner(cores_per_tile=...)` — the
   parameter exists (`runner/core.py:44`) and is honored (`core.py:208-212`).
3. One new `derive_arch_spec` argument for the explicit grid, replacing
   `_mesh_dims(n_tiles)` at `spec.py:172`, HARD-validated with
   `width*height == n_tiles` and capacity ≥ total_cores — a padded mesh SIGFPEs
   SANA-FE's C++ NoC (invariant documented at `spec.py:16-18`).
4. **Determinism requirement**: when the keys are 0/derived, the derivation must be
   a deterministic function of the *platform description* (declared core counts),
   never of the run's packed-core accident. Practical default: derive from the
   platform's declared `cores[].count` capacity, not from `Σ len(seg.cores)` of the
   mapping — then two runs on the same platform always share a floorplan, and the
   resolved values surface in `platform_constraints_resolved` for the record.
5. Tests: extend `tests/unit/chip_simulation/test_sanafe_arch_synth.py` (the
   contract tests already pin the current derivation verbatim — lines 106/123/155)
   with the explicit-key paths + the determinism property; fix the "custom" preset
   enum bug with its own test.

### II.2 (I.2) Advisories — current wiring and the move

The GUI is a FastAPI + vanilla-ES-modules app entirely inside
`src/mimarsinan/gui/` (no build step; vendored Plotly; offline ratchet test).
Advisories: computed by `advisories/engine.py:63::evaluate_config_advisories`
(config-time rules only — the graph/post-pretrain rule sets never reach the wizard),
folded into every `POST /api/config/resolve` payload
(`gui/wizard/schema_api.py:254`), rendered by
`static/js/wizard/review.js:360/382` into the right rail
(`static/wizard.html:200`, `.advisory-rail`). The **Review & Launch** section
already exists as a workbench section (`static/js/wizard/workbench.js:17`,
`data-section-id="review"` markup at `wizard.html:145-192`, holding Derived
values / Differs-from-defaults / Emitted config); the Launch button itself lives in
the rail (`wizard.html:214`). The move is a frontend relocation of the advisory
card list into the review section (a compact count badge may stay on the rail),
plus the pixel-verification pass. No backend change required; consider whether the
launch affordance should also gate on unacknowledged mandate-violation advisories
while in there (owner's call at review time).

### II.5 (I.5) Monitor failed-state — THREE distinct root causes

A. **`steps.json` never records a failed step.** The pipeline engine wraps step
   bodies in `try/finally`, not `try/except`
   (`pipelining/core/engine/pipeline.py:178-189`): on an exception the post-step
   hooks (where `gui.on_step_end` lives, `pipelining/session.py:177`) never run,
   and **no failure hook exists at all** — `GUIHandle` has no `on_step_failed`;
   `DataCollector.step_failed` exists (`gui/runtime/collector/mixins/steps.py:95`)
   with zero production callers. The step stays `"running"` forever. Only
   `run.py:131` writes run-level `failed` into `run_info.json`.
B. **Historical reads discard status.** `gui/runs.py:119,170` derive step status as
   `"completed" if end_time else "pending"` — the persisted `status` field is
   ignored and `run_info.json`'s `status`/`error` are never read; the overview
   viewmodel's failed-marker label can never receive the real error.
C. **The Stop button latches at page load.** `static/js/main.js:128-136` decides
   `_isActiveRun` once (any run still in `ProcessManager._runs` — kept up to an
   hour after finishing — answers 200), then `setupRunChrome()`
   (`main.js:519-541`) unconditionally shows `#header-stop-btn` and nothing ever
   hides it; the DELETE response for a dead run is ignored. The welcome page
   already does this correctly (`welcome.js:157/176-189`: Stop only when
   `is_alive`, re-evaluated on every poll).
Aggravator: a crash writes nothing to `steps.json`, so the mtime-watching WS tailer
(`runtime/active_run_tailers.py:121`) emits no frame; recovery waits on the 30 s
watchdog; orphan-recovered runs probe liveness via `os.kill(pid,0)` — PID reuse can
report a dead run alive indefinitely.
Fix set (matches the found seams): a real failure path on the engine →
`GUIHandle.on_step_failed` → `save_step_status(status="failed", error=…)` +
`collector.step_failed`; `runs.py` honors persisted status + `run_info.json`;
Stop visibility re-derived from `is_alive` on every render; a death event pushed
on the WS channel.

### II.6 (I.6) Heatmaps — why wrong, why slow

Naming note: "Hard Core Mapping" is the step; the tab is labelled **Hardware**
(`static/js/step-detail.js:360`). Renderer:
`gui/heatmap_renderer.py::render_heatmap_png_bytes` (matplotlib → PNG), driven by
lazy `HeatmapSource` descriptors from `snapshot/mapping_snapshot.py:127`
(per-core `get_core_matrix()`).

**Wrong** (all in `heatmap_renderer.py`): `aspect="auto"` (cells not square,
stretched to the axes box); `axis("off")` (no ticks/labels); **no colorbar and
per-core color limits** (symmetric p98-of-|W| per core — adjacent grid cells are
on different unlabeled scales; cross-core comparison is meaningless; >p98
saturates); silent nearest-resampling above `max_size=1024` (single pruned
rows/cols vanish — no explicit decimation policy); overlay drift (the frontend
positions placement overlays as percentages of the whole image box with
`object-fit:fill` while `tight_layout(pad=0)` leaves a margin —
`hardware-tab.js:549/642-668` fixed only the letterboxing half).

**Slow** (architecture, not constant-factor): GUI-spawned runs use
`ResourceRenderPolicy.DEFERRED` (`run.py:98`), so the **first browser attach
renders every PNG per-request** — one `<img>` per hard core
(`hardware-tab.js:536`), measured 474–555 heatmaps per real run (up to 19,725
persisted source arrays on big runs) — through non-thread-safe pyplot in
FastAPI's threadpool, at 1024 px/150 dpi (40–510 kB each) for display cells
capped at 200 px (`MAX_CORE_DISPLAY_PX`), plus one matplotlib line artist per
masked row/col, with no ETag on resource routes (cold caches re-fetch) and no
store eviction.

Fix directions: render at UI-resolution directly (decimate the matrix
server-side — max-pool/stride to ≤2× display px — then a colormap→PNG encode
without the matplotlib figure machinery; a pure-numpy LUT + PNG writer makes
each image sub-millisecond); shared, labeled color scale (global or per-segment
normalization + one colorbar asset); square cells; explicit mask-preserving
decimation (a pruned line must survive downsampling); batch/pre-render at
mapping-snapshot time or render-on-save for the common sizes; ETags; thread-safe
rendering (Agg figures per call or drop matplotlib entirely). Acceptance:
first-attach Hardware tab fully painted in ~instant time on a 500-core run,
pixels reviewed against a written spec.

### II.7 (I.7) Pruning tab — four failure modes + the dimensions live elsewhere

What renders today (`static/js/pruning-tab.js`): a layer list and ONE post-pruning
heatmap with mask lines — no mask map, no pre/post dims, no sparsity ratio. The
data chain: `snapshot/builders.py:205` (string-literal match on step name
`"Pruning Adaptation"`) → `snapshot/model_snapshot.py:112` (`snapshot_pruning_layers`
reads `prune_row_mask`/`prune_col_mask` buffers installed by
`transformations/pruning/seed_generators.py:230`).

Failure modes, in likelihood order: (1) the hard-coded step-name literal (any
rename/alias/backfill drops the whole tab); (2) **silent `continue`** when a mask
buffer is missing or length-mismatched (`model_snapshot.py:126-131`) — including
the `_get_model_perceptrons` fallback that wraps bare `nn.Linear` children which
never carry masks — yielding `layers: []` and "No pruning data" with zero
diagnostics (this silent skip is currently PINNED by
`tests/unit/gui/test_snapshot_pruning.py:42` — the pin must flip to loud); (3)
deferred render 404s render as blank `<img>` with no `onerror` fallback
(`pruning-tab.js:73`); (4) the `"model" in snapshot` precondition coupling.

The pre/post dimensions the owner wants **already exist** — on the IR-graph node
snapshot (`snapshot/ir_graph/ir_graph_nodes.py:126-150`,
`pre_pruning_axons/neurons` + an `ir_core_pre_pruning` descriptor) rendered only in
the Hardware tab's soft-core inspector (`hardware-tab.js:745-805`, and it too
silently drops the tile if any of three attrs is missing). The fix: fold the
pre→post pair (dims + masks + achieved-vs-configured sparsity) into the Pruning
tab from the same SSOT, make every skip loud (snapshot_error surface), and
de-literal the step-name gate (key off the step class/promise, not the display
string).

### II.11 (I.11) The Search button — exact root cause, verified by execution

The "Search" button is the `hw_config_mode` segmented control on the Hardware card
(`registry/entries_model.py:65`, options `fixed|search`; rendered as buttons by
`wizard/fields.js:392`); toggling seeds `arch_search={"optimizer":"nsga2"}`
(`wizard/main.js:400`), and the run reaches `ArchitectureSearchStep`.

**Root cause (one wiring hole), verified by executing the GUI starter baseline
with hardware-only search:** `architecture_search_step.py:97-104` builds
`fixed_platform_constraints` ONLY for `search_mode == "model"`; hardware-only
search leaves it `None`, so `_ensure_hw_only_cache` (`layout_hook.py:97`) passes
`{}` into `_collect_softcores` → **`KeyError: 'cores'` at `layout_hook.py:74`** for
every candidate. The exception is then swallowed per-candidate by
`nsga2_optimizer.py:76-87` (`except Exception` → penalty + `logger.warning`,
repeated pop×gen times), pymoo returns no feasible X, and the step raises
"Architecture search produced no candidates. Consider increasing pop_size…" —
blaming the budget for a hard wiring bug. `joint` and `model` modes work
end-to-end (verified); the Compilagent backend dies on the same KeyError in
hardware-only mode (unguarded `validate` at `backend/backend.py:182`).

Second, independent baseline trap: the Model-card Search raises
`NotImplementedError("No NAS search space defined…")` for every builder without
NAS options (lenet5, vgg16, resnet50, vit, …) — only builders with
`get_nas_search_options`/multi-option selects are searchable.

Additional verified drift (beyond the crash):
- `_decode_hw` (`problems/joint/problem.py:163-177`) synthesizes a platform of
  only `cores/target_tq/weight_bits/allow_coalescing`, **hardcodes
  `weight_bits=8`** (the starter pins 5), and drops
  `allow_coalescing/allow_neuron_splitting/…` for hardware-only mode — the search
  optimizes a *different chip* than the one deployment resolves
  (`build_platform_constraints_resolved` is bypassed entirely).
- The design intent was fail-loud (`layout_hook.py:87` documents it; the error
  contract test pins propagation) and the optimizer wrappers defeat it.
- GUI result channels: `GET /api/runs/{id}/discovered` does `.get("best")` on a
  JSON **list** → swallowed AttributeError → always `{"discovered": false}` (and
  no JS calls it); `NSGA2Optimizer` emits **zero** `search_event`s so the live
  search panel is blank for the classical baseline; the objective chips pre-select
  all 8 incl. accuracy while hardware mode silently filters accuracy out.
- Landmine: `search/multi_metric/multi_metric_search.py:66-81` has TOP-LEVEL
  executable code (a 1000×500 search + prints) imported by its package `__init__`
  — one stray import from a mystery hang. `scripts/import_path_inventory.py:65`
  references a module that no longer exists.
- Coverage: `tests/unit/search` is 159-green but no test constructs a real
  hardware-only problem, no test exercises `ArchitectureSearchStep`, and every
  template pins `hw_config_mode: "fixed"` — search has zero end-to-end coverage,
  which is how all of the above stayed invisible.

### II.10 (I.10) Optimizer surface — what exists, what the revision changes

What exists: a `SearchProblem` protocol + `EncodedProblem` vector encoding
(`search/problem.py`, `problems/encoded_problem.py`); a real pymoo NSGA-II
(`optimizers/nsga2_optimizer.py`); AgentEvolve and Compilagent drivers sharing an
LLM trace/event layer; candidate evaluation as a fast layout proxy
(`layout_hook`: build model → `LayoutIRMapping.collect_layout_softcores` →
`compute_mapping_stats` bin-packing → 8 static objectives; accuracy only via
short-train extrapolating evaluators — never the conversion pipeline).

Compilagent's introspection today (`backend/backend_layout.py`, `tools.py`): four
read-only tools — per-softcore records, a per-layer rollup whose layer identity is
a **name-string-split heuristic**, the full `LayoutVerificationStats`, and the
objective catalogue. The specific gaps against the owner's ask:
- **Shared weight banks are fully absent from the introspection surface** — the
  data exists upstream (`layout_ir_mapping.py:54,167-193` tracks
  `_sc_idx_to_bank_id`) but `LayoutSoftCoreSpec` (`mapping/layout/layout_types.py:10`)
  carries no `bank_id`, so the sharing map **dies at that boundary**. Threading
  `bank_id` through the spec is the single unlock.
- Softcores-per-layer exists but via the name heuristic — needs real IR-layer
  identity.
- Absent entirely: hard-core assignment (softcore → physical core), per-pass
  schedule structure (scalar count only), capability bits (`permission_kwargs()`
  forwards 3 of 7).

---

## Part III — Roadmap

Six workstreams. W0 items are ship-blockers and independent; W1/W2 parallelize;
W4 is the design core; W5 depends on W4 (with a W0-level minimal repair pulled
forward). Every task lists acceptance criteria; Part 0 rules bound everything
(fresh-run A/B for accuracy claims, pixel review for GUI, suite/typecheck/budget
gates, tests first).

### W0 — Ship-blockers

**W0.1 Pruned×streamed parity fix** (from II.3). Give the streamed NF↔SCM gate a
single neuron-index SSOT: the mapping's kept-neuron map (pruning masks → IR →
core neuron lists) applied to the NF capture before comparison
(`pipelining/core/nf_scm_parity.py` — `_capture_nf_streamed_counts` /
`_group_record_by_perceptron`). Never widen the SCM side; certify what deploys;
reconcile the ±1 row exactly. Tests: unit (a pruned two-layer flow through the
gate), plus a NEW tier-0 cell `lifs+pruned` via `templates/generate.py`.
Acceptance: the failing baseline config (lenet5+pruning, bare) runs green
end-to-end with all FATAL certificates; fresh tier spot-runs unchanged. Size: S–M.

**W0.2 Monitor failure states** (from II.5). (a) Engine: convert `_run_step`'s
`try/finally` into a real failure path invoking a new post-failure hook;
(b) `GUIHandle.on_step_failed` → `save_step_status(status="failed", end_time,
error)` + `DataCollector.step_failed` (already exists, zero callers); (c)
`gui/runs.py` honors persisted step `status` and folds `run_info.json`
status/error into overview payloads (`is_alive`, `error` included); (d) Stop
button visibility derived from `is_alive` on every render (copy the welcome-page
pattern, `welcome.js:157/176`); (e) push a terminal WS frame on failure so the UI
doesn't wait for the 30 s watchdog. Tests: unit for the hook chain + a
headless crash-run fixture asserting the persisted state machine; pixel review of
a failed run. Size: S–M.

**W0.3 Search minimal repair** (from II.11, pulled ahead of W5). One-line class:
build `fixed_platform_constraints` for `search_mode == "hardware"` too
(`architecture_search_step.py:97`); restore the documented fail-loud contract
(problem-level breakage must abort the search, not become pop×gen penalties —
distinguish infeasible-candidate from broken-problem in the optimizer wrappers);
fix the `"custom"`-preset-style seam here too: `_decode_hw` must stop hardcoding
`weight_bits=8` and must round-trip ALL platform keys through
`build_platform_constraints_resolved` so search and deployment see the same chip.
Defuse the `multi_metric` top-level-exec landmine. Tests: a real
hardware-only `JointArchHwProblem` unit test + an `ArchitectureSearchStep`
smoke test on the starter baseline. Acceptance: the Hardware-card Search runs
NSGA-II defaults to completion on the baseline and returns a result. Size: S.

### W1 — Config surface & platform truth

**W1.1 Weight-reuse knob removal** (II.1): the 6-site atomic removal + retired-key
machinery generalized to `platform_constraints` scope + unconditional reuse-phase
report + capability re-homing note for the future bank-aware consumer + template
regen + golden regen. Acceptance: suite/ratchets green; old saved configs get a
keyed remedy; one fresh tier spot-run byte-equal (modulo the report line). Size: S.

**W1.2 SANA-FE tile configuration** (II.8): `cores_per_tile` + tile-grid keys on
the hardware card; pass-through to the existing `SanafeRunner(cores_per_tile=...)`
seam; explicit-grid argument in `derive_arch_spec` with the
`width*height == n_tiles` hard invariant; derived defaults deterministic from the
platform's declared core capacity (never the packed count); resolved values
surfaced in `platform_constraints_resolved`; fix the `"custom"` preset enum
runtime error. Tests: extend `test_sanafe_arch_synth.py` (explicit keys,
determinism property, custom-preset). Acceptance: two runs of different models on
the same platform produce identical floorplans; NoC metrics comparable across
runs. Size: M.

### W2 — Monitor/GUI quality (pixel-verified)

**W2.1 Advisories → Review & Launch** (II.2): relocate the advisory cards into the
`review` section; keep a rail count badge; decide (with owner at pixel review)
whether mandate-violation advisories gate the Launch button. Size: S.

**W2.2 Heatmap correctness + instant loads** (II.6): renderer rewrite
(UI-resolution decimation with mask-preserving reduction, square cells, shared
labeled color scale, no per-index artists, thread-safe non-pyplot path) +
pipeline (pre-render at snapshot time or on-save; ETags; store eviction policy).
Acceptance: a 500-core run's Hardware tab paints fully in perceptually-instant
time from cold; pruned lines visible at any zoom; overlays aligned; pixels
reviewed. Size: M–L.

**W2.3 Pruning tab** (II.7): fold the pre→post dimension pair + masks + achieved
sparsity into the tab from the IR snapshot SSOT; make every silent skip loud
(flip the `test_snapshot_pruning.py:42` pin from silent-skip to surfaced
`snapshot_error`); de-literal the step-name gate; `onerror` fallback for resource
fetches. Acceptance: a pruned run shows per-layer map + pre/post dims + sparsity;
a missing-mask condition surfaces a visible diagnostic. Size: S–M.

### W3 — Recovery/monotone-adaptation rebalance (the II.4 program)

Owner-adjacent (MBH machinery). Goal: **no adaptation step exits below its
retention envelope; recovery budget lives where the loss occurs.** Program: use
the per-step envelope artifacts (`__mbh_retention_envelope.json`, retention
gauges) as the arbiter; reallocate the shared `endpoint_floor_steps` ledger by
envelope violation rather than fixed step order; give the violating step's
recovery a geometry that can climb (the §13 evidence shows the current LIF-leg
stalls flat while WQ climbs); keep the §13 curriculum insight (plain pretext →
deployed-composition grind) as the default shape unless measurement says
otherwise; the contained `experimental_walk_recovery` arm becomes a candidate
tool inside this program (its +10.3 pp catastrophic-case evidence stands, its
healthy-cell regressions forbid blanket arming). Protocol: fresh-run A/B on the
full lifsync/lifs tier subset for every candidate change; certificates FATAL
throughout. Acceptance for the mmixcore story specifically: LIF Adaptation exits
within envelope of its entry on the baseline, WQ's recovery no longer masks an
upstream collapse, and no tier cell regresses. Size: M–L (research-adjacent).

### W4 — Deployment formalization & formal cost model (II.9 → thesis §2)

The design core; mostly a **join and a registry** over measured surfaces that
already exist.

**W4.1 The deployment record.** One typed, versioned schema (a `DeploymentRecord`)
capturing the thesis §2 objects concretely: the **schedule** (ordered passes; per
pass: neural segments + interleaved ComputeOps; per segment: cores, parameter
counts AND bytes, connectivity entry counts AND bytes, latencies, spans,
reuse-vs-reprogram classification); **placement** (softcore→hardcore assignment,
banks, tiles); **utilization/area** (fold `CrossbarUtilizationReport` — currently
write-only — and `LayoutVerificationStats` in); **traffic** (tap
`flow_node_counts` for per-boundary window counts; join SANA-FE `noc_links`/tile
packets); **timing** (per-segment `timesteps_executed`, `ChipLatency`,
`_SegmentTiming`, sync counts, and NEW: timed host ComputeOps); provenance per
field (which stage/simulator produced it). Producers to wire (all identified in
II.9): `weight_reuse_plan_from_graph` + `weight_programming_report` +
`schedule_pass_count` + `programming_bits` → the dead `CostRecord` reprogramming
fields; the discarded relay-insertion count; the never-written
`ft_pass_walls.json` (producer shape exists in
`smooth_adaptation_cycle.py:148`); nevresim cycle extraction. `DeploymentPlan`
stays configuration-only; the record is a separate, measured artifact keyed by
the plan.

**W4.2 The parameterized cost model.** Over the record: segment initialization =
per-core constant reset/init + programming payload (params+connectivity bytes)
with the reuse band (`weight_reuse_cost_model.py` coefficients finally applied to
real quantities — the banded cost of the proposal); energy = SANA-FE measured +
traffic-term decomposition; latency = the full decomposition (programming,
compute passes, host ops, sync barriers, NoC); area = cores/occupancy/waste;
throughput. Every term carries its coefficient band and provenance
(measured vs modeled), preserving the "no proxies presented as measurements"
discipline.

**W4.3 Objectives registry v2.** Supersede `search/results.py::ALL_OBJECTIVES`
with a registry over the record: the existing 8 static objectives PLUS the
measured axes (deployed accuracy, mJ/sample, latency decomposition terms, spikes,
pass count, reprogramming bytes, NoC hop/traffic terms) — typed, direction-ed,
unit-ed, provenance-ed, extensible by registration (new objectives are additions,
never surgeries).

Acceptance: every tier run emits a schema-validated `DeploymentRecord`; the
legacy `cost_record.json` fields are reproduced exactly as a projection of the
record (continuity); a written formal spec maps each thesis-§2 term to its record
field 1:1; the three previously-disconnected cost surfaces have one owner each
inside the record. Size: L (the design doc + staged landings).

### W5 — Optimization surface revision (II.10/II.11 full; depends on W4)

**W5.1 Problem surface v2.** The joint problem consumes W4.3 objectives;
candidate platforms round-trip through the SAME `build_platform_constraints_resolved`
as deployment (kill the `_decode_hw` divergence permanently); fail-loud restored
end-to-end; hardware/model/joint modes share one evaluation contract; the
`EncodedProblem` adapter remains the generic-library surface (pymoo proven) with a
documented mapping for other MOO libraries.

**W5.2 Introspection channel.** A typed, versioned introspection payload registry
over the W4.1 record: thread `bank_id` through `LayoutSoftCoreSpec` (the one-field
unlock), real IR-layer identity (replace the name-split heuristic), hard-core
assignment, per-pass schedule structure, full capability bits. Compilagent's four
tools re-serve from this registry; AgentEvolve's prompting consumes the same
payloads. Clean-design bar: optimizers depend on the registry types only — no
reach-ins into mapping internals.

**W5.3 GUI reconnect.** NSGA-II emits the `search_event` stream the live panel
already handles; fix the `/discovered` list-vs-dict bug (or remove the dead
endpoint); objective chips honest per mode; add an end-to-end search template
cell (`hw_config_mode: "search"`) to the template matrix so search can never
silently drift again. Acceptance: baseline Search shows live generations and a
final Pareto set in the GUI; the search e2e cell is part of the guarded matrix.
Size: W5 total M–L.

### Sequencing

```
W0.1, W0.2, W0.3      — immediately, independent, small
W1.1, W1.2, W2.1–W2.3 — parallel after W0
W3                    — parallel track (MBH-adjacent, fresh-A/B protocol)
W4.1 → W4.2 → W4.3    — the design track (start with the W4.1 schema doc for owner review)
W5.1 → W5.2 → W5.3    — after W4.3 (W0.3 keeps Search alive meanwhile)
```

### Verification protocol (applies to every workstream)

1. Fresh-run vs fresh-run for any accuracy/runtime claim (the ledger persists in
   run dirs; resumes are starved — II.4/Part 0).
2. FATAL certificates green on every touched configuration; parity fixed at the
   SSOT, never budgeted.
3. GUI changes: screenshots against a written UX spec, reviewed by the owner.
4. Suite ≤2 min green, typecheck 0, template tests green, budgets/ratchets clean,
   goldens regenerated only via their scripts.
5. New coverage lands WITH the fix (the recurring lesson of this investigation:
   every one of II.3, II.8, II.11 was a hole precisely where the template matrix
   had no cell).

---

## Appendix

**Repro commands**
- II.3: run the saved config
  `generated/simplemlp_baseline_20260810_091219_…/_RUN_CONFIG/config.json`
  (lenet5+pruning, bare) → NfScmParityError at Soft Core Mapping.
- II.4: `generated/mmixcore_baseline_20260810_091759_…` — trail in
  `_GUI_STATE/live_metrics.jsonl`.
- II.11: starter baseline + `hw_config_mode="search"`,
  `arch_search={"optimizer":"nsga2"}` → KeyError:'cores' per candidate
  (visible as `mimarsinan.search.optimizers.nsga2_optimizer` WARNINGs), then
  "produced no candidates".

**Evidence artifacts**: `docs/research/findings/lif_deployment_exactness.md`
§10–§13 (parity classes, curriculum, ledger contamination, knob
characterization); `docs/CERTIFICATION_PROTOCOL.md`;
`chip_simulation/pareto.py::COST_BAND_DISCLAIMER` (the self-documented
instrumentation gap W4 closes).

**Glossary**: *pass* — one traversal of the NeuralOps/ComputeOps schedule;
*segment* — a contiguous on-chip neural stage; *tile* — hard-core group with
NoC-free internal communication; *ledger* — the run-total endpoint-recovery step
pool (`endpoint_floor_steps`); *envelope* — a step's retention band
(entry-vs-exit accuracy contract); *bank* — a shared weight matrix mapped once
and referenced by multiple cores.

