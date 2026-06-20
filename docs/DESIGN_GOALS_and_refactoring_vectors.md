# Design goals + the generic config matrix + refactoring vectors

Companion to `docs/research_artifacts_for_cascaded_ttfs_tuning/PROGRAM_SPEC_generic_deployment.md`
(§3 config space). This doc elaborates the *software-design intent* behind that genericity,
enhances the matrix, and proposes the refactoring vectors that keep the intent intact as
features pile on. Grounded in a 2026-06-20 audit of the actual seams (file:line below).

---

## 1. Software design goals (the intent)

The framework is a **compiler from a declarative deployment configuration to a deployed,
verified spiking network**. The genericity goal is not "support many features" — it is that
**the configuration is data and the pipeline is *derived* from it**, so the cross-product of
axes is covered without a combinatorial code explosion. Concretely:

1. **Declarative → derived.** The deployment config is the single input; everything
   downstream (which steps run, which forward trains, which soma deploys, which mapping
   strategy, which parity gates, which backend) is *resolved from it*. Adding an option must
   not mean editing the pipeline — only registering a new strategy.

2. **Orthogonal composition (cross-product, not flag-thicket).** `firing × sync × encoding ×
   mapping × backend × init × pruning × thresholding × spike-gen × bias × quantization ×
   calibration × driver` are independent concerns. Any cell of the product must assemble.
   Two concerns interacting is the exception (a declared, tested compatibility rule), never
   the default (a global `if`).

3. **One Source of Truth per concern.** Each decision is resolved in exactly ONE place; the
   rest of the code reads the *resolved decision*, never the raw flags. (Today the
   `SpikingDeploymentContract` does this for spiking semantics and `TtfsAdaptationPlan` for
   the adaptation thicket — but ~50 raw `config.get` reads and a 15+-site
   `cascaded`/`synchronized` dispatch still bypass it.)

4. **Open–closed / bounded blast radius.** The cost of adding one option to one axis should
   be ≈ one new strategy class + one registry entry — *not* scattered edits. Today a new
   `spiking_mode` touches 6–8 files (deployment_specs, behavior_config, neg_shift_bias,
   sanafe/net_synth, hybrid/flow, code_generation, …). The blast radius IS the debt metric.

5. **Behavior-carrying polymorphism, not string-isolating predicates.** A predicate like
   `requires_ttfs_firing(mode)` hides the *string* but each of its 10+ callers re-derives the
   *behavior* (which soma, which forward, which step). A **policy object** carries the
   behavior so callers just call a method. Predicates isolate the symptom; policies cure it.

6. **Capabilities vs. strategies are separate layers.** *Capabilities* = what the chip allows
   (cores, max_axons/neurons, has_bias, coalescing/splitting/scheduling permitted). They are
   **declared**. *Strategy* = how to fit a given model onto a capability-gated chip
   (coalesce a wide fan-in, split a wide channel, insert a sync-point). It is **derived**.
   Today these are tangled: three independent `allow_*` booleans threaded through the builder.

7. **Validate up-front, fail loud, never core-dump.** Every backend×mode combination is
   checked at *assembly* against a capability matrix, not reactively mid-run. Degenerate /
   unsupported configs raise actionable errors. (The SANA-FE SIGFPE — an unpinned dep
   upgrade — is the cautionary tale: a silent incompatibility that core-dumped instead of
   failing loud. Now guarded; the *pattern* must be standing policy.)

8. **Equivalence-preserving evolution.** Refactors are byte-identical (golden/parity/fidelity
   harness) or strictly better. The parity gates (`nf_scm_parity`, the torch↔sim fidelity
   lock, segment-policy goldens) are not just correctness checks — they are the **license to
   refactor aggressively**. No structural change without the equivalence lock green.

9. **Reproducible & pinned.** External deps are version-pinned; the deployment is
   deterministic given (config, seed, versions). Numerics that depend on reduction order
   (cuBLAS) are pinned. Determinism is a feature, not an accident.

10. **The code reads like the spec.** The abstractions mirror the domain axes, so a reader
    maps "synchronized cascaded TTFS, offloaded, coalescing, on nevresim" to objects, not to
    a trail of booleans. Mathematical/structural elegance is a maintainability goal, not a
    vanity one.

---

## 2. The enhanced configuration matrix

The original §3 matrix conflated **input axes** (what the user declares) with **derived
strategies** (what the framework resolves), and omitted several real axes (bias mode,
quantization, calibration, the optimization driver, the chip-capability gates). The clean
model separates the two:

### 2a. Input axes (declared in the deployment config)

| Group | Axis | Options (extensible) |
|---|---|---|
| **Workload** | Model | any PyTorch model, no non-mappable layer |
| | Dataset | any |
| | Weight init | pretrained · train-from-scratch |
| | Pruning | mode + fraction |
| **Spiking semantics** (→ the behavior contract) | Firing mode | LIF (rate) · TTFS analytical (`ttfs`/`ttfs_quantized`) · TTFS cycle-based |
| | Sync mode (× firing) | synchronized · cascaded |
| | Encoding placement | subsumed · offloaded |
| | Thresholding | `<` · `<=` |
| | Spike generation | TTFS · uniform · stochastic · spike-train |
| | **Bias mode** *(added)* | on-chip · param-encoded · subsumed |
| **Hardware platform / capabilities** | Core grid | per-core max_axons, max_neurons, count, has_bias |
| | **Weight quantization** *(added)* | bit width |
| | **Capability gates** *(reframed)* | coalescing-allowed · neuron-splitting-allowed · scheduling-allowed — *permissions, not strategies* |
| **Conversion process** *(added group)* | Activation quantization | on · off |
| | Calibration / conversion-health | gain-correction · distribution-matching · theta-cotrain · boundary-STE (composable) |
| | Optimization driver × ramp | controller · fast-ladder × {value-proxy · genuine · STE} |
| **Deployment target** | Backend | nevresim · SANA-FE · Lava · HCM/SCM sim · … |

### 2b. Derived (resolved by the contract — NOT user axes)

The framework computes these from the input; they must never be declared or branched on raw:

| Derived decision | Resolved from | Today's SSOT (or gap) |
|---|---|---|
| **Mapping strategy** (coalesce / split / sync-point) | model shape × capability gates | `compute_fc_tiling_mode` (coalescing only); splitting/scheduling are loose flags — *partial* |
| Training-forward kind | firing × sync | `contract.training_forward_kind()` — 5-way `if/elif` *(SSOT but not polymorphic)* |
| Soma / neuron model | firing × sync × spike-gen | `soma_hw_name_for_spiking_mode` + a 5-way chain in sanafe build — *duplicated* |
| Decode mode (count vs timing) | firing mode | scattered in flow/runner — *gap* |
| Which pipeline steps run | the whole contract | `deployment_specs.get_pipeline_step_specs` — *polymorphic but flag-heavy* |
| Which parity/acceptance gates apply | spiking mode × mapping | `nf_scm_parity_enabled` etc. — *SSOT, good* |
| Which backends are valid | backend-caps × spiking mode | `_BACKEND_CAPS` matrix — **informational only, not consulted at assembly** |

**The key reframing:** *the user declares a chip's capabilities and a workload; the framework
derives the mapping strategy to fit one onto the other.* Coalescing/splitting/sync-points were
listed as input axes in §3 — they are outputs.

---

## 3. Critical refactoring vectors

Ordered by leverage (how much each protects the design intent per unit of churn). Each cites
the debt it removes (audited file:line) and the principle (§1) it defends. All are
strangler-fig: introduce the seam, move behavior verbatim, lock byte-identical, then delete
the scattered branches.

### V1 — One contract-resolution layer for *every* axis (generalize `TtfsAdaptationPlan`)
**Principle 1, 3.** Today `SpikingDeploymentContract` covers spiking semantics and
`TtfsAdaptationPlan` covers the adaptation flags, but ~50 raw `config.get(...)` reads remain
scattered (`pipelining/core/**` reads `spiking_mode` 5×, `activation_quantization` 4×, the
`enable_*` flags, pruning, …). **Vector:** a single `DeploymentPlan` (the contract + the
derived table of §2b) that every module receives and reads; raw `config.get` of a deployment
flag outside the resolver becomes a lint-failure (grep-guarded, like the existing
`SpikingDeploymentContract` sole-reader guard). This is the backbone the rest hang off.

**V1 status — sole-reader guard BROADENED + closed (2026-06-20).** R1 scoped the grep-guard to
`pipeline_steps/**` only (the dir it fully migrated). This round migrated the remaining genuine
deployment-DECISION reads to `DeploymentPlan` and broadened the guard to **all of
`src/mimarsinan`** with a tight, documented allowlist
(`tests/unit/pipelining/test_deployment_plan.py::TestNoStrayDeploymentFlagReadsAnywhere`). The
guard has teeth (matches both `.get("flag")` and `["flag"]` read forms, receiver-agnostic, with a
write-target guard so `dp["flag"] = …` in the derivation layer is not a false offender) and is
non-flaky (a `test_allowlist_has_no_dead_entries` companion fails if any allowlisted carve-out
rots — file deleted or its read migrated away). Migrated: `simulation_runner/core.py` `spiking_mode`,
and `deployment_pipeline.py`'s `cuda_debug` (re-resolved AFTER the env `setdefault` so the plan
observes it), the `act_quant`/`wt_quant` display line, and the `pruning`/`pruning_fraction` log
reads — all byte-identical. Closed decisions:

- **`ttfs_cycle_schedule` stays OUT of the V1 forbidden set.** It already has its own src-wide
  sole-reader guard owned by `SpikingDeploymentContract`
  (`test_deployment_contract.py::TestSingleReaderInvariant`); duplicating it here would only widen
  the allowlist redundantly.
- **`simulation_runner/core.py` `weight_quantization` is NOT migrated (byte-identity carve-out).**
  The runner defaults it to `True` (legacy "quantized unless told otherwise"); `DeploymentPlan`
  resolves it with `False`. The key is genuinely absent for a `vanilla` config (the `phased`
  preset is what materialises it), so routing the read through the plan would FLIP the omitted-key
  value. Left verbatim, allowlisted with the reason. (Unifying the two defaults is a behavior
  change, deferred — not this finishing unit's mandate.)
- **The firing-semantics SSOT layer is allowlisted, not migrated.** `behavior_config.py`
  (`NeuralBehaviorConfig`), `firing_strategy.py` (`FiringStrategyFactory` — the C++-comparator
  factory), `deployment_contract.py`, `code_generation/generate_main.py`, and `config_schema/*`
  read the raw keys because they DEFINE / DERIVE / VALIDATE / DISPLAY the sub-contract (they are
  resolvers, not decision consumers). Forcing them through the plan would invert the dependency.
- **Tuner training-forward selection is allowlisted, not migrated.** `adaptation_manager.py` and
  the `activation_shift` / `lif` / `ttfs_cycle` tuners read `spiking_mode` /
  `cycle_accurate_lif_forward` to pick their training-forward family (a V2 `SpikingModePolicy`
  concern). The read sits on a per-perceptron `pipeline_config` dict / a tuner-local knob (and the
  ttfs_cycle read defaults to `"ttfs_cycle_based"`, not the plan's `"lif"`); threading a plan
  through those signatures removes no scattered branch and is not byte-identical — pure churn.

### V2 — `SpikingModePolicy` registry: behavior-carrying, replace the predicate-then-branch
**Principle 4, 5.** The single highest-blast-radius debt: `spiking_mode` /
`ttfs_cycle_schedule` (`cascaded`/`synchronized`) dispatch across **15+ sites**
(`deployment_specs.py:130-147`, `behavior_config.py:62-96`, `deployment_contract.py:89-102`
5-way `training_forward_kind`, `neg_shift_bias.py:125-140`, `sanafe/net_synth/build.py:184-218`,
`models/spiking/hybrid/flow.py:68,171-197`, `code_generation/generate_main.py`). **Vector:** one
polymorphic policy per (firing × sync) — `LifPolicy`, `TtfsAnalyticalPolicy`,
`TtfsSyncCyclePolicy`, `TtfsCascadePolicy` — each carrying `training_forward()`,
`soma_model()`, `decode()`, `calibration_forward()`, `applicable_steps()`, `valid_backends()`.
Mirrors `segment_policies.py` (already proven). Collapses the 5-way `training_forward_kind`,
the sanafe soma chain, the calibration-forward switch, and the flow forward-selection into
method overrides co-located in one class. **A new spiking mode = one new policy + one registry
line** (vs. 6–8 files today).

**V2 status — last dispatch site migrated; CLOSED (2026-06-20).** The final
`spiking_mode`-branching dispatch outside the policy — `code_generation/generate_main.py:resolve_exec_policy`
(the nevresim C++ `ComputePolicy`/`Execution` type-string choice) — now **delegates to
`SpikingModePolicy.nevresim_exec_policy(NevresimExecParams) -> ExecPolicySpec`**. The firing × sync →
codegen choice lives in one place per family (`LifModePolicy` / `TtfsAnalyticalModePolicy` /
`TtfsCascadeModePolicy`; the sync-cycle policy raises since nevresim never runs the synchronized
schedule). `resolve_exec_policy` keeps only the comparator/reset string selection (`resolve_compare_policy`
/ `resolve_lif_fire_policy`) and packs the chip-shape scalars into `NevresimExecParams`. Byte-identical:
the lock test in `tests/unit/code_generation/test_exec_policy.py` asserts the policy-driven output ==
the legacy golden string for every mode; torch↔sim fidelity + nf_scm parity green. Closed decisions:

- **The thresholding/firing → comparator-string selection stays on the codegen helpers, NOT the policy
  (DO NOT move).** `resolve_compare_policy` (`<`/`<=` → `Strict`/`InclusiveCompare`) and
  `resolve_lif_fire_policy` (`Default`/`Novena` + threshold → `LIFirePolicy<reset,compare>`) are the
  thresholding/firing comparator SSOT (mirrored on `FiringStrategyFactory`), orthogonal to the firing × sync
  *template family* the policy picks. They are passed into `NevresimExecParams` as data — folding them into
  the policy would entangle two independent axes and duplicate the `FiringStrategy` truth table. Kept as-is
  on purpose; **not** an open TODO.

### V3 — `Backend` interface + capability-validated registry (consult `_BACKEND_CAPS` at assembly)
**Principle 7.** `_BACKEND_CAPS` (`spiking_semantics.py:95-104`) declares which backends
support which modes — but it's **informational**; selection is `enable_*` flags
(`deployment_specs.py:104-169`) and validation is reactive inside `step.process()`. This is
exactly how the SANA-FE incompatibility reached the C++ and core-dumped. **Vector:** a
`Backend` ABC (`build`, `run`, `parity_gate`, `supports(contract)`) with a registry; pipeline
assembly *consults the capability matrix* to (a) select/validate backends up-front and (b)
append their steps — an unsupported backend×mode raises an actionable error at assembly, not a
crash mid-run. (The sanafe version guard added this for one failure mode; generalize it.)

### V4 — `ChipCapabilities` + derived `MappingStrategy` (untangle the three `allow_*` flags)
**Principle 6.** `allow_coalescing` / `allow_neuron_splitting` / `allow_scheduling` are three
independent booleans threaded through `hybrid_build_pool.py:35-68`,
`schedule_partitioner.py`, `mapping_structure.py:48-81`. **Vector:** a `ChipCapabilities`
value object (the declared permissions + core grid) and a `MappingStrategy` resolver that,
given a layer's shape and the capabilities, *derives* coalesce/split/sync-point (extending
`compute_fc_tiling_mode` from coalescing-only to the full decision) and raises a clear
infeasibility error when no strategy fits. Capabilities declared once; strategy derived once;
the builder consults the resolved strategy, not the raw flags.

**V4 status — strangler-fig cleanup CLOSED (2026-06-20).** The three back-compat bool kwargs
(`allow_coalescing` / `allow_neuron_splitting` / `allow_scheduling`) on
`build_hybrid_hard_core_mapping` are **deleted**; the only knob is `strategy=` (a resolved
`MappingStrategy`; omitted ⇒ all-permissions-off). The sole src caller already passed
`strategy=`; the ~25 test call sites were migrated to construct one via
`MappingStrategy.from_permissions(...)` / `ChipCapabilities.from_platform_constraints(...)`
(the new SSOT for wrapping loose raw bools), byte-identical (placement goldens + nf_scm parity +
torch↔sim fidelity green). Closed decisions:

- **The V4 leaf helpers keep their individual bool params (no-value-churn, DO NOT migrate).**
  `pack_layout` (`layout/layout_packer.py`), `verify_hardware_config`
  (`verification/.../mapping_verifier_hw.py`), `split_softcores_by_capacity`
  (`support/schedule/schedule_split.py` + `schedule_partitioner`), and the
  `suggest_hardware_config*` suggesters genuinely read **individual** permission bits at leaf
  level and are not called with the full three-bool tuple from a hot entry-point cross-section.
  Wrapping them in a `MappingStrategy` would add an indirection without removing any
  scattered branch (the blast-radius metric is unchanged) — pure churn. `permission_kwargs()`
  already exists for the entry points that DO want to spread a resolved capability object into
  these signatures. Kept as-is on purpose; this is **not** an open TODO.

### V5 — Contract-driven `StepPlan` (each step declares its applicability)
**Principle 1, 4.** `get_pipeline_step_specs` (`deployment_specs.py:94-177`) is the *least bad*
seam (semantic queries, no hardcoded list) but still hand-assembles with per-flag `append`s.
**Vector:** each step declares `applies_to(plan) -> bool` (and its requires/promises); the
plan filters the registry. "Which steps does this config need" becomes data each step owns,
not a 80-line conditional. Composes with V2 (`policy.applicable_steps()`).

**V5 status — FINISHED + CLOSED (2026-06-20).** The first part (`applies_to` + the ordered
`StepPlan` registry) landed earlier. This unit finished the second part: each `PipelineStep`
now declares its data contract at the CLASS level (`REQUIRES`/`PROMISES`/`UPDATES`/`CLEARS`
tuples; `PipelineStep.declared_contract()` exposes them), lifted byte-identically from the
former local `__init__` lists (`__init__` now reads the class attrs). `StepPlan.validate_data_contract(plan)`
asserts the requires/promises DAG at ASSEMBLY time — every consumed entry is promised by an
EARLIER selected step — raising `StepPlanContractError` that NAMES the missing producer (mirrors
the runtime `Pipeline.verify`/`set_up_requirements` semantics: promises+updates publish, clears
retract). `get_pipeline_step_specs` routes through it, so an unsatisfiable DAG fails loud at
assembly instead of as a bare `requires X` assertion deep in a run. Verified byte-identical to
the prior assembly across the full 7680-config cross-product (HEAD-vs-worktree diff = 0) + the
`test_step_plan.py` registry-integrity locks + the nf_scm parity gate (25). Closed decisions:

- **`TTFSCycleAdaptationStep` keeps its instance-specific `requires` extension (the ONE step
  not fully lifted, on purpose).** It appends `activation_scales` to `requires` in `__init__`
  ONLY when `ttfs_scale_aware_boundaries` is on — a per-instance, config-derived case that
  cannot be a static class constant byte-identically (the flag-off path must NOT declare it).
  Its `REQUIRES` declares the always-present static lower bound, which is what the assembly-time
  DAG validation checks; the opt-in extra is itself always satisfiable (the unconditional
  Activation Analysis step promises `activation_scales` earlier). Noted, not an open TODO.
- **Instance `requires`/`promises`/`updates`/`clears` stay `list`s (not the class tuples).** The
  base `__init__` does `list(...)` so the instance attribute is byte-identical to the pre-V5
  hand-built lists (the class declaration is an immutable tuple of constants; the instance gets
  its own mutable list). This keeps every existing reader — incl. the `ttfs_cycle` `requires + […]`
  concatenation and the `test_sanafe_simulation_step` list-equality assertion — unchanged.

### V6 — Mapper-node visitor (kill the `isinstance` chains)
**Principle 4, 5.** `isinstance(node, …Mapper)` chains re-implemented in 3+ files —
`per_source_scales.py:29-65`, `visualization/softcore_flowchart_dot.py:111-175`,
`spiking/scale_aware_boundaries.py:39-48` (and activation-unwrap loops in `lif_utils.py:16-24`,
`ttfs_spiking.py`). **Vector:** a polymorphic method on the `Mapper` base (or a `MapperVisitor`)
for the per-node operations (scale propagation, boundary handling, viz). A new mapper kind adds
one method, not edits in 3+ files.

### V7 — Finish the adaptation trifecta: `CalibrationPipeline` (P2) + `OptimizationDriver` (P3)
**Principle 2, 4.** `RampStrategy` (P1) and `TtfsAdaptationPlan` (Phase D) landed; the
`REFACTOR_PLAN_adaptation_strategies.md` P2/P3 remain. **Vector:** make calibration
(gain/theta/distmatch/boundary) a list of composable steps with explicit `compatible_with`,
and the controller-vs-fast-ladder split an `OptimizationDriver` — so the conversion-process
axes (§2a Group D) compose instead of interacting through `_configure` precedence. New
research calibrations (the Program-3 results) then drop in as steps.

### V8 — Config-schema collapse with provenance (P4) + dependency-pin policy
**Principle 9, 10.** ~90 flat `deployment_parameters` with overlapping prefixes. **Vector:**
namespace them under their owning concern (the §2 groups), with one translation table from the
legacy flat keys (tested, deprecation-shimmed), and a schema that records each key's
owner/derivation. Bundle the standing rule: **pin every external dependency** + a
version/capability guard at each integration boundary (the sanafe lesson, generalized).

### V0 (meta) — The equivalence harness is the guardrail for all of the above
**Principle 8.** None of V1–V8 are safe without the golden/parity/fidelity locks
(`nf_scm_parity`, the torch↔sim fidelity harness, the segment-policy + tuning goldens). The
discipline: extend the lock to *characterize* a seam before moving it, refactor to
byte-identical, then delete the old branch. This is what made P1/Phase-D safe (722 tests
byte-identical) and is the precondition for V2–V6.

---

## 4. Sequencing

V0 is always-on. Then: **V1** (the resolver backbone) → **V2** (the biggest blast-radius win,
spiking-mode policies) → **V3/V4** (backend + mapping capability contracts, which also close
the AC6 robustness gap class) → **V5/V6** (step plan + visitor cleanups) → **V7** (finish the
adaptation trifecta) → **V8** (config collapse + pin policy). Each is independently shippable
and test-locked; none is a big-bang.
