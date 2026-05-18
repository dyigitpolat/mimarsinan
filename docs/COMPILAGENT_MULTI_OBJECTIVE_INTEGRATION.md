# Compilagent × Mimarsinan: Multi-Objective Co-Search Integration

This document records the **end-to-end technical design** of the integration between **Compilagent** (agentic compiler-heuristic search core) and **Mimarsinan** (SNN + neuromorphic mapping / joint architecture–hardware search), with emphasis on **multi-objective evaluation**, **open hardware levers**, **agent guidance**, and **live observability**. It is written for maintainers who need both the architectural “why” and the implementation “where / how”.

---

## 1. Landscape and high-level objectives

### 1.1 Two frameworks, one problem

- **Mimarsinan** already solves **joint architecture + hardware** search via `JointArchHwProblem`: validation goes through `validate_detailed`, evaluation through `evaluate`, and objectives are a **named vector** (accuracy, parameter counts, layout-derived utilization, fragmentation, etc.) used by **NSGA-II** and **AgentEvolve** with Pareto semantics.
- **Compilagent** models search as an **`OptimizationSession`**: a workload spec, a pluggable **`Backend`**, a canonical **`Toolset`**, and harness-driven LLM loops (`pydantic-ai`, Claude SDK, …). Historically the session optimized a **single timing axis** (`TimingResult.median_ms`, speedup vs baseline, leaderboard sorted on latency).

The integration’s **north star** is to **reuse Mimarsinan’s single source of truth** for compile/validate/evaluate while **extending Compilagent** so that:

1. **All objectives** are first-class on the wire (JSON tool returns, leaderboard rows, observation events).
2. The agent can **query** the population by metric, **compute Pareto fronts**, and **compare** candidates without inventing parallel math in Mimarsinan.
3. **No single metric is privileged** as a “primary objective”; multi-objective runs are not silently collapsed onto latency.
4. The **search space for hardware sizing** is not over-constrained by JSON bounds meant for evolutionary algorithms; the agent can propose integers in wide validated ranges.
5. **Operators** get the same multi-objective signal as the model (live GUI: leaders, ranks, injected guidance text).

### 1.2 Architectural stance

- **Thin adapter, fat domain**: `MimarsinanLayoutBackend` delegates to existing `JointArchHwProblem` APIs; it does not fork layout math.
- **Generic MO primitives in Compilagent**: `session/multi_objective.py` is pure, backend-agnostic logic over normalized objective payloads.
- **Harness-specific ergonomics in Mimarsinan**: `GuidedToolset` is middleware on the tool surface (not a fork of Compilagent’s session loop), chosen because **tool under-use** was a practical failure mode.
- **Unified result path**: Final `SearchResult` still flows through `agent_evolve_support` (Pareto + minimax selection), so downstream pipeline steps see the same `Candidate` / objective shapes as other optimizers.

---

## 2. Compilagent: core library changes

### 2.1 Python version and `StrEnum` compatibility

- **`requires-python`** in `compilagent/pyproject.toml` is **`>=3.10`** (previously effectively 3.11-only due to `enum.StrEnum`).
- New module **`compilagent/_compat.py`**: on **3.11+** re-exports `StrEnum` from `enum`; on **3.10** provides a small **`str` + `Enum` mixin** backport with `__new__` / `__str__` aligned to stdlib semantics so string comparison and serialization behave like 3.11’s `StrEnum`.
- All in-tree `StrEnum` subclasses (`EventKind`, `WorkloadKind`, `StreamEventKind`, Triton internal enums, …) import **`from compilagent._compat import StrEnum`**, so **one policy** governs compatibility.

**Nuance**: This is strictly a **stdlib surface** fix; it does not change optimization semantics.

### 2.2 `IntFreeform`: open integer levers

**File**: `compilagent/src/compilagent/core/search_space.py`

- New frozen dataclass **`IntFreeform`** with fields: `min`, `max`, `step` (default `1`), `units`.
- **`serialize()`** emits `{"kind": "int_freeform", "min", "max", "step", "units"}` so agents and UIs see an **interval + step**, not a finite candidate list.
- **`LeverRange`** union extended: `IntRange | IntFreeform | FloatRange | EnumChoice | BooleanFlag | StructuredJsonRange`.

**Design intent**:

- **`IntRange`** = curated enumeration (good when the backend wants explicit discrete trials).
- **`IntFreeform`** = **scale reasoning** — the agent reads bounds and picks values; **`validate_intervention`** (backend) is the hard gate.

Exports updated in **`compilagent/core/__init__.py`** and top-level **`compilagent/__init__.py`**.

### 2.3 `session/multi_objective.py`: pure MO helpers

**File**: `compilagent/src/compilagent/session/multi_objective.py`

All functions operate on **row-shaped dicts** carrying at least:

- `candidate_id` (or `id` — helpers normalize via `_candidate_id`)
- `objectives`: `dict[str, Any]` where each metric maps to either:
  - a **serialized objective** `{"name", "value", "goal", "unit"}` (`goal` ∈ `{"min","max"}`), or
  - a bare numeric (treated as value only in `_objective_value`).

**API summary**:

| Function | Behavior |
|----------|----------|
| **`rank_by_metric(rows, metric, top_k=None)`** | Sorts by metric using `_goal_for`; rows missing numeric values go to the end with `rank: null`. **Dense ranking**: ties share the same rank; next rank skips (standard competition ranking style). |
| **`pareto_front(rows)`** | Filters to rows with non-empty `objectives`. Builds **union of all metric keys** across rows (`_common_metrics`). **Skips** rows that lack **any** common metric (cannot compare). Uses `_dominates` over full common metric set. **Degenerate**: ≤1 row → return as-is. |
| **`domination_count(rows)`** | Per-candidate count of others that dominate it; **0** ⇒ on front. |
| **`metric_summary(rows)`** | Per metric: `goal`, `unit` (first non-empty from rows), `best` / `worst` `{candidate_id, value}`, `median`, `count`. |

**Nuances**:

- **`_goal_for`**: first row that exposes a valid `goal` for that metric wins; backends should be **consistent** across candidates.
- **Plain numeric objectives** in a row are supported in **`rank_by_metric`** / **`metric_summary`** via `_objective_value`; **`_dominates`** requires **mapping** entries with `"value"` — so dominance is only well-defined for **serialized** shapes. Compilagent’s session serialization path (below) normalizes backend outputs to mappings when possible.
- **Pareto front complexity**: naive O(n²) pairwise dominance — acceptable for typical session sizes (`max_candidates` in the tens).

### 2.4 `OptimizationSession`: objectives on the wire, success rule, events, tools

**File**: `compilagent/src/compilagent/session/session.py`

#### 2.4.1 `Backend.objectives_for_candidate` integration (unchanged contract, widened consumer)

After compile/timing, the session builds **`objectives_map`**: `str` → serialized dict. Serialization rules:

- Objects with **`.serialize()`** → call it.
- **`Mapping`** → shallow `dict` copy.
- **Scalars** → coerced to `{name, value, float(value), goal: "min", unit: ""}`.

#### 2.4.2 **Success criterion** (`successful`)

Previously, “success” effectively required a **non-null `median_ms`** for many backends. For Mimarsinan, **`time_workload` now sets `median_ms=None`** intentionally (see §3.3), so the session would mark all runs failed unless generalized.

**New rule**:

```text
has_signal = (timing.median_ms is not None) OR bool(objectives_map)
successful = compile_ok AND has_signal AND (correctness is None OR correctness.ok)
```

So:

- **Classic backends**: still succeed on timing.
- **Multi-objective backends**: succeed on **non-empty objectives** even when `median_ms` is absent.

#### 2.4.3 **`run_candidate` JSON response**

The stringified JSON now always includes **`"objectives": objectives_map`**. Empty dict for single-axis backends; populated for MO backends.

#### 2.4.4 **`EventKind.OBJECTIVES_RECORDED`**

When `objectives_map` is non-empty, the session emits **`OBJECTIVES_RECORDED`** with `{candidate_id, objectives}`. This lets external sinks (e.g. Mimarsinan’s `MultiObjectiveSink`) **rebuild Pareto state** without scraping textual tool output.

#### 2.4.5 **New session methods (agent tools)**

Bound as first-class tools via **`build_session_toolset`** (`session/tools.py`):

| Method | Return shape (JSON) |
|--------|---------------------|
| **`query_top_candidates(metric, top_k=5)`** | `{"metric", "ranked": [{candidate_id, value, rank, goal}, ...]}` |
| **`pareto_front()`** | `{"pareto_size", "front": [{candidate_id, objectives, description, rationale}, ...]}` — **enriched** from `self.candidates` so the agent sees **plan text** alongside objectives. |
| **`metric_summary()`** | `{"metrics": {metric: {goal, unit, best, worst, median, count}, ...}}` |
| **`compare_candidates(candidate_ids, metrics=None)`** | Side-by-side `objectives` + `description` + `rationale`; unknown ids get `"error": "unknown candidate"`. |

**Implementation detail**: `_objective_rows()` only includes candidates whose **`objectives` dict is truthy** — baseline timing-only rows are naturally excluded.

**Note**: `session/tools.py` docstring for `build_session_toolset` still refers to an “8-tool” set; the table **`_TOOL_DESCRIPTIONS`** now includes **twelve** named tools (the original eight plus the four MO helpers). This is a **documentation drift** only.

#### 2.4.6 **Leaderboard and rejection reasons**

- **`build_leaderboard`** / **`compare_runs`** still sort by **`median_ms`** where present; each row also carries **`objectives`** (may be empty).
- **`CANDIDATE_REJECTED`**: if compile ok but no signal, reason remains **`"no_timing"`** — for MO backends this corresponds to **“no objectives and no median”** (e.g. `objectives_for_candidate` failed or returned empty). Worth remembering when debugging traces.

### 2.5 Tool descriptions (`session/tools.py`)

User-visible tool strings for **`run_candidate`**, **`run_candidates`**, **`compare_runs`**, **`synthesize_findings`** were updated to mention:

- The optional **`objectives`** dict in run results.
- That **`compare_runs`** rows include **multi-objective** payloads when the backend fills them.

**Non-change**: **`synthesize_findings`** remains **speedup-centric** (aggregates candidates with truthy `speedup`). For pure MO sessions with no speedup, this tool may be less informative; prompts steer the agent toward **`pareto_front` / `metric_summary`** for real conclusions.

---

## 3. Mimarsinan: integration layer

### 3.1 Search space rendering for Compilagent

**File**: `mimarsinan/src/mimarsinan/search/search_space_description.py`

**Method**: `SearchSpaceDescription.to_compilagent_levers(workload_id, backend_id)`

- **Architecture knobs**: still **`EnumChoice`** from NAS schema options — categorical structure is real.
- **Hardware dimensions** (`max_axons`, `max_neurons`, `count` per core type): now **`IntFreeform`** with **fixed wide bounds**:
  - `COMPILAGENT_AXON_BOUNDS = (8, 8192)`
  - `COMPILAGENT_NEURON_BOUNDS = (8, 8192)`
  - `COMPILAGENT_COUNT_BOUNDS = (1, 4096)`
- **Step**: `CORE_DIM_GRANULARITY` (8) for axon/neuron levers; **1** for core count.
- **`DerivationEvidence`**: `rule="mimarsinan.compilagent.open_range"`, `signal="compilagent_<dim>_bounds"` — explicitly documents that this surface is **not** derived from JSON `core_*_bounds` (those remain for NSGA2/AgentEvolve).

**Rationale (ties to observed pathology)**:

- JSON bounds like **[50, 300] cores** combined with **nominal chip capacity** in `param_utilization_pct` produced **tiny utilization percentages** (~1%) — not a broken metric, but a **scale mismatch**: the agent was encouraged to explore huge chips vs modest softcore counts.
- Open ranges + backend validation shift responsibility to **agent + guidance** to **right-size** cores toward the workload.

**`_derive_int_candidates`**: still generates **hint ladders** for documentation / sampling; comments clarify these are **hints**, not enforcement.

### 3.2 `MimarsinanLayoutBackend`

**File**: `mimarsinan/src/mimarsinan/search/optimizers/compilagent/backend.py`

#### 3.2.1 `validate_intervention` (hardware path)

For `target.kind == "hw.core"` with selector `"<idx>.<dim>"`:

- **Selector parsing**: must be two parts; `dim` ∈ `HW_DIM_NAMES`.
- **Numeric constraints**:
  - **`value <= 0`** → error.
  - **`value > _HARD_MAX`** (`65536`) → error (comment in code mentions alignment with “open range” — this is a **typos / runaway** guard; it is **wider** than the advertised lever max in `SearchSpaceDescription`).
- **`max_axons` / `max_neurons`**: must be **`value % 8 == 0`**; otherwise **validation fails** with a message suggesting **`(value // 8) * 8`** or **`((value + 7) // 8) * 8`** (agent-correctable).

**Note**: Earlier summary mentioned “warning” for mod-8; the implemented behavior is **hard validation failure** for non-multiples of 8 on those two dimensions.

#### 3.2.2 `compile`

Unchanged high-level story: `decode_plan` → `validate_detailed` → `_collect_layout_payload`; artifacts **`config.json`**, **`softcores.json`**, **`layout_stats.json`**; **`CompileResult.metadata`** carries a rich payload; **`_candidate_payloads[artifact_dir.name]`** caches:

- `config`, `softcores`, `per_layer`, `layout_stats`, `hw_objectives`, `objective_catalog`.

**`get_candidate_payload(candidate_id)`** exposes this for tools and **`GuidedToolset`**.

#### 3.2.3 `time_workload`

- Calls `problem.evaluate(configuration)` to obtain the **objective dict**.
- Returns **`TimingResult`** with:
  - **`timings_ms=()`**, **`median_ms=None`**, **`p20_ms=None`**, **`p80_ms=None`**
  - **`profile_metrics={"objectives": dict(objectives or {})}`**

**Intent**:

- Invalidate **latency-sorted leaderboard** as a decision driver so the agent **must** use MO tools / objective dicts.
- Avoid fabricating a fake `median_ms` that would **reintroduce an implicit primary**.

#### 3.2.4 `objectives_for_candidate`

Maps Mimarsinan’s objectives to **`compilagent.Objective`** using `ObjectiveSpec` goal directions from **`resolve_active_objectives`** / problem configuration (same semantics as other optimizers).

#### 3.2.5 Docstring drift

Module docstring at top of `backend.py` still mentions **“primary objective on `median_ms`”** in places — the implementation has moved to **MO-first**; treat the docstring as **stale** until updated.

### 3.3 `GuidedToolset` (tool result middleware)

**File**: `mimarsinan/src/mimarsinan/search/optimizers/compilagent/guided_toolset.py`

**Role**: Wrap `compilagent.Toolset` so that:

1. **First `inspect_workload` response** appends a **`[BASELINE FOOTPRINT]`** block built from **`backend.get_candidate_payload("baseline")`** (baseline compile during session bootstrap). Includes:
   - softcore count, layer count, max input/output counts across softcores, total softcore area, baseline **`fragmentation_pct`**, **`mapped_params_pct`** (labeled as param utilization in prose), neural/threshold group counts when present.
   - **Sizing prose** tying **`param_utilization_pct`** to “chip vs model” and recommending shrinking **`count`** / per-core dims toward largest softcore.
   - List of **active objectives** (name + goal).
2. **`run_candidate` / `run_candidates` responses** append **`[GUIDANCE]`** after the JSON text, built from **`MultiObjectiveSink.records()`** (successful, non-rejected, with objectives):
   - Pareto front listing (uses **`compilagent.session.multi_objective.pareto_front`**).
   - **Per-metric leaders** via **`metric_summary`**.
   - **Under-explored axes**: best within **10% of baseline** on that metric, or **all candidates equal** on a metric.
   - **Heuristic suggestions** (e.g. single-digit utilization → aggressive shrink; low accuracy spread → widen arch levers; diversity checks on **`0.count`** / **`0.max_axons`** from sink configurations).

**Harness compatibility (critical bugfix)**:

- Wrapped handlers use **`@functools.wraps(original_handler)`** so **`__wrapped__`** preserves the **original signature** for **pydantic-ai** tool schema generation. Without this, a `**kwargs`-style wrapper breaks the model’s ability to call tools.

**`Toolset` interface completeness**:

- **`by_name`**: caches wrapped `ToolDecl`s (same object for repeated introspection).
- **`tools` property**: materializes wrapped decls so harnesses that iterate **`.tools`** see augmented handlers.
- **`read_only_subset` / `with_extra`**: return new **`GuidedToolset`** instances that **share `_GuidanceState`** where appropriate so flags like **`baseline_injected`** persist.

**Sink mirroring**: **`MultiObjectiveSink.emit_guidance`** fires on each augmentation so the **GUI** can render the same text the agent sees.

**Minor nuance**: `_augment_run_result` labels guidance events with **`target_tool="run_candidates"`** even when the call was **`run_candidate`** — purely for the live event channel grouping.

### 3.4 `MultiObjectiveSink`

**File**: `mimarsinan/src/mimarsinan/search/optimizers/compilagent/sink.py`

- **`emit_guidance(text, target_tool)`** → search event **`type: "compilagent_guidance"`**.
- Taps **`OBJECTIVES_RECORDED`** (and related events) into **`CandidateRecord`** fields used by **`GuidedToolset._objective_rows`** and by final **`SearchResult`** construction.

**Baseline filtering**: compile events with **`candidate_id == "baseline"`** are **skipped** for candidate records so the record book stays **1:1 with agent proposals**.

### 3.5 `CompilagentOptimizer`

**File**: `mimarsinan/src/mimarsinan/search/optimizers/compilagent/compilagent_optimizer.py`

- **Removed** `primary_objective` configuration end-to-end.
- Builds **`GuidedToolset(session.toolset, ...)`** and passes it in **`HarnessRunRequest(toolset=guided_toolset, ...)`**.
- **`_DEFAULT_USER_PROMPT`** and **`_default_system_instructions`** rewritten around:
  - mandatory **inspect → diverse batch → run → read `[GUIDANCE]` → pareto/metric tools → next batch** loop,
  - **chip sizing intuition** (`param_utilization_pct` = used area / nominal capacity),
  - explicit catalog of **MO tools** and **layout introspection** tools.
- **`_build_result`**: pulls **`(model_config, platform_constraints)`** from **`backend.get_candidate_payload`** when available so **`Candidate.configuration`** passes **`problem.validate`** downstream (fixes earlier bug where only human-readable description was stored).

### 3.6 Example experiment JSON

**File**: `mimarsinan/examples/mnist_arch_search_compilagent.json`

- **`primary_objective`** removed.
- **`objectives`** intentionally **omitted** — runs through `default_objectives_for_mode("joint")` (`src/mimarsinan/search/results.py:46`), i.e. the exact same five defaults NSGA2 and AgentEvolve get with no `objectives` field: `estimated_accuracy`, `total_params`, `param_utilization_pct`, `neuron_wastage_pct`, `fragmentation_pct`. Compilagent does **no per-optimizer objective filtering** in code; the active objective set is governed entirely by `resolve_active_objectives(search_mode, user_selection)`, so every optimizer agrees on the surface presented to the agent / GUI by default.
- An explicit `objectives` list still works (any subset of `ALL_OBJECTIVES`, including `total_sync_barriers` for workloads where it actually varies) but is **opt-in per experiment**, not optimizer-specific.

---

## 4. GUI and operator-facing changes

### 4.1 Wizard schema and static UI

- **`mimarsinan/src/mimarsinan/gui/wizard/schema.py`**: removed **`primary_objective`** from compilagent fields.
- **`wizard.html` / `wizard.js`**: removed primary objective `<select>` and serialization / hydration logic; removed **`populateCompilagentPrimaryObjective`**.

### 4.2 Live monitor (Compilagent mode)

**Files**:

- `mimarsinan/src/mimarsinan/gui/static/js/compilagent-live.js`
- `mimarsinan/src/mimarsinan/gui/static/compilagent-live.css`
- `mimarsinan/src/mimarsinan/gui/static/index.html` (stylesheet link)
- `mimarsinan/src/mimarsinan/gui/static/js/step-detail.js` (routes live monitor by optimizer type)

**Features**:

- **Metric leaders strip** — quick view of best-known values per objective.
- **Per-candidate rank chips** — ranks per metric on candidate tiles; updates as objectives stream in.
- **`compilagent_guidance` event** — renders **`[GUIDANCE]` / `[BASELINE FOOTPRINT]`** text in a **collapsible** activity row (`preformatted` path in `_appendActivity`).
- **`step-detail.js`**: when `_liveMonitor === 'compilagent'`, **`syncSearchEventsFromState` / `renderLiveSearchTab`** delegate to **`compilagent-live`** (`initCompilagentLive`, `replayCompilagentEvents`, **`detachCompilagentLive`** on tab tear-down).

---

## 5. Tests added or updated

### 5.1 Compilagent

- **`compilagent/tests/compilagent/test_multi_objective.py`**: unit tests for **`IntFreeform.serialize`**, ranking ties, Pareto boundary cases, domination counts, metric summaries.
- **`compilagent/tests/compilagent/test_session_with_fakes.py`**: extended to cover **objectives in `run_candidate` JSON**, **`OBJECTIVES_RECORDED`**, and the **four new tools** with fake backends.

### 5.2 Mimarsinan

- **`tests/unit/search/test_search_space_description.py`**: expectations for **`IntFreeform`**, wide bounds, step granularity, JSON bounds ignored on compilagent path, collapsed-bound behavior updated.
- **`tests/unit/search/optimizers/compilagent/test_backend.py`**: **`median_ms is None`**, no **`primary_objective`** in profile metrics; **`validate_intervention`** cases (zero, huge, non-multiple-of-8 axons).
- **`tests/unit/search/optimizers/compilagent/test_optimizer.py`**: removed primary objective from factory; failure paths use realistic compile failures.
- **`tests/unit/search/optimizers/compilagent/test_guided_toolset.py`**: baseline block injection once, guidance suffix on runs, pass-through for other tools, **signature preservation** via `wraps`.

---

## 6. Documentation updates (high-level prose in-tree)

- **`mimarsinan/ARCHITECTURE.md`**: Compilagent optimizer subsection (multi-objective, introspection, guidance).
- **`mimarsinan/src/mimarsinan/search/ARCHITECTURE.md`** and **`.../search/optimizers/ARCHITECTURE.md`**: optimizers package listing, **`search_space_description`**, **`compilagent/`** submodule references.

---

## 7. Semantics deep dive: metrics that confused early runs

### 7.1 `param_utilization_pct`

In Mimarsinan this comes from **`LayoutVerificationStats.mapped_params_pct`**: roughly **packed softcore area / nominal chip capacity**. When **`count` × per-core nominal capacity** is enormous relative to the **true softcore area**, utilization **correctly** reads as a few percent.

The fix is **not** to redefine the metric ad hoc for Compilagent, but to **(a)** explain it in prompts and **`[BASELINE FOOTPRINT]`**, and **(b)** let the agent pick **smaller** `count` / tighter **`max_axons`/`max_neurons`** consistent with validation.

### 7.2 `total_sync_barriers`

Derived from **host-side segment topology** + **schedule sync** fallback. It is part of `ALL_OBJECTIVES` but **not** part of `default_objectives_for_mode(...)` for any search mode, so no optimizer enables it implicitly. Users who want it as a search axis opt in by listing it in their experiment's `objectives` array — independent of which optimizer drives the search. For topologies where it is constant across all candidates it will simply be a dead axis on the Pareto front; this is a property of the workload, not of compilagent. Earlier revisions of this doc described it as “removed from the compilagent example for MLP-Mixer” — that rationale was wrong: the flow is **model-agnostic** and the only difference between optimizers' surfaces is whatever the user wrote in `objectives`.

---

## 8. Dependency posture (project policy)

Per integration decision documented in conversation: **`dep_mode: hard_dep`** — Mimarsinan’s Compilagent optimizer path **expects `compilagent` to be installed** for the relevant Python (CI / local env), not a silent fallback. The **3.10 compatibility** work reduces environment friction without changing that policy.

---

## 9. Known limitations and honest edges

1. **`synthesize_findings`** is still speedup-oriented — MO-heavy sessions should rely on **`pareto_front` / `metric_summary` / compare tools** for narrative closure.
2. **`CANDIDATE_REJECTED` + `no_timing`** wording is legacy; for MO backends interpret as **missing both timing and objectives**.
3. **Pareto front** uses **intersection of metric keys** across rows; partially scored candidates may be excluded from dominance comparisons.
4. **`GuidedToolset` suggestions** encode **MNIST / single-core (`0.count`)** heuristics — useful for demos but **not universal** across all Mimarsinan search spaces; future work could derive suggestions from `SearchSpaceDescription` generically.
5. **`MimarsinanLayoutBackend` module docstring** partially outdated vs `time_workload` behavior.

---

## 10. File index (quick navigation)

| Area | Path |
|------|------|
| IntFreeform + exports | `compilagent/src/compilagent/core/search_space.py`, `core/__init__.py`, `compilagent/__init__.py` |
| StrEnum shim | `compilagent/src/compilagent/_compat.py` |
| MO primitives | `compilagent/src/compilagent/session/multi_objective.py` |
| Session + tools | `compilagent/src/compilagent/session/session.py`, `session/tools.py` |
| Open HW levers + evidence | `mimarsinan/.../search/search_space_description.py` |
| Backend adapter | `mimarsinan/.../optimizers/compilagent/backend.py` |
| Guidance middleware | `mimarsinan/.../optimizers/compilagent/guided_toolset.py` |
| Events + records | `mimarsinan/.../optimizers/compilagent/sink.py` |
| Optimizer orchestration | `mimarsinan/.../optimizers/compilagent/compilagent_optimizer.py` |
| Live GUI | `mimarsinan/.../gui/static/js/compilagent-live.js`, `compilagent-live.css`, `step-detail.js` |
| Example config | `mimarsinan/examples/mnist_arch_search_compilagent.json` |

---

*Document generated to capture the integration as implemented in-tree; when behavior and comments disagree, trust the code and treat this doc as the reconciliation layer until sources are aligned.*
