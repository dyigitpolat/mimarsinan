# Thesis-Support Source Features (TS-series) — campaign-grade search + sealed ledgers

## Status (2026-08-18)

| Stage | State |
|---|---|
| TS1 budget accountant + resource ledger core | pending |
| TS2 sampling optimizers (exhaustive / random / sobol) + grid enumeration | pending |
| TS3 LLM-driver threading (usage capture + budget) | pending |
| TS4 packer verdict trichotomy | pending |
| TS5 adaptation ledger + stall trace (artifact + record fragment) | pending |
| TS6 derived programming constants (all five profiles) | pending |

Deliverables: correctness, performance, **elegance** — the standing rule.
Scope guard: this series adds FRAMEWORK features only. Campaign runners,
metric batteries, calibration/oracle/defect/baseline studies live in
`mimarsinan_research` and consume only sealed artifacts; nothing in this
plan builds scaffolding into `src/`.

## Owner decisions (asked 2026-08-18)

| Question | Decision |
|---|---|
| Equal-budget stop semantics | **Boundary stop + exact ledger** — drivers stop at their natural boundary once the distinct-evaluation budget is spent; the sealed ledger records the exact spend; analysis normalizes. |
| Monetary cost placement | **Research-side pricing** — the framework seals raw facts (model name, calls, tokens in/out, wall); dollars are computed in `mimarsinan_research` from a price table. |
| Adaptation ledger home | **Step artifact AND an additive-optional DeploymentRecord fragment** (totals summary; the `invocations` schema precedent). |
| Derived programming constants | **All five profiles** (loihi, truenorth, odin, isaac_like, generic_estimated_22nm), NVM-class with wide literature bands + the endurance-unmodeled note. |
| Algorithm 7.1 semantics | **Code reality is authoritative** (owner directive): no `tuning_policy` change; the ledger reports per-increment spending as consumed, and the thesis algorithm is rewritten FROM the ledger. |

## Recon facts the design builds on (verified 2026-08-18)

- Optimizer contract: `SearchOptimizer.optimize(problem, reporter) ->
  SearchResult` (`search/optimizers/base.py`) — one method, clean plug-point.
- Factory: `create_optimizer` in
  `pipelining/pipeline_steps/config/architecture_search_helpers.py` with
  `OptimizerType = Literal["nsga2", "agent_evolve", "compilagent"]` and an
  if/elif ladder; `search_result_to_jsonable` lives beside it.
- THE distinct-evaluation identity already exists: `evaluate()`'s cache key
  (`json_key(resolved_configuration)`, `search/problems/joint/evaluate.py`)
  — a cache miss IS a distinct decoded evaluation; a hit IS a duplicate.
- Encoding (`search/problems/joint/encoding.py`, 194 LOC): core dims snap to
  `CORE_DIM_GRANULARITY` multiples, counts are ints, arch/option enum axes
  are index-coded — the hardware grid is finite; float option axes are the
  only unbounded case. `seed_vectors()` (R6) is the precedent for the
  encoding owning its own vector-production contracts.
- LLM call seam: `search/optimizers/llm/trace.py` creates a pydantic-ai
  Agent per call (usage available via the result's usage surface);
  `agent_evolve/llm_trace.py` is a SECOND trace module — consolidation
  candidate.
- Adaptation loop: `tuning/orchestration/adaptation_driver.py`
  (`AdaptationDriver.run` / `run_cycle`) with `acceptance_sensor.py`,
  `experimental_walk_recovery.py`, `endpoint_steps.py` as the
  verdict/recovery/endpoint seams.
- `LayoutPackingResult` (`mapping/layout/layout_types.py`) carries
  `feasible: bool` + `error: str` — room for an additive verdict field.
- No usage/token accounting exists anywhere in the search tree today.

---

## TS1 — budget accountant + resource ledger core

**Written from the shipped code** (the recon premise above — "the evaluate
cache IS the one seam" — did not survive contact: a problem answers about a
candidate through several channels, and the ones that are not `evaluate` do the
same work). `search/ARCHITECTURE.md` ("What a search spent") and the
`optimizers/budget.py` module doc are the authority; this section is the
summary TS2/TS3 read before extending it.

**The law.** The currency is the candidate IDENTITY — `json_key` of the
resolved configuration. A DISTINCT evaluation is the evaluator work that
identity costs the first time the run spends it; every later ask about it is a
DUPLICATE call that bought no new candidate. `charge_evaluation(budget, key,
hit=..., channel=...)` ROUTES, so no caller has to know the law: already spent
→ duplicate, whatever answered it; unspent and work about to run → the distinct
evaluation the run pays for; unspent and cache-answered → nothing at all (a
candidate a cheap predicate refused before anything was built is not an
evaluation, and neither are the re-asks a cache answers about it).

**Charged at three sites, across four channels** — because validation work IS
evaluator work: `validate_detailed` walks the same model build → conversion →
packing `_resolve_entry` that an evaluation walks, and NSGA-II asks it FIRST
through `constraint_violation`. Charging only the evaluate channel was measured
to seal a ledger claiming ZERO spend for a search whose every offspring was
screened out (the R6 note's 72/72 ViT case), with the budget bounding nothing.

| Site | When | Channel |
|---|---|---|
| `joint/validate.py` (past the caches and the declaration-only check) | a full resolution is about to run | the caller's |
| `joint/validate.py::_cached_verdict` | a recorded verdict answers the ask | the caller's |
| `joint/evaluate.py` (objective-cache HIT) | the one ask that never reaches `validate_detailed` | `evaluate` |
| `joint/validate.py::candidate_layout` | always — the introspection seam has no cache | `layout` |

`validate_detailed(configuration, *, channel=VALIDATE_CHANNEL)` takes its
channel from the caller (`constraint_violation` → `constraint`, `evaluate` →
`evaluate`, a backend calling it directly → `validate`): an ask belongs to the
seam a driver called, not to the helper that resolved it. Penalty (infeasible)
candidates ARE charged wherever they resolved — they consumed real evaluator
work. `constraint_fn` / `validate_fn` refusals are NOT: they read the
declaration and build nothing.

**The comparable axis is identities, not calls.** A driver asks through as many
channels as it likes, so a rate counted in calls reads the plumbing: NSGA-II
(screen, then score) sealed 0.5556 on a run that re-proposed 2 of 18 candidates
(0.111), a floor no single-channel driver can reach. The accountant therefore
counts ROUNDS of asking, per (channel, identity): a channel's first look joins
the round that candidate's proposal opened; a channel asking AGAIN about a
candidate it already asked about opens the next round. `duplicate_rate =
identities_reasked / identities_asked`, and both sides are sealed so an
analysis can re-derive or pool it. The rule is deliberately not "the previous
ask was about another candidate" — a driver that screens a whole batch and only
then scores it re-proposes nothing and must read 0.0.

**What shipped**

1. `search/optimizers/budget.py`: `EvaluationBudget(limit)` —
   `charge(channel, key, hit=...)`, `has_spent`, `distinct_spent`,
   `raw_calls`, `identities_asked`, `identities_reasked`, `duplicate_rate`,
   `exhausted` (distinct_spent >= limit); `problem_budget(problem)` (the one
   reader, fails loud on a wrongly typed attribute); `LlmUsage(model, calls,
   tokens_in, tokens_out)`; `ResourceLedger(wall_s, evaluations_raw,
   evaluations_distinct, identities_asked, identities_reasked, duplicate_rate,
   budget_limit, stopped_at_boundary, llm)` — frozen, `to_dict`/`from_dict`,
   facts only (no dollars; owner decision); `seal_ledger(...)`.
2. Problem injection: `JointArchHwProblem.evaluation_budget:
   Optional[EvaluationBudget]`, charged at the sites above. Absent budget =
   today's behavior byte-identical, artifact included.
3. `SearchResult.ledger: Optional[ResourceLedger] = None` (additive; old
   artifacts load) + serialization in `search_result_to_jsonable`, which omits
   the key entirely for an unmetered run.
4. NSGA-II threading: `GenCallback` checks `budget.exhausted` per generation
   and forces pymoo termination at the boundary (`terminate()` then
   `update(algorithm)` — pymoo updates the criterion BEFORE calling back);
   wall measured around `minimize`, ledger sealed there, so the post-search
   front re-read stays out of it. `stopped_at_boundary` means the budget CUT
   THE RUN SHORT. Campaign protocol note: choosing B as a pop-size multiple
   makes the boundary exact for NSGA by construction.
5. Declaration surface: `arch_search.evaluation_budget` (a positive count,
   refused by name otherwise) → `resolve_evaluation_budget`; it is a wizard
   `common_fields` key with no default, because an unset budget is an
   unmetered run.

Tests (`tests/unit/search/test_evaluation_budget.py`): duplicates never
charged; a re-ask about a candidate nobody built stays free through BOTH the
constraint and the evaluate channel; exhaustion at exactly B distinct; penalty
evals charged; every channel charged; the introspection seam spends what it
resolves; NSGA stops at the first boundary with `distinct_spent` sealed
exactly; the sealed rate equals the driver's re-proposal rate on the real
hardware-mode problem; a single-channel driver seals 0.0; ledger JSON
round-trip; no-budget byte-identity A/B.
Mutants killed: charge-on-hit (both channels); rate counted in calls; a
channel's re-ask never reopening the round; exhausted-on-raw-not-distinct;
callback ignores exhaustion.

## TS2 — sampling optimizers + grid enumeration

**One driver, three strategies** (the materializer pattern):

1. `search/optimizers/sampling_optimizer.py` (~230 LOC, new):
   - `SamplingStrategy` protocol: `vectors(problem, rng) ->
     Iterator[np.ndarray]`; `RandomStrategy` (uniform in [xl, xu]),
     `SobolStrategy` (`scipy.stats.qmc.Sobol`, scrambled, seeded),
     `GridStrategy` (consumes the encoding's enumeration).
   - `SamplingOptimizer(SearchOptimizer)`: stream vectors → decode →
     evaluate (the TS1 accountant does the counting; the cache absorbs
     duplicate vectors without spending budget) → stop on budget or
     stream end → nondominated front + minimax best via the existing
     `results.py` orderings (add a small pure `nondominated_front()`
     helper there if none exists — tested) → `SearchResult` with ledger.
     Emits the standard generation frames per batch so the live panel and
     event stream look identical across all six optimizers.
2. `encoding.py` gains `grid_vectors()` (~55 LOC; file lands ≈250):
   per-dimension value sets (index ranges for enums; snapped dim values
   within bounds; integer count ranges) → cartesian product. Fail-loud
   rules: any FLOAT option axis ⇒ refuse naming the axis ("exhaustive
   enumeration needs a discrete axis"); product > the caller's declared
   cap ⇒ refuse naming both numbers. Never silent truncation.
3. Factory elegance: replace the if/elif ladder with a declarative
   `OPTIMIZER_BUILDERS` table (name → builder function); `OptimizerType`
   grows to six names; unknown names keep failing loud. The wizard's
   optimizer choices read the same table (single source).

Tests: grid cardinality == closed-form product on a tiny space (mutant:
drop a dimension); float-axis refusal by name; cap refusal with both
numbers; Sobol seed determinism; bounds respected; driver front equals the
exhaustive front on a hand-computable toy problem; events emitted;
duplicate vectors spend no distinct budget.

## TS3 — LLM-driver threading (usage + budget)

1. Usage capture at the pydantic-ai seam: one traced-call helper that
   takes an `LlmUsage` accumulator and records model, call count, token
   counts per request — including retry/regen rounds. Boy-scout: if
   `agent_evolve/llm_trace.py` duplicates the calling path in
   `llm/trace.py`, consolidate both onto the one helper (two trace
   modules → one call path, per-driver formatting stays); if the
   duplication is superficial, instrument both and note why.
2. Budget threading: `agent_evolve` polls `budget.exhausted` per batch
   (its natural boundary), `compilagent` per proposal/tool-loop
   iteration; both fill `ledger.llm` and seal it through the same
   `SearchResult` path as TS1.

Tests: stubbed agent (injected fake) — usage accumulates across calls and
retries (mutant: retry path skips accumulation); batch loop stops at the
boundary after exhaustion; ledger sealed in the result JSON with model
name and both token directions.

## TS4 — packer verdict trichotomy

1. `mapping/packing/infeasibility_proofs.py` (~90 LOC, new): PURE, SOUND
   provers only, each with a one-line soundness argument —
   `no_core_type_fits(spec)` under the declared permissions (promoted
   from the verifier's existing largest-softcore check, one home), and
   `cells_exceed_total_capacity` (Σ softcore cells > Σ declared capacity
   — sound because no permission shrinks committed cells). Anything not
   provably sound stays out.
2. `LayoutPackingResult.verdict: str = "feasible"` (additive):
   `"feasible" | "proven_infeasible" | "heuristic_failed"`. `pack_layout`
   consults the provers on failure to upgrade the default
   `heuristic_failed`; the refusal message is prefixed with the verdict
   class, so search failure censuses (the ViT-style lines) become
   self-classifying. `feasible: bool` semantics untouched — downstream
   behavior byte-identical (pinned).
3. The research-side CP-SAT oracle study later VALIDATES the trichotomy;
   nothing oracle-shaped enters `src/`.

Tests: prover positive/negative pairs; verdict upgrade on a Σcells
overflow vehicle; `heuristic_failed` on the guards' UNPACKABLE_CORES
vehicle (packable in principle, greedy-refused — the exact case the
review names); existing-behavior A/B. Mutant: prover call removed ⇒
verdict pin red.

## TS5 — adaptation ledger + stall trace

Reports code reality (owner directive): the ledger seals what the
controller actually did and spent — per-increment budgets as consumed —
and the thesis's Algorithm 7.1 is rewritten from it.

1. `tuning/orchestration/adaptation_ledger.py` (~150 LOC, new): frozen
   event types — `Proposal(increment, params_digest)`,
   `Verdict(accepted, probe_metric, reading, rolled_back)`,
   `Refinement(kind, detail)`, `Escalation(path)` (taxonomy read off the
   driver's real branches at implementation: family switch / prefix /
   deploy-without-transform / controlled abort), `Recovery(steps)`,
   `Endpoint(steps)` — plus the `AdaptationLedger` accumulator with
   totals (proposed, accepted, rejected, retries, recovery_steps,
   probe_evals, endpoint_steps, total_steps) and `to_dict`.
2. Hooks: `AdaptationDriver.run_cycle` (proposals/attempts),
   `acceptance_sensor` (verdicts + probe readings),
   `experimental_walk_recovery` (recovery + escalations),
   `endpoint_steps` (endpoint consumption). Emission as a step artifact
   (`<Step>.adaptation_ledger.json`, the add_entry pattern).
3. Record fragment (owner decision): additive-optional
   `DeploymentRecord.adaptation: Optional[AdaptationSummaryRecord] = None`
   — totals + `stalls_by_path` + `completed_via` only (events stay in the
   artifact). Last position, no format-version bump, old JSON loads;
   mirror pins updated per the additive discipline. Not a priced
   quantity; no vocabulary change.

Tests: scripted scheduler stub driving accept/reject/refine/escalate
sequences — every branch class emits exactly one event (one mutant per
event class: dropped emission ⇒ red); Σ events == totals; record fragment
round-trip + legacy-JSON load; the artifact answers "which path completed
the run" mechanically (the Fig-7.2 question).

## TS6 — derived programming constants, all five profiles

Data + validation only; the mechanism is C6's (authored constants with
derivation strings, evidence kinds, bands):

| Constant | Profiles | Derivation rule (written into each constant) |
|---|---|---|
| `e_program_per_byte` | loihi, truenorth, odin (SRAM class) | 0.5–3× the profile's own read/access anchor per byte (loihi: `e_mac`); band IS the ratio spread |
| `e_program_per_byte` | isaac_like (NVM) | literature program-verify band, 10–100× SRAM class; note states endurance is UNMODELED |
| `e_program_per_byte` | generic_estimated_22nm | estimated exemplar value, evidence_kind `estimated` |
| `t_program_per_byte` | loihi, odin, isaac_like | donor-scaled from the two declaring profiles (truenorth, generic) via `validity.technology_node_nm`; band spans donors |
| `e_core_program` | all five | 0.5–2× `e_core_init` where declared (loihi anchor measured-derived); derivation names the anchor and why |

Companion `.md` derivation notes per profile. Validation pins:
vocabulary conformance (units/multiplicands); **non-perturbation** — the
silicon-correlation reference cases never reprogram, so their predictions
must be byte-identical (golden of predicted values asserted before/after);
one term-presence pin — a scheduled vehicle's energy/latency now carries
programming terms under loihi (previously absent-by-name).

---

## Sequencing and budgets

TS1 → TS2 → TS3 serial (they share the accountant/ledger core).
TS4, TS5, TS6 independent after TS1; any order; TS6 is data-first and can
land first if the campaign calendar wants loihi pricing early.

Estimated deltas: TS1 ~200 new + ~60 modified; TS2 ~300 new + ~80;
TS3 ~80–150 modified; TS4 ~120 new + ~40; TS5 ~250 new + record schema;
TS6 data + ~60 test. All files within the 300-LOC / 10-sibling budgets
(encoding.py lands ≈250; new siblings counted per directory).

## Verification protocol (every stage)

Tests first; `python -m pytest tests` ≤2 min green; `./scripts/typecheck.sh`
zero; ratchets/budgets only tighten; load-bearing guards mutation-checked
(cp-backup/mutate/expect-red/restore — the named mutants above);
byte-identity A/B wherever a default-absent feature claims it
(TS1 no-budget, TS4 feasible-bool, TS5 no-fragment); ARCHITECTURE.md per
touched module (`search/`, `mapping/`, `tuning/`, `deployment_record/`);
generic-only (no workload constants; no price tables — dollars live
research-side); per-stage commits, no AI-attribution trailers.
