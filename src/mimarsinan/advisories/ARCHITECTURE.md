# advisories — Programmatic deployment advisories

Formalizes the measured deployment failure modes of the theory memos
(`docs/research/findings/`) as GENERIC, programmatic rules: each rule detects
a mapping pattern and/or its interaction with the deployment configuration —
never a vehicle name — and fires an `Advisory` warning. Advisories are
**tentative theory** (each carries its memo citation and a `tentative` flag)
and NEVER fail a run; fail-loud does not apply to warnings.

The rule set doubles as the checklist of the **lossless-refinement mandate**:
for lif / synchronized-ttfs deployments, a fired rule that predicts a
deployment accuracy loss sets `mandate_violation=True` — the machine-readable
entry the refinement tooling consumes (via reporter events of kind
`deployment_advisory` or `Advisory.as_payload()`).

## Key files

| File | Role |
|---|---|
| `advisory.py` | The frozen `Advisory` record (id, severity ∈ {UNSUPPORTED, RISK, INFO}, title, one-paragraph detail with mechanism + measured basis + memo citation, `tentative`, `mandate_violation`, `suggested_levers`), `as_payload()`, and `lossless_mandate_applies` (lif/sync scope predicate). Separate from `engine.py` so rule modules can import the record while the engine imports the rules (no cycle). |
| `engine.py` | Evaluation entry points: `evaluate_config_advisories(plan_or_config)` and `evaluate_graph_advisories(model_repr, plan_or_config, channel_stats=...)` (each accepts a `DeploymentPlan` or a resolved config dict, resolving the plan once), `evaluate_post_pretrain_advisories(pretrain_acc, config, acceptance_target=...)`; `ALL_ADVISORY_IDS` pins the stable id set. |
| `rules_config.py` | Pure functions of the resolved `DeploymentPlan`: ADV-CASC-UNSUPPORTED (cascaded ttfs, casc memo), ADV-NOVENA-CHARGE (L1 Theorem 0), ADV-STRICT-LT-LATTICE (L1 V9, INFO), and the post-pretrain ADV-ENVELOPE-GATE (M4 §1). Decision axes come from the plan SSOT (the no-stray-flag-reads invariant); mode checks go through `chip_simulation/spiking_semantics.py` predicates, never string ladders. |
| `rules_graph.py` | Topology rules of the mapper `ModelRepresentation` + the `GRAPH_RULES` roster: ADV-STAIRCASE-DEPTH (L from `per_perceptron_cascade_depth` + S; sync composition law × lif 1/S law), ADV-FANIN-DEPTH-IMBALANCE (L1 V6; mapper-graph twin of `mapping/latency/depth_balancing.py`). Thresholds are module constants with one-line justifications. |
| `rules_graph_scale.py` | Scale/grid rules: ADV-SCALE-SPREAD and ADV-NORMFREE-CHAIN (M4; runtime q99 stats when provided, else weight-norm proxies stated in the detail), ADV-BIAS-GRID-DOMINANCE (M2; the same `PerceptronTransformer` effective view the WQ quantizes, gated on `resolve_wq_two_scale_projection`). |
| `graph_common.py` | Shared mapper-graph helpers (`exec_and_deps` over the public `execution_order`/`consumer_map` API, perceptron naming). |
| `surfacing.py` | `surface_advisories(reporter, advisories, context=...)`: loud `[ADVISORY][SEV]` prints + one `deployment_advisory` reporter event per advisory (`common/reporter.emit_reporter_event`). |

## Surfacing seams (owned by the callers)

- Headless / CLI: `pipelining/session.py` — config advisories at `run()` start,
  the envelope gate via a post-step hook on Pretraining; both under
  `best_effort` (warnings must never kill a run).
- Graph: `pipelining/pipeline_steps/config/torch_mapping_step.py` — after
  conversion, with channel q99 stats from
  `tuning/orchestration/install_resolution/capture.py`.
- GUI: `gui/wizard/schema_api.py` `resolve_payload()["advisories"]`, rendered
  in the wizard live rail (`gui/static/js/wizard/review.js`).

## Dependencies

Reads `chip_simulation/spiking_semantics.py` (mode predicates),
`pipelining/core/deployment_plan.py` (the decision-axis resolver the rules
consume), `spiking/gain_correction.py` + `spiking/segment_forward.py` (graph
depth walks), `transformations/perceptron/perceptron_transformer.py`
(effective parameters), `pipelining/core/platform_constraints_resolver.py`
(two-scale / bias-mode SSOT), `common/reporter.py`. Dependents: `pipelining`
(session, torch-mapping step) and `gui` (wizard resolve payload).

## Mandate semantics

`mandate_violation` is set exactly for rules predicting a DEPLOYMENT loss on a
lif/sync run (staircase depth, scale spread, norm-free chain, bias-grid
dominance, fan-in imbalance, Novena). ADV-CASC-UNSUPPORTED fires only outside
the mandate's scope; ADV-STRICT-LT-LATTICE is an exact-arithmetic hazard with
no measured loss on float lattices; ADV-ENVELOPE-GATE is a pretrain-side
deficit lossless refinement cannot eliminate — all three stay `False`.
