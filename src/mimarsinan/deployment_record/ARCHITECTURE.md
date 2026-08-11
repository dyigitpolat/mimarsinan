# deployment_record

The typed, versioned, provenance-carrying deployment artifact
(`deployment_record.json`) specified in `docs/deployment_record_schema.md`:
**a deployment is a schedule** — ordered passes over the alternating
NeuralOps/ComputeOps structure — and the record is the join of the run's
formerly-disconnected measured surfaces (schedule/placement census, layout and
crossbar utilization stats, SANA-FE physics, accuracy reads + certificates).
It is a join and a registry, not new measurement.

Discipline: all schema types are frozen dataclasses with JSON-safe
`to_dict`/`from_dict`; `from_dict` rejects unknown fields and wrong
`format_version` (the `CostRecord.from_dict` discipline — evolution is a
version bump + explicit migration, never silent tolerance). Every modeled
value carries a `(low, nominal, high)` band with a written basis; no proxy is
ever presented as a measurement.

## Key files

| File | Role |
| --- | --- |
| `schema/` | The fragment schemas: `provenance.py` (`Provenance`, `Band`, `ModeledValue`), `schedule.py` (`ScheduleRecord` + segment/compute-op stages), `placement.py` (softcores, banks, floorplan, tiles), `utilization.py` (typed mirrors of `CrossbarUtilizationReport.to_dict()` and `LayoutVerificationStats`, drift-guarded by mirror-fidelity tests), `traffic.py` (gate-reduced boundary totals + SANA-FE NoC), `physics.py` (timing + energy, banded modeled terms), `accuracy.py` (reads, certificates, adaptation walls), `record.py` (`RecordIdentity`, `DeploymentRecord`, `DEPLOYMENT_RECORD_FORMAT_VERSION`, atomic save/load), `serde.py` (strict unknown-field-rejecting serde helpers). |
| `build/` | `builder.py`: `DeploymentRecordBuilder` — attach-once fragment slots (double-attach raises), `seal(plan, resolved_step_names)` enforcing the spec §5 required-fragment matrix + cross-checks (pass count vs layout stats, Σ params_programmed vs the weight-programming report total, cores agree, one provenance entry per attached group). `SealPlanView` is the typed duck surface for plan predicates so this module never imports `pipelining`. `payload_sizes.py`: the sizing SSOT — `params_bytes = ceil(cells_used × weight_bits / 8)` (undeclared `weight_bits` FAILS LOUD, the documented None choice) and `core_connectivity_entries` (reads `HardCore.get_axon_source_spans()`, the same span SSOT behind the codegen export). `from_mapping.py`: pure-read converters from a live `HybridHardCoreMapping` — `schedule_record_from_mapping` (stages mirrored 1:1; per-segment programming via `weight_programming_report` on a single-stage view; the pass census formula is exactly `LayoutPlan.from_hybrid_mapping`'s so the seal cross-check holds by construction), `placement_record_from_mapping` (every placement dict key + banks with sharing degree; floorplan/tiles empty until SANA-FE joins), `utilization_record_from_mapping` (typed mirrors of the step's `CrossbarUtilizationReport` and `stats_dict_from_hybrid_mapping`, plus the threaded relay count). `from_certificates.py`: `boundary_traffic_from_node_counts` — the eager `{node_id: (B, n)}` → totals/maxima reduction the spike-count gate calls in-scope (raw tensors never persist). `from_simulators.py` (W4.4): pure reads of the `SanafeStepReport.to_snapshot_dict()` shape — energy aggregate + per-plane breakdown terms, per-segment measured timing (sample 0, `sim_time_s` includes NoC hops), NoC totals + link loads (packets summed over samples; static censuses over sample 0 only), the resolved floorplan (mesh dims already fold `floorplan_replicas`) and tile census; the scalar formulas are IMPORTED from `chip_simulation.cost_extraction` so §6 continuity is structural, never re-derived. |
| `cost/` | `legacy_projection.py` (W4.4): `cost_record_from_deployment_record` — THE §6 continuity contract: projects a sealed record onto a format-v3 `CostRecord` whose live fields are value-identical to `extract_cost_record` on the same SANA-FE snapshot (golden-fixture-pinned), with the formerly-dead fields (`reprogram_passes`, `reuse_passes`, `params_reloaded`, `max_ft_pass_wall_s`, `ft_pass_walls`) now live from the schedule/adaptation fragments; `activation_bytes_moved` stays 0 (no producer exists — stated disposition). Refuses records without the SANA-FE fragments: exactly the runs that never wrote `cost_record.json`. |

## Dependencies

Near-leaf: may import `mapping`, `chip_simulation`, `certification`,
`config_schema`, `common` — NEVER `pipelining`/`search`/`gui` (the mirror
contracts to `mapping` types are enforced by tests).
Consumers (stage 2): the mapping pipeline steps and the spike-count gate emit
fragments through the pipeline cache (`deployment_record_scm` /
`deployment_record_hcm`) using the `build/` converters. Stage 3: the nevresim
Simulation step emits `deployment_record_nevresim` (probe read in
`AccuracyReadRecord` shape, the driver's measured total-output-spikes figure,
host-op walls from `chip_simulation.hybrid_run.stage_timing.StageTimer`).
Stage 4: the terminal `DeploymentRecordStep`
(`pipelining/pipeline_steps/verification/deployment_record_step.py`) re-types
the cached fragments, joins the SANA-FE report via `build/from_simulators.py`,
seals against the plan (`SealPlanView` adapter in the step's assembly module),
writes `deployment_record.json` atomically, and emits the legacy
`cost_record.json` via `cost/legacy_projection.py` IFF the SANA-FE fragment is
present. Later stages: `search` candidate views and the GUI introspection
surfaces.
