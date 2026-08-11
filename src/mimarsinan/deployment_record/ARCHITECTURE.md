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
| `build/` | `builder.py`: `DeploymentRecordBuilder` — attach-once fragment slots (double-attach raises), `seal(plan, resolved_step_names)` enforcing the spec §5 required-fragment matrix + cross-checks (pass count vs layout stats, Σ params_programmed vs the weight-programming report total, cores agree, one provenance entry per attached group). `SealPlanView` is the typed duck surface for plan predicates so this module never imports `pipelining`. |

## Dependencies

Near-leaf: may import `mapping`, `chip_simulation`, `certification`,
`config_schema`, `common` — NEVER `pipelining`/`search`/`gui` (stage 1 imports
none of them; the mirror contracts to `mapping` types are enforced by tests).
Consumers (later stages): the pipeline's terminal Deployment Record step,
`search` candidate views, and the GUI introspection surfaces.
