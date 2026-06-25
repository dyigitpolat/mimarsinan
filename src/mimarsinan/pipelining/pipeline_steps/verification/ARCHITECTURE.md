# pipeline_steps/verification/ — Simulation Verification

| File | Step class |
|------|------------|
| `simulation_step.py` | `SimulationStep` |
| `loihi_simulation_step.py` | `LoihiSimulationStep` |
| `sanafe_simulation_step.py` | `SanafeSimulationStep` |

SANA-FE step uses `chip_simulation.sanafe` and optional HCM parity via `simulation_factory`.

**cost-emit:** after the SANA-FE run completes and its report is published, `SanafeSimulationStep._emit_cost_record` mines the in-memory `SanafeStepReport.to_snapshot_dict()` + the run's (firing × sync) `CertificationCell` + the deployed target metric into a measured `CostRecord` (`chip_simulation.cost_extraction.extract_cost_record`) and writes `cost_record.json` next to the run artifacts (`save_cost_record`). This is a pure ADDITIVE side-effect and is fully exception-isolated — any cost-extraction/write failure is logged and swallowed, so it NEVER crashes the deployment nor alters the step's result (accuracy / cache / return). It is the first standing site that emits a MEASURED cost record per deployment (closing the #1 cost gap E5 previously filled with a proxy).
