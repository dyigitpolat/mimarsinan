# certification

Deployment-faithfulness certificates. The certification observable is the
per-neuron, per-window SPIKE COUNT (an integer): post-WQ chip arithmetic is
integer weights x integer counts, so count equality across backends is exact
by construction under the integer-exact accumulation contract, and
accuracy(oracle) + counts(oracle == backend) derives accuracy(backend)
[spiking_deployment_calculus §17].

## Key files

| File | Role |
| --- | --- |
| `spike_certificate.py` | `certify_spike_counts` + `SpikeCountCertificate`: typed reference↔backend count comparison; per-backend exactness classes (`exact` / `counts-export`), fail-loud on unclassified backends. |
| `value_certificate.py` | `certify_twin_flow_values` + `ValueWindowCertificate` [mvm]: node-granular fp64 value-window twin edge (identity ↔ packed value programs) with the measured `VALUE_TWIN_FP64_ATOL`; no vacuous pass. |
| `count_alignment.py` | `PerceptronCountAssembler` (backend stage counts → perceptron channel vectors via IR provenance: `perceptron_index`/`perceptron_output_column`/`perceptron_output_slice`; placement = column × channels + slice), `certify_twin_flow_counts` (the FATAL edge: identity-IR twin ↔ packed program; returns `(cert, report, backend_node_counts)` so gate callers can reduce the backend's `{node_id: (B, n)}` counts — boundary traffic — without a second simulation), `certify_flow_counts`/`nf_perceptron_counts` (the model↔grid WQ-residual measurement, atol-governed elsewhere), `flow_perceptron_counts`, `intersect_aligned`. |
| `twin_schedule.py` | `twin_schedule_diagnostic`: per-node (stage, core-latency) table comparison printed on twin-cert FAILURE — triage aid, not a predicate (packed programs can preserve effective schedules while latency tables differ). |
| `record_certificates.py` | `certify_run_records`: typed Edge-B certificate over two backend `RunRecord`s (per-core in/out + segment output counts); used by the SANA-FE and Loihi steps after their first-diff asserts. |
| `randomized/` | X3a randomized deployments (plan Wave 1 section 10, theorem box): `instance` (canonical JSON integer networks), `generator` (seeded, declared space, named motifs), `cost_draws` (the registry's cost-only set partitioned; program-changing fields drawn one by one), `program` (mapper graph, integer IR at a declared geometry, IR relabelling), `executors` (HCM, TTFS contract, value and nevresim runs captured per neuron and window; the midpoint input rate, so the identity encoder emits each declared count, and `encoder_counts` to observe it), `network` (the integer network N of an instance document, and what Proposition 2 claims of it per semantics), `arms` (the hypothesis-on strata, each on its own generator stream, among them a long-window stratum beyond the declared window range, one hypothesis-off arm per exercisable hypothesis, one field each, and the base-violation arms, among them R4's phase-dithered encode declared on both programs), `runner` (identity reference against one drawn deployment, and the identity program against N), `attribution` (root cause by ablation: a position-free rotation under a dithered encode, dependency-ordered walks under timing-free executors and integer counts under the synchronized LIF count executor, alone and together, then relabelling, geometry, packing, cut), `screen` (Clopper-Pearson bounds, teeth, the toolchain-as-shipped, harness-setting and long-window lines, the network relation, coverage with weights, input dependence and the fixed axes, and the denominators over every row of the hypothesis table it is given). Driven by `scripts/random_deployments.py`; the hypothesis table is `tests/fixtures/theorem_box/hypothesis_table.json`. |

## Dependencies

torch + `spiking.segment_forward` (the NF reference walk). Consumers: the SCM
parity gate and the simulator steps (PR44/45) feed executor-specific count
gatherers; the `spike_count_parity_samples` config knob (registry:
entries_execution) sets the sample budget.

`randomized/` reads the mapping stack (mappers, IR mapping, packing through
`pipelining.core.simulation_factory.build_hybrid_mapping_for_pipeline`), the
executors (`models.spiking.hybrid`, `chip_simulation.value_run`,
`chip_simulation.ttfs`, `chip_simulation.simulation_runner`) and the registry's
decision classes (`config_schema.registry`); nothing in the toolchain imports
it. Its hypothesis-off switches patch one toolchain guard for the duration of
a run (`arms.Arm.engaged`) and never outside it.
