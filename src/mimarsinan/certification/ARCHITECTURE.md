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
| `count_alignment.py` | `PerceptronCountAssembler` (backend stage counts → perceptron channel vectors via IR provenance: `perceptron_index`/`perceptron_output_column`/`perceptron_output_slice`; placement = column × channels + slice), `certify_twin_flow_counts` (the FATAL edge: identity-IR twin ↔ packed program), `certify_flow_counts`/`nf_perceptron_counts` (the model↔grid WQ-residual measurement, atol-governed elsewhere), `flow_perceptron_counts`, `intersect_aligned`. |
| `twin_schedule.py` | `twin_schedule_diagnostic`: per-node (stage, core-latency) table comparison printed on twin-cert FAILURE — triage aid, not a predicate (packed programs can preserve effective schedules while latency tables differ). |
| `record_certificates.py` | `certify_run_records`: typed Edge-B certificate over two backend `RunRecord`s (per-core in/out + segment output counts); used by the SANA-FE and Loihi steps after their first-diff asserts. |

## Dependencies

torch + `spiking.segment_forward` (the NF reference walk). Consumers: the SCM
parity gate and the simulator steps (PR44/45) feed executor-specific count
gatherers; the `spike_count_parity_samples` config knob (registry:
entries_execution) sets the sample budget.
