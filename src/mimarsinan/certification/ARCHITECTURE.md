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

## Dependencies

torch only. Consumers: the SCM parity gate and the simulator steps (PR44/45)
feed executor-specific count gatherers; the `spike_count_parity_samples`
config knob (registry: entries_execution) sets the sample budget.
