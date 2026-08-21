# spiking/ — Spike-train encoding, the unified segment-aware NF forward, and deployed-cascade calibration

This module is the torch-side single source of truth for how spiking models
execute and how values cross host/chip boundaries. `SegmentForwardDriver`
partitions a model's mapper exec graph into host value nodes and neural spike
segments and runs one decode→compute→re-encode walk for every mode; per-mode
spike dynamics (cycle-accurate LIF, single-spike TTFS, pointwise analytical)
live in pluggable segment policies. The same boundary encode/decode helpers
are consumed by `SpikingHybridCoreFlow` and the SANA-FE/Lava/Nevresim runners,
so NF↔SCM parity holds by construction; the calibration helpers (scale-aware
boundaries, DFQ bias matching, gain correction, theta co-training) close the
remaining ANN→SNN conversion gap on the deployed cascade.

## Key files
| File | Purpose |
|---|---|
| `spike_trains.py` | Spike-train constructors: uniform, cycle-accurate signed-IF (`lif_spike_train`), materialized, and the legacy rate fallback. |
| `boundary_config.py` | `BoundaryConfig` dataclass — runtime knobs for boundary encode/decode (T, spiking mode, cycle accuracy, dtype, negative shift). |
| `segment_boundary.py` | SSOT boundary encode/decode: `encode_segment_input` (cached trains take precedence; missing non-raw slices are a hard error), `decode_segment_output(_torch)` (counts / T), the mode-agnostic wire transcode (`normalize_boundary_value`, TTFS alias kept) and its per-slice rate/LIF seam twins (`boundary_normalization_scales` — the derived view `kappa_fold / kappa_buf` of the stamped gauge tables, with a legacy wrapper-walk fallback for pre-stamp pickles — plus `normalize_boundary_slices_torch/_numpy`, applied by the hybrid twin and every deployed runner). |
| `segment_input_encoding.py` | `encode_segment_input` itself — the `(T, B, in_size)` segment input assembly: cached producer trains take precedence (except for retimed level stages, whose input is the COUNT re-encode by definition), missing non-raw slices are a hard error, raw-input gaps are uniform-encoded. Re-exported by `segment_boundary.py`. |
| `compute_boundary.py` | `normalize_boundary_value` (the wire transcode itself) and `encode_compute_boundary` — the uniform wire train `uniform(clamp(value/theta))` for subsumed plain-LIF-Perceptron ComputeOp boundaries; wrapper mappers and non-LIF ops stay rate-mode. |
| `segment_partition.py` | Exec-graph classification (spike producer vs host value boundary) and union-find partition into maximal spike segments; encoding perceptrons start fresh segments. |
| `segment_forward.py` | `SegmentForwardDriver` — the mode-agnostic walk: host ComputeOps run once on decoded values (with min recording and `_negative_shift`), spike segments delegate to `policy.run_segment`. |
| `segment_policies.py` | `LifSegmentPolicy` (per-cycle signed-IF cascade; entry/encoding boundaries emit the uniform wire train, mirroring `encode_compute_boundary`; host-op boundary re-encodes dispatch on representation — ABSOLUTE unarmed-chain producers transcode via the SSOT divide-first `normalize_boundary_value`, wire producers clamp; the domain map is `mapping.support.value_domain.value_domain_map`, so the guard and the gauge-establishment repair agree by construction and an unrepaired mixed fan-in fails loud; `node_value_recorder` side-channel) and `AnalyticalSegmentPolicy` (every node once on ideal values); re-exports `TtfsSegmentPolicy`. |
| `segment_policy_lif_serial.py` | `run_streamed_lif_cycles` — the streamed LIF hop's per-cycle loop (and its retime STE), plus the ODIN per-event twin: under `firing_granularity='per_event'` it arms a `SerialFoldSlot` from the mapper's own `get_effective_weight`/`get_effective_bias` and returns the hop's per-cycle emission MULTIPLICITY train; a multi-source hop, a rank>2 effective weight, or a comparator disagreeing with the point refuses by name. |
| `segment_policy_ttfs.py` | `TtfsSegmentPolicy` — latency-windowed single-spike sim (arrival latch, ramp decode, window-relative wire-normalized boundary trains per consumer input scale), drive-time effective-bias install, optional offload-boundary STE (`boundary_surrogate_temp`), plus the P4 frontier axes (`genuine_segments`, `genuine_hop_frontier`). |
| `segment_hop_frontier.py` | `run_segment_hop_hybrid` — [5v B2] the intra-segment k-hybrid: hops below the frontier run the genuine cycle loop (depth-prefix closed under deps), deeper hops run the trained proxy on the frontier's decoded values. |
| `chip_aligned_nf.py` | `chip_aligned_segment_forward` — thin LIF wrapper (driver + `LifSegmentPolicy`, `run_cycle_accurate` fallback); the torch mirror of HCM `_forward_rate`. |
| `lif_utils.py` | Unwrap wrapped `LIFActivation`s; toggle `use_cycle_accurate_trains` model-wide. |
| `scale_aware_boundaries.py` | Set per-block `activation_scale` (theta_out, encoding layer pinned) and forward-propagate `input_activation_scale` via the polymorphic mapper walk; `read_boundary_out_scales` is the pure (no-mutation) twin that DELEGATES non-perceptron nodes to `propagate_boundary_scale` (one implementation, both walks — armed buffer gauges, traffic lifts, residual rules; the §11.2 one-writer law); `verify_boundary_currency_coherence` is the install-seam fail-loud certificate. `establish_wire_gauge` is the gauge-establishment seam (arm the wrap slots, then propagate the currencies) every stage that trains against the deployed composition shares with WQ/SCM; `establish_gauge_for_mixed_domain_seams` is its scoped precondition repair — a graph carrying a HETEROGENEOUS fan-in has no domain there, so the seam is armed until the domain map classifies: the SCALE POLICY owns the gauge at every join whose producers differ (κ_T ≡ κ_S), and the loop's `_arm_domain_join` fallback only ever sees the leftovers, which are all-unity by construction and armed STRUCTURALLY at pass-through. A graph that already classifies is left untouched. `input_data_scale` is REQUIRED and stamped on the repr so pure re-reads agree with the propagated values. `boundary_scales_for_walk` is the one entry a twin walk reads its out-scale table through: `establish=True` runs the seam first (the per-event NF twin decomposes hops through `get_effective_weight`, so it reads the STAMPED per-source scales and owes the seam its precondition), `establish=False` is the pure read every default-point walk keeps. |
| `dfq_bias_correction.py` | Mode-agnostic DFQ core: teacher channel-mean capture (forward hooks, output- and input-side twins) reduced on the perceptron's declared channel axis, the mask-aware per-neuron `bias += eta*(ann − cascade)` loop with keep-best/early-stop over an injected deployed-behavior probe, and the [S3] `sequential_first_moment_fold` (per-hop deployed-vs-float pre-activation mean fold, own-offset-excluded, measured through the already-folded prefix). |
| `distribution_matching.py` | TTFS distribution matching: quantile scale-aware boundaries + the DFQ loop over the deployed single-spike cascade; returns gap/dead-fraction stats. |
| `lif_distribution_matching.py` | LIF DFQ bias correction over the deployed cycle-accurate cascade (read via the `node_value_recorder` side-channel); no boundary retune. |
| `gain_correction.py` | Per-cascade-depth theta trim (`gamma^d`, encoding/entry pinned) inverting the TTFS ramp-decode death cascade; `apply_gain_at_rate` for ramped tuning. |
| `theta_cotrain.py` | Promote non-encoding perceptrons' `activation_scale` to trainable per-output-channel Parameters (rebound on every referencing node) for cascade fine-tuning. |
| `per_channel_theta.py` | [S2/R3] Calibration-time per-channel theta: promote eligible matching-axis hops (channel-aligned consumer walk, structural perceptron paths only) to per-channel quantile thetas via in-place `.data` writes; armed for lif + synchronized by the `per_channel_theta` knob; weight-shared / axis-flipped / entry-boundary hops keep the scalar. |
| `sync_first_moment.py` | [S3/R6] Sync wrapper over `sequential_first_moment_fold`: derives each hop's own +theta/(2S) offset from the baked half-step flag (the §3.2 sign trap makes the exclusion load-bearing), orders hops by cascade depth, and marks folded hops (`sync_first_moment_fold` knob; applied at the AQ endpoint before endpoint recovery). |
| `seam_audit.py` | Seam-certificate auditor (`audit_model`): read-only per-edge/per-node B/C/G classification on one batch — currency-consistency, boundary encode round-trip vs the trained entry, armed wire-twin vs value-twin, LIF kernel vs its count staircase (spiking_deployment_calculus.md §9); validated by defect injection. |

## Dependencies
- `mapping` — IR types (`ComputeOp`, `IRSource`) and `HybridHardCoreMapping`/`HybridStage` for boundary encode; mapper classes (`InputMapper`, `ComputeOpMapper`) for node classification; `scale_propagation.walk_out_scales` for boundary-scale propagation and `arm_wrap_slots` for the mixed-seam arming; `support.per_source_scales.compute_per_source_scales` + `support.value_domain` (`value_domain_map`, `heterogeneous_domain_joins`) for gauge establishment and the domain map; `support.activation_scales` + `ScaleNormalizingWrapper` for the wire-divisor map; `channel_axis_walk` for the per-channel-theta eligibility walk; `support.bias_compensation.SYNC_ENTRY_HALF_STEP_FLAG` for the S3 own-offset derivation.
- `models` — `LIFActivation`, `run_cycle_accurate`, `TTFSActivation`, `TransformedActivation` (neuron dynamics + unwrap); `effective_preactivation_bias` (norm-folded TTFS bias); `activation_channel_axis` (owner-declared DFQ channel-axis ground truth); lazily `TTFSSegmentForward` as the DFQ cascade readout.
- `transformations` — `pruning.committed_masks.commit_perceptron_pruning`: DFQ starts from the committed-pruning raw-parameter state (the deployed executor never fires enforcement hooks); `PerceptronTransformer` for the S3 effective-bias fold writes.
- `chip_simulation` — `spike_modes` spike-timing encoders (Uniform / TTFS), shared with the simulators so encode timing cannot drift; `spiking_semantics` mode predicates for the per-channel-theta arming gate.
- `tuning` — lazily `LIFBlendActivation` in `lif_utils` (unwrap/toggle during the blend ramp; deferred import avoids a cycle); `shift_calculation.calculate_activation_shift` (the half-step magnitude SSOT) in `sync_first_moment`.

## Dependents
- `chip_simulation` — every deployed runner applies the wire-domain seam (`boundary_normalization_scales` + `normalize_boundary_slices_numpy`) and `decode_segment_output`; `spiking_mode_policy` installs `chip_aligned_segment_forward` as the NF probe.
- `mapping` — `support/bias_compensation` drives `SegmentForwardDriver` + `TtfsSegmentPolicy`.
- `models` — `nn/activations/lif.py` builds spike trains; `perceptron_mixer/perceptron.py` unwraps LIF activations.
- `pipelining` — the simulation factory toggles cycle-accurate trains; the SCM mapping and TTFS adaptation steps calibrate/propagate boundary scales.
- `tuning` — the LIF/TTFS adaptation tuners (chip-aligned forward, DFQ matching, gain correction) and the KD blend tuner (theta co-training).

## Exported API
`__init__.py` re-exports:
- `BoundaryConfig`, `encode_segment_input`, `encode_compute_boundary`, `decode_segment_output`, `decode_segment_output_torch` — the boundary contract.
- `SegmentForwardDriver`, `LifSegmentPolicy`, `TtfsSegmentPolicy` — the unified segment forward.
- `lif_spike_train`, `uniform_spike_train`, `rates_to_spike_train` — spike-train constructors.
- `unwrap_lif_activation`, `apply_cycle_accurate_trains_to_model` — LIF helpers.
- `calibrate_scale_aware_boundaries`, `propagate_boundary_input_scales` — boundary-scale calibration.
- `match_activation_distributions`, `match_lif_activation_distributions` — DFQ distribution matching.

`AnalyticalSegmentPolicy`, `chip_aligned_segment_forward`, gain correction,
theta co-training, and the partition helpers are imported from their modules
directly.
