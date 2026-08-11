# DeploymentRecord — the W4.1 schema specification (stage 0, for owner review)

2026-08-10 · deployment-formalization program, workstream W4 · status: **awaiting owner sign-off**
(implementation stages 1–6 do not start until this document is approved; the sign-off
checklist is at the end)

This document specifies the typed, versioned, provenance-carrying artifact that makes the
thesis-§2 deployment model concrete in the software: **a deployment is a schedule** — ordered
passes over the alternating NeuralOps/ComputeOps structure, each pass reprogramming or reusing
core-resident weights with a banded cost model for the difference; pass count, area, energy,
latency, and throughput emerge from workload × topology × hardware constraints × deployment
options. Today those quantities live on three disconnected surfaces (measured `CostRecord`,
static-mapping stats, physical coefficient bands) or are computed and dropped. The record is
**a join and a registry**, not new measurement.

Design theses:

1. **One artifact, sealed.** `deployment_record.json` per run, assembled from fragments the
   producing steps already compute, validated against a plan-derived required-fragment matrix,
   written by a new terminal pipeline step. Fail-loud: a record that cannot seal is a defect.
2. **Provenance is structural.** Every fragment group carries
   `Provenance{kind ∈ measured|modeled|derived|declared, producer, step}`; modeled values carry
   `(low, nominal, high)` bands with a written basis. No proxy is ever presented as a
   measurement.
3. **Same shape at two completenesses.** A search candidate populates the static fragments
   only; a pipeline run seals the complete record. "Is this objective/introspection payload
   available" ≡ "is its backing fragment populated" — the honesty discipline made structural.
4. `DeploymentPlan` stays configuration-only. The record is the *measured* artifact, keyed by
   the plan (`cell_key` + `config_digest`).

Module home: new top-level `src/mimarsinan/deployment_record/` (near-leaf: `pipelining`,
`search`, `gui` import it; it imports `mapping`, `chip_simulation`, `certification`,
`config_schema`, `common` only). Subpackages: `schema/`, `build/`, `cost/`, `objectives/`,
`introspection/` — every file ≤300 LOC.

---

## 1. Artifact, versioning, identity

- Filename: `deployment_record.json`, in the run's working directory, written atomically.
- `DEPLOYMENT_RECORD_FORMAT_VERSION = 1`, duplicated top-level for cheap sniffing.
- `from_dict` **rejects** unknown fields and wrong versions (the `CostRecord.from_dict`
  discipline). Schema evolution = version bump + explicit migration, never silent tolerance.
- All schema types are frozen dataclasses (repo idiom); JSON-safe `to_dict`/`from_dict`.

```python
RecordIdentity:
    format_version: int              # == 1
    run_dir: str
    cell_key: str                    # CertificationCell (firing × sync × backend axes)
    mode: str                        # spiking_mode / mvm dispatch string
    model_type: str;  model_name: str
    workload: str                    # ResolvedWorkloadProfile key
    config_digest: str               # sha256 of canonical deployment-config JSON
    platform: Mapping[str, Any]      # platform_constraints_resolved VERBATIM
                                     #   = the thesis' target-system contract
    deployment_options: Mapping      # schedule_policy, max_schedule_passes, weight_bits,
                                     #   target_tq, T, tolerances
    created_at: str                  # ISO timestamp
```

## 2. Fragment schemas

### 2.1 Schedule — the thesis' program, mirrored 1:1 from `HybridHardCoreMapping.stages`

```python
ComputeOpRecord:                     # one per host ComputeOp stage
    stage_index: int                 # position in the stage program (execution order)
    name: str;  op_type: str
    output_width: int
    wall_s_total: Optional[float]    # measured host wall (StageTimer); None until timed

SegmentCoreRecord:                   # one per hard core of a neural segment
    core_index: int
    axons: int; neurons: int         # physical geometry
    axons_used: int; neurons_used: int
    cells_used: int                  # programmed weight cells (CoreOccupancy semantics)
    params_bytes: int                # ceil(cells_used × weight_bits / 8) — exact, derived
    connectivity_entries: int        # compressed axon-source span count
    static_delay_levels: Optional[int]

SegmentRecord:                       # one per (neural segment × pass)
    stage_index: int; segment_index: int; pass_index: int
    pass_reason: "initial" | "capacity_overflow"      # pass_index>0 ⇒ capacity_overflow
    programming: "reprogram" | "resident"             # from schedule_weights_resident
    bank_ids: tuple[int, ...]
    cores: tuple[SegmentCoreRecord, ...]
    params_programmed: int           # Σ placement areas (0 when resident)
    params_unique: int               # weight-stationary ideal
    params_bytes: int                # Σ core params_bytes (0 when resident)
    connectivity_entries: int
    static_latency_levels: int       # dependence-level currency (LayoutVerificationStats)

ScheduleRecord:
    stages: tuple[SegmentRecord | ComputeOpRecord, ...]   # program order
    pass_count: int                  # cross-checked vs layout stats at seal
    sync_count: int
    reprogram_passes: int; reuse_passes: int              # deployed classification
    params_reloaded: int             # IR-level plan figure (provenance "planned@scm")
    compute_op_count: int
```

### 2.2 Placement — softcore→hardcore, banks, tiles

```python
SoftcorePlacementRecord:             # a pure read of soft_core_placements_per_hard_core
    ir_node_id: int
    segment_index: int; pass_index: int; hard_core_index: int
    axon_offset: int; neuron_offset: int; axons: int; neurons: int
    perceptron_index: Optional[int]  # REAL IR-layer identity
    weight_bank_id: Optional[int]
    bank_axon_range / bank_neuron_range: Optional[tuple[int, int]]
    split_group_id / split_fragment_index / coalescing_group_id: Optional[int]

BankRecord:      bank_id, rows, cols, params, placement_count   # sharing degree
FloorplanRecord: mesh_width, mesh_height, cores_per_tile, derivation: declared|derived
TileRecord:      tile_index, x, y, core_indices
PlacementRecord: softcores, banks, floorplan: Optional, tiles   # floorplan None w/o SANA-FE
```

### 2.3 Utilization / area — the two write-only reports, folded

```python
CrossbarUtilizationRecord:  # typed mirror of CrossbarUtilizationReport.to_dict(), 14 fields:
    cores_allocated, axons_used, axons_physical, axon_utilization,
    neurons_used, neurons_physical, neuron_utilization,
    cells_used, cells_physical, cell_occupancy,
    unusable_space, macs, weight_bits, programming_bits

LayoutStatsRecord:          # typed mirror of LayoutVerificationStats — all 34 fields, same names

UtilizationRecord:
    crossbar: CrossbarUtilizationRecord
    layout: LayoutStatsRecord
    relay_cores_inserted: int        # the currently-discarded depth-balancing return
```

### 2.4 Traffic — per boundary, per tile, per NoC link

```python
BoundaryTrafficRecord:               # REDUCED from flow_node_counts {node_id: (B, n)};
    node_id: int                     #   raw tensors are never persisted (reduced inside the
    producing_stage_index: Optional  #   gate scope, bounded memory)
    neurons: int; samples: int
    total_count: int; max_neuron_count: int

NocLinkLoadRecord:  from_x, from_y, to_x, to_y, packet_count
NocTrafficRecord:                    # SANA-FE scope, summed over segments/samples
    total_packets, inter_tile_packets, intra_tile_packets, input_path_packets,
    cross_tile_connectivity_edges, mapped_cross_tile_axons,
    link_loads: tuple[NocLinkLoadRecord, ...]

TrafficRecord:
    boundaries: Optional[tuple[BoundaryTrafficRecord, ...]]   # counts-observable + gate armed
    noc: Optional[NocTrafficRecord]                           # SANA-FE enabled
```

### 2.5 Timing + energy

```python
SegmentTimingRecord:  stage_index, timesteps_executed, sim_time_s
                      # sim_time_s INCLUDES NoC hop latency (charged by SANA-FE's C++ NoC)

LatencyDecomposition:
    programming_s: Optional[ModeledValue]    # payload bytes / bandwidth band — MODELED
    compute_steps: int                       # Σ timesteps_executed (measured; legacy latency_steps)
    compute_sim_time_s: Optional[float]      # Σ sim_time_s (measured, includes NoC)
    host_ops_s: Optional[float]              # Σ ComputeOpRecord.wall_s_total (measured)
    sync_s: Optional[ModeledValue]           # sync_count × barrier band — MODELED
    note: str                                # explicit no-double-count statement

TimingRecord:  s_global, depth, per_segment (empty w/o SANA-FE), latency

EnergyTermRecord:  name, mj, kind: measured|modeled, band_mj: Optional[(low, high)], basis
EnergyRecord:
    total_energy_mj, mj_per_sample, sample_count          # SANA-FE measured
    breakdown: tuple[EnergyTermRecord, ...]
    energy_proxy_neuron_steps: int                        # derived; legacy continuity
    total_spikes: int
```

### 2.6 Accuracy + adaptation

```python
AccuracyReadRecord:  metric, backend (hcm|nevresim|value_census|pipeline), samples,
                     kind: measured|carried, step
CertificateRecord:   name, backend, passed, neuron_windows_compared,
                     exact_match_fraction, max_abs_delta, detail
AccuracyRecord:      deployed: AccuracyReadRecord, reads: tuple, certificates: tuple
                     # "certified" = deployed read + FATAL gates green
AdaptationRecord:    max_ft_pass_wall_s, ft_pass_walls (per-pass {label, wall_s} bundles)
```

### 2.7 The record

```python
DeploymentRecord:
    identity: RecordIdentity
    schedule: ScheduleRecord
    placement: PlacementRecord
    utilization: UtilizationRecord
    accuracy: AccuracyRecord
    timing: TimingRecord                 # static part always; measured parts optional inside
    traffic: Optional[TrafficRecord]
    energy: Optional[EnergyRecord]       # None ⟺ SANA-FE disabled
    adaptation: Optional[AdaptationRecord]
    provenance: Mapping[str, Provenance] # one entry per attached fragment group (seal-enforced)
    format_version: int = 1
```

---

## 3. Thesis-§2 term ↔ record field (1:1)

| Thesis §2 term | Record field |
|---|---|
| Target-system contract (NeuralOp core types: geometries, counts, capabilities) | `identity.platform` (resolved `cores[]`, `weight_bits`, capability bits — verbatim) |
| Firing regimes / promised core semantics | `identity.cell_key` + `identity.mode` |
| ComputeOps (host ops for what cannot lower to MVMs) | `schedule.stages[*]: ComputeOpRecord` |
| Deployment IS a schedule (ordered passes over the alternating structure) | `schedule.stages` (program order) + `schedule.pass_count` |
| A pass | `SegmentRecord` (one per `(segment_index, pass_index)`) |
| Pass from blocking non-MVM interleave | segment boundaries between `ComputeOpRecord` stages |
| Pass from capacity overflow | `SegmentRecord.pass_reason == "capacity_overflow"` |
| Reprogram vs reuse core-resident weights | `SegmentRecord.programming` + `schedule.reprogram_passes`/`reuse_passes` |
| Parameters sent to the chip | `SegmentRecord.params_programmed` (count) + `params_bytes` (bytes) |
| Connectivity sent to the chip | `SegmentRecord.connectivity_entries` (count) + modeled bytes (cost model, banded) |
| Banded reuse-vs-reprogram cost | cost-model terms carrying `Band` over the fields above |
| Segment's cores | `SegmentRecord.cores` |
| Segment latencies | `static_latency_levels` (static) + `per_segment[*].timesteps_executed / sim_time_s` (measured) |
| Spans (connectivity structure) | `SegmentCoreRecord.connectivity_entries` |
| Mapping/placement (softcore→hardcore, banks) | `placement.softcores`, `placement.banks` |
| Tiles (NoC-free hard-core groups) | `placement.tiles` + `placement.floorplan` |
| Accelerator area: core counts and occupancy | `utilization.crossbar` + `utilization.layout` |
| Spike traffic per boundary | `traffic.boundaries` |
| Spike traffic per tile / NoC hop | `traffic.noc` |
| Energy | `energy` (measured breakdown) + modeled terms (cost model) |
| End-to-end latency decomposition | `timing.latency` |
| Throughput | cost-model output `throughput(record)` (samples/s from the decomposition) |
| Deployed accuracy ("read from the chip") | `accuracy.deployed` + `accuracy.reads` |
| Certified accuracy | `accuracy.certificates` (FATAL gates) |
| Deployment options (mapping strategy, tolerances → attainable quantization) | `identity.deployment_options` |
| Workload × topology | `identity.workload`, `identity.model_type/model_name/config_digest` |

---

## 4. Producer wiring (fragment → producing step → carry → today's producer)

Carry mechanism: pipeline-cache entries (`add_entry` / step `PROMISES`), JSON-safe dicts
re-typed on read — resume-safe, inspectable, and exactly how `ir_graph` /
`hard_core_mapping` already travel between steps.

| Fragment (cache key) | Producing step | Today's producer — status |
|---|---|---|
| identity | Deployment Record (assembly) | `DeploymentPlan.resolve`; `build_platform_constraints_resolved` — live |
| `deployment_record_scm` (reuse plan, relay count, IR latency) | Soft Core Mapping | `weight_reuse_plan_from_graph` (`mapping/weight_reuse.py:140`) — print-only, gated; relay count (`depth_balancing.py:255`) — printed+dropped; `IRLatency` — print-only |
| `deployment_record_hcm` (schedule, placement, utilization, programming, boundary traffic, hcm/value reads, certificates) | Hard Core Mapping | stage/pass census — print; `weight_programming_report` (`weight_programming.py:42`) — print+event; `CrossbarUtilizationReport` — write-only json; `LayoutVerificationStats` — GUI snapshot only; placements (`hard_core_mapping.py:174-206`) — never serialized; `flow_node_counts` (`count_alignment.py:160`) — collapsed to 3 certificate scalars |
| `deployment_record_nevresim` (probe read, total spikes, host-op walls) | nevresim Simulation | probe accuracy — live; `total_spikes` (`nevresim_driver.py:85-87`) — printed+dropped |
| `sanafe_simulation_results` (per-segment timing/energy/NoC/tiles, floorplan) | SANA-FE Simulation | `SanafeStepReport` — exists (only 6 scalars reach `CostRecord` today) |
| host-op wall times | inside simulator runs | NEW: opt-in `StageTimer` wrapping the `on_compute` invocation in `run_hybrid_stages` (default None ⇒ byte-identical) |
| `ft_pass_walls.json` | tuner-hosting adaptation steps | producer shape exists (`ft_pass_wall_metrics()` — zero callers); reader already waits (`cost_extraction.py:314`) |
| sealed record + `deployment_record.json` + projected `cost_record.json` | **Deployment Record (NEW terminal step)** | legacy emission `SanafeSimulationStep._emit_cost_record` — moves here |

## 5. Required-fragment availability matrix (seal-enforced, plan-derived)

| Condition on the resolved plan | Required fragments |
|---|---|
| always | identity, schedule, placement, utilization, accuracy, timing (static part) |
| counts-observable mode ∧ spike-count gate armed | `traffic.boundaries` |
| `enable_sanafe_simulation` | `energy`, `timing.per_segment`, `traffic.noc`, `placement.floorplan` |
| nevresim applies | a `"nevresim"` read in `accuracy.reads` |
| a tuner-hosting adaptation step in the resolved step list | `adaptation` |

Seal cross-checks (the three formerly-disconnected surfaces must agree):
`schedule.pass_count == utilization.layout.schedule_pass_count`;
`Σ SegmentRecord.params_programmed == weight_programming_report.params_programmed`;
`utilization.crossbar.cores_allocated == Σ len(segment.cores)`; every attached group has a
provenance entry. Double-attach raises; missing-required raises. No `best_effort`.

## 6. Continuity contract — legacy `cost_record.json`

- Written by the new terminal step as a **projection** of the record, same directory, same
  sanafe-conditional as today (no SANA-FE ⇒ no `cost_record.json`).
- Live fields (`cell_key, mode, backend, acc_deploy, mj_per_sample, spikes, latency_steps,
  cores, s_global, depth, energy_proxy_neuron_steps, provenance.run_dir, format_version=3`)
  are **value-identical** to today's `extract_cost_record` output — pinned by a
  golden-fixture test.
- Formerly-dead fields go live from the record: `reprogram_passes, reuse_passes,
  params_reloaded, max_ft_pass_wall_s, ft_pass_walls`.
- `activation_bytes_moved` stays 0: no producer exists anywhere in the codebase; activation
  movement is modeled via traffic terms instead. (Disposition stated here deliberately.)
- `SanafeSimulationStep._emit_cost_record` and its `best_effort` wrapper are deleted.

## 7. Cost-model coefficient inventory (bands, with bases)

Applied by `deployment_record/cost/` to **real record quantities** (never to module
constants; the VGG16 illustration in `weight_reuse_cost_model.py` stays as-is).

| Coefficient | low / nominal / high | Basis |
|---|---|---|
| `e_dma_per_byte_mj` | 31e-9 / 160e-9 / 320e-9 | HBM2 ~3.9 pJ/bit · DDR3 ~20 pJ/bit (Horowitz 45 nm) · off-chip worst case (existing `DEFAULT_COEFFICIENT_BAND`) |
| `bytes_per_param` | 0.5 / 1.0 / 2.0 | 4-bit · 8-bit · 16-bit weights (existing band) |
| `e_sync_barrier_mj` | 1e-4 / 1e-3 / 1e-2 | ~0.1 / ~1 / ~10 µJ per barrier (existing band) |
| `bytes_per_connectivity_entry` | **4 / 8 / 16** | **NEW — needs owner sign-off.** Span entries are exact counts; their byte size is modeled until a real chip wire format exists (`chip_spans.txt` is a simulator exchange format, not chip DMA truth). |
| `CoreInitCoefficients.energy_mj` (per core) | 8.96e-8 / 1.8637e-5 / 7.4547e-5 | **NEW — needs owner sign-off.** DERIVED at import from the SANA-FE per-event presets: a core init writes every neuron's soma state once, priced as `(soma_access + soma_update)` per neuron × a reference core — TrueNorth over 256 neurons (low), Loihi over 256 (nominal) and 1024 (high). Per-core SIZE uncertainty lives in the band, not in a hidden constant. |
| `CoreInitCoefficients.time_s` (per core) | 5.12e-7 / 2.4832e-6 / 9.9328e-6 | **NEW — needs owner sign-off.** Same derivation on the presets' `soma_access_latency_s + soma_update_latency_s`. |
| `programming_bandwidth_bytes_per_s` | **1e9 / 12.8e9 / 256e9** | **NEW — needs owner sign-off.** Programming-payload DMA bandwidth: ~1 GB/s serial configuration port · DDR3-1600 single channel ~12.8 GB/s · HBM2 stack ~256 GB/s. No chip programming port is measured anywhere in this program. A DIVIDING coefficient: the latency term reads the opposite corner so it stays monotone. |
| `sync_barrier_s` (latency twin of `e_sync_barrier_mj`) | **6.4e-8 / 8.0e-8 / 6.4e-7** | **NEW — needs owner sign-off.** DERIVED from the presets' `tile_hop_latency_s` (TrueNorth 4 ns / Loihi 5 ns) × a 16-hop (8-hop diameter, out and back) to 128-hop mesh traversal. |

Every modeled term ships `(value, band, kind="modeled", basis)`; measured terms ship
`band=None`. Latency decomposition never double-counts NoC (measured `sim_time_s` already
includes it; the record says so in `LatencyDecomposition.note` — the model REFUSES a record
whose note does not state the discipline).

Applied by `DeploymentCostModel` (`deployment_record/cost/model.py`), whose coefficient
fields ARE the table above. Two application rules are structural: `bytes_per_param` is
inventory-only — the record's `params_bytes` is already a byte count
(`build/payload_sizes.py`), so the payload goes through the model's per-BYTE DMA channel and
the weight width is never applied twice; and a **resident** segment costs exactly the per-core
reset constant (zero payload — nothing is reprogrammed into its cores).

## 8. Objectives registry v2 (consumer summary)

`ObjectiveSpecV2{key, direction, unit, provenance, availability, extractor}` over a
`RecordView` (full record | candidate static view). The legacy 8 re-register byte-equal
(`estimated_accuracy, total_params, total_param_capacity, total_sync_barriers,
param_utilization_pct, neuron_wastage_pct, axon_wastage_pct, fragmentation_pct`); added:
`deployed_accuracy, mj_per_sample, latency_steps, host_op_wall_s, total_spikes, pass_count,
reprogram_passes, reprogramming_bytes, params_reloaded, noc_inter_tile_packets,
noc_total_packets, programming_energy_mj, sync_barrier_energy_mj, throughput_samples_per_s`.
Duplicate registration raises — new objectives are additions, never surgeries. Unknown or
unavailable objective names fail loud (replacing today's silent drop).

**As implemented (stage 6, `deployment_record/objectives/`).** The spec carries one further
declared field, `requires` — the datum an axis needs, stated once next to its availability
predicate — so an unavailable objective can SAY what it is missing (the loud resolution error
and the wizard's `unavailable_reason` are the same string; no per-objective special cases).
Availability and extraction are one reader (`Backing`), so an axis can never advertise itself
and then fail to produce a number. `for_search_mode(mode)` asks availability of a *candidate
capability probe* — a maximally populated `CandidateStaticView` whose accuracy estimate exists
only where the mode trains — so hardware-only search drops `estimated_accuracy` because a
hardware candidate carries no accuracy read, not because a list says so. The resulting search
catalog is the legacy 8, byte-equal in keys/directions/ORDER, and `search/results.py` is its
projection. Two dispositions worth stating: `pass_count` is keyed to the SEALED schedule census
even though the static layout stats also carry `schedule_pass_count` (lifting a static pass axis
into the search catalog changes every optimizer's objective vector — a search-surface decision,
W5.1); and `total_sync_barriers`/`total_params`/`estimated_accuracy` are candidate-side axes with
no record twin (the record carries no host-slot census, no torch parameter census, and no
search-time proxy — `deployed_accuracy` is its measured counterpart).

## 9. Known limitations (stated, not hidden)

1. **nevresim cycle counts do not exist** — the C++ span machinery is connectivity
   compression, not a timer. v1 ships the nevresim fragment without cycles and the timing
   provenance says so; extraction is a follow-up requiring a vendored-C++ export.
2. **Boundary traffic is reduced at the gate** (per-node totals/maxima), never raw `(B, n)`
   tensors — bounded artifact size by construction.
3. **Resume DAG**: old run dirs lack the new fragment entries, so resumes of pre-record runs
   re-run from an earlier step than before. Accepted and documented.
4. Record emission is fail-loud; a run whose record cannot seal fails. This is the intended
   discipline (acceptance: every tier run emits a schema-valid record).

## 10. Owner sign-off checklist

- [ ] Fragment field lists (§2) — complete and correctly typed for the thesis defense story
- [ ] Thesis-term ↔ field table (§3) — 1:1, no term unrepresented
- [ ] Producer wiring (§4) — agreed carry mechanism and step ownership
- [ ] Availability matrix + seal cross-checks (§5)
- [ ] Continuity contract (§6) — incl. `activation_bytes_moved` staying 0 and the emission
      moving to the terminal step
- [ ] `bytes_per_connectivity_entry` band 4/8/16 B (§7) — bless or amend
- [ ] Objectives catalog additions (§8)
- [ ] Known limitations (§9) — acceptable as stated
