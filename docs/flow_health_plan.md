# Flow-Health Program (H-series) — closing the estimate↔signoff loop

## Status (2026-08-17)

| Stage | State | Commits |
|---|---|---|
| H0 investigation: profile + root-causes | **DONE** — findings below, all measured | (this doc) |
| H1 measured synaptic-event census | **DONE** — `chip_simulation/synaptic_events.py`; the SANA-FE finalizer measures per stage, `EnergyRecord.synaptic_events` seals the per-inference mean, `energy_per_inference_mj` prices on the record plane; refusal (None) propagates end-to-end when any trace is unparsed | `record(H1)` |
| H2 candidate carry census | **DONE** — one census over spans (`carry_census_from_spans`), two span producers; candidate == deployed on the vehicles, both disciplines; carry axes searchable (B's capacity-not-a-variable stands); known-zero semantics on BOTH planes | `search(H2)` |
| H3 fidelity as an instrument | **DONE** — twin armed via the shared `firing_semantics_kwargs`; full candidate-answerable surface; per-term zip with bands/evidence; utilization family = allocated denominator + committed-rectangle numerator BOTH planes (a second numerator fork found and closed); new `chip_occupancy_pct`; one-sided rows carry a written basis | `fidelity(H3)` |
| H3b host-rate calibration | **DONE** — `scripts/calibrate_host.py` + `platform_physics/host_calibration.py`; new `measured` evidence kind (requires the machine + method note); this host: 21.02 G/s (vs the 10 G/s estimate the study declared — a 2.1x host-term correction), RAPL unreadable so `p_host` stays declared | `physics(H3b)` |
| H4 performance + elegance consolidation | pending | |
| H5 study re-run + findings | pending | |

Repo: `mimarsinan` @ `e97cec1c` (E-series closed). The E-series made the candidate's
quantities MEAN what the record's mean; this program makes the loop between them
verifiable. Deliverables per the owner's directive: correctness, performance, and
code elegance — each stage names all three.

## H0 — what the investigation established (measured, not assumed)

**Hole 1 (energy unverifiable).** `from_record` deliberately claims no
`synaptic_events` ("total_spikes counts emissions, not synapse arrivals — no event
census is sealed yet"), so `energy_per_inference_mj` REFUSES on the record plane and
the candidate's energy is never checked against a deployment. Everything needed is
already sealed per stage: `SanafeSegmentRecord.per_neuron_spike_counts` (emissions),
`seg_input_spike_count` (per-wire input spikes), and the HCM connectivity. The census
is a pure join: static occupied-fanout ⊗ measured emissions.

**Hole 2 (search blind to carry).** The candidate produces NO carry quantity, while
`schedule_policy` is a search axis and carry is precisely the cost that axis moves.
The record census (E3) is `pass_carry.py` over hybrid stages; the candidate has the
same facts in different clothes — pass membership (`pass_placements`), adjacency
(`pair_wires`), widths (softcore `output_count`) — so the census function can be
shared verbatim over spans, with one span producer per plane.

**Hole 3 (fidelity one-sided + a 2.00x lie).**
- The twin (`fidelity_emission._fidelity_problem`) never passes `spiking_mode` /
  `ttfs_cycle_schedule` / `per_hop_retiming`: a re-timed run's twin prices the FUSED
  wall — E1's arming never reached the fidelity plane.
- The twin's active set is only the SEARCHED axes, so fragments other axes need are
  never computed and the prediction column is artificially empty.
- `param_utilization_pct`: candidate 5.354% = 68096 / 1,271,808 (whole declared
  chip, idle cores padded by `_stats_from_packing`); record 10.709% = 68096 /
  635,904 (allocated cores, `CrossbarUtilizationReport.cell_occupancy`). Same name,
  two denominators — the whole wastage family forks the same way. The catalog text
  ("share of ALLOCATED crossbar cells") sides with the record.

**Performance (measured on the live-path fixture, steady state).** Candidate eval =
4–6 ms, of which ~half is per-candidate re-derivation of per-PROBLEM constants:
`active_specs` resolved 5x per eval, `candidate_probe_without` built 4x per eval
inside `_requires_fragment`, the availability gate re-prices the cost report 10x per
eval (46 `cost_report()` reads), and `pack_layout` runs 11x per candidate
(`compute_mapping_stats` packs flat + per pass, then `collect_noc_fragments` re-plans
and re-packs every pass). On MLP-scale programs this is milliseconds; on conv-scale
softcore counts the packing multiplicity dominates search throughput.

**Other estimand suspects** — `total_sync_barriers` (candidate: host_segments +
schedule syncs; record answers nothing on the study runs) and `timesteps`
(candidate `simulation_steps` vs record `s_global`): DEFERRED past H3 with the
report's basis rows stating the mismatch per run; adjudication rides the H5
findings once live reports show the magnitudes. A SECOND utilization fork was
found during H3 and closed: the numerators differed too (candidate summed the
PLACED PIECES, the record's crossbar measures the committed used-rows x
used-columns RECTANGLE — 140 vs 740 cells on the token vehicle); both planes
now measure the rectangle, the IMC reclaim rule's meaning.

## Owner decisions (asked 2026-08-17)

| Question | Decision |
|---|---|
| Utilization estimand | **Allocated + new chip axis**: `param_utilization_pct` (and the wastage family) = allocated-cores denominator on BOTH planes; NEW `chip_occupancy_pct` (whole declared chip) carries the chip-sizing signal by name. Search-gradient change disclosed. |
| Host-rate calibration in scope | **Yes** — measure this host's `host_macs_per_s` (+ `p_host` where RAPL allows), declared back as `evidence_kind: measured` with machine identity. |
| Study re-run scope | **MLP × 4 profiles + the LeNet5 stretch**, one batch. |

## H1 — the measured synaptic-event census

The estimand, stated: one synaptic event = one spike arriving at one OCCUPIED cell
(used column) of a consumer row. This matches the candidate's model (`onchip_macs`
counts mapped cells; events = cells x activity) and the vocabulary's separation of
concerns (whole-row analog physics is `e_row_drive` x `boundary_events`, not events).
Events are counted over the EXECUTED window — the same window E1 sizes.

1. `synaptic_event_census(...)`: a pure function joining static per-source occupied
   fan-out (from the HCM: for each source neuron / input wire, the used cells in the
   consumer rows it drives) with measured per-neuron emissions and per-wire input
   spikes. One home; duck-typed over the sealed per-stage artifacts; no backend
   imports.
2. Emission wiring: `DeploymentRecordStep` computes the census from the SANA-FE
   per-stage records it already holds; `EnergyRecord` gains additive-optional fields
   (`synaptic_events: Optional[int] = None`, `synaptic_events_per_stage`), last
   position, no format bump — the `invocations` precedent.
3. `from_record` claims `synaptic_events` measured when sealed; the pricer's energy
   headline goes live on the record plane with no pricer change; the e_mac and
   e_synaptic_event_total supersession paths both verified.
4. Fidelity gains the energy triangle: candidate-priced (modeled events) vs
   record-priced (measured events, same constants) vs SANA-FE-measured
   (`mj_per_sample`) — the first validates the multiplicand model, the second the
   constants. Also derive and report the run's measured effective activity factor
   (measured events / cells / steps) beside the declared one.
5. Tests: hand-built vehicle with known fan-out and emissions (exact expected
   count); executed-window pin; absence pins (no SANA-FE run → census absent →
   refusal preserved); mutation kills (drop input side, drop fan-out weighting,
   count full rows, count semantic instead of executed window).

## H2 — the candidate carry census

1. Refactor `pass_carry.py`: the census becomes pure over SPANS
   (`(node, width, produced_in, last_consumed_in)`); `carried_wire_spans(stages)`
   stays the record's span producer; NEW `carried_softcore_spans(softcores,
   pass_placements, pair_wires)` is the candidate's. ONE census, ONE sizing (E3's
   `carried_wire_bytes` / `boundary_transfer_bytes`), two span producers.
2. Discipline threading, the E1 pattern: the search step resolves the run's
   transfer discipline through the deployment's own SSOT
   (`SpikingDeploymentContract.pass_boundary_transfer()`) and hands it to the
   problem; pinned end-to-end in `test_search_step_carries_firing_semantics`
   (streamed → VERBATIM, windowed → COLLAPSE). The fidelity twin gets the same.
3. Quantities: candidate claims `carried_raster_bytes`, `carry_peak_live_bytes`,
   `carry_out_bytes`, `carry_in_bytes` — same keys as the record, so the pricer's
   carry terms activate at candidate time and fidelity zips all four. Absent
   without a wire census (adjacency comes from the walk), without a discipline, or
   when nothing crosses a pass boundary.
4. Consistency pin with E5: the carry census and the NoC estimator's carried-input
   traffic must agree on WHICH wires cross (one pass-membership predicate, shared).
5. Registry/probes: carry axes classify as needing the census fragment; probe
   answers stay honest (`candidate_probe_without` semantics extended).
6. Tests: twin-equality pin (candidate census == record census on the bank-clustered
   vehicles' multi-pass programs, both disciplines), absence pins, mutation kills
   (wrong pass membership, buffer-sized transfer, one-direction census).

## H3 — fidelity as an estimate↔signoff instrument

1. **Twin arming fix**: `_fidelity_problem` passes the full semantics the search
   step passes (`spiking_mode`, `ttfs_cycle_schedule`, `per_hop_retiming`, H2's
   transfer discipline) from the run's resolved config — pinned with a re-timed
   fixture whose twin must price per-level windows.
2. **Full prediction surface**: the twin's active set becomes every
   candidate-answerable axis (registry-enumerated), not the searched subset — the
   report's prediction column stops depending on what the run happened to search.
3. **Term-level zip**: the absolute cost terms share names across planes by
   construction; fidelity compares every term (value, band, in-band, evidence kind)
   so "e2e off 100x" decomposes into "t_cycle x steps within X%; host term off Nx
   (estimated rates)". Axis-level rows stay.
4. **Estimand alignment**: implement the owner's utilization decision; adjudicate
   `total_sync_barriers` and `timesteps` the same way (one estimand per name, both
   planes, catalog text updated, drift pins updated with intent).
5. **One-sided rows say why**: each fidelity row carries a basis — measured-only by
   nature / prediction pending fragment X / refused: reason — the refusal-with-
   reason discipline extended to the report.
6. Report-first stays (standing owner decision); structural gates unchanged.

## H3b — host-rate calibration (if approved)

A small harness measures THIS host's `host_macs_per_s` (and states `p_host` from
declared TDP or measurement where available), written back as operator-declared
constants with `evidence_kind: measured` and the machine identity in the note. The
host terms stop being the 100x wildcard in every e2e/energy comparison. Refusal
behavior unchanged for runs that declare nothing.

## H4 — performance + elegance (measured targets only)

1. Cache per-problem constants: `active_specs` (one resolution), the fragment-
   requirement classification (probe construction), and the candidate view's cost
   report (one pricing per view). Target: ~2x on steady-state eval, measured.
2. One packing: `compute_mapping_stats` and `collect_noc_fragments` share a single
   pass plan + pack (the packer already seals placements and snapshots in one call).
   Target: pack_layout 11x → ≤4x per candidate, measured before/after.
3. Onchip census: one flow conversion serving both metrics (params + macs walk).
4. Cohesion: `candidate_fragments.py` split into `program_facts.py` (semantics,
   latency, programming, carry — the pass-structure facts) and the physics/context
   fragments; `_put` deduplicated across quantity extractors. Budgets/ratchets stay
   green; every move is a pure refactor under existing pins.

## H5 — study re-run + findings

Re-run the study per the owner's scope decision with the corrected machinery; every
run seals the event census, the carry quantities, and a full-surface, term-zipped
fidelity report. Findings doc (`docs/flow_health_study.md`): the energy triangle
per profile, measured-vs-declared activity, the utilization fork resolution, per-term
in-band table, remaining honest gaps. Update `ARCHITECTURE.md`s, this doc's status
table, and the project memory.

## Sequencing

H1 ∥ H2 → H3 (needs both planes' terms) → H3b → H4 → H5 (runs on the fast path).

## Verification protocol (every stage)

Tests first; `python -m pytest tests` ≤2 min green; `./scripts/typecheck.sh` zero;
ratchets/budgets clean; load-bearing guards mutation-checked; ARCHITECTURE.md per
touched module; per-stage commits, no AI-attribution trailers; behavior changes
carry A/B pins (no-census records load unchanged; no-profile runs byte-identical).
