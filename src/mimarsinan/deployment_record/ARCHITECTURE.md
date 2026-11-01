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
| `platform_physics/` (incl. `host_calibration.py` [H3b]: measure THIS host's ComputeOp-shaped MACs/s, per-op dispatch wall + RAPL package power where readable into an override-ready block with `evidence_kind: measured` — the host terms stop being the estimated 100x wildcard; [CL0] rows the calibration cannot measure (`p_host` without RAPL) are DECLARED beside the measured ones, each a constant with its own evidence, and a row both measured and declared refuses by name (`HostRowConflictError`); `host_block`/`load_host_block` read an artifact's block as the named `host` block of `merge_override_blocks`, refusing one that lacks a row the host terms multiply or carries a row outside the host group (`HostBlockIncompleteError`)) | The TARGET's half of the cost model: per-unit physical constants a vendor declares, which the cost model multiplies by the quantities the record already seals. The record counts (cells, MACs, spikes, hops, bytes, passes, timesteps); this names what each one costs in joules, square metres and seconds, so an absolute Area/Energy/Latency/Throughput number is attributable to a real chip instead of to a framework default. `vocabulary.py`: the closed constant list AS DATA — `(key, group, dimension, display_unit, multiplicand, doc)` per row, where `multiplicand` names the recorded quantity the constant is priced against, because a bare "energy" number is unusable and the pairing belongs in the declaration rather than at a pricing site. It also declares `SUPERSEDES`: a published AGGREGATE already contains the decomposition it supersedes, so `constants.resolve_supersessions` drops the absorbed terms and pricing can never charge both. The rule stays for a target that publishes only a whole-chip average, but NO shipped profile prices from `e_synaptic_event_total` any more: an average is a POINT model — it folds static power into a per-event rate, so it is exact where it was measured and wrong elsewhere (TrueNorth's own 26 pJ/event overshoots its published 96 Hz measurement by 256%, ODIN's 12.7 pJ/SOP undershoots its published biological-time point by 76%). Every profile now declares a MARGINAL per-event energy with static power beside it, and `correlation/` is what proves the difference. Execution POLICY (overlap, residency-across-batch) is deliberately NOT here — physics is banded numbers only. `constants.py`: `PhysicsConstantSpec` + the built registry, `spec_for`/`keys_in_group` (group order IS the wizard panel order); [U-E3] also the declared value `PhysicsConstantValue` and its one judge `constant_from_declaration` (the `source_ref` indirection below included), which the loader, the lane blocks and the converter's rows all read at module top: they live below the converter because `profile.py` imports the converter (a profile carries its converter table). `profile.py` (re-exporting the value): `PhysicsConstantValue` — the declared band in its declared unit PLUS its evidence, where `evidence_kind ∈ published/datasheet/derived/estimated` each REQUIRE their own field (a citation, a shown derivation, a written rationale), so an unevidenced number cannot be constructed; the SI `band` is a PROPERTY, never a stored field, so the declared and canonical forms cannot drift. `PlatformPhysics` has `has`/`band`/`missing`/`declares_all` and deliberately NO `get(key, default)`: an undeclared constant is ABSENT, which disables exactly the objectives that need it and may never be silently defaulted to a framework number masquerading as vendor data. `loader.py`: strict JSON reading (unknown constant, unknown field and wrong `format_version` all raise), the `source_ref` indirection — `sanafe_preset:<preset>:<field>` materializes a constant from the in-repo SANA-FE per-event presets so a number that already has a home is REFERENCED rather than forked — and `apply_overrides`, which requires an operator to say WHY they deviate and marks the result so reports and the wizard show the departure; [consts] a declaration stating `combine: slower` is one stage of a series link instead (a backing store's channel beside the chip's own programming port) and composes as the slower stage, corner by corner, so a store never speeds up a chip's declared port. `conversion.py`: the target's DATAFLOW, for the one quantity no record counts — an analog accelerator's cost is dominated by data conversion, but the conversion COUNT is a function of the target's geometry, converter sharing and input bit-slicing rather than of the workload, so the target declares a named model and the model derives `adc_conversions`/`adc_count` from quantities the record already carries. `digital` (the default) yields ZERO, which is a fact about a digital chip rather than an absence; `bit_sliced_crossbar` is the ISAAC/PRIME dataflow, where a partial column still costs a whole integration (the division rounds UP, since rounding down would price a read the chip performs at nothing). Derived counts are MODELED, never measured, and a view's own quantity always wins over a modeled one — a model may only ADD what the record could not know. The declaration is validated at LOAD time, so a dataflow the model cannot describe fails when the profile is read rather than as a conversion count nobody can defend at pricing time. `registry.py`: profiles discovered as data from `profiles/`, resolved via `__file__` so the cwd never matters; a vendor adds a target by dropping in `<name>.json` + `<name>.md` with no `src/` edit, and a profile whose description file is missing refuses to load. Its `profiles/` subdirectory is DATA, not code: one `<name>.json` of declared constants plus the `<name>.md` description file carrying citations, per-constant derivations, estimate rationales and the validity domain in prose. Eleven ship: [U-E4 A-1] the co-design study's own three tiles `study_compass`, `study_vr_like` and `study_isaac_like` (the study's estimator mapped one to one: its nangate45 probe tile area, its SRAM-DDR row's MAC energy, DDR3/4 backing store, write port and clock; each quantity the study does not price declared from a sourced band, never zero, and named in the profile's notes), so the toolchain reprices the chips the study priced; the three page-sourced Wave 1 profiles `hermes_64` (IBM HERMES PCM, measured MVM, a write time derived from the published programming sequence; [U-C3] its write energy refuses by name, the profile naming the missing quantity, the programming-rail voltage its core's paper prints only as "beyond 2.5 V"), `sram_cim_macro` (the 16 nm digital SRAM-CIM macro, macro-level constants and the cycle time of one core evaluation, read with a declared backing store) and `neurram` (130 nm RRAM, write-verify parameters and MVM time and energy page-sourced at one operating point; [U-C3] its write energy per pulse into one RRAM device a derived band, the chip printing no write current, from the paper's word-line capacitance and the code release's gate biases, amplitude ceiling and verify voltage), and the original five: `truenorth` (silicon, the reference, characterized at the 0.8 V corner where Akopyan's Fig. 17 firing-rate sweep lets the MARGINAL per-event energy be separated from static power by regression), `loihi` (SIMULATION — Davies 2018 Table 2 is explicitly pre-silicon SDF+SPICE, and the profile declares no `t_cycle` because Loihi is asynchronous with no fixed timestep, so it honestly cannot back latency, throughput, or the static energy that accrues over a wall it cannot price), `odin` (silicon, 28 nm FDSOI single core — the target whose paper publishes the affine power model outright, `P = P_leak + P_idle x f_clk + E_SOP x r_SOP`, and the reason the library stopped pricing from operating-point averages), `isaac_like` (SIMULATION, analog), and `generic_estimated_22nm` (every constant `estimated`, named so nobody mistakes it for a chip, existing so a first-cut study can run and so the estimated-evidence path is exercised by something unmistakable). `truenorth` is the reference profile — every constant published or derived-with-arithmetic-shown from Merolla 2014 (incl. Supplementary S1–S13) and Akopyan 2015, all measured on silicon, with the research pass in `docs/research/physics/`. Zeros are declared FACTS there (a fully digital chip has no ADC and no analog programming spread), distinct from absent constants, which mean "this target has not said". [TS6] The `programming` group is now populated on ALL FIVE profiles, by DERIVATION rather than by measurement — no paper measures a weight load, and refusing every programming term by name made every target unusable for a study that reprograms. `e_program_per_byte` is a new vocabulary row and a DISTINCT cost from `e_dma_per_byte`: the DMA constant is what MOVING a byte onto the chip costs (and a carried activation pays it per inference, on `carry_in_bytes`), while the program constant is what COMMITTING it to the weight store costs, charged on `reprogrammed_bytes` alone — one number for both would price an NVM program-verify write into every inference's carry, three orders of magnitude wrong. Two derivation families, each written into every constant's own `derivation`: RATIO-anchored (`e_program_per_byte` = [0.5x, 3x] the profile's own per-byte access anchor for the SRAM class, [10x, 100x] the SRAM class for the NVM one; `e_core_program` = [0.5x, 2x] `e_core_init` where declared, else the core-state sweep the geometry defines) and DONOR-scaled (`t_program_per_byte` for the three targets that publish no interface rate, from truenorth + generic along `validity.technology_node_nm`, band = the donors' envelope). A ratio band's nominal is its LOG-space centre, because the centre of a multiplicative spread is geometric. The isaac_like note once stated that ENDURANCE IS UNMODELED; since Wave 1 the write law counts it (`endurance_cycles`, below). |
| `units.py` | The dimensional-unit SSOT, above BOTH halves of the cost model because a constant and the quantity it multiplies must agree on what a unit means. A `Unit` carries a DIMENSION (energy/area/time/power/data/dimensionless) and an SI scale, so a vendor may declare in whatever unit they like (23600 fJ and 23.6 pJ are one constant) while a wrong-DIMENSION declaration fails loud; dimensionless symbols (`bit`, `levels`, `fraction`) are display labels that never rescale, which keeps a bit WIDTH from ever being converted like a DATA payload. It lives here rather than inside `platform_physics/` so that `quantities/` can import it without dragging the physics loader in behind it — the cycle that import order used to hide. |
| `quantities/` | The quantity catalog (C0): every priceable multiplicand as data — `spec.py` declares `QuantitySpec{key, dimension, unit, doc}` rows (the closed catalog the physics vocabulary's `multiplicand` column now references, pinned by test, so a pricing formula can never name a number nobody produces) plus `QuantityValue{value, provenance ∈ measured/static/modeled}` and the `Quantities` mapping, where ABSENCE is meaningful throughout (`get` raises; no default). The catalog keeps the two MAC censuses distinct on purpose: `macs` is the AS-MAPPED site count (replicas really fire — the energy multiplicand) while `total/onchip/host_macs` is the LOGICAL replication-free split census. `from_record.py` extracts the measured plane of a sealed record (reprogramming quantities count REPROGRAM passes only — a resident pass sends nothing; `noc_total_hops` derives from the sealed link loads rather than being re-sealed; `synaptic_events` is deliberately unclaimed — `total_spikes` counts emissions, not synapse arrivals). `from_level.py` is the level plane of a capability level's requirement placement on a declared chip (CL4, below). `from_candidate.py` extracts the static facts a search candidate can honestly answer (`CandidateQuantityContext` carries the run declarations the layout does not; `synaptic_events` is MODELED from the declared activity factor — the EDA switching-activity discipline — and says so). `probe.py` is the all-keys strictly-positive placeholder for capability questions only. |
| `correlation/` | SILICON CORRELATION: does a declared profile reproduce the measurements its own papers publish? The acceptance a physics profile must clear before it ships, and the thing that separates a characterization from a guess with citations. `case.py`: `ReferenceCase` — a published operating point as DATA, carrying the number, its citation AND the sentence it appears in, the workload census the paper's own structural statements determine, a per-entry `census_sources` map (a census number with no stated source is a fitted parameter in disguise, so construction refuses it), and an `independence` flag separating a genuine prediction from a row that re-derives the point its own constant was divided out of. `run.py`: prices the census with the named profile and scores each axis; an axis the physics REFUSES scores as a total miss, never as agreement, and a case may restate its operating point only by overriding a constant the profile already declares — never by introducing one. `derived.py`: the axes a paper prints that the pricer expresses as a ratio of two priced terms (`average_power_mw` = energy / latency), kept here so the four headline axes every registry and study table is keyed on stay unchanged. `library.py`/`report.py`: the shipped cases and the correlation table. `cases/` is DATA: seven cases over three devices, worst error 14.5%, all inside ±25%. [TS6] `report.correlation_payload`/`correlation_payload_json` is the ONE home of the JSON shape — `scripts/silicon_correlation.py --json` writes exactly those bytes and `tests/unit/deployment_record/correlation/reference_case_predictions_golden.json` pins them, so a physics edit that was supposed to touch nothing must prove it BYTE-identically (the band rides with the nominal: a moved band is a changed prediction). No shipped case reprograms, which is what makes that golden an invariant rather than a coincidence — asserted, not assumed. |
| `study.py` | [C7] The cross-platform comparison: ONE workload census priced by many declared targets — the question a chip designer actually asks. Because the pricer is one pure function of (quantities, physics), this is a fan-out over profiles rather than a second cost model. Two disciplines it ENFORCES rather than documents: an axis a target cannot back is ABSENT with its reason, never zero — on a minimized axis a zero would read as the best possible result, so Loihi's missing `t_cycle` must not make it the fastest chip in the table; and every row discloses its evidence (measurement kind plus a per-kind constant count), with a table mixing bases flagged explicitly, because a silicon measurement beside a pre-silicon simulation is a legitimate comparison and an illegitimate one to report silently. `scripts/cross_platform_study.py` runs it over a sealed record. |
| `fidelity.py` | [C4] The estimate-vs-signoff correlation, the EDA discipline this whole program is shaped by: what the SEARCH predicted against what the RUN measured, axis by axis. One pricer serves both completenesses, so the same objective keys carry the same term names on both sides and this is a ZIP rather than a second model. `in_band` is a verdict only where a BAND was predicted — a point prediction is not a claim about a range, and reporting one as out-of-band would invent a tolerance nobody declared — and relative error is measured AGAINST the measurement, since the measurement is the reference and the prediction is what was wrong. Either side may be absent and both cases are recorded: an axis only the record answers is what the search could not see. REPORT-FIRST by decision: it gates nothing numeric, because the tolerances are a decision to make ON this evidence rather than before it; the STRUCTURAL equality (the searched pass count == the deployed one) is gated where it belongs, in the mapping tests. |
| `fidelity_build.py` | [C4] Builds a report from a sealed record plus the candidate view of its own configuration — one registry read per side over the catalog's own axis order, and the predicted BAND recovered from the candidate's own priced term. `emit_fidelity_report` writes `fidelity.json` only when there is a prediction to compare against: an unsearched run has nothing to correlate, which is different from a fidelity of zero and is written as nothing at all. `scripts/fidelity_summary.py` aggregates the artifacts across run dirs. |
| `schema/` | The fragment schemas: `provenance.py` (`Provenance`, `Band`, `ModeledValue`), `schedule.py` (`ScheduleRecord` + segment/compute-op stages, plus the additive-optional `PassCarryRecord` — what crossed the INTRA-SEGMENT pass boundaries and under which discipline. A pass boundary is one the schedule INTRODUCES, so the intermediate signal is buffered either way and `transfer` says WHAT: `verbatim` keeps the spike raster and reproduces the fused execution bit-for-bit, `collapse` keeps window counts and re-emits an even train, which is cheaper and normalizes rhythm. They are DIFFERENT computations, so a program that carries but was handed no discipline RAISES rather than sealing a None; a program with no intra-segment boundary seals None because nothing crossed one), `placement.py` (softcores, banks, floorplan, tiles), `utilization.py` (typed mirrors of `CrossbarUtilizationReport.to_dict()` and `LayoutVerificationStats`, drift-guarded by mirror-fidelity tests, plus the additive-optional `ComputePartitionRecord` — the LOGICAL on-chip/host params+MACs split of §8b item 4, sealed on every run; pre-partition records load with `partition=None`, no format-version bump), `traffic.py` (gate-reduced boundary totals + SANA-FE NoC), `physics.py` (timing + energy, banded modeled terms), `accuracy.py` (reads, certificates, adaptation walls, plus the additive-optional [TS5] `AdaptationSummaryRecord` — what the adaptation controllers spent, TOTALS ONLY, with `stalls_by_path` and a per-step `completed_via` naming the path each rate search exited through; the per-event trace stays in the steps' own `<Step>.adaptation_ledger.json` artifacts, which `pipelining`'s `deployment_record_adaptation` folds), `record.py` (`RecordIdentity`, `DeploymentRecord` — whose `adaptation_ledger` field is LAST and defaults to None, the `invocations` precedent: records written before it existed load unchanged, no format-version bump —, `DEPLOYMENT_RECORD_FORMAT_VERSION`, atomic save/load), `serde.py` (strict unknown-field-rejecting serde helpers). |
| `build/` | `builder.py`: `DeploymentRecordBuilder` — attach-once fragment slots (double-attach raises), `seal(plan, resolved_step_names)` enforcing the spec §5 required-fragment matrix + cross-checks (pass count vs layout stats, Σ params_programmed vs the weight-programming report total, cores agree, one provenance entry per attached group). `SealPlanView` is the typed duck surface for plan predicates so this module never imports `pipelining`. `payload_sizes.py`: the sizing SSOT — `params_bytes = ceil(cells_used × weight_bits / 8)` (undeclared `weight_bits` FAILS LOUD, the documented None choice) and `core_connectivity_entries` (reads `HardCore.get_axon_source_spans()`, the same span SSOT behind the codegen export) — and the candidate's one programming census, `candidate_programming_census` with its type `CandidateProgramming` (one definition, re-exported by `program_census`; see "Physical-tile core counts" below). `from_mapping.py`: pure-read converters from a live `HybridHardCoreMapping` — `schedule_record_from_mapping` (stages mirrored 1:1; per-segment programming via `weight_programming_report` on a single-stage view; the pass census formula is exactly `LayoutPlan.from_hybrid_mapping`'s so the seal cross-check holds by construction), `placement_record_from_mapping` (every placement dict key + banks with sharing degree; floorplan/tiles empty until SANA-FE joins), `utilization_record_from_mapping` (typed mirrors of the step's `CrossbarUtilizationReport` and `stats_dict_from_hybrid_mapping`, plus the threaded relay count). `partition.py`: `compute_partition_record` — the ungated on-chip/host census, read from the SAME SSOT pair the validity gate uses (`compute_onchip_fraction` over the mapped IR for params, `estimate_onchip_fraction(metric="macs")` over the model for MACs), so the sealed fact and the gate can never disagree. `from_certificates.py`: `boundary_traffic_from_node_counts` — the eager `{node_id: (B, n)}` → totals/maxima reduction the spike-count gate calls in-scope (raw tensors never persist). `pass_carry.py`: `carry_record_from_mapping` — seals the intra-segment pass-boundary census as a `PassCarryRecord` under the RUN's transfer discipline, None when nothing crosses one, and RAISES when the program carries but the discipline or timestep count was not stated (sealing None there would drop the fact that verbatim and collapse are different computations); beside it the candidate's carry census (`candidate_carry_census`) and each wavefront group's on-chip carry (`wavefront_carry`); it holds no programming census. `from_simulators.py` (W4.4): pure reads of the `SanafeStepReport.to_snapshot_dict()` shape — energy aggregate + per-plane breakdown terms, per-segment measured timing (sample 0, `sim_time_s` includes NoC hops), NoC totals + link loads (packets summed over samples; static censuses over sample 0 only), the resolved floorplan (mesh dims already fold `floorplan_replicas`) and tile census; the scalar formulas are IMPORTED from `chip_simulation.cost_extraction` so §6 continuity is structural, never re-derived. [closure fix round, CL-SG] `payload_sizes.core_payload_bytes(cells, weight_bits, sign_rows)` is the one payload rule: a core's cells at the weight width plus its per-row sign register, one bit per used row of a per-axon core rounded up per core (plan section 21 item 11), read by `program_census.core_record` (which seals `SegmentCoreRecord.sign_register_rows`, declared-only, so a per-synapse record keeps its bytes), `candidate_programming_census`, the three `PassRow` builders (`PassRow.core_sign_rows`, priced by `period_census`) and the scaling ladder's write-once payload (`pass_program_bytes`) [N2] `producer_census.py`: the payload cells each pass's operand regions take per inference on the mapping and candidate planes (declared pass-major model: a region is written in every pass that holds it; each streamed instance holds its whole bank, `placed_operand_cells`); `PassRow.producer_cells` fills `PassCensus.written_cells_producer`, never `core_cells`. The period census's one slot rule places operand regions as it places static tiles (`mapping/support/core_slots.place`, read by `write_census.period_census`: R3 slots when the program rewrites, one core each where it keeps them), and `WriteCensus.producer_slots` gives each such core's static writes per period and producer writes per inference, so `write_law.cell_wear` and `max_writes_per_cell_per_inference` read the exact wear max(s / K + p), never a band divided by K. `ProgramPasses.operands_unstated` names a plane that cannot state operand regions (the sealed record's schema; the search's candidate path, which lowers no product), and `plane_writes` refuses its write terms by name under an operand core type. `write_law.py` reads the operand core type's write source and payload width; `cost/absolute/writes.py` adds the host-sourced payload transfer at `e_dma_per_byte` and counts producer write steps per pass. |
| `objectives/` | [N0] `context.py`: `candidate_context_from_platform(pcfg, ...)` — the ONE builder of a candidate's `CandidateQuantityContext` from its own RESOLVED platform (`simulation_steps`, `activity_factor` with 0-as-undeclared, `weight_bits`, resolved-floorplan tiles + `cores_per_tile`/`tile_mesh_height`, physical core/neuron/axon capacities, plus the caller-supplied host/on-chip census), so the axis gate and the view extraction answer from one source; the search's layout hook attaches it (with `physics` off `pcfg["platform_physics_resolved"]`) to every live candidate view — before N0 the live path attached neither fragment and every admitted priced axis raised at extraction. [N3] `CandidateStaticView` gained the `noc_fragments` fragment (duck-typed `LayoutNocFragments`; the wireload estimate is DERIVED via the SANA-FE estimator when the fragments and the declared activity/window/floorplan allow one, memoized in `_derived`), `DeploymentRecordView` gained `quantities` (from_record — measured), `extractors.quantity_field(key)` reads a produced quantity at EITHER completeness (same key, so fidelity zips modeled vs measured automatically), and the catalog appends `noc_total_hops` LAST (min, no physics needed — hops are a count). `probe_without("layout")` also drops `noc_fragments` (they are layout-derived, so a NoC axis counts as needing the layout); `run_capability_probe(..., activity_factor=)` narrows spike-dependent modeled axes the way physics narrows priced ones, `registry.requires_activity(key)` asks the probes, and the wizard rows carry `requires_activity` beside `requires_physics` so an activity-less draft is never offered a chip whose run refuses at resolution. [C2] `probes.py` owns the capability questions, at two levels: `candidate_capability_probe(mode)` (what a candidate CAN carry in this mode) and `run_capability_probe(mode, physics)` (that, narrowed by what THIS run declared). `CandidateStaticView` gained the `physics` and `quantity_context` FRAGMENTS — the quantities they price are DERIVED, never stored, because a pre-baked quantity set would let `candidate_probe_without('layout')` still answer an energy axis and the run would then die at extraction. Both views' `cost_report()` now carry the vendor-priced plane (a candidate's is priced terms alone; a record's is the measured report with the priced terms appended), which is what makes `chip_area_mm2` / `energy_per_inference_mj` / `e2e_latency_s` / `throughput_inferences_s` answerable at BOTH completenesses through one Backing. `cost_term` is therefore TOLERANT of a missing term: a partial report is a legitimate answer (this physics refused this axis), and availability==extraction must never raise. `registry.requires_physics(key)` asks the probes rather than a list, and `resolve_active(..., probe=)` lets a RUN refuse a priced axis by name while the parameterless form stays capability-level (the wizard offers the whole catalog). The priced axes register LAST, so the legacy-eight prefix every optimizer's objective vector depends on is byte-unchanged, and the per-mode DEFAULT sets are untouched — the new axes are opt-in by name. The objectives registry v2 (spec §8) — the ONE catalogue of optimization axes, superseding `search/results.py::ALL_OBJECTIVES` (which is now a projection of it). `spec.py`: `ObjectiveSpecV2{key, direction, unit, provenance ∈ measured\|modeled\|static\|training_proxy, requires, availability, extractor, doc}` with the legacy `.name`/`.goal` properties every optimizer reads, plus the `RecordView` protocol (what an objective may read; an accessor is `None` exactly when its datum is absent) and the `LayoutStatsView` duck surface both `LayoutVerificationStats` and its record mirror satisfy. `views.py`: the two completenesses — `DeploymentRecordView` (a sealed record; the cost surface evaluated ONCE and cached) and `CandidateStaticView` (exactly what the joint layout hook computes: layout stats + chip capacity + total params + the host-segment census + an optional accuracy estimate), the `chip_param_capacity` formula SSOT (the layout hook calls it), and the candidate probes that turn capability into an availability question instead of a hand-maintained list: `candidate_capability_probe(mode)` (the maximally-populated candidate of a search mode — "which axes can this mode carry") and `candidate_probe_without(fragment)` over `CANDIDATE_FRAGMENTS` (the same candidate MINUS one fact — "which axes need this fact", which is how the search decides whether a candidate's layout is worth computing). `extractors.py`: `Backing` — one reader per axis that returns the value or `None`, so availability and extraction are the SAME function and can never disagree. `registry.py`: `ObjectiveRegistry` (duplicate key RAISES; `available_for(view)`, `for_search_mode(mode)`, `search_catalog()`, `modes_available(key)`, `extract(view)`, and `resolve_active(mode, names)` which FAILS LOUD on an unknown, repeated, or mode-unavailable name — the search surface's silent drop is gone). `catalog.py`: the contract order — the legacy 8 re-registered byte-equal (keys, directions, ORDER) followed by the §8 record axes (`deployed_accuracy`, `mj_per_sample`, `latency_steps`, `host_op_wall_s`, `total_spikes`, `pass_count`, `reprogram_passes`, `reprogramming_bytes`, `params_reloaded`, `noc_*_packets`, `programming_energy_mj`, `sync_barrier_energy_mj`, `throughput_samples_per_s`), the modeled ones reading `cost/`'s terms rather than re-deriving them. |
| `introspection/` | The introspection channel (spec W5.2): typed, VERSIONED payloads over the same two completenesses the objectives registry uses. `payloads.py`: frozen payload/row types carrying their own `payload`/`payload_version`, refusing to load under another name or version (the `CostRecord.from_dict` discipline) — `SoftcoresPayload` (shape + `bank_id` + `perceptron_index`), `LayerRollupPayload` (keyed by REAL `perceptron_index`, the core NAME only a label), `BankCompositionPayload` (sharing degree; real bank sizes on a sealed record), `PlacementPayload` (softcore→hard core, sealed only), `SchedulePayload` (pass structure under the DECLARED `schedule_policy`), `CapabilitiesPayload` (every declared capability bit, geometry INCLUDED -- `max_axons`/`max_neurons` as the EFFECTIVE per-core limits and `hardware_bias`, resolved from the declared grid by the platform SSOT, so a served `null` means "no grid declared" rather than "unread"), `LayoutStatsPayload` (typed through `LayoutStatsRecord`). `views.py`: `CandidateLayoutView` — `packed(...)` for a consumer that ALREADY holds a packing (the compilagent backend, feeding it the search problem's own `candidate_layout` census, so the channel never re-answers what the producer answered) and `from_platform(...)` for one that holds only (softcores, platform), which is then the ONE place that turns them into a layout answer and forwards the capability declaration WHOLE via `layout_kwargs()` — and `RecordIntrospectionView` over a sealed record. `builders.py`: the rollups as pure functions of ROWS, so candidate specs and sealed placements aggregate through ONE implementation. `registry.py` + `catalog.py`: register/serve/availability (duplicate name RAISES; serving an unavailable payload raises naming what it requires); `channel_envelope` stamps `INTROSPECTION_FORMAT_VERSION` -- the ENVELOPE convention, distinct from each payload's own version -- around anything PERSISTED (the compilagent `introspection.json` artifact). Optimizers depend on these types and nothing else — an AST test pins that no compilagent module imports `mapping` |
| `design_point/` | [CL7] Design-point instances: one network deployed on several chip rows, each row certified, censused, timed, judged and priced, or refused by name, as one closed document. `schema_types.py`: a small closed schema language (scalars, enums, nullable, lists, string-keyed maps, objects that refuse unknown fields) with path-named errors and JSON Schema export. `schema.py`: the instance document (`SCHEMA_ID`, version 1) and `validate_instance` (closed fields, then row rules: a judged row carries every judged field, a refused row names a refusal, the energy split names dynamic, static, static write, operand write and host, every census executor is declared); workload-free, so a convolutional network fills the static fields and a dynamic-operand block (attention: operand cores, operand writes per inference, host softmax) fills the operand fields of the same schema. `placement.py`: per operator (a perceptron's IR cores, or one host op) its place, role (static; operand where its weight bank is one of the program's `dynamic_bank_bindings`; host; wiring), the requirement table's reason (`mapping/capability`) and the passes or host stage it runs in. `executors.py`: `certify_program`, the atol-0 edges of the executed program on two executors (the fp64 `ValueHybridCoreFlow` and the int64 golden, `run_golden` keeping its node outputs) against the converted model and the identity mapping, and the per-operator census of nonzero outputs per executor and mismatches between them (not vacuous when every observed operator has nonzero outputs on both). `timeline.py`: the record plane's period census of the built program (`plane_write_census` over `mapping_program_passes`) as a stage-ordered timeline with write units and events per pass at the declared granularity, operand writes per inference, and the period's write summary. `endurance.py`: lifetime at a batch from `price_write_terms`, and K* per endurance corner as the smallest batch the pricer admits (doubling then bisection), each boundary priced on both sides, with the closed form and `min_batch` beside. `chip_rows.py`: a row's core types (role, geometry, write technology) and its refusals by name (`per_core_type_write_technology` when the core types name several write technologies, the write law pricing one physics per chip; `operand_core_type` while the registry's core-type schema declares no `operand` field). `pricing.py`: a scaling-ladder priced point read as the row's priced variants (exploration plane): the headlines at the variant's batch K from the write law (writes of a period over K and static power over the longer latency included; refused by name where the write law refuses), the steady headlines without writes beside, and the split into the headline's own parts (`scaling_ladder_reading.ENERGY_HEADLINE_PARTS`) with a closure check; and the facts the executed census compares with it. `rows.py`: `judged_row`, the judged fields and the admission verdict at the declared batch (a certificate mismatch or a refused lifetime refuses by name), and `plane_agreement`, the executed census against the priced program's field by field. [fix_b] `wear.py`: wear per declared core type (role, write technology, cells written, the most-written cell's writes per inference, endurance band, hours, verdict); a row's hours are the corner-wise minimum and its binding type is named, and `validate_instance` recomputes every entry from endurance, wear and rate, refuses a row of several write technologies without its per-type wear, and refuses a judged row whose planes disagree on an unnamed field. Driver: `scripts/design_point_instance.py` (one row per run, then assembly and validation). |
| `cost/` | `absolute/` (C1) is the VENDOR-PRICED plane, a second engine beside the default-band one: `price_absolute(quantities, physics)` multiplies the target's declared constants by the catalog quantities a view produces, so ONE pure function prices a search candidate's static shape and a sealed record's measurements identically. `context.py` holds what every metric shares — the `PricingContext` (supersession resolved ONCE at construction, so an aggregate absorbs its components and nothing is charged twice), term/refusal accumulation, the component-table walk that also reports work the target left UNPRICED, and `HostTime`, the one host rule energy and latency share (a measured wall over `host_compute_rate`, else the declared `host_macs_per_s`; host work that cannot be priced REFUSES the headline rather than omitting it, because unpriced host work makes `subsume` look free). `metrics.py` prices area (the published per-core footprint when there is one, else the component decomposition + state storage), steady-state e2e latency (`t_cycle` x `latency_steps` + sync + core init + host — `t_hop` deliberately NOT added, since timestep-synchronous execution already contains hop time, the same no-double-count discipline the measured `sim_time_s` carries; programming is its own amortized term, never the steady-state headline) and throughput (inverted latency, corners flipping). `energy.py` prices the compute aggregate or per-MAC constant + periphery components + static power over the priced latency + host. `formulas.py` is the band algebra (elementwise for products and sums; corners FLIP on inversion and division) plus the component tables as data — including [TS6] `PROGRAMMING_BYTE_ENERGIES`, the per-byte program-load costs in the order a byte incurs them (`e_dma_per_byte` moves it onto the chip, `e_program_per_byte` commits it to the weight store): both multiply the SAME payload bytes into the amortized `energy_programming_mj`, a target declaring both pays both, and a target that published only one prices only that. `report.py` folds the priced terms into the EXISTING report groups so candidate and record reports carry the same term NAMES by construction (what the fidelity contract zips), appends every refusal to the notes, and `emit_physics_report` writes `deployment_cost_report.json` iff the record declared physics — a no-profile run is byte-identical, pinned by the step test. Priced terms are never summed into the measured totals; the report says so. `coefficients.py` (W4.5): the §7 coefficient inventory — the DMA/sync-barrier energy bands are IMPORTED from `chip_simulation/weight_reuse_cost_model.py` (`DEFAULT_COEFFICIENT_BAND`, never copied numbers) and applied through ITS `phase_cost_model` (`dma_energy_mj` = per-byte channel, `sync_energy_mj` = barrier channel); the stage's new coefficients carry their written basis in the `Band` itself — `CoreInitCoefficients` (per-core reset energy+time, DERIVED at import from the SANA-FE per-event soma-reset presets over a 256–1024-neuron reference core), `BYTES_PER_CONNECTIVITY_ENTRY` (4/8/16 B), `PROGRAMMING_BANDWIDTH_BYTES_PER_S`, `SYNC_BARRIER_S`; `banded()` evaluates any corner-indexed function into a `Band`, `opposite_corner` keeps DIVIDING coefficients monotone. `terms.py` (W4.5): `CostTerm{name, unit, value, band, kind, source}` with the epistemic invariants enforced at construction (measured ⇒ no band, modeled ⇒ band required, band must bracket the value), `SegmentInitCost`, and the JSON-safe, version-checked `DeploymentCostReport`. `combine.py` (W4.5): the term constructors + corner-wise band algebra. `segments.py` (W4.5): per-segment initialization cost — per-core reset constant × `len(segment.cores)` plus the programming payload (`params_bytes` + `connectivity_entries` × modeled byte width) through the DMA channel and the bandwidth band; a `resident` segment's payload is identically zero. `model.py` (W4.5): `DeploymentCostModel` — `segment_initialization` / `energy` (measured passthrough + modeled programming/core-init/sync) / `latency` (modeled programming, core-init, sync + measured compute and host-op walls; REFUSES a record whose `timing.latency.note` does not state the no-NoC-double-count discipline) / `area` (utilization census + its derivations) / `throughput` (inverted latency band) / `evaluate` → `DeploymentCostReport`. The record's OWN modeled terms (`energy.breakdown[kind="modeled"]`, `timing.latency.programming_s`/`sync_s`) are re-derived here, never passed through, so nothing is double-counted. `legacy_projection.py` (W4.4): `cost_record_from_deployment_record` — THE §6 continuity contract: projects a sealed record onto a format-v3 `CostRecord` whose live fields are value-identical to `extract_cost_record` on the same SANA-FE snapshot (golden-fixture-pinned), with the formerly-dead fields (`reprogram_passes`, `reuse_passes`, `params_reloaded`, `max_ft_pass_wall_s`, `ft_pass_walls`) now live from the schedule/adaptation fragments; `activation_bytes_moved` stays 0 (no producer exists — stated disposition). Refuses records without the SANA-FE fragments: exactly the runs that never wrote `cost_record.json`. |

## Value NoC census and the candidate carry census (Wave 2 closure)

From the closure unit's step 1 (KA-G3-1, IE-1, IE-3). `schema/schedule.py`: `ScheduleRecord.value_noc_wire_hops`
(declared-only) is the value-domain build's NoC wire census in value-hops on the resolved floorplan; a record without
it keeps its bytes (every archived record). `quantities/value_domain.py`: `record_value_noc_wire_hops(hybrid_mapping,
platform)` is the build's fill (value domain and a resolved floorplan, else None) and `value_noc_reading(hops, bits)`
the one reading of a value NoC census on both planes (byte-hops at `activation_bits`; zero traffic is zero without the
grid; other traffic is withheld by the grid's name). `record_value_facts` reads the sealed census as `noc_value_bytes`;
a record that seals none keeps `NO_RECORD_WIRES`, and the candidate view (`objectives/views.py`) reads the same rule.
`build/from_mapping.schedule_record_from_mapping(value_noc_wire_hops=)` takes it, and the hard-core mapping step's
emission passes `record_value_noc_wire_hops` (pipelining). `build/program_census.py`: the candidate's carry census and
period-census rows read the fragments' segmentation (`mapping/noc/fragments.spec_segments(softcores, noc.runs)`) and
the delayed slices (`delayed_carry_wires`); the built program's boundary widths count `delayed_input_slices`, so both
planes' carry census and widths are equal on the deployed program (`test_integration_edits.py`). The candidate carry
census itself lives in `build/pass_carry.py` (`candidate_carry_census`, `boundary_widths_of`; moved there at the pack
integration, re-exported from `program_census`). `introspection/views.py`: `CandidateLayoutView.from_platform(runs=)`.

## Physical-tile core counts and the per-tile period census (Wave 2 pack)

From the pack unit's fix round and its integration (fragments `11_wave2/pack/arch_fragments/deployment_record.md` and
`11_wave2/integration/pack/arch_fragments.md`). `build/program_census.py`: every core count of both planes is of
physical tiles (a fused core counts its members). `candidate_programming_census` (`build/payload_sizes.py`, the one
definition; `program_census` re-exports it) reads one tile accessor, per core `PassProgram.core_tiles` (one tile per
core on a program that states none), for `segment_cores`, `reprogrammed_cores` and `pass_cores` (the payload stays
sized per allocated crossbar, as `SegmentCoreRecord.params_bytes` is). [U-F4, U-N] `segment_cores` and
`committed_cells` count each pass's evaluated cores (`PassProgram.evaluated()`: a wavefront head's dark cores are loaded,
never evaluated; every core when none is dark), as `from_record` leaves a sealed segment's `dark_core_indices` out of
`segment_cores` and `cells_used`; the written cores, bytes and cells and `pass_cores` keep the dark cores, and the
candidate plane states `sum(pass_cores)` under `cores_allocated`, the record's allocated crossbars with the dark cores
(`quantities/from_candidate.py`, `objectives/context.py`). `candidate_program_passes`, `mapping_program_passes` and
`record_program_passes` state one `PassRow` entry per physical tile, its cells, extent, type and residency mark
(`mapping/packing/residency_tiles.expand_to_tiles`), with the closure's sign rows split over a fused core's members as
its used rows fill them (`residency_tiles.tile_rows`; on the candidate plane `PassProgram.core_tile_sign_rows`) and N2's
operand cells on a core's one tile (`build/producer_census.operand_cells_per_tile`, which refuses by name a fused core
holding operand regions). A sealed core reads its tiles off its geometry against the declared core types
(`declared_core_geometries`, `residency_tiles.declared_tiles`). `quantities/from_record.py`: `segment_cores`,
`reprogrammed_cores` and `cores_allocated` count physical tiles; a sealed geometry the declared types do not determine
withholds the three by name. The sealed fields (`utilization.crossbar.cores_allocated`, `utilization.layout`) keep their
meaning. Boundary: a fused core whose summed geometry equals another declared type is read as one core of that type
on the sealed record, until the record seals each core's members (`SegmentCoreRecord`).

## Scaling ladder (Wave 2 exploration: CL3, W03, CL5, CL7)

From the exploration unit's ladder fragments (`11_wave2/explore/{cl3,w03,cl5,cl7}/arch_fragments/`), merged at CL9 of
integration pass 2 and restated on the integrated tree. `introspection/scaling_ladder.py` is free of workloads: the
caller hands in a built mapper representation, resolved platform dicts and the candidate plane's program census as a
callable. `shape_census` lays one network out per core geometry and keeps the layout walk's program graph
(`ShapeCensus.program`, `ProgramGraph.of_walk`) and the walk's runs (`ShapeCensus.runs`, the builder segment of each
spec) beside its wire census. [U-AX] On a per-axon pool the walk holds the network's weights, so it reads each core's
rows on the declared weight grid (`mapping/layout/ir_mapping/sign_rows.py`: 1 + mixed per row, the build's rule) and
the census and every row state them (`physical_rows`: deployed, or two per input, an upper bound named with its
cause); `design_point.pricing.candidate_facts` carries the statement and `rows.plane_agreement` names every field an
upper-bound candidate disagrees on by that cause, so plane parity compares only planes that read the same rows.
`schedule_program` packs and schedules one core budget with the builder's planner: U1 on
the walk's runs, or, when the platform declares `residency_plan`, the residency composition on the program graph,
so a declared step is planned as the builder plans it (tile-granular units, the declared step, its pins and its
period census); a program whose passes fit the declared cores side by side carries the planner's fit pins
(`residency_rules.fit_pins`), so at C >= F the ladder reads 0 rewrite bytes per period, as the record does. It
returns a `ScheduledProgram` (or a `LadderScheduleRefusal` by name). `compute_mapping_stats` plans a declared
`residency_plan` only when it holds the walk's cut census and program, and refuses the key by name without them
(`refuse_without_cut_census`). `price_scheduled` prices a scheduled program on a platform that may differ only in its
physics and the row keys (batch K, rate, lifetime); anything else refuses by name (`LadderDeclarationError`), so one
schedule is priced under several override-block sets (host, backing store) and batch sizes without scheduling again;
`price_program` composes the two. Write-once costs count physical tiles (`PassProgram.tiles`, the pack unit's rule)
and the closure's sign-register bytes (`pass_program_bytes`). At a declared `inference_batch_K` the candidate context
carries the pass rows, so the value rule's latency window and G3's write law are priced at that batch (write energy,
exposed write time, lifetime and its admissibility verdict, the headlines at K). The ladder's one write source is G3's
candidate-plane period census (`scaling_ladder_reading.period_writes`: `plane_write_census` over the program facts'
passes, then `write_quantities` at K = 1 for `max_writes_per_cell_per_inference`, the K*_census multiplicand, and
`period_rewrite_bytes`); the passes it rewrites decide `multiplexed` (regime reading, the period writes in M, the
schedule reading), and the ladder's own write rule is removed. [consts] Pi's numerator and M's period writes are the
same write law's (`scaling_ladder_reading.tile_writes`, K = 1, serial, producer writes left out): Pi prices the
program's deployment load (`period_census(every_core=True)`: every core of every pass written once from an empty
chip), M the program's own period census, so a commit energy, the per-core `e_core_program` and a write-verify wall
the law prices are Pi's, and where the law refuses writing every tile once (a chip that declares no commit energy) the
axis is absent by the
law's reason rather than a transfer-only number. `ladder_row` reads one priced program at one K;
`scaling_ladder_reading` adds the reading columns: MACs per weight R (on-chip MACs over the stationary weight cells,
`stationary_weight_cells`), K x R and Pi x R per axis; the two-ceiling minimum batch (`min_batch`: endurance m r T / E
from the census beside the closed forms at x_fluid and x_packed, the write-bandwidth ceiling r t_w, the crossover
endurance, the binding ceiling and the verdict per corner); and the latency semantics (`latency_semantics`).
`write_law_columns` carries the write law's terms at the row's K, among them the energy headline's parts
(`ENERGY_HEADLINE_PARTS`), so `energy_per_inference_mj` at K is their corner-wise sum by `cost/absolute/energy.py`'s
construction and a reader splits it without subtracting; the row's `headlines` stay the steady headlines without
writes. The exploration script declares keys as JSON values (`scripts/scaling_ladder.py --declare`, `declared_values`),
so `--declare 'residency_plan={"step": "floor"}'` explores the floor on every target; a value that is no JSON refuses
by name.

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
present. Stage 5 adds `cost/`'s `DeploymentCostModel` as a pure reader of a
sealed record. Stage 6 adds `objectives/` and gives the cost model its first
caller: `search/results.py` PROJECTS `objectives/`'s search catalogue onto the
legacy `ObjectiveSpec` tuple (byte-equal by test) and delegates resolution to
the registry, `search/problems/joint/layout_hook.py` calls the shared
`chip_param_capacity`, and `gui/wizard/schema.py` serves the per-mode
availability rows (`objective_catalog`) the wizard's objective switch will
consume in W5.3. The import direction is one-way: `search`/`gui` import this
module, never the reverse. Stage W5.2 adds `introspection/`: the compilagent backend re-serves every
introspection tool from `INTROSPECTION_REGISTRY` and no longer imports
`mapping` at all. Later stages: the joint problem consuming the objectives
registry directly (W5.1).

## Write law, value domain and validity (Wave 1, G3)

- **Value-domain quantity rule** (`quantities/value_domain.py`). The resolved platform names the value domain
  (`core_semantics` rides it exactly for MVM runs, S0), and both planes then read the value rule instead of the
  spiking one: `synaptic_events` = committed cells (MAC events, static), `timesteps` = 1, `latency_steps` =
  Σ over execution stages of (L_s + 1), carry at `activation_bits` read off the dense crossing census, no
  `carried_raster_bytes`, and a declared `activity_factor` refused by name (`ValueDomainActivityError`). A carry
  the census shows empty is 0 whatever the grid; a nonzero one without `activation_bits` is WITHHELD
  (`Quantities(withheld=)`: absent with a reason), so `PricingContext.component_bands` names its carry terms
  unpriced instead of dropping them, on the headline and on `energy_dynamic_mj` alike (both per-inference
  energy terms name every component of their sum they cannot price, withheld or undeclared, and so does the
  per-load `energy_programming_mj`: a transfer or commit row the target never declares while the payload is
  positive, and `e_core_program` while cores are reprogrammed); the value NoC
  census follows the same rule. The candidate
  plane applies the rule in `objectives/context.py`, the record plane in `quantities/from_record.py`; archived
  records carry no surface key and keep their pricing.
- **Period census** (`build/write_census.py`, E49). One rule over each plane's passes (`PassRow`), the executed
  rotation (`mapping/support/period_census.rotation_rewrites`, shared with G5's `plan_period_census`): a pass
  keeps the cores the residency seam marks (`resident_core_indices`, an executed residency plan) or U1's
  whole-pass flag keeps; when two or more passes write, each rewrites its other cores once per period; a lone
  writing pass is never displaced and writes nothing. Capacity never makes a program of several writing passes
  loaded once (interaction finding 4, option (a)): every cost is the cost of the program the runner executes,
  and a fit is resident only where the executed plan pins it. [W03] The residency planner pins every fit: under a
  declared plan the floor at F <= C, and without one every core of a program whose passes that load their own
  cores fit the declared pool side by side (`residency_rules.fit_pins`, the floor's pinned rule at r = 0, marked on
  the build by `schedule_residency_plan.mark_fit_residency` and on the candidate plane by `collect_noc_fragments`),
  so a chip that holds the program is programmed once on every plane. [fix_b] U1's passes alone judged that fit,
  and on larger programs they spread over more cores than the pool while the toolchain's unscheduled pack of the
  same program fits it: where U1's passes are not pinned and the unscheduled pack holds on the declared pool, that
  pack is the program, pinned, on both planes (the builder's single shared pool, `packing/hybrid_build_fit.py`;
  the candidate's one pack of every spec, `residency_rules.unscheduled_pack_holds`). [fix_c] The rule judges every
  plan but the stateless reload baseline (`residency_rules.fit_rule_applies`): a declared floor or step index whose
  passes are not pinned whole is the same pack where the declared pool admits it, and the built mapping then carries
  no composition plan (`schedule_residency_plan`: the unit plan the fit replaces). A pass is
  `resident` exactly when it writes nothing per period. Endurance belongs to the physical core, so the census
  also counts each core's writes per period (`SlotWrites`): the runner loads a pass's cores of a type from that
  type's first core not pinned by a residency mark, so the first core takes one write from every pass that
  rewrites a core of its type, and `WriteCensus.most_written_core` is the endurance multiplicand. The physical
  cores live in `mapping/support/core_slots.py` (moved from `build/` by the explore integration, module budget): a stated core is a declared core or a fused one that occupies its
  components (`canonical_fuse_hardcores`, read by G5's `core_type_tiles`); pins the builder fits only with a tile
  on a larger core (its dominance matching, `residency_rules.check_pinned_fit`) state the census with no rewrite
  placed on a physical core (said in the label), and only pins no dominance matching fits refuse by name
  (`WriteCensusError`), as a core that is neither declared nor fused does. Where a plane states no core types on
  a chip of several (the shape-only plane until `PassProgram.core_geometry` lands), the rotation is stated by the
  same rule and no rewritten core is placed (said in the label). A
  rewritten row that states several execution stages (per-hop re-timing, one per depth level, on the candidate
  plane) places no rewritten core, so its lifetime is the pigeonhole bound below. A sealed record states one
  stage per row and no re-timing, so a re-timed record is placed per row (a stated boundary until the record
  seals its stages).
  `WriteCensus.pass_core_types` carries each pass's cores per type. Plane adapters in `build/program_census.py`: `candidate_program_passes` (shape-only
  fragments), `mapping_program_passes` (the built program, with per-boundary carried widths) and
  `record_program_passes` (a sealed record, which keeps no per-wire spans); the same module holds the record
  build's programming census and copies the executed residency marks into S0's declared-only record fields.
- **Write law** (`quantities/write_law.py`, `cost/absolute/writes.py`, CH-A1 and E85.W). The S0 policy keys ride
  the resolved platform to both planes; every write quantity and term is presence-based on
  `inference_batch_K`. `price_write_terms(census, physics, policy, *, most_written=)` prices the write energy per
  inference (DMA of the rewritten bytes / K plus the commit: `e_write_per_cell` x cells x verify iterations / K,
  producer writes per inference; [consts] plus `e_core_program` x W / K where the chip declares it, W the period's
  rewritten cores holding a static program (`programmed_cores`: [U-C C-6] a core whose every placement reads a bound
  bank holds none, and its operand writes stay the operand law's), its basis carrying the constant's evidence class,
  so the deployment load charges it once per tile and the step choice reads it), the exposed write time (serial Σ t_w / K; overlapped
  Σ max(0, t_w(q+1) / K - t_c(q)) after the shadow-room check, per declared core type: every pass computes on at most
  floor(count_t x (1 - s)) cores of each type t, s read exactly as its decimal, and an untyped plane on a
  chip of several types refuses the overlapped term by name; never the planner's 0.8x budget) and the lifetime
  (`endurance_cycles` / the most-written cell's wear per inference, [U-C C-1] in programming pulses: its writes x
  the technology's `pulses_per_write`, one where undeclared, `write_lifetime.endurance_in_writes` the one reading
  every endurance reader takes; the writes are the most-written core's period writes / K,
  with N2's producer writes folded into the same slot rule, `core_slots.place`; producer cells on cores a census
  does not place bound the wear from below only). A census that
  places no core (only where it provably rewrites, above) passes no `most_written`: the pigeonhole bound
  ceil(W / C) prices an UPPER-bound lifetime, which can prove a program inadmissible but never admit one. A
  lifetime floor yields a separate admissibility verdict and suppresses no priced term. With K declared both headlines add the write
  terms.
- **Per-core-type write technology** (N2b: `quantities/write_technology.py`, `cost/write_lifetime.py`,
  `platform_physics/resolve.with_write_technology`). A core type declares `write_physics`, a shipped profile whose
  write rows (`WRITE_TECHNOLOGY_KEYS`: the CH-A1 rows and `e_program_per_byte`) price its cells beside its own
  write granularity and units; the transfer channel (`e_dma_per_byte`, `t_program_per_byte`) and every other
  constant stay the chip's. Where the core types write by several technologies (`WritePolicy.core_technologies`,
  None otherwise, so a chip of one technology is priced as one, byte for byte), `plane_writes` splits the period
  census by technology (`split_by_technology`: the one census over the chip, restricted pass by pass to each
  technology's cores; `PassRow.producer_core_cells`, `WriteCensus.written_cores` and `type_producer_slots` carry
  what the restriction reads) and `price_write_inputs` prices each share by G3's one law under its physics and
  joins them: energies add, the per-pass walls add before the exposure rule (the technologies share the write
  ports), the lifetime is the shortest technology's and the floor verdict the worst (`join_lifetimes`); every
  term a share refuses is refused for the chip with the share's core type named. `price_write_terms` handed a
  policy of several technologies refuses by name. Two technologies on one geometry, a write physics no profile
  ships, a plane that states no core types, and payload cells on two technologies without per-core cells refuse
  by name; the resolver seals each named physics as `write_physics_resolved`. The stationary choice's price
  (`mapping/support/operand_price.py`) reads the operand core types' technology. `shadow_room_refusal` (the
  overlapped mode's per-type room) moved to `write_technology.py`, and `domain_bytes_of` to `value_domain.py`.
- **Step choice** (`cost/absolute/step_choice.py`, E85's priced plan, CL3). The residency planner states the step
  function and prices nothing; `cheapest_step(censuses, physics, policy, *, corner=)` picks the step whose period
  census prices the least `price_write_terms` write energy plus the `carry_inject` and `carry_readout` components
  per inference at the declared K (`x9_0_v3.price`'s argmin; the compute term is the same for every step of one
  program and never enters). Ties go to the first step, the smaller r: prices within `TIE` (2^-48, relative) of
  each other count as equal, because two steps of equal exact price differ in the last place once each term is
  rounded. A step whose writes the target does not price (no commit energy) or whose carried bytes it does not
  price refuses by name (`StepChoiceError`), and so does every step on a profile whose external component is
  undeclared (`context.external_refusals`, the gate `price_absolute` reads too: the SRAM-CIM macro chooses no step
  without its backing store). [fix_c] The write law (`price_write_terms`) reads the same gate, so on such a profile
  every write term (energy, exposed time, lifetime and its verdict) refuses by its own name on every plane: the
  executed program's design-point pricing (`design_point/endurance.priced_at`) and the exploration plane's
  `price_absolute`, which states the write law's named refusals beside `absolute_pricing`.
- **Validity** (`cost/absolute/validity.py`, E190). A profile's structured validity geometry (core axons and
  neurons, cell bits, chip core count, optional `cells` scaling rule) bounds its aggregate constants; outside it
  `chip_area_mm2` (and a per-event aggregate's energy) refuses by name. A caller of `price_absolute` that states
  no `geometry` is judged by what the census proves: a single core proves its type; several prove only their
  cell count (`CensusCells`), since types that share a dimension average to any whole mean core (384 x 256 +
  128 x 256 reads as two 256 x 256 cores). A declared `cells` rule then reads a footprint exactly by the
  census's cells; at a point validity a mean core off the point refuses (one core lies off it), a pooled census
  refuses, and a mean core AT the point is judged there with that basis stated in the term's source
  ("judged at the census's mean core ..."). `CENSUS_PROVED` refuses that mean-core verdict by name instead: the
  cross-platform table (`study.compare_platforms`), whose rows carry no source, judges under it unless the
  caller passes the declared core types. `UNJUDGED` reads the aggregates as declared, by name at the call site.
  `validity_refusals` is the CL2 check: judged at each physics target bundle's own geometry, only ODIN's refuses
  an aggregate (4-bit weights on 3-bit cells: `chip_area_mm2` refuses by name); ISAAC-like's 16-bit weights lie
  outside its 2-bit cells too, but its profile declares no aggregate, so nothing it prices is bounded.
- **External components and target refusals** (CL2). A profile that describes part of a priced system names the
  component it leaves out (`PlatformPhysics.external_components`, serialized only when present: the SRAM-CIM
  macro's backing store, `e_dma_per_byte` and `t_program_per_byte`); until a run declares those constants beside
  the profile (a named override block), `price_absolute` prices nothing and refuses `absolute_pricing` by name,
  and [U-CR C-2] each headline it suppresses by name with the gate's reason (`GATED_HEADLINES`), so a reader that
  looks a headline up finds the gate (the scaling ladder prints 'no physics declared' only where no profile exists).
  A target bundle that declares no physics (a geometry-only target) reports its
  `target_bundles.absolute_pricing_refusal` on both planes: `absolute_pricing_for_record` (so
  `emit_physics_report` writes that refusal) and the candidate view, through
  `CandidateQuantityContext.pricing_refusal` and `candidate_cost_report(..., refusal=)`; a run with neither
  physics nor a target still reports nothing.
- **Chip descriptions and declared departures** (U-C C-2, C-5). A target bundle is the chip description: its
  geometry, the core fields its cores carry beside the geometry's (`core_fields`, the cores' write path: NeuRRAM one
  cell at a time, HERMES-64 one core per write step) and its keys, the input width (`activation_bits`) among them
  where the chip states one. A run may declare another value only through `target_overrides` with a note, which the
  faithfulness label names; the sealed record prints each such departure by name (`DeploymentRecord.
  declared_departures`, `build/builder.with_declared_departures` at seal, emitted only when the run departs), the
  scaling ladder's rows print the keys `--declare` and `--core-field` set otherwise (`declared_departures`), and the
  design point's chip block names the build's own (`target_bundles.declared_departures`; on a row of several core
  types, each against its own bundle). A synthetic point whose
  bundle states no input width (the generic estimated chip) takes the run's declaration as its description.
- **Operand-only cores** (U-C C-6). A core whose every placement reads a bound bank holds no static program: the
  built and candidate planes state such cores per pass (`PassRow.operand_only_cores`, `PassCensus.
  operand_only_cores`, serialized only when nonzero), the build seals a mark on every core of a chip with an operand
  core type (`SegmentCoreRecord.operand_only`, declared-only), and neither `e_core_program` charge counts them: the
  write law's (`programmed_cores`) and the per-load term's (`reprogrammed_cores`, withheld by name on a record of such
  a chip sealed without the marks: the term names the charge unpriced, or refuses by name where it is the term's only
  part). Their operand writes stay priced by the operand write law, and their used
  rectangle's cells stay in the census as on every core.
- **Value NoC, lanes, host dispatch, override blocks.** `noc_value_bytes` (byte-hops of a value program,
  `mapping.noc.value_traffic`; a neuron-split softcore's wires are counted at one fragment's core, the
  `estimate_noc` approximation, a stated boundary) is priced by `e_noc_value_per_byte_hop`; the lane rows (`area_per_lane`,
  `e_lane_op`, `t_lane_op`) price only a lane census, through the component tables (`formulas.DerivedFactor`:
  lanes = `lanes_per_tile` x `tiles` beside the footprint on either area path, the lane pass `lane_ops` /
  lanes on latency), so a lane census on physics without the rows names `area_per_lane(lanes)`,
  `t_lane_op(lanes)` and `e_lane_op(lane)` unpriced on the three headlines, and lane work without its lanes is
  withheld by name; host dispatch with no host MACs is priced through
  `t_host_op_overhead` or named unpriced; `platform_physics/resolve.merge_override_blocks` merges named override
  blocks into the one `platform_physics_overrides` value and refuses a constant two blocks declare.
- **Level plane** (`quantities/from_level.py`, `cost/absolute/level_pricing.py`, CL4). A capability level's
  requirement placement (`mapping/capability`: the field map `level_quantities` and, on a core budget, the
  schedule `level_schedule.prepare_level(...).schedule(C)`) is a third plane beside the record and the candidate:
  `from_level(row, schedule, platform, *, counters, lanes_per_tile, timesteps)` prices it on `C` cores of the
  platform's one declared core type. Its period census is G3's one rule reached through the seam both other planes
  use (`write_law.plane_writes`, the executed rotation of finding 4 (a)): each tile of a pass is one physical core,
  and where the program fits its budget (F <= C) the residency floor marks every tile kept, so it is programmed
  once and writes 0 B per inference (W03); a dynamic product's stationary operand joins the census as
  producer-sourced cells per inference, on producer-written tiles stated as cores of no static cells
  (`PassRow.producer_cells`, `producer_cores`), so N2's one slot rule places them as on the other planes. A value-domain target reads the value rule (one evaluation per stage,
  MAC events = committed cells plus on-chip operator MACs, carry at `activation_bits`, else withheld by name); a
  spiking target is priced on the time axis only, at its declared LIF window (`synaptic_events` withheld: no
  record of a census member gives an activity factor). The level places no core on the mesh, so the traffic
  between its cores (`noc_value_bytes`, spike packets and hops) is withheld by name and its terms are named
  unpriced. Above C0 every census field is `modeled`. `price_level(view, physics)` is `price_absolute` with the
  view's write inputs (`WriteInputs.of`) and the chip's declared core type; `lane_area_quantities(chip,
  lanes_per_tile)` is what a chip's area reads, so a study can trade tiles for lanes at one declared area.
- **Host energy split** (`cost/absolute/host_split.py`, CL0). The host is not the chip: `host_energy_split` reads a
  priced report's energy with and without `energy_host_mj`, end by end (None where the headline is refused), and
  `host_share` per end; `dual_verdict(verdict, splits)` evaluates any verdict over a set of points on both energies,
  and a verdict that differs is `HOST_BOUND` ("host-bound"). Stock verdicts: `corner_order` (end by end, the
  corner-aligned reading of bands, under which a host energy two points share cancels), `band_order` (the interval
  rule), `lowest` and `cap_verdict`.
- **Share rule** (`cost/absolute/host_split.py`, U-SH). One rule prices the dynamic energy's MAC and carry shares on
  every plane: `price_shares` prices a program's quantities in the two `SHARE_VARIANTS` (compute-only: no carry,
  programming, sync, op count or host-side quantity; no-host: no op count or host-side quantity; both with
  `host_macs` 0) with the plane's compute split, readout bits and geometry; `shares` reads a term's compute-only share
  and its boundary share (no-host less compute-only, `sub_corners`), naming each absent one by `share_absence` (U-CR
  C-2's compute-absence rule, on the pricing that lacks it); `share_column` and `absent_shares` write and read the
  row column form. The scaling ladder (`introspection/scaling_ladder.py`, `scaling_ladder_law.axis_columns`) and the
  design point's record plane (`design_point/pricing.record_variant`) both call it, and the instance schema refuses a
  null share without a name (`dynamic_shares_refusals`).

## Lane block and output stage (Wave 2 W08)

- **Lane block.** `platform_physics/lane_blocks/` ships lane blocks as data (`ng45_os_lane.json`: the output-stage
  lane of `arith_toggle.md` section 6, area, energy and time per lane, every row `estimated`, at the block's declared
  `technology_node_nm` and `supply_v`, 45 nm and 1.1 V); a chip declaring `onchip_arithmetic` with lane classes and a
  `lane_block` resolves its physics with the block's G3 lane rows (`area_per_lane`, `e_lane_op`, `t_lane_op`)
  wherever the profile and the overrides declare none (`resolve.resolve_platform_physics(lane_block=)`), never on its
  own: a block adds no physics to an unpriced chip. Each row is scaled from the block's node and supply to the
  physics' own (`validity.technology_node_nm`, `supply_v`) by the block's declared rule (area `node^2`, energy
  `node*V^2`, or by node alone where the target declares no supply, time `none`), the factor named in its note
  (`lane_blocks.lane_block_rows(target=TargetNode)`); a block beside physics of no declared node, or without a rule
  for a dimension, is refused by name (`LaneBlockError`). The value-domain target bundles declare the block (one lane
  per 64 output columns).
- **Lane quantities.** One rule on both planes (`quantities/lanes.lane_block_quantities`): `lanes_per_tile` is lanes
  per core times the resolved cores per tile (0 on the off ablation, `classes` []), withheld by name where lanes
  carry no `lanes_per_core` (`NO_LANE_COUNT`: the block's area and time named unpriced), and `lane_ops` the lane work
  per inference (0 where the lanes do not run). The record plane (`quantities/from_record.py`) reads the executed work
  from the record's optional `output_stage`, which `schema/record.py` emits only when set (sealed from the Soft Core
  Mapping fragment), so every record of a chip without the declaration is byte-identical; the candidate plane
  (`objectives/context.candidate_context_from_platform(lane_ops=)`) prices the work the lane plan estimates on the
  candidate's model (provenance `modeled`) and carries an estimated output stage that names what it cannot see (its
  layout and host census are the lane-free shape's). A fused add's operand rows are no crossbar cells, MACs or
  programming bytes (`mapping/crossbar_utilization.CoreOccupancy` subtracts them): the add is priced once, as lane
  work; a pool lane's work is its requantized outputs, its window sums priced as its reduction cores' cells.
- **Chip-only energy and the output-stage line.** A point that prices a lane block (`lanes_per_tile` present or
  withheld by name) gets `energy_chip_only_mj` beside the headline (compute, periphery, static and write terms, no
  host energy; priced when the host refuses the headline too), naming the chip components it cannot price as the
  dynamic term does (and refused by the headline's own reason where the headline refuses before any host term,
  e.g. a candidate's static power without a priced latency), and `cost/output_stage.output_stage_note`
  prints beside it the lane work, host round
  trips (with the dependency barriers), host boundary bytes and the chip-only energy with its unpriced components,
  labelled `executed` or `estimated` (a requirement placement where the lanes do not run), in the record's cost
  report and in `candidate_cost_report` (a candidate's stage prints its lane plan's work).

## On-chip carry buffer over capacity (U-G)

U-F4's on-chip carry buffer is a run's declaration (`carry_buffer_bytes`, the quantity `carry_buffer_capacity_bytes`
on both planes), its access energy, leakage and area per byte physics constants. An SRAM's access energy depends on
its size, so a profile may state the buffer over capacity: `carry_buffer` (`platform_physics/profile.CarryBufferTable`,
judged by `loader.with_carry_buffer`) holds rows of `bytes` with any of `CARRY_BUFFER_KEYS`, each a profile constant
with its evidence; a constant that holds at every size (bitcells' leakage and area per byte) is stated flat in
`constants` and never also in a row. `price_absolute` prices the physics at the declared capacity
(`PlatformPhysics.at_carry_buffer`): each row constant at the capacity's row, a flat constant (an operator's override
included) at every capacity, and a row constant the table leaves unstated there refused by the table's reason through
`formulas.carry_buffer_refusals`, so only a headline whose buffer work exists is refused. A profile without the
table, or a run without a buffer, prices exactly as before; the table is serialized only when present. The three
study chips state it (45 nm SRAM rows of jouppi2021ten Table 2; static power and area per byte from their own leakage
and cells per bit).

## Core types with their own weight bits (Wave 2 WI-5g)

On a chip whose core types carry their own `weight_bits`, every core's programmed payload is sized at its own type's
width (`build/program_census.core_record`, `mapping/crossbar_utilization.CrossbarUtilizationReport.core_widths` for
`programming_bits`), the record's platform lists each type's width, and the one-width quantity `weight_bits` is
withheld where the types carry different widths (`quantities/from_record.py`), so every per-bit term (cell area per
bit, an aggregate's bit validity) refuses by name instead of pricing every cell at the platform width.

## Event-driven compute time and the per-core programming wall (U-C2)

`cost/compute_time.py` holds the one compute-time law the latency headline, the overlapped write exposure and the
ladder read. A chip with a fixed timestep prices its compute per step (`t_cycle` x `latency_steps`); a chip without
one, an event-driven core that runs its synaptic operations one at a time, declares `t_synaptic_op`, the time twin of
`e_mac`, and leaves `t_cycle` undeclared: its compute is `t_synaptic_op` x `synaptic_events`, the event census the
energy reads (one operation per input event and neuron the core processes; a core that sweeps every neuron per input
event counts the whole sweep). The census total states one core's wall exactly. On several declared cores the cores
of one (pass, segment) row run concurrently and the rows in turn, so the wall is each row's busiest core summed over
the rows (`synaptic_events_busiest_core`). [targeted M2] The candidate plane states it from the energy's own model (per
physical tile its as-mapped cells x timesteps x `activity_factor`, `compute_time.busiest_core_events`), formed in
`objectives/context.py` only where the resolved physics prices compute per event, so no other chip's quantities move;
the same per-row events reach the write law (`PlaneWrites.pass_sops`), so an overlapped write hides behind each
pass's busiest-core compute (`compute_time.pass_compute_walls`). A view that states the census as one total (a sealed
record) refuses the latency by name on several cores and the overlapped exposure on any core count, as it does without
a census or a declared `activity_factor`; with the latency refused, the throughput, the static energy over it, Pi's
time denominator and the regime labels refuse by the same name. A declared `t_cycle` takes precedence (a step's wall
holds its events). The ladder's layer-pipelined bound is the whole compute-only time on one event-serial core and the
busiest core's operations of one pass on several (`scaling_ladder_law.layer_pipelined`, `compute_time.busiest_stage`). `t_core_program` is the time twin of `e_core_program`: the wall of
programming one core beyond its payload bytes, priced in `latency_programming_s` (x `reprogrammed_cores`) and in the
write law's per-pass walls (x the rewritten cores holding a static program, `write_census.pass_programmed_cores`),
so Pi's numerator, M's period writes and the write-bandwidth batch read it. An interface that writes a whole core
image per load, whatever the program occupies (ODIN's SPI), declares its image's wall there and no
`t_program_per_byte`, whose occupied-cell payload would underprice a partly used core.

## Cells per stored weight (U-C4)

A core type whose stored weight takes several array cells, in the unit its chip description's per-cell rows are
declared for, declares `cells_per_weight` and `weight_cell_rows` (`mapping/platform/weight_cells.py`; absent: one cell
per weight): a weight is a block of `weight_cell_rows` cells on adjacent rows of its column by `cells_per_weight /
weight_cell_rows` on adjacent columns of its row. `max_axons` x `max_neurons` keep counting the weights the mapping
packs; the write law counts every cell of each written weight (`WritePolicy.cells_per_weight`: `runtime_written_cells`,
the cell, row and column write units, the commit energy and write steps, the timeline's written cells), while payload
bytes count weights and the wear stays per cell (both cells of a pair take each write of their weight, so the
most-written cell and the lifetime are the weight's). The declared chip's `cells_physical`, `axons_physical` and
`neurons_physical` and the core the aggregates' validity is judged at (`validity.geometry_of`) are the array's cells,
rows and columns, so area, row drivers, converters and per-slot rows scale with cells; on a chip of one cell per weight
they are its weights, unchanged. Core types that differ in their block are priced per write technology
(`write_technology.CoreTechnology.key`), and [targeted M2] every written-cell quantity of such a chip (written cells,
write units and write events on the record, candidate and level planes, and the design-point timeline's written cells,
units and events) is the sum of the census's shares by write technology, each at its own block
(`write_technology.cell_shares`, `write_law.summed_cells`: the shares the write terms are priced on), withheld by name
(the timeline refuses) where no split holds every written weight. A target bundle's geometry counts cells and its cores
hold that geometry over the block (`target_bundles`, refused by name where the cells hold no whole weights): `neurram`
2 cells on 2 rows (wan2022a Methods, PDF p.10, the differential-row mapping), `isaac_like` 8 cells on 1 row
(shafiee2016isaac p.5, the count its bit-sliced conversion model declares; the run door refuses a core type stating
another beside it), [targeted M1] `prime_mat_256x256` 2 cells on 1 row (chi2016prime Section III-A1, p.30: positive and
negative weights in two crossbars sharing one input port, the reading of `docs/research/physics/prime_constants_research.md`
section 3.3(b), at the bound 4-bit weights), every other bundle 1. Both doors refuse a count above 1 without its rows,
and rows that do not divide it.

## Edge host and the host-chip link (U-J)

Owner rule of 2026-10-28: the host is priced from a low-power Jetson-class device with published constants, never
measured on the experiment machines. `platform_physics/hosts/`: a host described as data (`<name>.json` beside its
`.md`): its published power modes, a mode chosen by a declared rule (`default`: the documented default mode), and
per row its class and band (`hosts/description.py`: `host_description_block` is the run's `host` override block,
every row naming its host; `host_attach_block` composes the link into the programming channel, `combine: slower`).
`hosts/rows.py` holds the vocabulary's HOST group, now with `p_host_idle`, `e_host_mem_per_byte`,
`t_host_mem_per_byte`, `e_link_per_byte`, `t_link_per_byte` and `t_link_transfer`. The loader keeps one host per
deployment (`require_one_host`): an identified host replaces the profile's host rows whole, and a host row naming no
host beside it (a measured server row) refuses by name (`HostBlockMixError`). `cost/host_terms.py` prices what the
block declares and nothing else: every byte crossing the boundary, both directions (the pass carry, the segment I/O
`segment_in_bytes` / `segment_out_bytes`, the write law's payload from the host) at the link and host memory, a
latency per transfer (`host_link_transfers`), a MAC-free host operator's bytes at the host memory time
(`host_memory_bound_bytes`, in the host's time), and `p_host_idle` over the inference's latency less the host's
time (`energy_host_idle_mj`; `latency_host_link_s` the link time). [U-J2] Whose energy each term is, the host's
description states per energy row (`energy_attribution`; every block row carries its `energy_plane`, the physics
keeps the system-level rows as `system_energy_rows`). Owner answer of 2026-10-29, "Host's work only": the headline's
host energy `energy_host_mj` is the attributable terms, the host's work above its idle floor (`energy_host_work_mj`,
(p_host - p_host_idle) x host time), the link (`energy_host_link_mj`) and host memory (`energy_host_mem_mj`) per
crossing byte; the idle floor under the work (`energy_host_floor_mj`) and over the wait (`energy_host_idle_mj`) are
system-level, and `energy_system_mj` (labelled the whole-system line, never the headline) prints beside. [targeted M1]
The work's power is a difference of two bands and takes its cost ends (`power_above`: the high end the power's high
less the floor's low, the low end never below zero, the nominal one setup's); the system line keeps the whole host at
its corners: `energy_host_system_mj` is the whole host (p_host over its time, p_host_idle over the wait, the link and
host memory) and `energy_system_mj` the headline's chip side (`PricingContext.energy_chip_side`) plus it, composed
from parts at every end (at the nominal, the headline plus the floor and the idle). The wait is the latency's parts the host does not execute, summed directly
(`PricingContext.host_wait`), never e2e less host time corner by corner (on a point-band chip that unordered the
band). Host time always adds to latency. A measured host wall prices the host unless the physics describes
a host of its own (`describes_host`: a host description's whole model) without relating the wall's machine to it
(`host_compute_rate`): an edge host prices its own time, and a partial host block (a server's measured calibration)
keeps the wall's rule, which needs `host_compute_rate`. The candidate plane forms the boundary quantities from the
walk's census (`mapping/support/host_boundary.py`, sized at the carried value grid in `objectives/host_link.py`)
only where the physics declares the link, so no other quantity moves; a carried pass boundary is a transfer for its
crossing part only (`crossing_widths`: under a declared `carry_buffer_bytes`, net of the widths a wavefront group
holds, `ProgramPasses.onchip_boundary_widths`, by the carry census's own rule, `pass_carry.group_held`). A row whose
source departs from the source rule (a preprint, another module's measurement) is named in the description's
`departures`, and the block carries the departure into the row's note.

## The candidate's schedule census (U-PP)

`quantities/from_candidate.py`: a candidate's `pass_count` and `sync_count` are the schedule census of the pass
structure that supplies its `latency_steps` and programming census (`CandidateQuantityContext.pass_count`,
`sync_count`, set by `objectives/context.py` from `build/payload_sizes.CandidateProgramming.schedule_census`, the
fragments' `LayoutNocFragments.schedule_census`), so they equal the sealed record's `ScheduleRecord.pass_count` and
`sync_count` for the same program; the layout statistics answer only for a candidate without a pass structure. Callers
that pack the statistics on the builder's segments (the joint search, the scaling ladder) see no change; a caller that
packed them on depth segments (the baseline rerun's candidate plane) now prints the record's counts, and its
`e_sync_barrier`/`t_sync_barrier` terms price the record's syncs.

## Where the backing store lives (U-J3)

A chip description declares its backing store (`platform_physics/hosts/store.py`; the profile's `backing_store`:
`{"chip": {<source>}}`, the chip's own DRAM through its memory PHY, or `{"host": {<source>}}`, the host's memory over
the host-chip link). A run may restate it as a declared departure, a `backing_store` entry of a named override block
(`store_block`), marked overridden in the resolved physics. A declaration stating both stores or none, an unknown
field or no source refuses by name (`BackingStoreError`); the chip's area components agree with it where it declares
them (`judged_store`: a memory PHY exactly when the store is the chip's, a host-owned store over a link PHY to the
host), a host-owned restatement drops the memory PHYs and re-derives `area_global_fixed`, and a host attach row (a
chip channel fed over the link) on a chip-owned store refuses by name. The store routes the store traffic, the pass
carry and the write law's backing-store payload (a period's payload over K), and a program load's payload: on a
chip-owned store the chip's rows price it alone (`e_dma_per_byte`, `e_readout_per_byte`, the write port), on a
host-owned store the link and host memory do (`host_terms.link_bytes`; the write walls at the slower of the port and
the store's per-byte stage plus one `t_link_transfer` per written pass, `compute_time.store_walls`; the load at the
link and host memory, `formulas.programming_byte_energies`, `hosts.store.slower_port`) and the chip's store rows
(`chip_store_table`, the write transfer's `store_transfer_note`) price none of it. The program's own crossing (the
segment I/O, host-sourced operand payloads) is on the link on either store. The candidate plane counts a carried
pass boundary as a transfer only to a host-owned store (`objectives/host_link.py`); the residency step choice prices
its carry and write transfer on the declared store (`step_choice`). Beside the host link a chip that states no store
refuses its store traffic by name (`hosts.store.store_refusal`: the latency and energy headlines), as a host-owned
store without the link does; a physics without the link and without a declaration prices as before. The study chips
declare chip-owned stores (their two DDR4 channels, `cd41_chip.CTX`, wang2020shuhai Table V).

**The store rule's scope, a declared rule** (U-J4 K-2: U-J3's departure D2, accepted by the conductor on 2026-10-30,
vetoable). The no-store refusal applies only where the physics prices the host-chip link (`hosts.store.LINK_ROWS`).
The reason: beside the link the same bytes have two places (the chip's own store rows and the link with host
memory), so a chip that states no store would price them on both or on neither; a chip with no link rows has one
store place, its own channel rows (`e_dma_per_byte` in, `e_readout_per_byte` out, its write port), so without a
declaration it prices as on RC1 (`tests/unit/deployment_record/test_store_rule_without_link_uj4.py`, which also runs
on RC1's tree). A chip that declares a memory PHY still has to declare its store (`judged_store`).

## Bytes that cross the host-chip link (U-J4)

One rule (conductor decision of 2026-10-30, from U-J3's review minor m2): where the physics prices the host-chip
link, every byte that reaches the chip over the link or leaves it that way pays the link and host-memory rows
(`e_link_per_byte`, `e_host_mem_per_byte`, `cost/host_terms.py`) and never the chip's backing-store rows
(`e_dma_per_byte` in, `e_readout_per_byte` out). The crossing bytes are the program's
own on either store (the segment I/O, a host-sourced operand payload) and, on a host-owned store, the store traffic
(U-J3). Sites: the write law prices a host-sourced payload's cells only and notes the transfer on the link
(`writes._write_energy`, `PlatformPhysics.payload_link_note`); the residency step choice adds it at the link and host
memory (`step_choice.price_step`); N2's lowering price adds it there too and injects its streamed vectors at the
carry's per-byte energy on the declared store (`mapping/support/operand_price.py`, `hosts.store.injection_byte_energy`);
the record pricer's static/operand split (`design_point.pricing`) reads the write law. N2's cost-only lowering
choice can move (targeted pass, review M1): both lowerings place the same cells and move the same payload, so the
streamed vectors order them, and on equal prices the tie rule picks fewer bank cells, the lowering streaming more. On
a host-owned store the injection moved from `e_dma_per_byte` to the link and host memory, so a chip without
`e_dma_per_byte` beside the edge host now takes the lowering streaming less (it tied before), and a block declaring
neither link energy row ties (it took that lowering before); a row the lowering's price reads and the physics does
not declare is named in its printed reason (`operand_form.LoweringPrice`, `hosts.store.unpriced_link_row`). The
residency step choice does not move to another step: every step of one plan writes the same operand-bank cells per
inference (the units' producer cells over the step's phases, `residency_units.plan_period_facts`), so the payload adds
one constant to every step; beside a hand block declaring neither link energy row it refuses by name
(`StepChoiceError`), where it priced the payload on `e_dma_per_byte` before. The chip's own end of the link is
priced by no row of the vocabulary: the link's energy names it unpriced in its basis (`hosts.link_ends.link_end`, a
boundary), as does a program load's link part. A crossing byte whose link or host-memory row the physics does not declare (a hand block stating part of the
link; a host description declares every row, `REQUIRED_HOST_ROWS`) is named unpriced on the headline. Without link
rows nothing changes: `e_dma_per_byte` is the chip's one channel and prices the payload, as on RC1.

## U-J4's minors closed and the host-sourced payload's link time priced (U-J5)

- **Where a transfer row prices** (L-1, U-J4 review m1). A chip description may place a per-byte transfer row
  (`e_dma_per_byte`, `e_readout_per_byte`) in a chip-internal buffer every arriving byte passes (`transfer_rows`,
  `platform_physics/hosts/transfer.py`); undeclared, the row is the backing store's (the vocabulary's meaning). The
  link-crossing rule and the host-owned store's rule drop only a backing-store row: a chip-buffer row is paid by every
  byte that enters the buffer, crossing or not (the write law's transfers, the carry rows `hosts.store.chip_store_table`,
  a program load `formulas.programming_byte_energies`, the lowering's injection `hosts.store.injection_byte_energy`,
  the step choice's carry). isaac_like declares its `e_dma_per_byte` a chip buffer (a byte read through the tile's
  eDRAM buffer, its own text); no other shipped profile declares one. Without the link the two places price alike.
- **The link's ends per corner** (L-2, m2). A host description declares which ends of the link its `e_link_per_byte`
  counts at each corner (`link_ends`, `platform_physics/hosts/link_ends.py`; the Jetson: the host's end at low and
  nominal, both ends at high); its block row carries it into the resolved physics, and the link's energy names the
  chip's end unpriced only at the corners where it is not counted. A description that declares none keeps U-J4's
  text at every corner.
- **One link energy row declared** (L-4, m4): the step choice's basis names the missing row
  (`hosts.store.unpriced_link_row`), as the headline and the lowering's reason do; no price changes.
- **The host-sourced payload's link time** (L-5, U-J4 finding F1): where the physics prices the link and the operand
  payloads come from the host, a pass's payload bytes stream at the link's per-byte stage, so its producer wall is the
  slower of its producer write steps and that link wall, plus one `t_link_transfer` per pass carrying a payload
  (`cost/compute_time.producer_walls`, U-J3's `store_walls` its precedent). The producer writes stay the write law's
  per-inference term (serial in both overlap modes), so on a host-owned store the period's backing-store payload and
  the host-sourced payload take the one link in turn, each in its own write phase (the exposed write's basis says so,
  `payload_link_note`). Where the physics prices the link and declares no per-byte stage, or no transfer latency, the
  missing part is named unpriced on the latency (`hosts.store.payload_time_unpriced`). Without link rows nothing
  changes.
- **The streamed vectors' row on the chip's own channel** (L-6): where the lowering price injects them at an
  undeclared `e_dma_per_byte` (the chip's own store, or no link), its printed reason names the row; no choice or price
  moves.

## Device samples (U-HV)

A run whose plan runs an ODIN step (`enable_odin_fpga_simulation`, `enable_odin_hacc_export`) seals the optional
`DeploymentRecord.device_samples`: one row per step (`backend`, `declared_by` = `odin_sample_split`, `split`,
`indices`), joined by the Deployment Record step from each step's `sample_read` entry
(`pipelining/pipeline_steps/verification/odin/sample_split.py`). The seal requires the fragment where the plan view
says an ODIN step ran (`SealPlanView.device_samples`); `schema/record.py` emits it only when set, so every record of a
run without an ODIN step keeps its bytes. A cached ODIN entry with no `sample_read` (written before the declared split)
is refused by name (`UnrecordedSampleRead`).
