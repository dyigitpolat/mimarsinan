# Dynamic-Operand MVM (DOM) — attention products as packable constructs

**Status: DESIGN — direction approved 2026-08-18, implementation unscheduled.**

Owner directives folded in verbatim:

> "for loihi (or other platform templates that this information is missing), we
> can derive the programming energy / time constants from other platforms,
> having the relative fractions at reasonable values compared to other
> constants."

> "about the SNN flows, I know that we cannot 'tune' the QAT for spike
> semantics if inputs are encoded as parameters, however, I believe that the
> magnitude scaling etc can still work somehow. not a priority, but something
> to keep in mind."

---

## 1. Motivation, with the measured context

The U3 ViT study decomposed the deployment wall precisely (loihi profile,
measured host block, `docs/scheduler_unification_plan.md`):

- t1_02 TTFS flow: **33.1% of MACs on chip**; host = GELU MLP (11.2 G) +
  attention products (0.72 G) → host wall 0.514 s, host energy 10.28 J.
- t1_09 MVM flow: fc1/fc2/patch/heads chip-mapped; the host residue is
  **attention products + softmax/LayerNorm** — the last host MACs standing.
- Chip terms vary honestly (dynamic 471→669 mJ, chip latency 20→120 ms across
  the U3 Pareto front) but sit under the constant host pedestal.

Attention products are activation×activation matmuls: `S = Q·Kᵀ`, `O = A·V`.
The construct: **program one operand into crossbar cells as a per-inference
weight payload, stream the other operand through as inputs** — turning the
last big host MACs into chip MACs. Per head: an S-core programmed with `K`
(d_head axons × tokens neurons) consuming the token-streamed `q` vectors, and
an O-core programmed with `V` (tokens axons × d_head neurons) consuming the
softmaxed rows. Softmax and LayerNorm stay host.

ViT-B/16 @224 sizes: 144 (layer, head) pairs × two products → **3.63 M cells**
(1–3% of any U3 Pareto chip) and a **~3.6 MB per-inference programming
payload** at 8-bit. Feasibility (analyzed 2026-08-18): value-domain flow,
write-tolerant (SRAM-CIM-class) targets, fixed sequence length; the ceiling
after conversion is the softmax/LayerNorm round-trip (marshalling-bound, the
R5 residual), and the NVM kill-switch is write endurance.

## 2. The design thesis: one new axiom, everything else derived

Today the packing universe rests on an implicit axiom: *a weight bank's
payload is a training-time constant.* DOM replaces it with:

> **A weight bank's payload may be an IR-produced value, quantized onto a
> declared grid, loaded at a program event.**

Everything else — instance slicing, packing, streaming residency, pass
boundaries, carry, programming census, pricing, certification — is already
payload-agnostic or becomes so with small, additive changes. The design's
job is to prove that claim mechanism by mechanism and mark the deltas.

## 3. Existing-mechanism inventory (what DOM reuses verbatim)

| Mechanism | Where | What DOM reuses |
|---|---|---|
| `WeightBank` identity + instance slicing | `mapping/ir/weight_bank.py`, `mapping/ir/types.py` (`NeuralCore.weight_bank_id`, `weight_row_slice`, `column_slice` shared views) | The bank IS the identity every downstream law keys on (residency, programming, dedup). DOM adds a payload *source*, never a new identity. |
| Value-domain packaging | `mapping/platform/packaging_contract.py` (`PackagingContract.is_value_domain`: "package any weight-stationary affine op") | DOM extends "weight-stationary" to "weight-stationary *per inference*". The contract is the legality gate: dynamic banks are value-domain-only, fail-loud elsewhere. |
| Boundary value grids | `NeuralCore.boundary_grid` (`[mvm AQ]`, `activation_bits` registry key, owner `boundary_quantization`) | The realized symmetric-grid type + quantize/clip discipline is exactly what the payload needs on its way into cells. |
| Calibrated ranges | Activation Analysis step (per-source scales walk, `mapping/support/activation_scales.py`) | Per-(layer, head) K/V ranges → calibrated payload grid scales. No new calibration machinery. |
| Segmentation law | `mapping/layout/segmentation.py` (`partition_ir_graph`: every `ComputeOp` is a host barrier) | DOM's payload-quantize op is a ComputeOp ⇒ the producer/consumer chip-program cut falls out of the EXISTING partition rule. Zero segmentation changes. |
| Unified pass planner | `mapping/support/schedule/pass_planner.py` (U1: residency-first, `bank_clustered_law`, `resident_passes`) | Within one inference, a programmed dynamic bank IS a resident bank: the streaming law composes "program K once, stream 197 tokens" verbatim. Across inferences it reprograms — which is already the record's per-program-load accounting. |
| Pass-cut / carry | `mapping/support/schedule/pass_cut.py`, `pass_carry.py` (one census, both planes) | Streamed-operand rasters price as carry exactly as today. The payload does NOT ride carry (see §4.5 — no double count). |
| Programming census | `mapping/weight_programming.py`; `deployment_record/build/from_mapping.py` (`programming ∈ {resident, reprogram}`, `params_bytes = 0 if resident`); candidate side `candidate_programming_census` | Payload bytes land in the SAME `reprogrammed_bytes` quantity; DOM adds only an attribution field (§4.5). |
| Physics vocabulary | `deployment_record/platform_physics/vocabulary.py`: `e_program_per_byte` (multiplicand `reprogrammed_bytes \| carry_in_bytes`), `e_core_program` (× `reprogrammed_cores`), `t_program_per_byte` | The pricing formulas for DOM **already exist**. Declaration state today: `t_program_per_byte` in generic_estimated_22nm + truenorth only; the energy constants declared NOWHERE — §4.6 fixes this per the owner directive. |
| Executor harness | `chip_simulation/hybrid_run/hybrid_stage_runner.py` (`run_hybrid_stages` callbacks over `state_buffer`), `host_compute.execute_compute_op_torch`, `StageTimer` | The payload is a host-op output sitting in `state_buffer`; materialization is one hook at neural-stage entry (§4.4). |
| Per-stage backend programs | SANA-FE per-stage arch synthesis (R3), nevresim VERBATIM replay (G1), lava (G2), SCM `ValueHybridCoreFlow` | Every backend already builds per-stage programs; a per-sample payload is an injection at build time, with the G-series harness as the parity instrument. |
| Certification | `pipelining/core/gates/value_gates.py` (fp64 window-exact twin, `value_parity_samples`, FATAL twin certs on mvm cells) | The correctness gauge for the whole construct: packed dynamic-MVM vs the float reference, per neuron-window. |
| Fidelity | R4 armed gates, term zip, `activity_warning` | New rows enter report-only, gate after evidence — the standing discipline. |

## 4. The deltas

### 4.1 IR: `DynamicBankBinding`

New sibling `mapping/ir/dynamic_bank.py`:

```python
@dataclass(frozen=True)
class DynamicBankBinding:
    """A bank whose payload is the output of an IR node, loaded at a
    program event. The payload source is the quantize op's OUTPUT SPAN,
    row-major over (axons, neurons); `transpose` covers the K-vs-Kᵀ case."""
    bank_id: int
    source_node_id: int          # the bank_program ComputeOp (§4.2)
    payload_shape: tuple[int, int]   # must equal the bank's core_matrix shape
    transpose: bool
```

`IRGraph` gains `dynamic_bank_bindings: Dict[int, DynamicBankBinding]`
(default empty — the `__getattr__` legacy-unpickle pattern already in
`graph.py` covers old graphs). The bank's `core_matrix` at build time is a
zero placeholder carrying the SHAPE authority (extents, packing, occupancy);
`pre_pruning_snapshot`/pruning stay vacuous for dynamic banks (validated:
a pruned dynamic bank is a build error until §7 revisits it).

**Validity rules, enforced at graph finalize (fail loud, by name):**
1. `bank_id` bound at most once; the bank exists; shape matches.
2. Dynamic banks are legal iff the run's `PackagingContract.is_value_domain`.
3. The source node precedes every instance of the bank in graph order
   (segmentation then guarantees producer-chip-program ≠ consumer-chip-program).

### 4.2 Lowering: the `bank_program` ComputeOp

The attention lowering (value-domain path of the ViT builders —
`models/vit_leaf.py` / the torchvision bridge that today emits MHA as host
ComputeOps) becomes, per (layer, head):

```
chip: QKV projection instances            (existing affine packages)
host: bank_program[K]  — gather K, quantize onto the bank grid   (ComputeOp)
chip: 197 S-instances reading dynamic bank K, streaming q tokens (NeuralCores)
host: softmax(/√d)                                               (ComputeOp)
host: bank_program[V]                                            (ComputeOp)
chip: 197 O-instances reading dynamic bank V, streaming A rows   (NeuralCores)
chip: output projection                    (existing)
```

Why a ComputeOp and not a new node kind — three birds, one stone:
- Its presence IS the host barrier `partition_ir_graph` already cuts on, so
  producer and consumer land in different chip programs with no new
  segmentation semantics.
- It is the host-side quantize step (grid from Activation Analysis
  calibration, the boundary-AQ discipline), and its measured wall prices
  under the EXISTING host terms (dispatch + rate) with zero new vocabulary.
- Its output in `state_buffer` is the payload handle the executor reads —
  payloads ride the state like every host-op output.

The dynamic instances themselves are ordinary `NeuralCore`s
(`weight_bank_id` = the dynamic bank, `perceptron_output_column` = token) —
the per-token instance pattern the planner already streams. `bank_id`
backfill in `_segment_specs` (U1) carries them into the planner unchanged.

Opt-in config key: `attention_mapping ∈ {"host", "dynamic_mvm"}` (registry
entry, value-domain relevance predicate, default `"host"` — byte-identical
runs unless declared; the C3 option-axis chain makes it searchable later for
free).

### 4.3 Scheduling: nothing

`plan_segment_passes` needs no change. Within an inference the dynamic bank
is one shared bank over 197 same-latency instances: the residency law
composes the streamed program (program once, stream tokens) and
`mark_bank_residency` verifies its geometry — verbatim U1 machinery. Across
inferences the programming record already accounts per program load
(`from_mapping.py`: `params_bytes` per non-resident pass), which is exactly
DOM's per-inference reprogramming.

### 4.4 Execution: one materialization hook

`hybrid_stage_runner` gains a payload step at neural-stage entry (inside the
runners' `on_neural`, before the stage executes): for each stage core whose
bank has a binding, read the `bank_program` output from `state_buffer`,
reshape/transpose per the binding, and install it as the stage's core payload
for THIS sample. Concretely per backend:

- **SCM / `ValueHybridCoreFlow`** (the mvm deployment executor): fill the
  stage's core matrices (torch tensors) before the stage forward. This is
  the primary path and the cheapest.
- **nevresim / SANA-FE / lava**: per-stage programs are already synthesized
  per stage (R3); the payload is injected at program-build time per sample.
  Cost note: per-SAMPLE program rebuild on the simulators — a simulation-
  wall cost, not a correctness issue; the G1/G2 VERBATIM harnesses are the
  parity instrument. Spot-verification tiers keep `max_simulation_samples`
  small, as today.

Staleness is the failure mode to design against: a payload must be
re-materialized every sample. The hook therefore keys on
`(sample, stage)` — never cached across samples — and the V&V plan carries a
mutation check for exactly this (§8).

### 4.5 Census, quantities, pricing — and the no-double-count law

- **On-chip fraction**: the flow walk classifies the S/O instances as chip
  MACs (they are NeuralCores now); `estimate_onchip_fractions` and the host
  census move ~0.72 G MACs from `host_macs` to `onchip_macs` with no code
  change beyond the lowering.
- **Programming**: per-inference payload bytes land in `reprogrammed_bytes`
  (record: sealed from the schedule as today; candidate:
  `candidate_programming_census` counts dynamic banks from the binding
  table). One additive-optional attribution field on the segment programming
  record: `payload_source: "model" | "runtime"` (the `invocations`
  additive-schema precedent) so studies can split trained-weight
  reprogramming from operand programming.
- **The split, stated once**: payload bytes price under PROGRAMMING terms
  (`e_program_per_byte`, `t_program_per_byte`, `e_core_program`); the
  `bank_program` op's wall prices under HOST terms; streamed-operand rasters
  price under CARRY. Three disjoint term families, one byte never in two.
- **Fidelity**: the new rows enter report-only; gates arm only after a
  sealed-vs-candidate evidence batch (the R4/R5 protocol).

### 4.6 Derived programming constants (owner directive)

Current state: `t_program_per_byte` declared by generic_estimated_22nm and
truenorth; `e_program_per_byte` and `e_core_program` declared by NO profile —
so today a DOM headline would refuse by name everywhere except partially on
two profiles. Per the directive, the missing constants are AUTHORED into the
profile JSONs as `evidence_kind: "derived"` values with ratio-based
derivations — data, not a new code path, exactly the C6 discipline (every
constant carries its derivation string; comparisons disclose evidence kinds).

Proposed derivation rules (each written into the constant's `derivation`):

| Constant | Profile class | Rule | Band discipline |
|---|---|---|---|
| `e_program_per_byte` | digital SRAM (loihi, truenorth, odin) | SRAM write ≈ 1–3× read energy per bit; anchor on the profile's own access-scale constant (loihi: `e_mac` 23.6 pJ/op as the read-path anchor) | low = 0.5× anchor/byte-scale, high = 3× — the band IS the ratio spread; nominal mid. A derived band must be wide enough to be honest. |
| `e_program_per_byte` | analog NVM (isaac_like) | program-verify write, nJ/cell scale from the ISAAC/PRIME literature — 10–100× the SRAM class | band spans the literature spread; the note must state endurance is UNMODELED (a written-per-inference NVM array wears out; §9 open question) |
| `t_program_per_byte` | loihi, odin, isaac_like | donor = the two declaring profiles; scale by technology node where stated (`validity.technology_node_nm`) | band covers both donors' values after scaling |
| `e_core_program` | all | same control-path scale as the profile's `e_core_init` where declared (loihi: 31.13 nJ measured-derived) → band [0.5×, 2×] `e_core_init` | derivation names `e_core_init` as the anchor and why (program setup ≈ state-reset control work) |

Validation requirement: the V2/V3 silicon-correlation harness re-runs green —
the reference cases never reprogram, so their predictions must be
bit-unchanged (asserted, not assumed), and one new programming-dominated
self-consistency case is added where a donor paper supports it. The refusal
behavior remains for anything underivable: a derived constant is written down
or the term stays absent-by-name — never a silent zero (the standing E-series
rule).

### 4.7 SNN flows — the keep-in-mind note (not in scope)

The owner's read is right on both halves. Parameter-encoded inputs cannot be
QAT-tuned (the exact-QAT pairing needs trained weights to co-adapt with the
spike semantics — a runtime payload has no training loop), so the
window-exact certificate calculus does NOT extend to spiking DOM. What can
survive is the magnitude calculus: the scale-migration machinery can gauge a
calibrated |K| onto weight scales the same way it gauges trained weights, and
Q would ride the existing spike encoding. That path is accuracy-gauged only
(deployed-accuracy delta, no exactness cert), and the pre-softmax score range
is the hazard (timing codes have narrow dynamic range). Design consequence
NOW: the legality gate in §4.1 is a single predicate
(`PackagingContract.is_value_domain`) — relaxing it later is one condition +
its own verification program, and nothing else in this design assumes the
value domain.

## 5. What does not change (the protection contract)

- `dynamic_bank_bindings` defaults empty and `attention_mapping` defaults
  `"host"`: every existing run is **byte-identical** (A/B pinned on a
  fixed-config cell, the C1 discipline).
- Packer, planner, segmentation, carry, record schema: untouched or
  additive-optional. Layout specs stay payload-blind — the shape-only planes
  need no payload, which is what keeps DOM fully candidate-representable
  (search can price it without ever holding a K tensor).
- Spiking flows: bindings rejected at graph finalize by the packaging
  contract, loudly.
- Suite ≤2 min, typecheck 0, ratchets only tighten, no workload constants
  framework-side (token counts, head dims — all flow from the model config).

## 6. Engineering analysis

**Budgets (ViT-B/16 @224, per inference):** payload 2×197×64×144 cells ≈
3.63 M cells ≈ 3.6 MB @8-bit. SRAM-class write energy ≈ 3–30 µJ (invisible
next to the 10.3 J host term it helps remove); programming wall at 1–10 GB/s
config bandwidth ≈ 0.4–3.6 ms — comparable to the chip compute window and
the term the search should see. +24 program events (2/layer) on the critical
path unless the target overlaps program-with-compute (§9). Expected host-wall
move in the mvm flow: ~45 ms → the softmax/LayerNorm marshalling floor
(~15–30 ms) — bounded, real, and honestly below the naive MAC arithmetic.

**Sizing/fragmentation:** S/O cores are small (64×197, 197×64); they pack
into the U3-class chips' idle capacity trivially; the planner streams them
per head with the existing law. Fixed sequence length is assumed; an
autoregressive growing-cache regime is a different (write-quadratic) problem,
named out of scope.

**Module budget (≤300 LOC, ≤10 siblings respected):**

| Change | Where | Est. |
|---|---|---|
| `DynamicBankBinding` + graph field + finalize validation | `mapping/ir/dynamic_bank.py` (new), `graph.py` (+field) | ~120 LOC |
| `bank_program` lowering in the value-domain attention path | the mvm packaging pass + `models/vit_leaf.py` pattern | ~150 LOC |
| Payload materialization hook | `chip_simulation/hybrid_run/` (one helper + call sites in SCM runner; simulator injection under the per-stage builders) | ~100 LOC + per-backend glue |
| Census attribution (`payload_source`) + candidate binding count | `deployment_record/build/from_mapping.py`, `search/problems/joint/candidate_fragments.py` | ~60 LOC |
| Registry key `attention_mapping` + retired-keys none | `config_schema/registry/entries_execution.py` | ~20 LOC |
| Derived constants | profile JSONs + `.md` derivation notes | data only |

## 7. Implementation pointers (verified seams)

- Bank anatomy and shared column views: `mapping/ir/weight_bank.py:14–36`.
- Instance fields the lowering populates: `mapping/ir/types.py:72–94`
  (`weight_bank_id`, `weight_row_slice`, `perceptron_output_column`,
  `boundary_grid`).
- Barrier law the `bank_program` op rides: `mapping/layout/segmentation.py`
  `partition_ir_graph` (every ComputeOp is a `HostSegment`).
- Streaming/residency law: `mapping/support/schedule/pass_planner.py`
  (`plan_segment_passes`, `resident_passes`), verification twin pins in
  `tests/unit/mapping/test_unified_scheduler.py`.
- Spec backfill that carries dynamic `bank_id` into the planner:
  `mapping/packing/hybrid_build_pool.py::_segment_specs` (U1 backfill).
- Programming record classes: `deployment_record/build/from_mapping.py:102–150`
  (`resident`/`reprogram`, `params_bytes=0` when resident).
- Pricing vocabulary rows: `deployment_record/platform_physics/vocabulary.py`
  (`e_program_per_byte`, `e_core_program`, `t_program_per_byte`).
- Executor loop + state buffer: `chip_simulation/hybrid_run/hybrid_stage_runner.py:65–130`
  (`run_hybrid_stages`; the neural/compute callback seam; `StageTimer`).
- Host-op execution + measured walls: `chip_simulation/hybrid_run/host_compute.py`
  (`execute_compute_op_torch` — also the calibration estimand, H3b/R2).
- Value certificates: `pipelining/core/gates/value_gates.py`
  (`value_parity_samples` registry key; FATAL twin certs on mvm cells).
- Calibrated ranges for grids: the Activation Analysis step artifacts
  (`Activation Analysis.activation_scales.json` in any sealed run dir).

## 8. Verification & validation

Tests first, per layer; every load-bearing guard mutation-checked
(cp-backup/mutate/expect-red/restore, the standing practice).

**Unit (new files):**
1. `tests/unit/mapping/test_dynamic_bank_binding.py` — binding legality:
   value-domain-only (spiking contract rejects, by name); source precedes
   instances; shape mismatch fails; legacy graph unpickles with empty table.
   *Mutants:* drop the domain check; drop the ordering check.
2. `tests/unit/mapping/test_dynamic_bank_planning.py` — on a
   `bank_clustered_vehicles`-style token graph with a dynamic bank: specs
   stay payload-blind; the planner composes the streamed program
   (`residency_applied`, resident flags) exactly as for a static bank;
   builder==planner parity (the U1 pin pattern).
3. `tests/unit/chip_simulation/test_dynamic_payload_execution.py` —
   (a) identity grid: dynamic-MVM stage output == `torch` matmul at fp64;
   (b) quantized grid: within the grid's stated band vs float reference;
   (c) **freshness**: two samples with different K produce different outputs.
   *Mutant:* cache the payload across samples → (c) must fail.
4. `tests/unit/search/test_dynamic_programming_census.py` — candidate
   `reprogrammed_bytes` includes the payload; `payload_source` attribution;
   host census moves the product MACs on-chip; candidate == sealed-record
   census on the vehicle (the E2/H2 cross-plane discipline).
5. `tests/unit/deployment_record/test_programming_pricing.py` — payload ×
   `e_program_per_byte`/`t_program_per_byte` terms appear; undeclared
   profile → absent-by-name with the note; derived constants load with
   `evidence_kind: derived` and their derivation strings; **no
   double-count**: the payload bytes appear in programming terms and NOT in
   carry terms (assert both sides).
6. Byte-identity A/B: a fixed mvm cell with `attention_mapping` absent is
   bit-identical to pre-DOM output (the C1 no-profile precedent).

**Cross-backend parity:** a 2-token synthetic dynamic bank through SCM vs
nevresim (VERBATIM replay harness, G1) — same outputs, same event counts;
SANA-FE spot-run for the per-stage injection path.

**Certification:** the fp64 window-exact twin extended over dynamic stages,
FATAL on the acceptance cell — the same gate class that certifies mvm today.

**Integration/acceptance (templates via `generate.py`, the F1 rule):**
- New tier-0 cell `t0_xx_mvm_attn_dyn` — a small `vit_leaf` attention with
  `attention_mapping: dynamic_mvm`: cert FATAL green, record sealed,
  fidelity report emitted, programming terms priced (generic profile) and
  refused-by-name (a profile left underived on purpose, pinning the refusal).
- t1_09 variant with dynamic attention: measured host wall target ≤30 ms
  (from ~45 ms), reprogrammed_bytes ≈ 3.6 MB/inference sealed, fidelity zip
  on the new terms report-only; A/B against stock t1_09.
- Deployed accuracy within the cell's `degradation_tolerance` — the grid is
  a real quantization of a live operand; this is the number that catches a
  bad calibration.

**Derived-constants validation:** V2/V3 correlation harness green with the
new constants loaded; reference-case predictions asserted bit-unchanged
(they never reprogram); evidence kinds disclosed in every comparison
artifact.

**Protocol:** suite ≤2 min green; `./scripts/typecheck.sh` zero; ratchets
clean; ARCHITECTURE.md updated for `mapping/ir`, `chip_simulation`,
`deployment_record`; per-stage commits, no AI-attribution trailers.

## 9. Open questions (tracked, not blocking the design)

1. **Program-while-compute overlap** — can the target program head h+1's K
   while computing head h? Determines whether the 24 program events serialize
   into e2e. Proposal when reached: a capability bit
   (`allow_program_overlap`, default False = serialized — conservative), the
   pricer charging the serial case until declared.
2. **Per-inference dynamic grid scales** vs calibrated static scales —
   calibrated first (this design); dynamic scales need a per-program-event
   scale register write and a hardware capability claim.
3. **NVM endurance** — unmodeled; a written-per-inference ReRAM array has a
   lifetime measured in days. If an analog profile ever prices DOM, the
   comparison must carry an endurance disclosure (candidate: a
   `writes_per_cell_per_inference` quantity with a profile-declared endurance
   budget — refusal machinery fits naturally).
4. **Growing K/V caches (autoregressive)** — write volume becomes quadratic;
   out of scope; the fixed-length assumption is stated at every entry point.
5. **Spiking DOM** (§4.7) — magnitude-scaling path, accuracy-gauged only;
   revisit after the value-domain construct has sealed evidence.

## 10. Suggested staging (when implementation is scheduled)

D1 IR + binding + validation pins → D2 lowering + registry key (byte-identity
A/B) → D3 SCM execution + freshness/exactness pins + value certs → D4 census,
pricing, derived constants + correlation-harness validation → D5 simulator
injection + cross-backend parity → D6 cells + the host-wall acceptance study.
Each stage lands with its tests, gates, and docs per the protocol above.
