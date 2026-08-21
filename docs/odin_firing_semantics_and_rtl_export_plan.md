# ODIN firing semantics as declared axes + variable-geometry RTL export — engineering plan

2026-08-21 · status: **approved plan of record, revised under adversarial review** (owner
directives 2026-08-21 in §3; four-judge review record in §14) · baseline `17be1010` on
branch `odin-fpga-infra` in the `mimarsinan_fpga` clone · prerequisite: none.

**Trigger:** the owner is realizing the thesis deployment system physically — ODIN cores
(Frenkel, TBioCAS'19; `ChFrenkel/ODIN`) instantiated multi-core on HACC@NUS Alveo FPGAs as a
mimarsinan execution backend, with host segments on the same node's CPUs/MI210 GPUs. ODIN's
soma law (event-serial threshold checks, hard zero reset, saturating unsigned fixed-width
membrane, possibly >1 spike per neuron per timestep, per-pre-row weight sign) is not a point
the current axes system can declare, and no exporter emits physical memory images.

## Status (2026-08-21)

| Stage | State | Commits |
|---|---|---|
| Plan of record | committed, then revised under 4-judge adversarial review (§14) | docs(ODIN0) |
| P0 headroom extractions (subpackages, corrected census) | **DONE** — five pure extractions create the LOC headroom P1/P2 need; behaviour is relocation only (31/31 moved definitions AST-byte-identical) | refactor(ODIN0) — suite 12,048 passed / 32 skipped / 0 failed (identical to the parent), typecheck 0, budgets exit 0, architecture 90 passed |
| P1 axes + unified point-keyed capability + cell identity | **DONE after one adversarial fix cycle** — the two axes are DECLARED, the capability query is re-keyed on the resolved point, and cell identity discriminates it; every backend refuses the new point by name and nothing executes it. The verifier REFUTED the first commit on two defects (the cross-key contract was not total over a raw draft grid; loihi's step-level refusal named the mode instead of the axis); both are closed and covered | semantics(ODIN1) — golden regen +68 lines / **0 deletions** (4 keys x 17 fixtures), suite 12,324 passed / 32 skipped / 0 failed in ~61 s, typecheck 0, budgets exit 0, architecture 90 passed · fix(ODIN1) — suite **12,547 passed / 32 skipped / 0 failed** in ~56 s, golden snapshot UNCHANGED, typecheck 0, budgets exit 0, architecture 90 passed · mapping(ODIN1) — the canonical event-order SSOT (adjacency + row-pair contracts) with 15 tests; suite 12,562 passed |
| P2 torch serial-fold kernels | open | |
| P3 nevresim integration policy + resolver consolidation | open | |
| P4 exporter + platform declaration + feasibility gates | open | |
| P5 vendored RTL + cosim harness (semantic-equivalence gate R11a) | open | |
| P5.5 synthesis/implementation gate + resource table | open | |
| P6 geometry/semantics generator | open | |
| P7 FPGA kernel + XRT backend; **stock-core board bring-up first** (R11b) | open | |
| P8 measured campaign + fidelity + compile-limits study | open | |

## 0. Verified findings this plan is built on

Framework findings (verified at `17be1010`; full investigation and review reports in the
session record):

- **F1 — the variant axis governs transcoding, not the soma law.** `spiking_variant`
  feeds `is_streamed`/`is_windowed` → `pass_boundary_transfer()` → `VERBATIM`/`COLLAPSE`
  (`chip_simulation/activation_semantics.py:87-97`, `deployment_contract.py:111-120`,
  `mapping/support/schedule/pass_cut.py:27-34`). Reset lives in `firing_mode`
  (`models/nn/lif_kernels.py:58-66`), the compare in `thresholding_mode` (`:32`). ODIN's
  event-serial evaluation is a *fourth*, currently absent axis — a soma law orthogonal to
  all three (the streamed plan's §9 lesson: never conflate the execution discipline with
  topology). Note: `is_windowed` already means `variant == synchronized`, so the new axis
  must not reuse the word "windowed" (§14, J2-4).
- **F2 — ODIN's reset and compare already exist.** Hard zero reset is exactly
  `firing_mode="Novena"`; ODIN's `v ≥ θ` is `thresholding_mode="<="`. No new values on
  either axis.
- **F3 — every capability gate keys on the legacy mode string and would pass ODIN
  silently.** `_BACKEND_CAPS` is four bools keyed by `spiking_mode`; `supports_spiking_mode`
  ends in a bare `return caps.lif` (`chip_simulation/spiking_semantics.py:219-245`); a
  *second* capability table with a permissive fallback sits on
  `FiringStrategy.capabilities` (`firing_strategy.py:52-62`); the production guards route
  through `pipeline_helpers.py:23-32` → `require_backend_supported` → the same bare
  `caps.lif`. A traced ODIN config raises **nothing** end-to-end, and the coverage/cert
  identity collapses to the streamed-LIF cell in-tree
  (`hypervolume_axis_encoder.py:171`, three variant-less `CertificationCell` builders:
  `cost_extraction.py:336`, `spike_count_gate.py:134`,
  `deployment_record_assembly.py:122-123`). The fix is a **re-keying of the one existing
  query**, not a new gate beside it (§2.5).
- **F4 — nevresim already carries counts everywhere except three chokepoints.**
  `spike_t = std::int_fast8_t` flows through core state, inter-core wires, the output
  buffer, stdout, and SPKREC unmodified. Multiplicity is destroyed only at: all five input
  spike generators (`spike_train_spike_generator.hpp:35-36` re-binarizes the carry on
  ingest), the SPKTRN recorder (`spike_train_recorder.hpp:58-59`), and the SPKTRN parser
  (`execute_nevresim.py:84-86`).
- **F5 — the C++ integration step is the one hard-coded reduction.**
  `spiking_neuron_compute.hpp:27-49` is one `std::inner_product` then one `fire()`;
  `FirePolicy::fire` never sees the axon vector. The generic opening is an
  **integration-policy** template axis (`WholeVectorIntegrate` default /
  `EventSerialIntegrate<Bits>`), a defaulted parameter on `SpikingCompute` so every emitted
  policy string stays byte-identical when off. The membrane type belongs to the integration
  policy (`MembranePotential<W>` is a bare alias of the weight type).
- **F6 — the torch packed executor already materializes the per-axon vector and discards it
  one line later** (`packed_cycle.py:229-233`). `LIFCyclePolicy.advance()` takes a
  pre-reduced contribution and cannot express event-serial firing; the per-core reference
  loop (`lif_step.py:186`) already passes `(weight, axon-vector, θ, bias)`. The parity gate
  exercises the reference loop, the deployed metric the packed path — a serial kernel must
  land in both.
- **F7 — the NF cannot reach a serial kernel through `nn.Linear`.** The decomposition point
  is `segment_policies.py:226` using `PerceptronTransformer().get_effective_weight` — the
  mapper's own function — which makes the axon order a provable, not merely assertable,
  parity contract.
- **F8 — ascending axon index is already the implicit order in both implementations**
  (span gather fills monotonically; spans are RLE'd in axon order). Nothing asserts it. The
  bias has no axon index in nevresim (`bias_` added once per cycle) while the always-on
  axon is delivered at its mapped slot — the bias's position in the order is semantics and
  must be declared. **Adjacency of a multiplicity `e[a]` is free in the software twins
  (per-axon counts) but absent from the hardware's raw AER stream (k spikes = k separate
  row events, interleavable)** — adjacency is count-changing (§14, J3-1 counterexample) and
  must be manufactured by the v1 host router and the testbench driver, not assumed.
- **F9 — `ChipModel` is a uniform-geometry, semantics-free envelope; `HardCoreMapping`
  preserves per-core geometry.** Export pads to the global max and backfills phantom
  rows; a packer driven from `ChipModel` would program real cells from padding. All eight
  `ChipModel` consumers are terminal; a sibling exporter consuming `HardCoreMapping`
  disturbs nothing. Mutation hazards: the memoized shared `get_core_matrix()` grid;
  `ChipLatency`'s in-place `core.latency` writes.
- **F10 — the param-encoded bias is exactly one always-on tail row on the shared weight
  grid** (`compute_core_input_count` hard-codes `+1`; emitters write `core_matrix[-1,:]`;
  the quantizer clips the bias row with the weights). Multi-row bias is new machinery —
  out of scope (§11).
- **F11 — the additive-change protocol is mechanized, and the budget landscape is tight.**
  Golden snapshot: a default-preserving key produces a pure-addition diff. **Nineteen**
  non-allowlisted directories sit at exactly the 10-sibling cap, including `config_schema/`
  and `models/spiking/hybrid/` — the flat-sibling extraction is illegal there; relief must
  be subpackages. LOC headroom of files this plan edits: `activation_semantics.py` 291/300,
  `deployment_derivation.py` 295/300, `lif_step.py` 289/300, `packed_cycle.py` 289/300,
  `segment_boundary.py` **299/300**.
- **F12 — the streamed rollout is the template.** Plan lands alone; the numerically-inert
  phase first; the legal set opens **in the same commit** as its atol=0 FATAL gate with a
  teeth test and a non-degenerate witness; derived defaults are functions; every new
  predicate must be total over configs without the axis (the `t0_44` scar).

ODIN-target findings (verified in the upstream RTL, all 3,368 lines read; upstream HEAD
`1781931` "release-ready"):

- **F13 — stock geometry/interfaces.** 256×256 crossbar; 20-bit SPI, 40 SCK per
  transaction, **byte-granular with write masks** — a full core program is
  8192×4 + 256×16 = 36,864 transactions ≈ **236 ms/core at SCK = CLK/4 = 6.25 MHz**
  (25 MHz CLK), and multi-pass deployments pay it per reprogrammed pass — this number IS
  the measured `reprogram` physics; AER in 17-bit / out 8-bit 4-phase; neuron memory
  256×128 b (LIF layout: parameters + **state fields `vmem[77:70]`, `calcium[80:78]`,
  `caleak_cnt[85:81]`** — the state fields are part of the packer tables and the per-sample
  CLEAR); synapse memory 8192×32 b, 4 b/synapse = mapping bit + 3-bit magnitude.
- **F14 — stock soma law.** 8-bit unsigned membrane, saturate at 255, floor at 0;
  per-neuron 8-bit θ; `v ≥ θ` checked after **every** synaptic event; hard reset to 0;
  >1 spike per neuron per timestep is physical and order-dependent; leak off in deployment.
- **F15 — weight sign is per pre-synaptic row** (`SPI_SYN_SIGN`); the magnitude field is
  3-bit. The representable weight set is **symmetric** `{−7…+7}` — the framework's
  `quantization_bounds(4) = (−8, 7)` includes a value stock ODIN cannot hold, so the
  platform must declare a symmetric-magnitude range and the export must refuse `q_min`
  (§14, J3-3).
- **F16 — inference weight-freeze is doubly guaranteed** (`ca_en=0`; mapping bits 0 +
  `PROPAGATE_UNMAPPED=1` + `UPDATE_UNMAPPED=0`). `SPI_OPEN_LOOP=1` = externally-routed
  engine. Config registers have **no SPI readback and no reset value** (doc §4) — the
  memories do have readback; §7 rows 20–21 gate accordingly.
- **F17 — the stock RTL is not FPGA-synthesizable as vendored:** both memories are
  behavioral models that upstream explicitly instructs to replace with SRAM macros or BRAM
  (`synaptic_core.v:154`, `neuron_core.v:301`, upstream doc §5). Geometry is not parametric
  (128-bit neuron word, `N>>4` SPI generate loops, 32-bit synapse word, 13-bit synapse
  address, controller sweep bounds all hard-wired to 256). Full-fan-out event = 512 cycles;
  scheduler FIFO depth 32; no testbench ships upstream.
- **F18 — HACC@NUS toolchain is sufficient, with these operational facts** (from
  `Xtra-Computing/hacc_demo`): Slurm; Vitis 2021.2/2022.2 + XRT + shells for
  U250×6/U280×4/U55C×6/U50/VCK5000×3; hw_emu + VNC waveform sim; host-driven xclbin
  programming; `/data/${USER}` staging; `#SBATCH --account=slurm`. **The true U55C+MI210
  nodes are hacc-gpu2 and hacc-gpu3 only** (hacc-gpu1 pairs U55C with U250/U280; gpu3 also
  hosts VCK5000); their pools (`mi210_*`) run 7-day limits, while the bare U55C *shell*
  partitions are capped at **1 hour** and the 12 h pools are U250/U280/U50. Hardware
  builds run on `cpu_only` (hacc-node2, 7-day) or hacchead with `/tools/xilinx` Vitis
  2022.2 — never on a board reservation. The repo's `.slurmech.toml` targets a different
  cluster entirely (xlog1/H100); HACC submission is hand-written `sbatch` v1 (§5.4).

## 1. The semantic model — ODIN firing as declared points on two new generic axes

### 1.1 Authored keys (no target-specific config keys)

| Key | Section | Values (default first) | Meaning |
|---|---|---|---|
| `firing_granularity` | deployment | `per_cycle` \| `per_event` | when the threshold is evaluated: once per cycle on the reduced contribution (today's law), or after every arriving event occurrence in canonical order, emitting one spike and resetting per crossing (≥0 spikes per neuron per cycle) |
| `membrane_arithmetic` | deployment | `unbounded` \| `saturating_unsigned` | the membrane's arithmetic: today's unbounded signed accumulator, or clamp to `[0, 2^membrane_bits − 1]` on every update |
| `membrane_bits` | platform_constraints | int ≥ 0, default `0` | fixed membrane register width; `0` = not fixed-width. **Bits-driven like `weight_bits`:** `membrane_bits > 0` derives `membrane_arithmetic="saturating_unsigned"`; an explicit `unbounded` against a declared width is a keyed, remediable contradiction (mirror of the weight-quantization doctrine) |
| `weight_sign_granularity` | platform_constraints | `per_synapse` \| `per_axon` | where the weight sign physically lives. `per_axon` (ODIN's `SYN_SIGN`) implies: symmetric magnitude range `[−(2^(weight_bits−1)−1), +(2^(weight_bits−1)−1)]` (export refuses `q_min`), physical row-pair expansion with declared factor 2, and the expansion factor feeds the physical-cell/energy accounting (§5.2) |

Names deliberately avoid the taken vocabulary: `windowed` already means
`variant == synchronized`, and the default membrane is not "exact" (it is lattice-exact
only under armed conditions) — hence `per_cycle`/`per_event` and `unbounded` (§14, J2-4).

Registry shape (streamed precedent): `group="spiking"`, `domain="event"`,
`Category.ADVANCED`, `legal_values` + `derived_default` (functions, not constants) +
provenance, **no** `DEFAULT_DEPLOYMENT_PARAMETERS` entry for the two deployment keys — so
the golden-snapshot diff is a pure addition. Platform keys get defaults beside
`weight_bits`.

**The resolved point is one named object** — frozen
`SomaLaw(firing_mode, thresholding_mode, firing_granularity, membrane_arithmetic,
membrane_bits, bias_slot)` with exactly one constructor from the resolved contract — the
SSOT the capability query, the kernels, the cache key, the exporter manifest, and the
generator descriptor all consume (§14, J4-5). `bias_slot` is fixed `"tail"` in v1 (§2.3).

The stock-ODIN point, fully written:

```
core_semantics="spiking", spiking_family="lif", spiking_variant="streamed",
firing_mode="Novena", thresholding_mode="<=",
firing_granularity="per_event",
platform: membrane_bits=8 (⇒ saturating_unsigned), weight_bits=4,
          weight_sign_granularity="per_axon",
          cores=[{max_axons:128, max_neurons:256, count:P, has_bias:false}],
          allow_coalescing=false, allow_neuron_splitting=false
```

`has_bias:false` ⇒ param-encoded bias ⇒ effective logical fan-in **127** per core — a
named acceptance-scope limit (§11): stock ODIN has no inter-core partial-sum transfer, so
wide fan-in refuses via the existing `WideFanInUnsupportedError`, honestly.

The sync-fire generator variant (P6) is `per_cycle` with a wider `membrane_bits` and
`per_synapse` signs — the same vocabulary.

### 1.2 Legality and defaults (byte-identical default-off)

- `legal_firing_granularities(cfg)`: `("per_cycle",)` unless the resolved point is
  `(lif, streamed)` — i.e. `per_event` is declarable **only under the streamed variant**
  (a windowed hop collapses counts and would destroy multiplicity; a legal-but-refused
  point is what the legality harness exists to prevent; §14, J2-5). Total over partial
  configs; mvm short-circuit (the `t0_44` scar).
- `legal_membrane_arithmetics(cfg)`: `("unbounded",)` for TTFS; both for lif; derived
  default is bits-driven (§1.1).
- Cross-key validation: `per_event` requires `resolve_bias_mode(cfg) == "param_encoded"`;
  `saturating_unsigned` requires `membrane_bits ≥ 1`; under `per_event`,
  `lif_membrane_init` must satisfy `V0·θ` integral and `0 ≤ V0·θ < θ` (a keyed error, not
  a projection — the row-pair lemma's precondition, §2.3).
- Resolution ordering: the two deployment axes are folded **early**, by a sibling fold
  beside `fold_spiking_axes` (which itself still writes exactly four keys), so the recipe
  fold and `sim_enables` derivation see a resolved point without any raw `config.get`
  (§14, J1-5).

### 1.3 What stays unchanged

`spiking_family`, `spiking_variant`, `firing_mode`, `thresholding_mode`,
`spike_generation_mode` vocabularies and legality; `ActivationSemantics`, `mode_id`, both
legacy bridges, `fold_spiking_axes`; `NeuralBehaviorConfig`'s four default-free fields.

## 2. The execution contract

### 2.1 Scope and composability

The axes are a per-neuron soma law executed inside a neural segment; window **counts**
(integers) cross boundaries, and the boundary transform is deterministic on counts, so the
atol=0 gate composes across spans (the §9 test for correct scoping). An ODIN deployment
declares `spiking_variant="streamed"`, so the streamed gates and VERBATIM pass-carry arm
with zero edits. Per-cycle multiplicity is the new load-bearing degree of freedom inside a
segment; the NF↔SCM gate gains a raster-level comparison under `per_event` (§7 row 6).

### 2.2 The serial fold, defined once

For one neuron per cycle, axon slots `a = 0 … A−1` in canonical order, multiplicities
`e[a] ≥ 0`, signed logical weights `w[a]`, threshold θ, membrane `m`:

```
for a in 0..A-1:
    repeat e[a] times:                      # occurrences of one slot are ADJACENT (§2.3)
        m := sat(m + w[a])                  # saturating_unsigned: clamp to [0, 2^bits−1]
        if compare(θ, m): emit one spike; reset(m)     # Novena ⇒ m := 0
return per-neuron count emitted this cycle
```

Under `(per_cycle, unbounded)` the same interface degenerates to today's
integrate-then-fire-once, byte-identical. Four implementations — NF torch, HCM torch,
nevresim C++, RTL — implement exactly this; §7 proves it.

**Emission bound** (revised per §14, J3-5): `e_max` is a **propagated** per-wire quantity —
seeded `1` at every segment entry (the encode emits ≤1/cycle), and for each core,
`e_out(n) = ⌈(Σ_a max(w[a],0)·e_in(a) + (θ−1)) / θ⌉` (the `θ−1` term is the entering
membrane), propagated in topological order within the segment. The deployment refuses at
mapping/export time when any `e_out > 127` (the count-currency ceiling), **and** every
implementation carries a runtime assertion at the same bound so a violation fails loud
identically everywhere rather than overflowing one of the four.

### 2.3 The canonical event order (SSOT in `mapping/`)

The order is authored where axon slots are assigned — a new module beside the axon-source
assignment in `mapping/` (§14, J2-7). Codegen span compression, the exporter, the torch
kernels, the NF decomposition, the v1 host router, and the cosim testbench driver all
**consume** it; NF-order ≡ mapper-order is provable because both read
`get_effective_weight`.

1. Axon slots ascend: 0, 1, …, A−1.
2. Bias (always-on) rows sit at the tail (`SomaLaw.bias_slot="tail"`); nevresim's per-cycle
   `bias_` is 0 under `per_event`; torch folds `hw_bias` at the same declared position.
3. **Occurrences of one slot are adjacent.** Adjacency is count-changing (θ=5,
   `w=[+3,−3]`, `e=[2,1]`: adjacent = 1 spike; interleaved = 0). The software twins get it
   free (per-axon counts); the hardware does not — so the **v1 host router and the P5
   testbench driver are required to buffer one timestep, count per destination slot, and
   drain ascending slots with multiplicity adjacent**. A §7 gate feeds a deliberately
   sweep-interleaved stream and shows it is normalized (or refused), never silently
   executed (§14, J3-1).
4. Row-pair expansion maps logical slot `a` to physical rows `(2a, 2a+1)` =
   (excitatory, inhibitory). At most one of the pair is nonzero per (axon, neuron), and a
   zero add is a no-op **given `m < θ` on entry to the zero-magnitude event** — which
   holds after any event (both fire-paths leave `m < θ`) and at window start once §1.2's
   `V0·θ < θ` constraint holds. The lemma and its precondition are locked by a kernel-level
   property test.
5. The v2 on-fabric router inherits two hard constraints: preserve this order, and keep
   the boundary transform host-side (§14, J3-10).

### 2.4 Membrane representation, init, lattice

- Membrane type/bounds belong to the integration policy (torch kernel argument; C++
  `IntegrationPolicy::membrane_t`). `COMPUTE_DTYPE` and default-point state allocations
  untouched.
- `lif_membrane_init` under `per_event`: constrained per §1.2 (integral, `< θ`,
  non-negative), programmed into the neuron word's **state field** (F13), and applied once
  per window.
- The per-event lattice snap is **scoped strictly to `per_event`** (the default points'
  `membrane_integer_lattice` machinery is untouched — §14, J1-12); the quantum under a
  fixed-width integer membrane is the register LSB, derived on the `SomaLaw`.
- The membrane-readout decode (`Q_T = θ·c_T + m_T`) is refused under
  `saturating_unsigned` — typed refusal, not a silent skip.

### 2.5 Carrier and the unified refusal architecture

- `SpikingDeploymentContract` gains two frozen defaulted fields + resolved
  `membrane_bits`; the `SomaLaw` is constructed from the contract in exactly one place;
  executor entry points take it keyword-only, default-free.
- **One capability query, re-keyed — not a new gate** (§14, J2-1/J1-4 reconciled): the
  policy object returned by `policy_for_spiking_mode` carries the `SomaLaw`;
  `supports_backend`/`require_backend_supported` and `Backend.supports`/`require_supported`
  answer on the point when given a contract/policy (their production callers all have
  one), while a raw mode-string query keeps today's mode-keyed answer (the pinned legacy
  surface). `BackendSpikingCapabilities` gains `per_event_firing: bool = False` and
  `saturating_membrane: bool = False`; `FiringStrategy.capabilities`'s second table and
  its permissive fallback fold into the same query. Under the ODIN point: hcm (P2) and
  nevresim (P3) support it; sanafe, lava/loihi, training refuse **by name**;
  `sim_enables` derives from the same query via the early fold (§1.2), so unsupported
  backends are off by derivation and an explicit enable gets a keyed error.
- **Cell identity is in-tree and lands in P1** (§14, J1-3): `AxisCoordinates.firing`
  becomes a function of the resolved point (default-preserving for every existing point),
  and the three production `CertificationCell` builders pass the point's tag as `variant`
  — with a test asserting the ODIN point and streamed-LIF yield different `cell_key`s.
  `certification_observable()` and the conformance matrix become point-aware in P1.
- Cycle-atomic theorems refuse under `per_event`: the synchronized count executor,
  `advance()` on a pre-reduced contribution, per-hop retimed level stages. Mapping
  transformations that re-threshold a partial sum are refused under the point
  (`allow_neuron_splitting=false`, `allow_coalescing=false` on the platform; a keyed
  refusal guards the general case), and the fold-invariant ones (output tiling — per-neuron
  disjoint; identity relays — a θ=1/w=1 relay maps k events to k spikes exactly) are
  admitted with a §7 gate comparing NF against the **packed hard-core** executor, closing
  the identity-mapping blind spot (§14, J3-8).

### 2.6 The wire contract (v1) and honest reporting

Unchanged inter-segment contract: rates ∈ [0,1]. Multiplicity lives only inside a neural
segment. Decode is unbounded; final outputs exact; interior re-encode clamps at the
declared ceiling 1.0 with the currently-missing symmetric upper-rail warning + a seam-audit
ceiling class. **In v1 this holds on hardware because routing is host-mediated — the
boundary transform is host-side by construction; that property is a named v2-router
constraint, not an accident** (§14, J3-10). Raster-bytes accounting prices
`log2(k_max+1)` bits per slot from the declared point (default 1 — byte-identical). The
honesty ledger gains the discipline's row at P3.

## 3. Owner decisions (2026-08-21, verbatim where load-bearing)

| # | Question | Decision |
|---|---|---|
| 1 | ODIN-style firing support | "generically support ODIN-style firing, by extending the current SSOT contracts. The elegance of the spike semantics must be retained and the design integration must stay generic (no ODIN-specific config). Mathematical parity must be retained throughout the entire pipeline." |
| 2 | Non-default semantics | "a code generator for non-default spike semantics in ODIN deployments. LIF datapath modification should be allowed. But, the default ODIN chip must be supported out of the box." |
| 3 | Non-default geometry | "Supporting non-default core geometry on ODIN must be supported… default ODIN chip must be deployable. This is a hard requirement." |
| 4 | Compile limits | "study limits of what we can compile into FPGA later once we have the infrastructure" (→ P8) |
| 5 | Isolation | "Create a mimarsinan_fpga clone repo… we only merge to real mimarsinan once we are sure that the infrastructure is elegant and correct." |
| 6 | No-impact bar | "this new support cannot impact previously existing behavior, correctness, code elegance, and performance." |
| 7 | Target node | U55C + MI210 (hacc-gpu2/3), Vitis RTL-kernel flow (approved). |
| 8 | Semantics path | Stock-exact contract first, sync-fire variant as generator flagship (approved). |
| 9 | Signed weights | Row-pair expansion below the mapper; logical geometry declared (approved). |
| 10 | v1 routing | Host-mediated inter-core routing, on-fabric router v2 (approved). |

"Deployable" (directive 3) is read in two stages, both required: **R11a** semantic
equivalence of the exported images against the stock RTL (Verilator, P5) certifies
correctness; **R11b** the stock core synthesizes/implements (P5.5) and is the **first**
hardware bring-up on the board (P7) — cosim alone does not discharge the word (§14, J4-2).

Designer decisions within the directives: two axes over a variant value (§1.1); bits-driven
membrane arithmetic (§1.1); `per_event` restricted to the streamed variant (§1.2); wire
ceiling 1.0 in v1 (§2.6); bias at the order's tail (§2.3); exporter consumes
`HardCoreMapping` (F9); no new top-level module — exporter at `mapping/export/odin/`,
backend at `chip_simulation/odin_fpga/`, RTL-only under `hw/` (§5.1); BARRIER by verified
deterministic cycle bound, vendor tree strictly untouched (§5.4).

## 4. SSOT map (every mechanism has exactly one home)

| Mechanism | SSOT home | Consumers | Pin |
|---|---|---|---|
| Axes vocabulary + legality + derived defaults + early fold | `chip_simulation/activation_semantics.py` + P0 subpackage | registry entries, wizard, derivations, recipe fold | golden snapshot; legality harness |
| `SomaLaw` (the resolved point) | one frozen dataclass, one constructor from the contract | capability query, kernels, cache key, exporter manifest, generator descriptor, cell identity | point-construction tests |
| Point-keyed capability query | the re-keyed existing query (`supports_backend` chain) | registry validation, steps, cross-sim applicability, sim_enables | refusal tests + capability-guard audit |
| The serial fold (torch) | `models/spiking/hybrid/serial/` kernel | `SerialLIFCyclePolicy.step`/`advance_events`, NF decomposition | packed↔reference equivalence; NF↔SCM atol=0 |
| The serial fold (C++) | `nevresim …/integration_policy/event_serial_integrate.hpp` | `SpikingCompute<Fire, Integration>` | nevresim↔HCM parity; constexpr self-tests |
| Canonical event order (incl. adjacency + drain rule) | new module in `mapping/`, beside axon-slot assignment | codegen spans, exporter, torch kernels, NF, v1 host router, cosim tb driver | cross-implementation order test; row-pair lemma property test; sweep-order normalization gate |
| Sequencer program format (versioned schema: CONFIG/CLEAR/INJECT/TREF/BARRIER/READOUT) | one emitter/decoder in `mapping/export/odin/` | exporter (write), cosim tb (read), XRT runtime (read) | golden round-trip; ≥2-sample cosim |
| Memory-image bit layouts (params + state fields) | `mapping/export/odin/` packer tables | exporter, cosim tb, runtime, SPI readback gate | pack/unpack golden; SPI readback byte-compare |
| nevresim reset+compare resolver (the two duplicated pairs) | one total resolver, consolidated at P3 | codegen + behavior config | mutation-tested equivalence; raises on unknown |
| Boundary ceiling + upper-rail warn | `spiking/segment_boundary.py` (post-P0 relief) | HCM, NF, compute-boundary, v1 runtime | boundary locks; seam-audit class |
| Platform physics | `profiles/odin.json` (exists) + FPGA-instance profile | cost/fidelity | correlation case; area pin vs 86,400 µm² with the expansion factor |

## 5. The exporter, the RTL tree, and the generator

### 5.1 Repo layout (revised per §14, J4-12/J2-12)

- `hw/` holds **only** RTL, templates, Makefiles, constraints: `hw/vendor/odin/`
  (byte-identical to `ChFrenkel/ODIN` @ `1781931`, LICENSE + PROVENANCE.md + a recorded
  file-manifest hash gated at P5; a copy rather than a submodule because HACC nodes are
  offline and the license notice for the `hw/` subtree is self-contained), `hw/fpga/mem/`
  (the BRAM wrapper substitution for the two behavioral memories — the upstream-mandated
  implementation step (F17), a reviewed overlay selected by file order, never an edit to
  `hw/vendor`), `hw/gen/` (variant templates; generated and template RTL carry the
  UCLouvain/Solderpad header + a statement-of-changes per SHL-2.0 §4(b)), `hw/tb/`
  (Verilog testbench sources), `hw/fpga/` (kernel wrapper, xo/xclbin scripts). A root
  NOTICE paragraph declares the subtree's license.
- **Every line of Python lives under `src/mimarsinan/`**, typechecked and budgeted: the
  packer/exporter/program emitter at `mapping/export/odin/` (subpackage beside
  `chip_export.py`), the cosim harness driver and later the XRT backend at
  `chip_simulation/odin_fpga/` (the nevresim/sanafe/lava_loihi precedent). No new
  top-level module; the root doc's module count fix (18→19) lands at P0 as the D7
  correction.
- `.slurmech.toml` gains `hw/**` at **P5** (the suite reads the tree from tests);
  `verilator` is version-pinned with a loud-skip guard; heavy gates run under a named
  runner (`scripts/hw_tests/`), not the default suite (§7 markers).

### 5.2 The exporter (P4)

Consumes `HardCoreMapping` after `ChipLatency.calculate()`; fresh arrays only. Emits per
core: the neuron-memory image (parameters **and state fields**: `vmem`, `calcium`,
`caleak_cnt` initialized per §2.4), the synapse-memory image (row-pair expanded, symmetric
magnitudes — a matrix containing `q_min` refuses with a keyed error, gated), `SYN_SIGN`,
the config-register list, the sequencer program (now including a per-sample **CLEAR**
stage rewriting the neuron state bytes — the software twins zero membranes per sample and
the chip must too, §14 J3-2), and the manifest (SomaLaw, geometry, ordering, emission
bounds, feasibility evidence).

Feasibility gates at mapping/export time (§14, J3-3/4/5): θ within
`[1, 2^membrane_bits−1]` — and because θ *is* the folded quantization scale
(`θ = ⌊q_max/w_max⌋` or the trained integral scale), this is a real scope predicate whose
remediation path is the existing scale/threshold adaptation ladder targeting the ceiling,
stated with the keyed error; the propagated emission bound; `q_min` refusal; fan-in ≤ 127.
Ledgers stay logical **and** the declared expansion factor is wired into the physical
accounting (`cells_physical`, `params_bytes`' physical twin, `synaptic_events`, the
physics multiplicands), pinned against `odin.json`'s 86,400 µm²/64k-synapse core (§14,
J2-6). Platform registration goes through the non-literature seam in `imc_platforms.py`
with its own provenance and eligibility class; the literature-transcription literal `12`
is untouched (§14, J2-10).

### 5.3 The generator (P6)

`CoreSpec` is a **projection** of one core-type mapping (reusing the `CORE_FIELDS` names,
`has_bias` included) plus the resolved `SomaLaw` (§14, J2-8) — it cannot drift from the
platform declaration. It emits variant RTL from `hw/gen/`, packer tables, and the
descriptor; the stock spec is a vendored passthrough asserted byte-identical to
`hw/vendor/odin`. Flagship variant: sync-fire (`per_cycle`, 16-bit membrane, per-synapse
signs), bit-matching the existing streamed-LIF contract when its no-saturation bound
holds (gate-checked).

### 5.4 The physical backend (P7, summarized; own design doc lands with P7)

Vitis RTL kernel (P cores + SPI masters + AER bridges + sequencer + tagged capture);
`hw_emu` first; **stock-core bring-up precedes any generated variant** (R11b). BARRIER is
a **verified deterministic cycle bound** (the sequencer knows its injected event count;
worst-case drain = f(count, 512-cycle sweeps, FIFO 32, output-handshake stalls), verified
against cosim on adversarial cases) — the vendored tree carries no patch at all. Config
registers have no readback: the runtime re-programs them on every power-cycle/session
start and the P5 tb shadow-asserts them; memories are verified by SPI readback
byte-compare (§7 rows 20–21). Build host: `cpu_only`/hacchead (F18); staging via
`/data/${USER}`; submission = hand-written `sbatch` v1 (a HACC slurmech profile is a
recorded option, not assumed); `pyxrt` gets `OPTIONAL_TOP_LEVEL_DEPS` + lazy-import
entries with stated reasons and a `DependencyBoundary` with non-empty guards. Per-pass
SPI reprogramming (~236 ms/core, F13) is the measured `reprogram` physics and is recorded
into the deployment record's timing fragments.

## 6. Config schema + wizard surface

Registry entries per §1.1; domain tagging auto-injects the existence gate and mvm
refusals; wizard options render from `legal_values`; the hardcoded
`firing_modes_by_spiking` duplicate in `gui/wizard/schema.py:31-36` is replaced by the
legality call **within P1's touched work** (it duplicates a table P1 edits); a `docs/ux/`
entry + wizard-representability audit + screenshot run land with P1 (§14, J4-18).

## 7. Verification & no-regression matrix

Markers: rows tagged [slow] run as `slow`+`integration` under the named runner
(`scripts/hw_tests/` for RTL rows), skip loud on missing toolchain; everything else lives
in the default suite (§14, J1-7).

| # | Surface | Gate | Phase |
|---|---|---|---|
| 1 | Resolved config surface | golden snapshot regen; diff pure-additive; global recipe surface bit-identical | P1 |
| 2 | Default suite + typecheck | 12,048+ passed / 0 failed ≤2 min; typecheck 0 | every |
| 3 | Backend registry outputs | pinned selection tests untouched; frozen legacy baseline untouched | P1, P7 |
| 4 | Unified refusal | ODIN point × {sanafe, lava, loihi, training, nevresim-pre-P3} refuses by name at the step-level guards (the seven existing call sites); sim_enables off; explicit enable ⇒ keyed error; conformance matrix + `certification_observable` point-aware | P1 |
| 5 | Cell identity | ODIN point and streamed-LIF produce **different** hypervolume coordinates and `CertificationCell.cell_key`s; default points unchanged | P1 |
| 6 | NF↔SCM exactness | window-count atol=0 gate arms unchanged; + raster-level atol=0 under `per_event`; teeth test; non-degenerate witness pinned | P2 |
| 7 | Kernel equivalence | packed (`advance_events`) ↔ reference (`step`) bit-equal under the point | P2 |
| 8 | Sweep-order normalization | a deliberately interleaved event stream is normalized to canonical adjacency (or refused); never silently executed | P2 |
| 9 | Row-pair lemma + order | kernel property test incl. the `V0·θ < θ` precondition; cross-implementation order test; bias-tail test | P2, P4 |
| 10 | NF ↔ packed HCM under the point | mapping-transformation invariance: output tiling + relays admitted and exact; splitting/coalescing refused | P2/P4 |
| 11 | Default-point byte-identity | identical policy objects; nevresim `main.cpp` byte-identical; cache keys unchanged (a `semantics_point` sub-object carrying all three values, omitted when default; discrimination test incl. `membrane_bits` 8 vs 16) | P2, P3 |
| 12 | Typed refusals | path-C, `advance()`, retimed levels, membrane readout, `V0` violations, `q_min`, θ-ceiling, emission bound, fan-in — each a keyed error + runtime assertions for the emission bound in all implementations | P2–P4 |
| 13 | nevresim ↔ HCM [slow] | parity rows at atol=0 (counts) + counted-raster carry round-trip; SPKREC↔SPKTRN sum identity under counts | P3 |
| 14 | Exporter | pack/unpack golden round-trip incl. state fields; sequencer-program schema round-trip; feasibility gates fire on constructed violations | P4 |
| 15 | **R11a — semantic equivalence (hard-req half 1)** [slow] | end-to-end map→quantize→export→Verilator cosim of vendored stock RTL (+BRAM overlay): per-neuron per-window counts ≡ nevresim ≡ HCM at zero difference, over **≥2 consecutive samples**, with a θ-near-ceiling witness | P5 |
| 16 | Vendor integrity | `hw/vendor/odin` manifest hash ≡ recorded upstream `1781931` manifest | P5 |
| 17 | SPI programming | tb programs via SPI, reads back **full** neuron+synapse memories, byte-compares to the exporter image; config regs shadow-asserted in tb | P5 |
| 18 | BARRIER bound | the deterministic drain bound ≥ observed drain on adversarial cosim cases; an event after the bound is a test failure | P5 |
| 19 | **P5.5 — synthesis/implementation gate (hard-req half 2a)** [slow] | vendor+overlay synthesizes and implements for the U55C shell; emits the per-core LUT/FF/BRAM/URAM table (P8's input artifact) | P5.5 |
| 20 | Variant cosim [slow] | per-`CoreSpec` cosim vs nevresim twin; stock spec byte-identical to vendor | P6 |
| 21 | **R11b — stock board bring-up (hard-req half 2b)** [slow] | the stock core runs on a U55C board first; on-board certificate campaign vs nevresim (counts, atol=0); measured Timing/Energy fragments | P7 |
| 22 | Wizard surface | representability audit + `docs/ux/` entry + screenshot run for the new keys | P1 |
| 23 | Performance non-impact | suite wall within baseline band; tier-0 spot check on pipeline-behavior phases | every |

## 8. Phasing

- **P0 — headroom extractions** (pure moves): `config_schema/derivation/` **subpackage**
  (the flat sibling is illegal — `config_schema/` is at the 10-cap and unallowlisted);
  `models/spiking/hybrid/serial/` **subpackage** seeded with extracted helpers from
  `lif_step.py` (289) and `packed_cycle.py` (289); relief for `segment_boundary.py` (299)
  and `activation_semantics.py` (291); the D7 module-count doc fix. Every target checked
  against the corrected 19-at-cap census (F11). *Acceptance:* §7 rows 2, 23; moves only.
- **P1 — axes + unified capability + cell identity** (§1, §2.5; `SomaLaw`; early fold;
  order SSOT module lands here so P2∥P4 holds; wizard row). *Acceptance:* §7 rows 1–5, 22.
- **P2 — torch serial-fold kernels** (kernel SSOT; policies; NF decomposition; refusals;
  upper-rail warn; per-event lattice). Legal set opens for hcm **with** its gates.
  *Acceptance:* §7 rows 6–12 (torch half).
- **P3 — nevresim** (integration policy; counted rasters; cache sub-object; resolver
  consolidation — the two nevresim reset resolvers **and** the two compare resolvers,
  total, raising on unknown; tier cell; ledger row). *Acceptance:* §7 rows 11–13.
- **P4 — exporter + platform + feasibility** (§5.2). *Acceptance:* §7 rows 9–10, 12, 14.
- **P5 — vendored RTL + BRAM overlay + cosim harness**. *Acceptance:* §7 rows 15–18.
- **P5.5 — synthesis gate + resource table**. *Acceptance:* §7 row 19.
- **P6 — generator** (§5.3). *Acceptance:* §7 row 20.
- **P7 — FPGA kernel + XRT backend + HACC packaging; stock-core first** (§5.4; own doc).
  *Acceptance:* §7 rows 3, 21.
- **P8 — measured campaign, fidelity, compile-limits study** (consumes P5.5's table).

**Order:** P0 → P1 → (P2 ∥ P4) → P3 → P5 → P5.5 → (P6 ∥ P7) → P8. One commit per phase +
separate behavior-free extraction commits; each lands independently green.

## 9. Tier cells and coverage

New cells only via `templates/generate.py` (P3). Cell identity is point-aware from P1
(§2.5) — in-tree, not deferred. What *is* merge-time: the campaign workspace's ROADMAP
D-layer row and the screening artifacts for any axis-collapse claims; the physical backend
is not a collapsed member of the faithfulness axis until a screening artifact earns it.

## 10. Findings reported, not acted on (owner sign-off required)

| # | Finding | Severity |
|---|---|---|
| D1 | The synchronized count executor never reads `firing_mode`; latently wrong for `Novena` today; it is the oracle for `count_alignment` certificates (`sync_counts.py:11-79`, `lif_step.py:106-108`) | high |
| D2 | Reset-law and compare-law resolvers are each duplicated with opposite unknown-value fallbacks (`behavior_config.py:67-73` vs `generate_main.py:12-19`). **P3 consolidates the two nevresim pairs** (this plan's touched work); the lava/sanafe/training reset dispatchers remain reported | med |
| D3 | Membrane-readout gates never consult the reset law — the `Q_T` identity is already invalid under `Novena` today | high |
| D4 | `recording/records.py:2` states order-independence as a theorem; false under `Novena`. P2 rewrites the comment (doc-only) | low |
| D5 | The boundary clamp-loss proof is one-sided (no upper-rail policy anywhere). P2 adds the warn for the new point; the historical silence is recorded | med |
| D6 | ~20 raw `config.get` reads of semantics keys outside the contract (list in the R1 report). The `gui/wizard/schema.py` hardcode is **fixed in P1** (it duplicates a table P1 edits); the rest are reported | med |
| D7 | `AGENTS.md:11-33` contradicts CLAUDE.md on doc placement; root module count stale (fixed at P0) | low |
| D8 | SANA-FE TTFS input path passes spike times into a bool-mask attribute (`neuron_model.py:137-143` vs `sana_fe/src/models.cpp:816`) | recorded |
| D9 | `spike_t = std::int_fast8_t` width is implementation-defined — a cross-compiler bit-exactness hazard for nevresim generally | low |

## 11. Risks / open items

- **θ-ceiling scope** (J3-4): θ is the folded quantization scale; `membrane_bits=8`
  refuses layers whose scale lands above 255. The remediation is the adaptation ladder
  targeting the ceiling; how much of tier-0 fits stock ODIN is a *measured* P4/P5 result,
  and the answer bounds what "out of the box" covers. The generator's wider-membrane
  variants are the designed relief.
- **Fan-in ≤ 127 logical** on stock cores (no partial-sum transfer): named scope limit.
- **Emission-bound refusals** on real networks: measured at P4; remedy is scale/θ
  adaptation, never a wider count clamp.
- Generalized multi-level comb (rates >1 on wires), retimed levels under `per_event`,
  serial-aware QAT (training gap measured and carried on the ledger), `saturating_signed`
  membranes, multi-row bias: all deliberately not built; each becomes real work only when
  a study demands it.
- **Board-side**: node contention on hacc-gpu2/3 (both busy in the sampled `sinfo`); 1-h
  shell partitions vs multi-hour builds (build on `cpu_only`); per-pass ~236 ms/core SPI
  reprogramming dominating multi-pass latency (it is also the *measurement* we want).
- First-run cache-cold flake (13 sanafe-arch tests on a cold clone): re-run is
  authoritative; watch the file.

## 12. Prose surfaces to re-check per phase

Registry `doc=`/`empty_means=`; advisory `detail=`; recipe rationale constants; CLAUDE.md
SSOT list; root + touched module ARCHITECTURE rows; `how_networks_reach_the_chip.md`;
`templates/generate.py` coverage notes; this plan's Status table.

## 13. Merge protocol (owner directive 5)

Work lands on `odin-fpga-infra` only; nothing is pushed. Merge to real mimarsinan on the
owner's judgment, as one reviewed branch, after re-running the full gate set on a fresh
clone of real main, re-basing, and extending the campaign-workspace surfaces (ROADMAP row,
screening artifacts). History stays reviewable: one commit per phase, subjects as claims,
bodies carrying the gates' numbers.

## 14. Adversarial review record (2026-08-21)

Four independent judges (behavior preservation; genericity/elegance; mathematical parity;
directive completeness) attacked the committed plan of record; every finding was verified
against source before acceptance. Accepted and applied: the P0 subpackage restatement and
the corrected 19-at-cap census (J1-1/2/9, J2-2); in-tree cell identity moved to P1
(J1-3/J4-3); the unified point-keyed capability query replacing a third gate
(J2-1 + J1-4 reconciliation: production seams typed, raw-string queries stay mode-keyed);
the early fold for `sim_enables` (J1-5); the three-field cache sub-object (J1-6); §7
markers and the named runner (J1-7/J4-11); point-aware conformance matrix at P1 (J1-8);
lattice subsumption scoped to `per_event` (J1-12); order SSOT into P1/`mapping/`
restoring P2∥P4 (J1-13, J2-7); vocabulary renames `per_cycle`/`per_event`/`unbounded`
(J2-4); bits-driven membrane arithmetic (J2-3); streamed-only legality for `per_event`
(J2-5); expansion factor wired into physical cost accounting with the area pin (J2-6);
`CoreSpec` as a projection (J2-8); `has_bias:false` + coalescing declarations in the
worked point (J2-9, J3-9); IMC registration via the non-literature seam (J2-10); resolver
consolidation scope stated once (J2-11, J3-11); modules re-homed with no Python under
`hw/` (J2-12, J4-12); wire-adjacency as a v1 host-router/tb requirement with the
normalization gate (J3-1); the per-sample CLEAR stage, state fields in the packer, and the
≥2-sample cosim (J3-2/6); symmetric-magnitude declaration with `q_min` refusal (J3-3);
the θ-ceiling feasibility predicate and near-ceiling witness (J3-4); the propagated
emission bound with runtime assertions (J3-5); the untouched vendor tree with the BRAM
overlay + cycle-bound BARRIER replacing the probe-wire patch (J3-7, J4-1); the NF↔packed
invariance gate (J3-8); v1-scoped boundary claim + the second v2-router constraint
(J3-10); R11a/R11b split with P5.5 (J4-1/2); SPI readback + shadow-assert gates (J4-6);
the corrected ~236 ms SPI figure (J4-7); HACC operational corrections — build host,
partitions, hacc-gpu2/3, hand-written `sbatch` v1 (J4-8/9/14/15); the program-format SSOT
row (J4-10); license notices for derivative RTL (J4-16); the vendor manifest gate and
pinned SHA `1781931` (J4-17); the wizard row and the in-P1 fix of the schema hardcode
(J4-18); P8's input artifact from P5.5 (J4-19).

## Deliberately unchanged (the protection contract)

`ChipModel` and all eight consumers; `hard_cores_to_chip`; `fold_spiking_axes` (exactly
four keys); `ActivationSemantics`, `mode_id`, both legacy bridges;
`NeuralBehaviorConfig`'s shape; `_legacy_backend_specs`; every clamp/kernel/encoder on
`(per_cycle, unbounded)` points, including `membrane_integer_lattice`; `VERBATIM_BACKENDS`
for existing points; existing tier cells, numbers, axes; the default suite's wall-time
band; the streamed plan's invariants; `hw/vendor/odin` byte-for-byte.

## Verification protocol (every stage)

Tests first; `python -m pytest` ≤2 min green; `./scripts/typecheck.sh` zero; ratchets and
budgets only tighten (no new function-level imports, no new broad excepts, no allowlist
growth); templates only via `templates/generate.py`; ARCHITECTURE.md per touched module;
load-bearing guards mutation-checked; byte-identity A/B wherever a default-off claim is
made (golden-snapshot zero-deletion proof; emitted-artifact byte-compare); generic-only —
no target constants framework-side; GUI changes verified in pixels against a written UX
entry; per-stage commits, no AI-attribution trailers.

## Program-level acceptance ("ODIN support" defined)

1. The stock ODIN chip is declarable in generic vocabulary and deployable end-to-end
   within its measured feasibility scope (θ ceiling, fan-in 127, emission bounds — §11):
   exported images reproduce nevresim and torch per-neuron window counts at zero
   difference under RTL cosimulation of the byte-identical vendored core with the
   upstream-mandated memory overlay (R11a), the design closes synthesis/implementation
   for the U55C shell (P5.5), and the stock core is the first configuration brought up on
   the board (R11b).
2. A `CoreSpec` with non-default geometry and/or the sync-fire law generates a core +
   packer + descriptor passing the same cosim gate.
3. Every pre-existing configuration resolves, executes, and prices byte-identically, at
   unchanged suite wall time, with zero weakened checks.
4. The physical backend reproduces nevresim counts on HACC hardware and lands measured
   timing/energy fragments (including per-pass reprogramming cost) in the deployment
   record.
5. An ODIN-point run can never be recorded under a streamed-LIF identity: coverage
   coordinates, certification cell keys, certificates, and refusals are point-aware
   in-tree (P1), gated by §7 row 5.
