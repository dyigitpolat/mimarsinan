# ODIN firing semantics as declared axes + variable-geometry RTL export — engineering plan

2026-08-21 · status: **approved plan of record** (owner directives 2026-08-21, recorded verbatim in §3) ·
baseline `17be1010` on branch `odin-fpga-infra` in the `mimarsinan_fpga` clone ·
prerequisite: none (P0 extractions create their own headroom).

**Trigger:** the owner is realizing the thesis deployment system physically — ODIN cores
(Frenkel, TBioCAS'19; `ChFrenkel/ODIN`) instantiated multi-core on HACC@NUS Alveo FPGAs as a
mimarsinan execution backend, with host segments on the same node's CPUs/MI210 GPUs. ODIN's
soma law (event-serial threshold checks, hard zero reset, saturating unsigned fixed-width
membrane, possibly >1 spike per neuron per timestep, per-pre-row weight sign) is not a point
the current axes system can declare, and no exporter emits physical memory images.

## Status (2026-08-21)

| Stage | State | Commits |
|---|---|---|
| Plan of record | this document | docs(ODIN0) |
| P0 headroom extractions | open | |
| P1 axes + refusal architecture | open | |
| P2 torch serial-fold kernels | open | |
| P3 nevresim integration policy | open | |
| P4 exporter + platform declaration | open | |
| P5 vendored RTL + cosim harness (stock deployability) | open | |
| P6 geometry/semantics generator | open | |
| P7 FPGA kernel + XRT backend (HACC) | open | |
| P8 measured campaign + fidelity + compile-limits study | open | |

## 0. Verified findings this plan is built on

Framework findings (verified at `17be1010`; full investigation reports in the session record):

- **F1 — the variant axis governs transcoding, not the soma law.** `spiking_variant`
  feeds `is_streamed`/`is_windowed` → `pass_boundary_transfer()` → `VERBATIM`/`COLLAPSE`
  (`chip_simulation/activation_semantics.py:87-97`, `deployment_contract.py:111-120`,
  `mapping/support/schedule/pass_cut.py:27-34`). Reset lives in `firing_mode`
  (`models/nn/lif_kernels.py:58-66`), the compare in `thresholding_mode` (`:32`). ODIN's
  event-serial evaluation is a *fourth*, currently absent axis — a soma law orthogonal to all
  three (the streamed plan's §9 lesson: never conflate the execution discipline with topology).
- **F2 — ODIN's reset and compare already exist.** Hard zero reset is exactly
  `firing_mode="Novena"` (`lif_kernels.py:64-65`, `nevresim .../zero_reset.hpp`,
  `sanafe_reset_mode → "hard"`); ODIN's `v ≥ θ` is `thresholding_mode="<="`. No new values
  on either axis.
- **F3 — every capability gate keys on the legacy mode string and would pass ODIN silently.**
  `_BACKEND_CAPS` is four bools keyed by `spiking_mode`; `supports_spiking_mode` ends in a
  bare `return caps.lif` (`chip_simulation/spiking_semantics.py:219-245`). A traced ODIN
  config raises **nothing** end-to-end: lava/SANA-FE/training all claim `lif=True`, the run
  executes different physics, and the number is recorded as coverage of the streamed-LIF cell
  (`hypervolume_axis_encoder.py:171`, `certification.py:82-85`). The refusal architecture is
  the load-bearing half of P1.
- **F4 — nevresim already carries counts everywhere except three chokepoints.**
  `spike_t = std::int_fast8_t` flows through core state, inter-core wires, the output buffer,
  stdout, and SPKREC unmodified (`nevresim/include/common/types.hpp:16`,
  `compute_policy_base.hpp:59-63`, `spike_count_recorder.hpp:59-76`). Multiplicity is
  destroyed only at: all five input spike generators (`spike_train_spike_generator.hpp:35-36`
  re-binarizes the carry on ingest), the SPKTRN recorder (`spike_train_recorder.hpp:58-59`
  clamps to 1, one ASCII char per cycle), and the SPKTRN parser
  (`execute_nevresim.py:84-86` rejects non-binary symbols).
- **F5 — the C++ integration step is the one hard-coded reduction.**
  `spiking_neuron_compute.hpp:27-49` is one `std::inner_product` then one `fire()`;
  `FirePolicy::fire(threshold, membrane)` never sees the axon vector. The generic opening is
  an **integration-policy** template axis (`WholeVectorIntegrate` default /
  `EventSerialIntegrate<Bits>`), a defaulted parameter on `SpikingCompute` so every emitted
  policy string stays byte-identical when off. The membrane type belongs to the integration
  policy (`MembranePotential<W>` is a bare alias of the weight type, `types.hpp:27-28`).
- **F6 — the torch packed executor already materializes the per-axon vector and discards it
  one line later.** `packed_cycle.py:229-233`: `grouped = signals.reshape(B, group, n_axons)`
  feeds an `einsum` that destroys arrival order. `LIFCyclePolicy.advance()` takes a
  pre-reduced contribution and **cannot** express event-serial firing; the per-core reference
  loop (`lif_step.py:186`) already passes `(weight, axon-vector, θ, bias)` and needs zero
  call-site change. The parity gate exercises the reference loop
  (`forward_with_recording ⇒ recording=True`), the deployed metric the packed path — a
  serial kernel must land in both or gate and metric measure different physics.
- **F7 — the NF cannot reach a serial kernel through `nn.Linear`.** The segment policy loops
  cycles and re-invokes the fused layer (`spiking/segment_policies.py:219-240`); axon
  identity is erased before the neuron sees it. The decomposition point is
  `segment_policies.py:226` using `PerceptronTransformer().get_effective_weight` — the same
  function the mapper uses (`mapping/mappers/perceptron_mapper.py:23-24`), which makes the
  axon order an explicit, assertable parity contract.
- **F8 — ascending axon index is already the implicit order in both implementations.**
  nevresim's span gather fills `signals[dest++]` monotonically (`compute_policy_base.hpp:35-64`,
  spans RLE'd in axon order by `compress_sources_to_spans`); the torch span fill mirrors it.
  Nothing asserts it. The bias has **no** axon index in nevresim (`bias_` added once per
  cycle, `spiking_neuron_compute.hpp:34`) while the always-on axon is delivered at its mapped
  slot — under event-serial evaluation the bias's position in the order is semantics and must
  be declared.
- **F9 — `ChipModel` is a uniform-geometry, semantics-free envelope; `HardCoreMapping`
  preserves per-core geometry.** Export pads every core to the global max and backfills
  phantom neurons/`is_off` rows (`mapping/export/chip_export.py:31-46,117-122`); an ODIN
  memory-image packer driven from `ChipModel` would program real cells from padding. All
  eight `ChipModel` consumers are terminal emitters or profiling reads — a sibling exporter
  consuming `HardCoreMapping` disturbs nothing. Two mutation hazards: the memoized shared
  `get_core_matrix()` grid (must not be mutated) and `ChipLatency.calculate()`'s in-place
  `core.latency` writes.
- **F10 — the param-encoded bias is exactly one always-on tail row on the shared weight
  grid.** `compute_core_input_count` hard-codes `+1`; the emitters write `core_matrix[-1,:]`
  and append `IRSource(node_id=-3)`; `assert_bias_scale_param_encodable` refuses a second
  scale; the quantizer clips the bias row to `±q_max` with the weights
  (`mapping/platform/mapping_structure.py:13-21`, `mapping/ir_mapping_class_emit.py:77-88`,
  `mapping/export/chip_quantize.py:58-59`). Multi-row bias would be new machinery — out of
  scope (§11).
- **F11 — the additive-change protocol is mechanized.** Golden resolution snapshot: a
  default-preserving key produces a pure-addition diff, provable by
  `git diff -U0 … | grep '^-[^-]' | wc -l == 0`. Registry keys must appear in
  `CONFIG_KEYS_SET`, carry `legal_values` + `derived_default` + provenance, and pass the
  legality/derived-defaults/wizard-representability audits. `activation_semantics.py` is at
  291/300 LOC and `deployment_derivation.py` at 295/300 — extractions must precede the
  feature (precedent `1c18fa21`). Eleven directories are at the 10-sibling cap.
- **F12 — the streamed rollout is the template.** Plan lands alone; the numerically-inert
  phase first (vocabulary in the enum, legality locked); the legal set opens **in the same
  commit** as its atol=0 FATAL gate with a teeth test and a non-degenerate witness; derived
  defaults are functions; every new predicate must be total over configs without the axis
  (the `t0_44` mvm-leak scar, `activation_semantics.py:264-277`); docs and tier cells close.

ODIN-target findings (verified in the vendored-to-be RTL, all 3,368 lines read):

- **F13 — stock geometry/interfaces.** 256×256 crossbar; 20-bit SPI (40 SCK/transaction,
  SCK ≤ CLK/4; ≈60 ms full-core programming at 25 MHz); AER in 17-bit/out 8-bit 4-phase;
  neuron memory 256×128 b (LIF layout occupies 57 bits; 71 spare); synapse memory 8192×32 b,
  4 b/synapse = mapping bit + 3-bit magnitude; word address `{pre[7:0], post[7:3]}`.
- **F14 — stock soma law.** 8-bit unsigned membrane, saturate at 255, floor at 0; per-neuron
  8-bit θ; `v ≥ θ` checked after **every** synaptic event; hard reset to 0; >1 spike per
  neuron per timestep is physical and order-dependent; leak is a per-neuron subtractive
  amount on explicit time-reference events (deployment programs `leak_en=0`).
- **F15 — weight sign is per pre-synaptic row** (`SPI_SYN_SIGN`), not per synapse. Signed
  weights on stock silicon require row pairs; the framework-visible geometry is the logical
  one (`max_axons=128`), the exporter owns the physical expansion.
- **F16 — inference weight-freeze is doubly guaranteed** (`ca_en=0` forces the SDSP flags to
  0 so `WSYN_NEW = WSYN_CURR`; mapping bits 0 + `PROPAGATE_UNMAPPED=1` + `UPDATE_UNMAPPED=0`
  deselects SDSP entirely). `SPI_OPEN_LOOP=1` makes the core an externally-routed engine.
- **F17 — geometry is not parametric in the stock RTL.** `N/M` exist but the 128-bit neuron
  word, `N>>4` SPI generate loops, 32-bit synapse word packing, 13-bit synapse address, and
  controller sweep bounds are hard-wired to 256. Variable geometry requires a generator; the
  stock core ships vendored and unmodified. Full-fan-out event = 512 cycles; scheduler FIFO
  depth 32; no testbench ships upstream.
- **F18 — HACC@NUS toolchain is sufficient** (verified from `Xtra-Computing/hacc_demo`):
  Slurm; Vitis 2021.2/2022.2 + XRT + shells for U250×6/U280×4/U55C×6/U50/VCK5000×3; hw_emu +
  VNC waveform sim; host-driven xclbin programming; `/data` staging; 12 h/2 d FPGA
  reservations, 7-day compile partitions. hacc-gpu1/2/3 co-locate U55C boards with MI210
  GPUs (ROCm 6.3.2; PyTorch-ROCm serves them through the `torch.cuda` API) on 64-core EPYC.
  Chosen target: U55C `xilinx_u55c_gen3x16_xdma_3_202210_1`, Vitis 2022.2, RTL-kernel flow.

## 1. The semantic model — ODIN firing as declared points on two new generic axes

### 1.1 Authored keys (no target-specific config keys)

Two new authored deployment keys and one platform key. All names are laws, not devices.

| Key | Section | Values (default first) | Meaning |
|---|---|---|---|
| `firing_granularity` | deployment | `windowed` \| `event_serial` | when the threshold is evaluated: once per timestep on the reduced contribution (today's law), or after every arriving event occurrence in canonical order, emitting one spike and resetting per crossing (≥0 spikes per neuron per cycle) |
| `membrane_arithmetic` | deployment | `exact_signed` \| `saturating_unsigned` | the membrane's arithmetic: unbounded signed accumulation (today), or clamp to `[0, 2^membrane_bits − 1]` on every update |
| `membrane_bits` | platform_constraints | int ≥ 0, default `0` | fixed membrane register width; `0` = not fixed-width (inert). A hardware capability, beside `weight_bits` |

Registry shape (per the streamed precedent): `group="spiking"`, `domain="event"`,
`Category.ADVANCED`, `legal_values` + `derived_default=frozen_default(...)` +
`provenance="derivation rule"`, **no** `DEFAULT_DEPLOYMENT_PARAMETERS` entry — so both keys
materialize through `derive_pipeline_runtime_parameters` and the golden snapshot diff is a
pure addition (§7 row 1). `membrane_bits` gets a platform default `0` beside `weight_bits`.

The stock-ODIN point is then, in entirely generic vocabulary:

```
core_semantics="spiking", spiking_family="lif", spiking_variant="streamed",
firing_mode="Novena", thresholding_mode="<=",
firing_granularity="event_serial", membrane_arithmetic="saturating_unsigned",
platform: membrane_bits=8, weight_bits=4, cores=[{max_axons:128, max_neurons:256, …}]
```

and the sync-fire RTL variant (P6) is `firing_granularity="windowed"` with a wider
`membrane_bits` — the generator consumes the same vocabulary (§5.3).

**Not chosen** (recorded because both were argued): a new `spiking_variant` value (it would
conflate the soma law with the transcoding law, force a seventh `mode_id`, new legacy-bridge
values, and cascade into every mode-string consumer — non-additive by construction); a new
`FiringMode` (its legality is exhaustively pinned three-valued, and five reset dispatchers
with three different fallbacks sit behind it, §10 finding D2).

### 1.2 Legality and defaults (byte-identical default-off)

- `legal_firing_granularities(cfg)`: `("windowed",)` for the TTFS family and for mvm-domain
  documents; `("windowed", "event_serial")` for lif. Total over partial configs (`_ttfs_or`
  precedent) and short-circuiting for mvm (the `t0_44` scar, F12).
- `legal_membrane_arithmetics(cfg)`: `("exact_signed",)` for TTFS; both for lif.
- Cross-key validation: `membrane_arithmetic="saturating_unsigned"` requires
  `membrane_bits ≥ 1`; `firing_granularity="event_serial"` requires a param-encoded bias
  (`resolve_bias_mode(cfg) == "param_encoded"`) because the bias must occupy a declared slot
  in the event order (§2.3). Keyed, remediable errors per the legality harness.
- The derived defaults are frozen to today's law, so every existing document resolves
  byte-identically and no legacy value can reach the new points (the streamed invariant:
  reaching a new point is always an explicit post-migration choice).

### 1.3 What stays unchanged

`spiking_family`, `spiking_variant`, `firing_mode`, `thresholding_mode`,
`spike_generation_mode` vocabularies and legality; `ActivationSemantics` (still a 2-field
dataclass), `mode_id`, both legacy bridges, and `fold_spiking_axes` (still writes exactly
four keys); `NeuralBehaviorConfig`'s four default-free fields. The new axes ride the
contract, not the behavior config (§2.5).

## 2. The execution contract

### 2.1 Scope and the composability question

The axes are a **per-neuron soma law** executed inside a neural segment; what crosses a
segment boundary remains per-neuron window **counts** (integers), and the boundary transform
is deterministic on counts — so the atol=0 exactness gate composes across spans unchanged,
which is the §9 test for correct scoping. An ODIN deployment declares
`spiking_variant="streamed"`, so `is_streamed_lif` and every streamed gate (VERBATIM
pass-carry, the NF↔SCM exact gate) arm **with zero edits**.

The per-cycle multiplicity is the new load-bearing degree of freedom *inside* a segment: two
runs with equal window counts but different per-cycle multiplicity are different computations
at the next hop. The NF↔SCM gate therefore gains a raster-level comparison under
`event_serial` (§7 row 6), and the raster invariant `raster.sum(0) == counts` stays pinned.

### 2.2 The serial fold, defined once

For one neuron per cycle, with axon slots `a = 0 … A−1` in canonical order (§2.3), event
multiplicities `e[a] ≥ 0`, signed logical weights `w[a]`, threshold θ, and membrane `m`:

```
for a in 0..A-1:
    repeat e[a] times:
        m := sat(m + w[a])            # sat clamps to [0, 2^bits−1] under saturating_unsigned
        if compare(θ, m):             # thresholding_mode
            emit one spike; reset(m)  # firing_mode (Novena ⇒ m := 0)
return per-neuron count emitted this cycle
```

Under `(windowed, exact_signed)` the same interface degenerates to today's
integrate-then-fire-once and is byte-identical by construction. The four implementations —
NF torch, HCM torch, nevresim C++, RTL — implement exactly this definition; §7 is the gate
chain that proves it.

Bounds: the per-cycle emitted count must fit the count currencies (`spike_t` holds ≤127). A
static bound `⌈(Σ_a max(w[a],0)·e_max + bias⁺)/θ⌉ ≤ 127` is checked at export and at
contract build; violation refuses loudly (never clamps).

### 2.3 The canonical event order (new SSOT)

One module owns the order; every implementation asserts against it:

1. Axon slots ascend: slot 0, 1, …, A−1, exactly the span-gather order both implementations
   already produce (F8) — now asserted, not assumed.
2. The always-on (bias) row(s) sit at the tail of the slot order, where the mapper already
   appends them (F10). nevresim's per-cycle `bias_` is 0 under `event_serial` (the bias is
   the tail row's weight); the torch kernels fold `hw_bias` at the same declared tail
   position; the RTL receives bias rows as ordinary tail rows.
3. Physical row-pair expansion (stock signed silicon) maps logical slot `a` to physical rows
   `(2a, 2a+1)` = (excitatory, inhibitory). Because at most one of the pair holds a nonzero
   magnitude for a given (axon, neuron), and a zero add is a no-op under saturation with
   reset-on-crossing, folding the signed weight once per logical slot equals the physical
   exc-then-inh delivery — the row-pair lemma, locked by a kernel-level property test (§7
   row 9). All-zero physical rows emit no event (cycle-count optimization, semantics-free).
4. The v2 on-fabric router must preserve this order (per-timestep buffering + ordered
   drain); recorded now as a hard constraint on P7.

### 2.4 Membrane representation, init, lattice

- The membrane type/bounds belong to the integration policy (torch: the kernel's
  `membrane_bounds`; C++: `IntegrationPolicy::membrane_t`, F5). `COMPUTE_DTYPE` and every
  existing state allocation are untouched for the default point.
- `lif_membrane_init` is a once-per-window pre-charge `V0·θ`; under `saturating_unsigned` it
  is projected into the representable range and negative `V0` is refused.
- The integer-lattice snap moves from once-per-cycle to once-per-event under `event_serial`,
  inside the same `measurement_plane()` on both twins (the n8e tie canary); the lattice
  quantum under a fixed-width integer membrane is the register LSB, derived on the resolved
  point, subsuming `membrane_integer_lattice` rather than coexisting with it.
- The membrane-readout decode (`Q_T = θ·c_T + m_T`) is refused under
  `saturating_unsigned` (charge is destroyed at both rails; the identity is false) — a typed
  refusal, not a silent skip (§7 row 12).

### 2.5 Carrier and refusal architecture

- `SpikingDeploymentContract` gains two frozen defaulted fields (`firing_granularity`,
  `membrane_arithmetic`) + resolved `membrane_bits`, read once in `from_pipeline_config`.
  `FiringStrategy` carries them so the whole neuron law resolves in one object. Executor
  entry points take them keyword-only, default-free (`test_semantic_defaults` discipline);
  they are never read via raw `config.get` (the ~20 existing raw-read sites are a recorded
  finding, §10 D6, not a pattern to extend).
- **The refusal keys on the resolved semantics point, not the mode string** (F3).
  `BackendSpikingCapabilities` gains `event_serial: bool = False` and
  `saturating_membrane: bool = False` (dataclass defaults keep every existing entry literal
  unchanged); a new `require_semantics_supported(point, *, backend, context)` is the single
  gate, called from the registry's up-front validation, the mode policies, and
  `cross_sim_parity.derive_applicability`. Under the ODIN point: hcm/nevresim support it
  (after P2/P3); sanafe, lava/loihi, training refuse **by name** (SANA-FE's
  `NeuronStatus` and lava's boolean `spiking_activation` cannot represent multiplicity);
  `ConversionPolicy.derive`'s `sim_enables` consults the same gate so unsupported backends
  are off by derivation, and an explicit `enable_*=true` gets a keyed error.
- Executors that are theorems about cycle-atomic integration refuse under `event_serial`:
  the synchronized count executor (path C), `advance()` on a pre-reduced contribution, and
  per-hop retimed level stages (their hop re-encode cannot carry multiplicity; declared
  illegal under the point rather than generalized — §11 open item).

### 2.6 The wire contract (v1) and honest reporting

The inter-segment wire contract is **unchanged**: rates ∈ [0,1]. Multiplicity lives only
inside a neural segment (intra-segment hops are value-preserving; pass-carry rasters carry
counts, §4.3). At a host boundary the decode (`counts/T`) is unbounded and final network
outputs are exact and unclamped; a *re-encode* into a following segment clamps at the
declared ceiling 1.0 — identically in every backend and in the physical runtime, since all
share the boundary SSOT — and gains the currently-missing symmetric upper-rail warning plus
a seam-audit ceiling class (the one-sided clamp-loss proof is finding D5). Raster-bytes
accounting (`raster_bytes`, `carried_wire_bytes`) prices `log2(k_max+1)` bits per slot from
the declared point (default 1 bit — byte-identical). The honesty ledger
(`docs/how_networks_reach_the_chip.md`) gains the discipline's row when P3 closes.

## 3. Owner decisions (2026-08-21, verbatim where load-bearing)

| # | Question | Decision |
|---|---|---|
| 1 | ODIN-style firing support | "generically support ODIN-style firing, by extending the current SSOT contracts. The elegance of the spike semantics must be retained and the design integration must stay generic (no ODIN-specific config). Mathematical parity must be retained throughout the entire pipeline." |
| 2 | Non-default semantics | "a code generator for non-default spike semantics in ODIN deployments. LIF datapath modification should be allowed. But, the default ODIN chip must be supported out of the box." |
| 3 | Non-default geometry | "Supporting non-default core geometry on ODIN must be supported… default ODIN chip must be deployable. This is a hard requirement." |
| 4 | Compile limits | "study limits of what we can compile into FPGA later once we have the infrastructure" (→ P8) |
| 5 | Isolation | "Create a mimarsinan_fpga clone repo… we only merge to real mimarsinan once we are sure that the infrastructure is elegant and correct." |
| 6 | No-impact bar | "this new support cannot impact previously existing behavior, correctness, code elegance, and performance." |
| 7 | Target node | U55C + MI210 nodes, Vitis RTL-kernel flow (approved: "I agree with your target node decisions"). |
| 8 | Semantics path | Stock-exact contract first (no RTL risk), sync-fire variant as generator flagship (approved via §6 of the analysis report). |
| 9 | Signed weights | Row-pair expansion below the mapper; logical geometry declared (approved). |
| 10 | v1 routing | Host-mediated inter-core routing, on-fabric router v2 (approved). |

Designer decisions within those directives (this plan's authority, overridable):
two-axes shape over a variant value (§1.1); wire ceiling stays 1.0 in v1 (§2.6); bias at the
tail of the canonical order (§2.3); exporter consumes `HardCoreMapping`, never `ChipModel`
(F9); RTL lives in a top-level `hw/` tree on the `nevresim/` precedent (§5.1).

## 4. SSOT map (every mechanism has exactly one home)

| Mechanism | SSOT home | Consumers | Pin |
|---|---|---|---|
| Axes vocabulary + legality + derived defaults | `chip_simulation/activation_semantics.py` (+ P0 extraction module) | registry entries, wizard, derivations | golden snapshot; legality harness |
| The serial fold (torch) | `models/spiking/lif_serial_step.py::lif_serial_fold` | `SerialLIFCyclePolicy.step`/`advance_events` (HCM paths A+B), NF decomposition branch | packed↔reference equivalence test; NF↔SCM atol=0 |
| The serial fold (C++) | `nevresim …/integration_policy/event_serial_integrate.hpp` | `SpikingCompute<Fire, Integration>` via `NeuronCompute` | nevresim↔HCM parity rows; constexpr self-tests |
| Canonical event order | new `models/spiking/event_order.py` (name final at P2) | torch kernels, NF decomposition, exporter, sequencer, cosim tb | cross-implementation order test; row-pair lemma test |
| Semantics-point refusal | `require_semantics_supported` beside `_BACKEND_CAPS` | backend registry, mode policies, `cross_sim_parity`, `sim_enables` | capability-guard audit + new refusal tests |
| Boundary ceiling + upper-rail warn | `spiking/segment_boundary.py` / `BoundaryConfig` | HCM, NF, compute-boundary, physical runtime | boundary lock tests; seam-audit class |
| Reset-law C++ string | one resolver (P2 consolidates the five dispatchers, §10 D2) | codegen + behavior config | mutation-tested equivalence |
| Memory-image bit layouts (stock ODIN) | `hw_export/odin` packer tables (from F13/F14 bit tables) | exporter, cosim tb, FPGA runtime | pack/unpack golden round-trip |
| Geometry/semantics descriptor | generator `CoreSpec` (§5.3) | RTL emitter, packer, contract check | per-variant cosim |
| Platform physics | `deployment_record/platform_physics/profiles/odin.json` (exists) + new FPGA-instance profile | cost/fidelity | correlation case |

## 5. The exporter, the RTL tree, and the generator

### 5.1 Repo layout

Top-level `hw/` (the `nevresim/` precedent: non-Python implementation trees live at the
root, outside module budgets and pyright scope, referenced by a driver path):

```
hw/vendor/odin/     # stock RTL, byte-identical to ChFrenkel/ODIN @ <pinned SHA>, LICENSE
                    # (Solderpad SHL-2.0) + PROVENANCE.md (upstream URL, SHA, date, cite)
hw/gen/             # generator templates for variant cores (RTL assets, not .py strings —
                    # deliberate deviation from the cpp-template-as-python-string precedent,
                    # recorded here because Verilog templates would blow the 300-LOC budget)
hw/tb/              # Verilator/testbench harness (SPI driver, AER driver/capturer, cosim)
hw/fpga/            # Vitis RTL-kernel wrapper, package_xo / v++ scripts (P7)
```

Python: `src/mimarsinan/hw_export/` — a new top-level module (`__init__.py`,
`ARCHITECTURE.md`, root map row + the "19 modules" count fix) with an `odin/` subpackage;
budgets respected by construction. The stock-RTL path is held by a driver-class attribute
(`NevresimDriver.nevresim_path` precedent). `.slurmech.toml` gains `hw/**` at P7.

### 5.2 The exporter (P4)

Consumes `HardCoreMapping` (per-core geometry, F9) after `ChipLatency.calculate()`;
allocates fresh arrays (never mutates the memoized grid); emits per core:

- **Neuron-memory image** (256×128 b): per-neuron word with `lif_izh_sel=1`, `leak_str=0`,
  `leak_en=0`, `thr=θ` (gated `1 ≤ θ ≤ 2^membrane_bits−1`), `ca_en=0`, SDSP fields 0,
  `neur_disable=1` for unused neurons (F13/F14 bit layout tables, reproduced in the packer
  module as the cross-language contract comment).
- **Synapse-memory image** (8192×32 b): row-pair expanded weights (`|w|` in the 3-bit field,
  mapping bit 0), nibble/byte/word packing per F13.
- **`SYN_SIGN` vector** (per physical row) + the config-register list (open-loop,
  freeze registers per F16, AER source select).
- **The sequencer program** per segment/pass: CONFIG / INJECT(events in canonical order) /
  TREF(if leak or windowed variant) / BARRIER / READOUT — the deployment schedule's neural
  stages compiled to the chip; host stages stay in the stage list.
- **A manifest** naming the semantics point, geometry, ordering, and emission bound — the
  contract the runtime and cosim check against.

Ledgers stay logical; the physical expansion factor is a declared capability from which
derived physical figures are computed (never a re-interpretation of `cells_used`).
Platform declaration lands beside it: `IMCPlatform` registration (provenance-quoted;
registry-content test literal 12→13), `membrane_bits`, per-core `weight_sign_granularity`
if promoted to `CORE_FIELDS`, `core_value_granularity: {"threshold": "per_neuron"}`
available unchanged.

### 5.3 The generator (P6)

`CoreSpec(axons, neurons, weight_bits, signed_weights, membrane_bits, firing_granularity,
firing_mode, thresholding_mode)` — the same §1.1 vocabulary — emits: (a) variant RTL from
`hw/gen/` templates (a parametric, LIF-only, open-loop core in ODIN's architecture: SDSP,
IZH, and burst logic omitted because deployment never uses them, F16/F17); (b) the matching
packer tables; (c) the semantics descriptor consumed by the contract check and the cosim
harness. The **stock spec is a vendored passthrough** asserted byte-identical to
`hw/vendor/odin` — the out-of-the-box requirement never depends on generation. Flagship
variant: sync-fire (`windowed` granularity, 16-bit membrane, subtractive or zero reset,
per-synapse signed weights at `weight_bits=4`), which bit-matches the existing streamed-LIF
contract when its no-saturation bound holds (gate-checked).

### 5.4 The physical backend (P7, summarized; detailed design doc lands with P7)

Vitis RTL kernel: P cores + per-core SPI-master config engines + AER bridges + the sequencer
+ timestep-tagged event capture behind AXI; `hw_emu` first. XRT session driver implements
the abstract `Backend` methods for real; the deployment step drives `run_hybrid_stages`
(host ops through `execute_compute_op_*` on the contract's device — MI210 parity frozen
once); `RunRecord` emission feeds the existing certificate stack; registry wiring per the
four-registries checklist (backend registry default-off entry, `_BACKEND_CAPS` row,
`BACKEND_CLASSES: "exact"`, parity-harness arm) + a `DependencyBoundary` for `pyxrt` with
non-empty guards. The only sanctioned stock-RTL touch: exporting `SCHED_EMPTY` +
controller-idle as two wires for an exact BARRIER, kept as a clearly-diffed additive patch
with a conservative cycle-bound fallback.

## 6. Config schema + wizard surface

Registry entries per §1.1 with `doc=` strings written against the law (not the device);
domain tagging auto-injects the `core_semantics=spiking` existence gate and the mvm
domain-rules refusal; wizard options render from `legal_values` (locked single-value fields
render locked); an advisory row explains the point's backend availability. Prose surfaces
re-checked per phase (§12).

## 7. Verification & no-regression matrix

| # | Surface | Gate | Phase |
|---|---|---|---|
| 1 | Resolved config surface | golden snapshot regen; diff **pure-additive** (`grep '^-[^-]' == 0`); global recipe surface bit-identical | P1 |
| 2 | Default suite + typecheck | 12,048+ passed / 0 failed, ≤2 min wall (baseline ~56 s); `typecheck.sh` 0 errors | every phase |
| 3 | Backend registry outputs | pinned selection tests untouched and green (`test_backend_registry.py:84-103`); frozen legacy baseline untouched | P1, P7 |
| 4 | Refusal architecture | ODIN point × {sanafe, lava, loihi, training, nevresim-before-P3} refuses **by name**; `sim_enables` derivation off; explicit enable ⇒ keyed error; capability-guard audit extended | P1 |
| 5 | Kernel equivalence | packed (`advance_events`) ↔ reference (`step`) bit-equal under the point (the only guard spanning metric and gate paths, F6) | P2 |
| 6 | NF↔SCM exactness | existing atol=0 window-count gate arms unchanged (streamed variant); + raster-level atol=0 comparison under `event_serial`; teeth test (mutate θ ⇒ raises); non-degenerate witness pinned | P2 |
| 7 | Default-point byte-identity | `(windowed, exact_signed)` returns the identical policy objects; nevresim emitted `main.cpp` byte-identical; compile-cache keys unchanged (new hash field default-omitted) | P2, P3 |
| 8 | nevresim ↔ HCM | parity-harness rows for the point at atol=0 (counts) + counted-raster carry round-trip; SPKREC↔SPKTRN sum identity restored under counts | P3 |
| 9 | Ordering + row-pair lemma | cross-implementation canonical-order test; property test: signed-once fold ≡ exc-then-inh physical delivery; bias-tail position test | P2–P4 |
| 10 | Exporter | pack/unpack golden round-trip (bit-exact images); θ/emission-bound refusals; logical-ledger invariance (utilization/record outputs byte-identical for non-ODIN configs) | P4 |
| 11 | **Stock deployability (directive-3 hard requirement)** | tiny network end-to-end map→quantize→export→**Verilator cosim of vendored stock RTL**: per-neuron per-window counts ≡ nevresim ≡ HCM at zero difference | P5 |
| 12 | Typed refusals | path-C executor, `advance()`, retimed levels, membrane readout, negative `V0` — each raises a typed error under the point; conformance matrix answers for every cell (no silent skips) | P2 |
| 13 | Variant cosim | per-`CoreSpec` cosim vs nevresim twin; stock spec byte-identical to vendor | P6 |
| 14 | Board parity | on-board certificate campaign vs nevresim (counts, atol=0); measured Timing/Energy fragments with `kind="measured"` | P7 |
| 15 | Performance non-impact | suite wall time within baseline band per phase; tier-0 spot check for pipeline-behavior phases | every phase |

## 8. Phasing

- **P0 — headroom extractions** (pure moves, no behavior): split legality/derivation helpers
  out of `activation_semantics.py` (291/300) and `deployment_derivation.py` (295/300) into
  sibling modules; ARCHITECTURE rows. *Acceptance:* §7 row 2; diff shows moves only.
- **P1 — axes + refusal architecture** (vocabulary, registry entries, contract/FiringStrategy
  fields, `membrane_bits`, capability re-keying + `require_semantics_supported`, sim_enables
  derivation, legality/validation, wizard rendering). Numerically inert: the point is
  declarable but every executor refuses it. *Acceptance:* §7 rows 1–4.
- **P2 — torch serial-fold kernels** (kernel SSOT + `SerialLIFCyclePolicy` + packed
  `advance_events` + NF decomposition + ordering SSOT + typed refusals + boundary
  upper-rail warn + per-event lattice snap). The legal set opens for the hcm backend **in
  the same commit** as gate rows 5–6. *Acceptance:* §7 rows 5–7, 9, 12.
- **P3 — nevresim integration policy** (C++ integration-policy axis + counted raster
  protocol + cache-key field + `NevresimExecParams` threading + reset-dispatch
  consolidation §10 D2 + tier cell via `templates/generate.py` + honesty-ledger row).
  *Acceptance:* §7 rows 7–8.
- **P4 — exporter + platform declaration** (§5.2). *Acceptance:* §7 rows 9–10.
- **P5 — vendored RTL + cosim harness** (§5.1 `hw/vendor`, `hw/tb`; the missing testbench;
  local `verilator` bring-up). *Acceptance:* §7 row 11 — the hard requirement.
- **P6 — generator** (§5.3). *Acceptance:* §7 row 13.
- **P7 — FPGA kernel + XRT backend + HACC packaging** (§5.4; own design doc).
  *Acceptance:* §7 rows 3, 14.
- **P8 — measured campaign, fidelity, compile-limits study** (owner directive 4; own doc).

**Order:** P0 → P1 → (P2 ∥ P4) → P3 → P5 → (P6 ∥ P7) → P8. Each phase lands independently
green (suite + typecheck + tier-0 spot check per repo rule 11), one commit per phase, plus
separate behavior-free extraction commits where budgets force them.

## 9. Tier cells and coverage

New cells only via `templates/generate.py` (a `lifes`-tagged mode-table entry at P3;
existing cells keep numbers and hypervolume axes). Certification cells disambiguate via the
existing `CertificationCell.variant` field (`#event_serial` tag) so no historical cell key
is disturbed. The hypervolume `firing`/`backend` axes live in the campaign workspace and are
extended **at merge time** (§13): an ODIN run must never be counted as coverage of the
streamed-LIF cell, and the physical backend is *not* a collapsed member of the faithfulness
axis until a screening artifact earns it.

## 10. Findings reported, not acted on (owner sign-off required; physics-plan §8d precedent)

| # | Finding | Severity |
|---|---|---|
| D1 | The synchronized count executor never reads `firing_mode`; its staircase is the subtractive-reset theorem, so it is latently wrong for `Novena` today, and it is the oracle for `count_alignment` certificates (`sync_counts.py:11-79`, `lif_step.py:106-108`) | high |
| D2 | Five reset-law dispatchers with three different unknown-mode fallbacks (`behavior_config.py:67-70` vs `generate_main.py:17-19` disagree for the same backend). P3 consolidates the two nevresim ones as part of touched work; the lava/sanafe/training three are reported | med (becomes high if `FiringMode` ever widens) |
| D3 | Membrane-readout gates never consult the reset law — the `Q_T` identity is already invalid under `Novena` today (`membrane_readout.py:90-96`, `membrane_export.py:41-59`) | high |
| D4 | `recording/records.py:2` states order-independence as a theorem; false under `Novena`, catastrophically false under `event_serial`. P2 rewrites the comment (doc-only) | low |
| D5 | The boundary clamp-loss proof is one-sided: no upper-rail policy or warning exists anywhere (`negative_boundary.py`, `segment_boundary.py:169-187`). P2 adds the warn for the new point; the historical silence is recorded | med |
| D6 | ~20 raw `config.get` reads of semantics keys outside the contract (list in the R1 report); `gui/wizard/schema.py:31-36` hardcodes a duplicate of `legal_firing_modes` | med |
| D7 | `AGENTS.md:11-33` per-directory-docs claim contradicts CLAUDE.md and the ratchet; root `ARCHITECTURE.md` says "18 modules" while 19 exist (20 after P4) | low |
| D8 | The SANA-FE TTFS input path passes spike *times* into an attribute the C++ coerces to a per-timestep bool *mask* (`neuron_model.py:137-143` vs `sana_fe/src/models.cpp:816`) | out of scope, recorded |
| D9 | `spike_t = std::int_fast8_t` has implementation-defined width — a cross-compiler bit-exactness hazard for nevresim generally | low |

## 11. Risks / open items

- **Emission-bound refusal** (§2.2) may bite real networks; the remedy is θ/scale adaptation,
  not a wider count type — measured when P4 gates run.
- **Generalized multi-level comb** (re-encoding rates >1) is deliberately not built; the v1
  wire ceiling stays 1.0. Becomes real work only if a study needs over-unity wires.
- **Retimed levels under `event_serial`** are refused, not generalized (the level-boundary
  re-encode cannot carry multiplicity). Open until a use case demands it.
- **Training through the serial fold** (surrogate per crossing): evaluate-only v1; the
  train↔deploy gap is measured and carried on the ledger like the retiming gap (−2.5 pp
  precedent). Serial-aware QAT is genuine future work.
- **`saturating_signed` membranes** (sync-fire variant with tight bounds): deferred until the
  no-saturation gate refuses a wanted configuration.
- **Multi-row (wide) bias** on stock silicon: F10 machinery, deferred; the quantizer's
  existing ±q_max bias clip applies as today.
- **Board-side hazards** (P7): Slurm 12 h windows vs parity-campaign length (checkpointing);
  one xclbin per shell/Vitis pairing; MI210 host-op determinism frozen once and pinned.
- **First-run cache-cold flake**: 13 sanafe-arch tests failed once on a cold clone and pass
  on re-run; treat a green re-run as authoritative, watch the file.

## 12. Prose surfaces to re-check per phase

Registry `doc=`/`empty_means=` strings; advisory `detail=` texts; recipe rationale constants
(`conversion_rationales.py`); `CLAUDE.md` SSOT list; root + touched module ARCHITECTURE
rows; `docs/how_networks_reach_the_chip.md` honesty ledger; `templates/generate.py`
coverage notes; this plan's Status table.

## 13. Merge protocol (owner directive 5)

Work lands on `odin-fpga-infra` in this clone only; nothing is pushed. Merge to real
mimarsinan happens only on the owner's judgment of elegance + correctness, as one reviewed
branch, after: re-running the full gate set on a fresh clone of real main, re-basing, and
extending the campaign-workspace surfaces this clone deliberately does not touch
(hypervolume axes, ROADMAP D-layer row, screening artifacts). The clone's history stays
reviewable: one commit per phase, subjects as claims, bodies carrying the gates' numbers.

## Deliberately unchanged (the protection contract)

`ChipModel` and all eight of its consumers; `hard_cores_to_chip`; `fold_spiking_axes`
(still writes exactly four keys); `ActivationSemantics`, `mode_id`, both legacy bridges;
`NeuralBehaviorConfig`'s shape; `_legacy_backend_specs` (frozen baseline); every existing
clamp/kernel/encoder on `(windowed, exact_signed)` points; `VERBATIM_BACKENDS` for existing
points; existing tier cells, their numbers, and their hypervolume axes; the default suite's
wall-time band; the streamed plan's invariants.

## Verification protocol (every stage)

Tests first; `python -m pytest` ≤2 min green; `./scripts/typecheck.sh` zero; ratchets and
budgets only tighten; templates only via `templates/generate.py`; ARCHITECTURE.md per
touched module; load-bearing guards mutation-checked; byte-identity A/B wherever a
default-off claim is made (golden-snapshot zero-deletion proof; emitted-artifact
byte-compare); generic-only — no target constants framework-side; per-stage commits, no
AI-attribution trailers.

## Program-level acceptance ("ODIN support" defined)

1. The stock ODIN chip is declarable in generic vocabulary, deployable end-to-end, and its
   deployed per-neuron window counts match nevresim and the torch reference at zero
   difference, witnessed by RTL cosimulation of the unmodified vendored core (§7 row 11).
2. A `CoreSpec` with non-default geometry and/or the sync-fire law generates a core +
   packer + descriptor that passes the same cosim gate.
3. Every pre-existing configuration resolves, executes, and prices byte-identically, at
   unchanged suite wall time, with zero weakened checks.
4. The physical FPGA backend reproduces nevresim counts on HACC hardware and lands measured
   timing/energy fragments in the deployment record (P7).
5. The honesty surfaces (ledger row, coverage identity, refusals-by-name) make it
   impossible to record an ODIN number as a streamed-LIF number anywhere.
