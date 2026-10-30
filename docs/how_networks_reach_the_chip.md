# How a trained network actually reaches the chip — the plain-language guide

**Date:** 2026-08-05. Written as a wake-up-call audit: every statement below
describes what the code does today (verified against the source and against
two real artifacts: the lenet5_baseline lif run and the 4,925-core ViT mvm
run), with the idealizations called out instead of smoothed over.

---

## Part 1 — The journey from checkpoint to chip program

**1. Reorganize.** The trained torch model is reorganized into a chain of
"perceptrons" — matrix-multiply units (a weight matrix, a bias, an
activation) — plus everything between them (pooling, normalization, attention
internals, reshapes).

**2. Quantize.** Weights are snapped to integers (e.g., 8-bit): the chip
stores integer matrices; the scale factors move into per-core thresholds.
From this point on, the weight arithmetic is integer-exact and every later
transformation is gated by bit-identity tests.

**3. Translate to the intermediate representation.** Two node kinds:

- **Neural cores** — crossbars: a weight matrix with input lines ("axons")
  and output neurons, a threshold, optionally a bias. This is what the chip
  executes.
- **Host compute ops** — everything that is not a matrix multiply. These do
  NOT run on the chip. They run in software, in float, between chip stages.

  This is a real hybrid-execution assumption, and it is bigger than it
  sounds. In the lenet5 lif run, the FIRST convolution, both max-pools, and
  the final scale correction ran as host ops — the chip executed conv2 and
  the two FC layers. In the ViT run, 76 of 151 stages are host ops (every
  attention block internal, every LayerNorm, every GELU). A physical
  deployment needs a companion processor for these, and their cost is not
  chip cost.

**4. Share weights.** When many cores use the same matrix (convolution
positions, transformer tokens), the matrix is stored once as a "weight bank";
cores hold references to slices of it. The ViT: 4,925 cores over 25 banks —
57 MB of actual weights.

**5. Eliminate dead structure.** The analysis proves which rows, columns,
and whole cores can never carry signal (or carry a known constant), deletes
them, and writes an attributed report. Bit-exactness of everything the
network still computes is the gate.

**6. Pack.** Logical cores are placed into the platform's fixed-size physical
crossbars: small cores can be merged into one crossbar ("coalescing" — the
lenet's 196 conv cores packed into 55 physical cores), too-wide layers are
split across crossbars, leftover space is zero-padded. If the whole network
fits the pool of physical cores (lenet: needed ~57 of 120), it becomes ONE
chip program. If not, it is run in installments ("passes"): program weights,
run, buffer the outputs, reprogram, continue. For weight-shared models a
scheduler keeps each bank resident on a physical core while its instances
stream through — weights are programmed once, not per pass.

**7. The program.** The result is an ordered list of stages: neural segments
(groups of cores executed together on-chip) alternating with host ops. This
list — cores, wiring, thresholds, stage order — is the deployment artifact.

---

## Part 2 — What "running" means: the value currencies

The single most important fact of the execution model:

> **Between stages, values travel as plain numbers. Spikes, where they exist
> at all, live only INSIDE a neural segment's execution.**

Every stage boundary is a buffer of numbers per line ("how much"), not a live
spike stream. Host ops compute on those numbers directly. What happens inside
a neural segment depends on the mode:

**Value mode (`core_semantics=mvm`)** — your ViT pipeline. No spikes
anywhere. The crossbar is used as an integer matrix-multiply engine
(in-memory-computing style); correctness is bit-exact value parity against
the trained model. Everything in this guide about spike timing is irrelevant
to this mode.

**Streamed LIF (`spiking_family=lif, spiking_variant=streamed` — the default
since 2026-08-07; redefined per-segment 2026-08-08, plan §9)** — streaming is
the execution discipline of each Neural Segment: within a segment, binary
spikes flow cycle-by-cycle through its latency groups with no interior
transcode; at segment boundaries, host compute ops (pooling, encode, readout)
operate on window counts and the next segment re-encodes. Any hybrid
architecture deploys — the build step reports the span topology instead of
gating on it — and a model whose host ops are confined to an encode prefix +
readout suffix earns the **end-to-end** property (ONE segment, one continuous
window, values transcoded exactly twice). Scheduling is locked off; every
segment stays resident in one chip program. The NF train forward IS the
deployed streaming cascade, gated bitwise: per-neuron window counts equal the
identity executor at atol=0 ACROSS segments, with no mismatch budget (host
ops are deterministic on counts, so exactness composes).

**Windowed LIF (`spiking_variant=synchronized`)** — the number system IS counts: a value in [0,1]
is represented as "n spikes within a T-tick window" (T = simulation_steps).
That is textbook rate coding, and the activation-quantization training chain
exists precisely because it makes every activation live on a 1/T grid.
Inside a segment, execution is genuinely tick-accurate: each neuron
integrates charge per tick, fires on threshold crossings, subtracts on fire;
dependent cores stream spikes tick-by-tick through pipeline latencies. At
segment boundaries the train is collapsed to its count, stored as a number,
and — when the next stage is neural — re-emitted as an EVENLY SPACED train.
Timing never survives a stage boundary, by construction.

**The re-timing knob (`lif_per_hop_retiming`)** applies that same
collapse-and-re-emit at every layer-to-layer boundary, because a spiking
neuron is charge-sensitive to input rhythm (same count in, different rhythm →
possibly different count out), and training assumed even spacing. Measured on
this codebase: without the per-layer reset, a model trained under the default
recipe loses ~2.5 accuracy points; deploying a model through mismatched
semantics collapses train↔deploy agreement to ~0.84. The knob was once
implemented by splitting the mapping into one segment per layer (the split
you caught); since 2026-08-07 the mapping stays FUSED and the reset executes
as per-level stages inside the one segment (docs/lif_hop_fused_mapping_design.md
— implemented), so the program artifact is single-segment while every hop
boundary keeps the identical re-encode.

**Synchronized LIF variant** — drops the tick loop entirely: each layer is
one closed-form evaluation of "how many spikes would this neuron emit for
these input counts" (the staircase function). Locked bit-equal to the tick
loop under its stated conditions.

**TTFS family (`ttfs`, `ttfs_quantized`, `ttfs_cycle_based`)** — the value
is encoded in WHEN a single spike occurs (earlier = larger). The analytical
modes compute those times in closed form; the cycle-based mode ticks.

**Decode.** The final counts (optionally plus the leftover membrane charge —
a diagnostic refinement, excluded from chip-claim reads) are scaled back to
logits.

---

## Part 3 — The honesty ledger

| aspect | what runs today | faithful to what | gap status |
|---|---|---|---|
| weight arithmetic | integer matrices, exact | any crossbar chip | bit-gated everywhere |
| host ops | float, in software | a chip WITH a companion processor | cost not charged to chip; count is per-model (lenet: first conv + pools; ViT: 76/151 stages) |
| within-segment dynamics (lif) | tick-accurate, latency-pipelined | windowed chip execution | faithful |
| between stages | counts buffered, re-emitted evenly | pass-based execution (buffer + replay is physically what weight-swapping deployments do) | faithful for windowed chips; NOT free streaming |
| per-layer re-timing | every layer boundary behaves like a pass boundary | the training-time assumption | free-streaming chips do not do this; the ~2.5 pp / 0.84 experiments MEASURE that gap |
| finite window T | all activations on a 1/T grid | any windowed rate-coded chip | trained-for (activation quantization) |
| input boundary | first layer runs value-side, once; inputs enter as even trains | input encoding hardware | idealized, uniform encode |
| silicon backends | lava (Loihi), sanafe, nevresim exporters consume the same program | real silicon | each has its own consistency checks; fidelity claims end at the exporter seam |
| within-segment dynamics (per-event soma point) | the threshold is evaluated after EVERY arriving event occurrence, in ascending slot order with one slot's multiplicity adjacent, on a saturating unsigned membrane of declared width; a neuron may emit several spikes in one cycle and those COUNTS travel the wire inside the segment | an event-driven chip whose soma fires per arriving synapse event (the ODIN-style law) | executed by the torch twins (`models/spiking/serial`) and by nevresim (`EventSerialIntegrate`), gated per-neuron at atol=0 between them; every other backend refuses the point BY NAME. NOT yet run on RTL or silicon — the cosimulation and board gates are later phases, so no hardware fidelity is claimed here |

The clean summary of the deployment contract: the toolchain now carries BOTH
disciplines honestly. **Windowed** (`lif_sync`, the ttfs family): a
pass-structured machine in which timing is normalized at every stage/hop
boundary — the re-timing executes as per-level stages inside ONE fused
segment since 2026-08-07. **Streamed** (`lif`, the default): each neural
segment free-runs as a binary-spike cascade, counts cross host boundaries,
and train↔deploy agreement holds at atol=0 BY CONSTRUCTION (the raw
per-segment streaming cascade is the train forward), enforced by a fatal
per-neuron window-count gate; `segments == 1` is reported as the end-to-end
special case. Every tier-0 accuracy number means:
"this trained model, executed under THESE semantics, scores X." The historical
2.5-point / 0.84-agreement numbers measured what deploying MISMATCHED
semantics costs — which is exactly why the discipline is a trained-for axis,
never a deploy-time toggle.

Since 2026-08-21 a THIRD, orthogonal thing is declarable inside the streamed
discipline: the **soma point** (`firing_granularity` × `membrane_arithmetic` ×
`membrane_bits`). The default point — one threshold evaluation per cycle on the
whole reduced contribution, unbounded accumulator — is what every number above
was measured under and is byte-identical. The per-event point evaluates the
threshold after each arriving event on a fixed-width saturating register, which
is a genuinely different physics: it is its own hypervolume cell
(`lif+per_event-sat8`), its own certification cell, and its own tier row
(t0_54), so a per-event run can never be reported against a per-cycle
regression floor. What is honest to say today: two independent implementations
(torch and nevresim) execute it and agree per neuron at zero difference on
compiled fixtures; no hardware has run it yet; and the end-to-end tier cell
(t0_54, simple_mlp) is **GREEN as of 2026-08-21** — 1542.2 s wall, deployed
accuracy **0.9525 on the HCM torch metric and 0.96 on nevresim** (25 subsampled
test images) against a trained 0.9533 read, with both streamed NF↔SCM arms
(window counts AND the per-cycle raster) exact at atol=0 and both spike-count
certificates exact (max|Δcount| = 0 over 20 neuron-windows, hcm streaming-twin
and nevresim-vs-hcm). Its first run was RED at Soft Core Mapping — 1004/3152
per-cycle emission mismatches, worst nf=1 vs scm=7 — because the chip-aligned
NF forward the LIF adaptation step installs on the model carried no soma point
and silently ran the DEFAULT per-cycle law (at most one spike per cycle) as the
twin of a per-event deployment. The raster arm of the gate is what caught it;
the window counts alone would not have. The point's honest price is wall time:
the event-serial fold costs roughly 7× the per-cycle walk, and every endpoint
stage that trains against the deployed composition pays it (1122 s of this
cell's 1542 s is Weight Quantization's endpoint recovery).

---

## Part 4 — The decision, resolved (2026-08-07)

Both arms shipped as first-class disciplines of the `(spiking_family ×
spiking_variant)` taxonomy:

1. **Streamed lif is the default** (`spiking_variant=streamed`): the plain
   cycle-accurate LIF adaptation trains the raw streaming cascade, so the
   train forward IS the deployed forward — parity by construction, gated
   bitwise. Since 2026-08-08 (plan §9) the discipline is per-Neural-Segment,
   so pooling architectures deploy too: tier-0 carries end-to-end cells
   across mlp/mixer/conv topologies (t0_45–t0_49, incl. the spiking-native
   `stream_cnn` vehicle) plus the multi-span flagship t0_50 (lenet5,
   3 segments, pools as host ops between them).
2. **Windowed lif** stays fully supported as `spiking_variant=synchronized`
   (the historical `lif` cells, re-tagged `lifsync`, keep their numbers);
   the per-hop reset executes as level stages inside one fused segment.
3. The A/B instrument is now the variant switch itself, on any streamable
   model.
