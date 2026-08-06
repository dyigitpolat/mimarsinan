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

**Rate-coded LIF (`lif`)** — the number system IS counts: a value in [0,1]
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
semantics collapses train↔deploy agreement to ~0.84. Until now the knob was
implemented by splitting the mapping into one segment per layer (the split
you caught); the fix (docs/lif_hop_fused_mapping_design.md) moves the reset
inside the executor and un-splits the mapping.

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

The clean summary of the deployment contract as it exists: **the simulated
chip is a windowed, pass-structured machine in which timing is normalized at
every stage boundary — not a free-running asynchronous chip.** Every tier-0
accuracy number means: "this trained model, executed under THESE semantics,
scores X." Within that contract the toolchain is honest and bit-gated
end-to-end. What the contract does NOT claim is free-streaming behavior;
where the two diverge, this codebase measured the divergence instead of
hiding it — that is what the 2.5-point and 0.84-agreement numbers are.

---

## Part 4 — The three decisions this leaves open

1. **Accept the windowed contract as the deployment story** (status quo,
   now to be made structurally clean by the fused-mapping fix). Defensible:
   windowed execution is how pass-based and weight-swapped deployments
   physically run.
2. **Pursue free-streaming fidelity as a research axis** — training that
   tolerates bursty rhythm, or training-time modeling of cascade timing.
   That is a research program (the −2.5 pp is its baseline), not a knob.
3. **Carry both as measured arms** — the re-timing toggle already gives the
   A/B instrument on any trained model.
