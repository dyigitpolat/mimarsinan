# Loihi 1 — platform physics profile

Intel Loihi (first generation), 14 nm FinFET, 128 neuromorphic cores of 1024 neurons.
The full research pass, with verbatim quotes and page references for every number, is
[`docs/research/physics/loihi_constants_research.md`](../../../../../docs/research/physics/loihi_constants_research.md).

## The one thing to know before using this profile

`measurement_kind` is **`simulation`**, not `silicon`. Davies 2018 Table 2 — the
source of nearly every per-event constant here — is titled *"Loihi **pre-silicon**
performance and energy measurements"*, and the body text qualifies it:

> "Table 2 provides a selection of energy and performance measurements from
> pre-silicon SDF and SPICE simulations, consistent with early post-silicon
> characterization." (§Results / Silicon Realization, p. 95)

So this is a **design-intent** profile, corroborated by silicon but not replaced by
it. That matters for exactly one thing this program does: a cross-platform
comparison against `truenorth` is comparing a simulated chip against a measured one,
and the report must say so. The panel's `simulation` badge and this field exist for
that reason.

The single exception is `p_static_per_core`, derived from **measured** board-level
data (Frady 2020) — the only published Loihi leakage figure found.

## Sources

- `davies2018loihi` — Davies et al., "Loihi: A Neuromorphic Manycore Processor with
  On-Chip Learning", *IEEE Micro* 38(1), 2018. Table 2 is the per-event SSOT.
- `frady2020neuromorphic` — Frady et al., "Neuromorphic Nearest Neighbor Search Using
  Intel's Pohoiki Springs", NICE 2020. The only published static-power source.
- `davies2021advancing` — Davies et al., *Proc. IEEE* 109(5), 2021. Weight precision.
- `dey2021mapping` — Dey & Dimitrov, *Front. Neurosci.*, 2022. Independent
  (non-Intel) characterization; the source of the 24-bit state width.

## Validity domain

| Quantity | Value |
|---|---|
| Technology | 14 nm Intel FinFET |
| Supply for the 58 GSOPS / peak-efficiency operating point | **0.75 V** |
| Basis | **Pre-silicon SDF + SPICE simulation** (see above) |
| Structure | 128 cores × 1024 neurons; 4096 in/out axons per core; 16 MB synaptic memory |
| Temperature | **not stated by any source** |
| Clock | **none — Loihi is fully asynchronous** |

That last row is a real constraint, not a footnote: *"All logic in the chip is
digital, functionally deterministic, and implemented in an asynchronous bundled data
design style."* Any Loihi profile quoting an operating FREQUENCY is wrong; the
`t_cycle` this profile declares is a **measured sustained per-timestep wall**, not
a clock (below).

## Notes on individual constants

- **`t_cycle` is a MEASURED workload band, not a clock** (owner decision
  2026-08-16, reversing the earlier refusal). Loihi has no global clock and no
  fixed timestep — a timestep is *"throttled to a real-time scale (one millisecond
  per timestep is common), or the mesh may operate unthrottled"*. What IS
  published is the unthrottled sustained wall measured on Pohoiki Springs:
  *"just over 13 µs per timestep for the 1M-pattern dataset workload and 5.8 µs
  per timestep when processing 76,800-pattern datasets"* (frady2020neuromorphic
  §5.1). The band's ends are those two workloads; the nominal takes the LARGER
  figure because the same paper reports late-window slowdowns to ~60 µs/step
  (x86 spike-I/O bound, Fig 8) that a flat wall already under-charges. The
  declaration is validated by `loihi_knn_query_latency` — a SELF-CONSISTENCY
  reference case (same paper): the pricer's `t_cycle × 140 steps + the measured
  520 µs CPU wall` lands at −22.8% of Table 3's published 3.03 ms query latency,
  and the inverted throughput at +16.8% of the published 366 s⁻¹. An operator
  who runs throttled (1 ms/step) or on a different mesh scale overrides it.
- **`host_compute_rate` is the identity (1.0).** The derate a MEASURED host wall
  is divided by. Loihi-hosted deployments measure their CPU walls end-to-end
  (§5.4), and derating a measurement would invent headroom; no host THROUGHPUT
  rate (`host_macs_per_s`) is declared, so a run that never measured its wall
  still refuses host-inclusive axes by name.
- **The per-op energy is MARGINAL, not an aggregate.** Davies' 23.6 pJ is declared as
  `e_mac`, beside the neuron-update and static-power constants the same table reports
  separately — not as `e_synaptic_event_total`, which would suppress them. Filed as an
  aggregate it made Loihi look like it could price a per-inference energy without a
  timestep; it cannot, and now says so.
- **`e_neuron_update` is ONE banded constant, not two.** Davies publishes "81 pJ / 52 pJ"
  for the *active / inactive* neuron update — one quantity at two activity levels. The
  profile previously declared both it and `e_leak_per_neuron_step` (52 pJ), and the
  pricer sums both over `neurons_used × timesteps`, so every neuron was charged twice.
  Only `e_neuron_update` survives, banded 52–81, with the nominal at the INACTIVE cost:
  the multiplicand is every update, a census cannot say which of them fired, and in a
  sparse SNN nearly none do.
- **One-sided bands stay one-sided.** `e_mac` is published as a
  *minimum* (23.6 pJ) and `t_array_read` as a *maximum* (3.5 ns). Both are declared as
  point values at the published number: centring a band on a published extremum would
  fabricate the end the paper withheld.
- **Anisotropy is preserved.** The NoC costs 3.0 pJ / 4.1 ns per E–W hop and
  4.0 pJ / 6.5 ns per N–S hop. A hop count does not say which axis it took, so the two
  published values are the band and the nominal is their mean — collapsing them would
  hide a 1.33×/1.59× directional spread the paper actually reports.
- **`area_per_core_total` is a die-area SHARE**, not a measured core layout: 60 mm² ÷
  128 cores, on a die that also carries three x86 cores and I/O. The real
  neuromorphic core is therefore *smaller*. What makes the division trustworthy is
  that it reproduces the paper's own figure — 1024 neurons / 0.46875 mm² = 2184.5
  neurons/mm², against the stated *"maximum neuron density of 2,184 per mm²"*.
- **Zeros are facts.** `area_per_adc`, `e_adc_conversion`, `t_adc_conversion`,
  `write_sigma` and `read_sigma` are 0.0 because Loihi is fully digital — no
  converter, no analog programming spread. That is different from an undeclared
  constant, which means "this target has not said".

## The programming group — DERIVED, and how (TS6)

No Loihi paper measures a weight load, and refusing every programming term by name
made the target unusable for any study that reprograms. The three constants below are
therefore **authored derivations**: each states its anchor, its ratio and the arithmetic,
and each is `evidence_kind: derived` so a comparison discloses that it is not a
measurement. A ratio band's nominal always sits at the band's **log-space centre**
(`sqrt(low x high)`) — the centre of a multiplicative spread is geometric.

- **`e_program_per_byte` = 11.8 / 28.904 / 70.8 pJ/B.** Anchor: this profile's own
  `e_mac` (23.6 pJ per synaptic spike op = one read of one synapse entry). A weight is
  up to 9 bits with sign, so one entry is one byte and the per-byte access anchor is
  23.6 pJ/B. An SRAM write costs ~1–3× its read, and the anchor overstates the bare
  array read, so the band is `[0.5x, 3x]` the anchor — the band *is* the ratio spread.
  It is **not** `e_dma_per_byte`: that would be what MOVING a byte onto the chip costs,
  which Intel does not publish, and which a carried activation also pays. This is the
  commit into synaptic SRAM, charged on `reprogrammed_bytes` alone.
- **`t_program_per_byte` = 2.545 / 142.7 / 636.4 ns/B.** Donor-scaled along
  `validity.technology_node_nm` from the two profiles that declare an interface rate:
  TrueNorth's scan chain (800 ns/B at 28 nm → 400 at 14 nm) and the generic 22 nm
  exemplar (4/80/1000 ns/B → 2.545/50.91/636.4). The band is the envelope of both
  scaled donors; the nominal is the geometric mean of their scaled nominals.
- **`e_core_program` = 15.46 / 31.025 / 62.26 nJ.** Anchor: this profile's own measured
  `e_core_init` (30.92–31.13 nJ) — programming a core walks the same per-core control
  path a state reset walks. Band `[0.5x, 2x]` on the anchor's own corners. It is the
  per-LOAD twin of `e_core_init`, which stays per-inference; the pricer never charges
  both for one event.

## What this profile deliberately does not declare

`e_dma_per_byte` (Intel publishes no host-interface transfer energy, so the byte's
*journey* stays unpriced even though its *commit* is now derived), `e_sync_barrier`
(only its *time* is published), `e_row_drive`, `e_leak_per_neuron_step` (already inside
`e_neuron_update`, see above), and — of the `host` group — everything except the
measured-wall identity `host_compute_rate`. None are published; each absence disables
exactly the objectives that need it.

## Cross-check against what the repository already claims

The research pass compared this profile against the Loihi numbers already in the
tree, and found a **provenance defect worth recording**:

`chip_simulation/sanafe/presets.py::LOIHI_PRESET` and the vendored
`sana_fe/arch/loihi.yaml` both claim their numbers come from "Davies 2018 SPICE".
Only ~4 of ~22 values are traceable there. The real source is Boyle 2023's SANA-FE
paper: **micro-benchmark measurements on Nahuku silicon** with linear-regression
fits — arguably a *better* basis than pre-silicon simulation, which makes the defect
the attribution rather than the numbers. It also explains the anomalies: the preset's
35.5 pJ synapse energy sits above the paper's 23.6 pJ minimum, and its 3.8 ns / 4.7 ns
latencies *exceed* the published 3.5 ns maximum.

This profile therefore declares its constants **directly from the papers** rather than
referencing the presets through `source_ref`: a reference would inherit the preset's
provenance, and the preset's stated provenance is wrong.

One further in-repo disagreement, reported but not fixed here (it is geometry, not
physics): `imc_platforms_literature.py::loihi_dense_equiv_128x1024` declares
`max_axons=128`, which matches no Loihi constant — its own provenance string quotes
1024 neurons and 4096 axons. It looks like the *kilobyte* count of synaptic memory
reused as an axon count. It also sets `has_bias=False`, contradicting Davies 2018
Eq. (1) (*"b_i is a constant bias current"*).
