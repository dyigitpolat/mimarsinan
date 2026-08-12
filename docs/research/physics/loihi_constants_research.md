# Loihi (1st generation) — published per-unit physical constants

Research note for the hardware "physics profile". Every number below is either
(a) transcribed verbatim from a published source, or (b) an explicit arithmetic
derivation from published numbers with the division shown, or (c) listed in
§4 as **UNSOURCED**. Nothing here is guessed.

Status: research only — no code changed. Read §6 before trusting any Loihi
constant already living in `src/`.

## 0. Sources used

| key | source | type | how used |
| --- | --- | --- | --- |
| `davies2018loihi` | M. Davies et al., "Loihi: A Neuromorphic Manycore Processor with On-Chip Learning", *IEEE Micro* 38(1):82–99, 2018 | peer-reviewed (magazine, IEEE) | **primary**; Table 2 is the per-event energy/latency SSOT |
| `davies2021advancing` | M. Davies et al., "Advancing Neuromorphic Computing With Loihi: A Survey of Results and Outlook", *Proc. IEEE* 109(5):911–934, 2021 | peer-reviewed | per-core memory split, weight precision, timestep scale |
| `frady2020neuromorphic` | E. P. Frady et al., "Neuromorphic Nearest Neighbor Search Using Intel's Pohoiki Springs", NICE 2020 | peer-reviewed | **only** published source found for Loihi static/leakage power; multi-chip barrier sync |
| `dey2021mapping` | S. Dey, A. Dimitrov, "Mapping and Validating a Point Neuron Model on Intel's Neuromorphic Hardware Loihi", *Front. Neurosci.* 16, 2022 | peer-reviewed, independent (non-Intel) | state-variable bit widths |
| `boyle2023sanafe` | J. A. Boyle et al., "SANA-FE: Simulating Advanced Neuromorphic Architectures for Fast Exploration", *IEEE TCAD* 44:3165–3178, 2025 | peer-reviewed | provenance of the in-repo SANA-FE Loihi preset (§6) |
| `blouw2018benchmarking` | P. Blouw et al., "Benchmarking Keyword Spotting Efficiency on Neuromorphic Hardware", NICE 2018 | peer-reviewed | system-level anchor only; no per-unit constants taken |

PDFs cached at
`/tmp/claude-1005/-home-yigit-repos-research-stuff/45df81fa-3078-4a3f-8bb8-1b58f5a3b81c/scratchpad/papers/`.

**Critical framing for the whole table below.** Davies 2018 Table 2 is titled
"Loihi **pre-silicon** performance and energy measurements" and the body text
qualifies it as:

> "Table 2 provides a selection of energy and performance measurements from
> pre-silicon SDF and SPICE simulations, consistent with early post-silicon
> characterization." (§Results / Silicon Realization, p. 95)

So every Table-2 number is **simulated (SDF + SPICE), corroborated but not
replaced by silicon measurement**. Treat them as a design-intent profile, not a
measured chip characterization. Column `basis` records this per row.

---

## 1. Constants table

`evidence_kind` ∈ published | datasheet | derived | estimated.
`basis` ∈ simulated | measured | projected | architectural (a structural
constant, not a measurement).

### 1a. ENERGY

| quantity | value | band | unit | evidence_kind | basis | citation | verbatim quote |
| --- | --- | --- | --- | --- | --- | --- | --- |
| energy per synaptic operation (no learning) | 23.6 | ≥23.6 (stated as a **min**; no max published) | pJ | published | simulated | `davies2018loihi` Table 2, p. 96 | "Energy per synaptic spike op (min) \| 23.6 pJ" |
| energy per synaptic operation **with learning** (pairwise STDP weight update) | 120 | — | pJ | published | simulated | `davies2018loihi` Table 2, p. 96 | "Energy per synaptic update (pairwise STDP) \| 120 pJ" |
| energy per neuron update — active | 81 | — | pJ | published | simulated | `davies2018loihi` Table 2, p. 96 | "Energy per neuron update (active / inactive) \| 81 pJ / 52 pJ" |
| energy per neuron update — inactive (= leak/idle energy per neuron per timestep) | 52 | 52–81 across inactive→active | pJ | published | simulated | `davies2018loihi` Table 2, p. 96 | "Energy per neuron update (active / inactive) \| 81 pJ / 52 pJ" |
| energy per spike **within a tile** | 1.7 | — | pJ | published | simulated | `davies2018loihi` Table 2, p. 96 | "Within-tile spike energy \| 1.7 pJ" |
| energy per **inter-tile hop**, E–W | 3.0 | 3.0–4.0 over the two axes | pJ | published | simulated | `davies2018loihi` Table 2, p. 96 | "Energy per tile hop (E-W / N-S) \| 3.0 pJ / 4.0 pJ" |
| energy per **inter-tile hop**, N–S | 4.0 | 3.0–4.0 over the two axes | pJ | published | simulated | `davies2018loihi` Table 2, p. 96 | "Energy per tile hop (E-W / N-S) \| 3.0 pJ / 4.0 pJ" |
| static/leakage power **per chip** | 104 | 104.3–104.4 (two independent rows) | mW | derived (see §3.2) | measured | `frady2020neuromorphic` Table 1 + §5.2, p. 6–7 | "Static \| 3.34 W" (Table 1, 76,800-pattern rows); "Estimates of the different power components for the Loihi chips are obtained by extrapolating measurements on an instrumented board containing 32 Loihi chips and running 76,800-pattern search queries." (§5.2); "Static power is due to leakage when all circuits are fully powered." (§5.2) |
| static/leakage power **per neuromorphic core** | 0.815 | upper bound | mW | derived (see §3.2) | measured | `frady2020neuromorphic` §5.2 + `davies2018loihi` p. 86 | "Almost all leakage can be attributed to the neuromorphic cores, which dominate chip area." (`frady2020neuromorphic` §5.2) |
| per-chip **reset / state re-initialization** energy (whole-chip network reset between queries) | 5.94 | 5.94–5.98 (two rows) | µJ | derived (see §3.3) | measured | `frady2020neuromorphic` Table 2 + §5.2, p. 6–7 | "Reset \| 0.19" (Table 2, mJ, 76,800-pattern rows); "A reset phase occurs after each query to prepare the system for the next query." (§5.2) |

### 1b. TIME

| quantity | value | band | unit | evidence_kind | basis | citation | verbatim quote |
| --- | --- | --- | --- | --- | --- | --- | --- |
| time per synaptic operation | 3.5 | ≤3.5 (stated as a **max**) | ns | published | simulated | `davies2018loihi` Table 2, p. 96 | "Time per synaptic spike op (max) \| 3.5 ns" |
| time per synaptic update (pairwise STDP) | 6.1 | — | ns | published | simulated | `davies2018loihi` Table 2, p. 96 | "Time per synaptic update (pairwise STDP) \| 6.1 ns" |
| time per neuron update — active | 8.4 | 5.3–8.4 inactive→active | ns | published | simulated | `davies2018loihi` Table 2, p. 96 | "Time per neuron update (active / inactive) \| 8.4 ns / 5.3 ns" |
| time per neuron update — inactive | 5.3 | 5.3–8.4 inactive→active | ns | published | simulated | `davies2018loihi` Table 2, p. 96 | "Time per neuron update (active / inactive) \| 8.4 ns / 5.3 ns" |
| spike hop latency — **intra-tile** | 2.1 | — | ns | published | simulated | `davies2018loihi` Table 2, p. 96 | "Within-tile spike latency \| 2.1 ns" |
| spike hop latency — **inter-tile**, E–W | 4.1 | 4.1–6.5 over the two axes | ns | published | simulated | `davies2018loihi` Table 2, p. 96 | "Latency per tile hop (E-W / N-S) \| 4.1 ns / 6.5 ns" |
| spike hop latency — **inter-tile**, N–S | 6.5 | 4.1–6.5 over the two axes | ns | published | simulated | `davies2018loihi` Table 2, p. 96 | "Latency per tile hop (E-W / N-S) \| 4.1 ns / 6.5 ns" |
| **barrier-sync time, single chip** (mesh-wide, scaling with tile count) | 113 → 465 | 113–465 for 1→32 tiles | ns | published | simulated | `davies2018loihi` Table 2, p. 96 | "Mesh-wide barrier sync time (1-32 tiles) \| 113-465 ns" |
| barrier-sync time, **multi-chip** — 256-chip column (4×64) | 14.3 | — | µs | published | measured | `frady2020neuromorphic` §4.3, p. 5 | "due to the highly asymmetric dimensions of each column of Loihi chips (4×64), the barrier synchronization time per column of 14.3µs is only marginally faster than the barrier synchronization time across all 768 (12×64) chips, 16.2µs." |
| barrier-sync time, **multi-chip** — full 768-chip mesh (12×64) | 16.2 | — | µs | published | measured | `frady2020neuromorphic` §4.3, p. 5 | (same sentence as above) |
| algorithmic timestep wall time — throttled real-time mode | 1000 | — | µs | published | measured | `davies2021advancing` §II-A, p. 914 | "Depending on application needs, timesteps may be throttled to a real-time scale (one millisecond per timestep is common), or the mesh may operate unthrottled, proceeding as fast as the communication patterns in the workload allow. The latter results in each timestep consuming varying amounts of real time." |
| algorithmic timestep wall time — unthrottled, multi-chip workloads | 5.8 / 13 | 5.8–13 (two workloads) | µs | published | measured | `frady2020neuromorphic` §5.1, p. 7 | "the system typically sustains just over 13µs per timestep for the 1M-pattern dataset workload and 5.8µs per timestep when processing 76,800-pattern datasets." |
| cross-sectional spike bandwidth per tile | 3.44 | — | Gspike/s | published | simulated | `davies2018loihi` Table 2, p. 96 | "Cross-sectional spike bandwidth per tile \| 3.44 Gspike/s" |
| **operating frequency / clock period** | **N/A — no global clock** | — | — | published | architectural | `davies2018loihi` §Design Implementation, p. 87 & p. 94 | "All logic in the chip is digital, functionally deterministic, and implemented in an asynchronous bundled data design style." (p. 87); "because the activity in SNNs is highly sparse in both space and time, the activity gating that comes automatically with asynchronous flow control eliminates the power that would often be wasted by a continuously running clock." (p. 94) |
| (corroboration for the above) | — | — | — | published | architectural | `davies2021advancing` §II-A, p. 913 | "Loihi's neuromorphic mesh and cores are built with asynchronous circuits and, at the transistor level, communicate event-driven tokens of information between logic stages. This allows spike messages and iterative processes within each core to proceed as fast or slow as the computation and pipeline activities allow without ever waiting for clock edges or needlessly expending clock power during periods of inactivity." |

### 1c. AREA / CAPACITY

| quantity | value | band | unit | evidence_kind | basis | citation | verbatim quote |
| --- | --- | --- | --- | --- | --- | --- | --- |
| die area | 60 | — | mm² | published | measured | `davies2018loihi` §Silicon Realization, p. 95 | "Loihi was fabbed in Intel's 14-nm FinFET process. The chip instantiates a total of 2.07 billion transistors and 33 MB of SRAM over its 128 neuromorphic cores and three x86 cores, with a die area of 60 mm2." |
| technology node | 14 (FinFET) | — | nm | published | measured | `davies2018loihi` §Silicon Realization, p. 95 | (same sentence as above) |
| transistor count | 2.07 | — | ×10⁹ | published | measured | `davies2018loihi` §Silicon Realization, p. 95 | (same sentence as above) |
| total SRAM, chip | 33 | — | MB | published | measured | `davies2018loihi` §Silicon Realization, p. 95 | (same sentence as above) |
| total synaptic memory, chip | 16 | — | MB | published | measured | `davies2018loihi` §Silicon Realization, p. 95 | "Loihi includes a total of 16 MB of synaptic memory." |
| **per-core area** | 0.469 | — | mm² | **derived** (see §3.1) | derived from measured die area | `davies2018loihi` p. 95 | derived as 60 mm² ÷ 128 cores; self-consistency check against the paper's own "Loihi's maximum neuron density of 2,184 per mm2" in §3.1 |
| SRAM per neuromorphic core (total, incl. ECC) | 2 | — | Mb (= 256 KB) | published | architectural | `davies2018loihi` §Core Microarchitecture, p. 92 | "The core's total SRAM capacity is 2 Mb, including ECC overhead." |
| synaptic state per core | 128 | — | KB | published | architectural | `davies2018loihi` p. 89 constraint 2; corroborated `davies2021advancing` §II-A p. 913 | "The total synaptic fan-in state mapped to any core must not exceed 128 KB (Nsyn × 64b, subject to compression and list alignment considerations)." / "Each core contains 128 kB of synaptic state, and another 20 kB of routing tables that can be flexibly allocated over its 1024 neurons" |
| routing-table memory per core | 20 | — | kB | published | architectural | `davies2021advancing` §II-A, p. 913 | (same sentence as above) |
| neurons (compartments) per core | 1024 | — | — | published | architectural | `davies2018loihi` p. 86 & p. 89 constraint 1 | "Each neuromorphic core implements 1,024 primitive spiking neural units (compartments) grouped into sets of trees constituting neurons." (p. 86); "The total number of neurons assigned to any core may not exceed 1,024 (Ncx)." (p. 89) |
| neuromorphic cores per chip | 128 | — | — | published | architectural | `davies2018loihi` §Chip Overview, p. 86 | "Loihi features a manycore mesh comprising 128 neuromorphic cores, three embedded x86 processor cores, and off-chip communication interfaces that hierarchically extend the mesh in four planar directions to other chips." |
| neurons per chip | 131,072 | — | — | published | architectural | `davies2021advancing` §II-A, p. 913 | "Loihi implements 131 072 leaky-integrate-and-fire neurons using a digital, discrete-time computational model partitioned over 128 cores" |
| max synapses per core, 64-bit-word budget (N_syn) | 16,384 | — | 64-b words | **derived** (see §3.4) | architectural | `davies2018loihi` p. 89 constraint 2 | derived: 128 KB ÷ 64 b = 16,384 words |
| max synapses per core, **densest 1-bit format** | 1,048,576 | — | synapses | **derived** (see §3.4) | architectural | `davies2018loihi` p. 89 & p. 95 | "With its densest 1-bit synapse format, this provides a total of 2.1 million unique synaptic variables per mm2" |
| **axon fan-in limit per core** (N_axin, distribution lists) | 4096 | — | — | published | architectural | `davies2018loihi` p. 89 constraint 4 | "The total number of distribution lists, associated by axon_id, in any core must not exceed 4,096 (Naxin). This is the number of input-side axon_id routing slots highlighted in red in Figure 3." |
| **axon fan-out limit per core** (N_axout, core-to-core edges) | 4096 | — | — | published | architectural | `davies2018loihi` p. 89 constraint 3 | "The total number of core-to-core fan-out edges mapped to any given core must not exceed 4,096 (Naxout)." |
| binding constraints in practice | constraints 2 & 4 | — | — | published | architectural | `davies2018loihi` p. 89 | "In practice, constraints 2 and 4 tend to be the most limiting." |
| max neuron density | 2184 | — | neurons/mm² | published | derived-by-authors | `davies2018loihi` §Silicon Realization, p. 95 | "Loihi's maximum neuron density of 2,184 per mm2 is marginally worse than TrueNorth's." |
| synaptic-variable density (1-bit format) | 2.1 | — | ×10⁶ /mm² | published | derived-by-authors | `davies2018loihi` §Silicon Realization, p. 95 | "With its densest 1-bit synapse format, this provides a total of 2.1 million unique synaptic variables per mm2, over three times higher than TrueNorth, the previously most dense SNN chip." |

### 1d. BIT WIDTHS / NUMERIC FORMAT

| quantity | value | band | unit | evidence_kind | basis | citation | verbatim quote |
| --- | --- | --- | --- | --- | --- | --- | --- |
| synaptic weight bit width (supported range) | 1 … 9 | 1–9, signed or unsigned, mixable within one fan-out | bits | published | architectural | `davies2018loihi` §Network Connectivity Architecture, p. 87 | "Variable synaptic formats. Loihi supports any weight precision between one and nine bits, signed or unsigned, and weight precisions may be mixed (with scale normalization) even within a single neuron's fan-out distribution." |
| synaptic weight field in the learning engine | 9 (signed) | — | bits | published | architectural | `davies2018loihi` Table 1, p. 92 | "9 \| wgt+C \| 9b (S) \| Synaptic weight" |
| corroboration of weight range | 1 … 9 (signed) | — | bits | published | architectural | `davies2021advancing` §II-A, p. 913 | "variable weight precision (from 1- to signed 9-b values)" |
| synaptic delay field | 6 (unsigned) | — | bits | published | architectural | `davies2018loihi` Table 1, p. 92 | "10 \| dly+C \| 6b (U) \| Synaptic delay" |
| synaptic tag field | 9 (signed) | — | bits | published | architectural | `davies2018loihi` Table 1, p. 92 | "11 \| tag+C \| 9b (S) \| Synaptic tag" |
| spike-trace field | 7 (unsigned) | — | bits | published | architectural | `davies2018loihi` Table 1, p. 92 + §Trace Evaluation p. 91 | "The Loihi hardware computes a low-precision (7-bit) approximation of this first-order filter using stochastic rounding." |
| learning-engine accumulator | 16 | — | bits | published | architectural | `davies2018loihi` §Learning Rule Functional Form, p. 91 | "The multiplications and summations of Equation 4 are computed iteratively by the hardware and accumulated in 16-bit registers." |
| **membrane potential + synaptic current state width** | ±23 (i.e. signed, magnitude bound 2²³) | 8–24 across state/config variables | bits | published (independent) | measured/architectural | `dey2021mapping` §3.1.2 | "State variables—membrane potential and current—are allotted ±23 bits each." |
| membrane threshold field | 17 (as the high bits of a 23-bit word) | — | bits | published (independent) | architectural | `dey2021mapping` §3.1.2 | "the membrane potential threshold is assigned 17 bits interpreted as the 17 high bits of a 23 bit word" |
| membrane decay constant field | 12 | — | bits | published (independent) | architectural | `dey2021mapping` §3.1.2 | "The membrane potential decay constant δ_v is allotted 12 bits" |
| general state/config variable width | 8 … 24 | — | bits | published (independent) | architectural | `dey2021mapping` §4.3 | "most state and configuration variables are in the range of 8–24 bits" |
| synaptic delay range (default provisioning) | 8 | 8 … 62 (62 only when fewer compartments are mapped) | timesteps | published | architectural | `davies2018loihi` §Core Microarchitecture, p. 92 | "The parameter Nsdelay indicates the minimum number of synaptic delay units supported, eight in Loihi. Larger synaptic delay values, up to 62, may be supported when fewer neuron compartments are needed by a particular mapped network." |
| learning epoch period (T_epoch) | 2 … 8 typical | max 63 | timesteps | published | architectural | `davies2018loihi` §Learning Rule Functional Form, p. 91 | "The epoch period is globally configured per core up to a maximum value of 63, with typical values in the 2 to 8 range." |
| spike message width | 32 | — | bits | published | architectural | `davies2021advancing` §II-A, p. 913 | "All communication between neurons occurs over spike events, 32-bit messages containing destination addressing, and, sometimes, source addressing and graded-value payloads that the network-on-chip routes between cores." |
| neuron model has a per-compartment bias | yes | — | — | published | architectural | `davies2018loihi` §Spiking Neural Unit, p. 83, Eq. (1) | "where w_i,j is the synaptic weight from neuron j to neuron i, α_u(t) = τ_u^-1 exp(-t/τ_u)H(t) is the synaptic filter impulse response parameterized by the time constant τ_u with H(t) the unit step function, and b_i is a constant bias current." |
| mesh scaling limits | 4096 on-chip cores / 16,384 chips | — | — | published | architectural | `davies2018loihi` §Chip Overview, p. 86 | "The mesh protocol supports scaling to 4096 on-chip cores and, through hierarchical addressing, up to 16,384 chips." |

---

## 2. Note on Davies 2018 Table 3 — do NOT read it as Loihi

Table 3 ("Comparison of solving ℓ1 minimization on Loihi and Atom") and the
"5 mm² active silicon" figure describe a **predecessor** chip, not Loihi:

> "On an earlier iteration of the Loihi architecture, we quantitatively assessed
> the efficiency of Spiking LCA to solve LASSO... Both chips were fabbed in
> 14-nm technology, were evaluated at a 0.75-V supply voltage, and required
> similar active silicon areas (5 mm2)." (§Algorithmic Results, p. 96)

The figure caption confirms it: "Image reconstruction from the sparse
coefficients computed using the Loihi **predecessor**." (Fig. 8, p. 96). Do not
use the 5 mm² number as a Loihi area.

---

## 3. Explicit derivations

### 3.1 Per-core area (system-level ÷ count)

```
per_core_area = die_area / n_neuromorphic_cores
              = 60 mm² / 128
              = 0.46875 mm²  ->  0.469 mm²
```

Self-consistency check against the paper's own derived density:

```
128 cores × 1024 neurons/core = 131,072 neurons
131,072 / 60 mm²              = 2184.53 neurons/mm²
```

which reproduces the paper's stated "maximum neuron density of 2,184 per mm2"
exactly. That confirms the paper itself normalizes by the **full 60 mm² die**.

**Caveat (important):** 0.469 mm² is therefore a *die-area share*, not a measured
neuromorphic-core layout area. It silently absorbs the 3 embedded x86 cores, the
off-chip interfaces, the NoC routers and the I/O ring. A true per-core layout
area is **UNSOURCED** (§4). The real neuromorphic-core area is strictly smaller
than 0.469 mm².

### 3.2 Static / leakage power per chip and per core

Frady 2020 Table 1 reports system static power for two workload sizes. The
32-chip figure is fully grounded (both the wattage and the chip count are stated
in the paper):

```
static_per_chip = 3.34 W / 32 chips = 0.104375 W = 104.4 mW/chip
```

Cross-check using the 1M-pattern row (chip count **inferred**, see caveat):

```
53.4 W / 3.34 W = 15.99  ->  16 × 32 = 512 chips
53.4 W / 512 chips       = 0.104297 W = 104.3 mW/chip
```

The two agree to 0.1%, which corroborates the paper's own modelling assumption
("static power and x86 idle power are assumed to remain constant per chip",
§5.2). **Caveat:** the paper never states that the 1M-pattern row used 512
chips — that is my inference from the exact 16× ratio (768 chips would give
69.5 mW/chip and would contradict the paper's constant-per-chip assumption).
Use the 32-chip derivation as the primary; the 512-chip line is corroboration
only.

Per neuromorphic core:

```
static_per_core = 104.4 mW / 128 cores = 0.8155 mW = 816 µW/core
```

**Caveat:** this is an **upper bound**. The paper says "*Almost* all leakage can
be attributed to the neuromorphic cores"; assigning 100% of chip leakage to the
128 cores over-attributes by the (unquantified) x86 + I/O + NoC share. Also, the
supply voltage of the Pohoiki Springs measurement is **not stated** — this
number is *not* pinned to the 0.75 V of Davies 2018 Table 2. Do not mix it with
Table 2 energies without flagging the voltage mismatch.

### 3.3 Per-chip reset / re-initialization energy

Frady 2020 Table 2, "Reset" column (units mJ), 76,800-pattern rows (32 chips):

```
0.19 mJ / 32 chips = 5.9375 µJ per chip per reset
```

Cross-check, 1M-pattern rows (512 chips per §3.2):

```
3.06 mJ / 512 chips = 5.977 µJ per chip per reset
```

Agreement to 0.7%, consistent with the paper's "Reset energy per chip is
constant for every query" (§5.2). Per core:

```
5.9375 µJ / 128 cores = 46.4 nJ per core per reset
```

**This is a whole-network state reset, NOT a synapse-memory programming
(DMA/weight-load) energy and NOT a barrier-sync energy.** See §4 and §6.

Order-of-magnitude context for §6: a single chip-wide timestep in which every
neuron takes an *inactive* update costs

```
131,072 neurons × 52 pJ = 6.82 µJ   (all-inactive timestep, chip-wide)
131,072 neurons × 81 pJ = 10.6 µJ   (all-active timestep, chip-wide)
```

so the measured 5.94 µJ per-chip reset sits just below one whole idle timestep.

### 3.4 Synapse capacity per core

```
N_syn budget  = 128 KB / 64 b per word = 131,072 B × 8 b / 64 b = 16,384 words
1-bit synapses/core (binary MB) = 128 × 1024 × 8 = 1,048,576
```

Chip level, and the reason the paper's density figure has a small ambiguity:

```
binary  MB: 128 cores × 1,048,576 = 134.2 × 10⁶ synapses ; /60 mm² = 2.24 ×10⁶/mm²
decimal MB: 16 × 10⁶ B × 8 b      = 128   × 10⁶ synapses ; /60 mm² = 2.13 ×10⁶/mm²
```

The paper's stated "2.1 million unique synaptic variables per mm2" matches the
**decimal-MB** reading. Prefer the paper's own 2.1 ×10⁶/mm² over either
re-derivation, and treat 128–134 M as the chip-level 1-bit synapse capacity band.

### 3.5 Tile geometry

Table 2 reports quantities "per tile" and a barrier-sync range "(1-32 tiles)",
which together imply 32 tiles per chip and hence

```
cores_per_tile = 128 cores / 32 tiles = 4
```

This is an **inference**, not a stated constant. The paper never states the
number of tiles, the cores-per-tile packing, or the mesh aspect ratio. The
8×4 tile grid used in the in-repo SANA-FE arch (§6) is **not** in the paper.

---

## 4. UNSOURCED — no published per-unit value found

These must be reported as unknown in any physics profile. Do not fill them by
analogy.

| # | quantity | why it is unsourced / what exists instead |
| --- | --- | --- |
| U1 | **energy per synapse-memory read/write (programming / DMA / weight load)** | No source found in any of the six papers. Frady 2020 reports programming *time* only — "The programming time for a 192-chip column was measured to be 893 seconds, or about 4.6 seconds per Loihi chip. Incrementally adding additional data points to the system requires on the order of 1ms to encode and program." (§5.3) — with **no** associated energy. The per-chip reset energy in §3.3 is a *different* quantity. |
| U2 | **barrier-sync ENERGY** (per barrier, per core or per chip) | Davies 2018 publishes barrier *time* only (113–465 ns). No barrier energy anywhere. See §6 finding F9. |
| U3 | **measured neuromorphic-core layout area** | Only the 60 mm² die area is published. §3.1's 0.469 mm² is a die-area share and is an over-estimate of the core proper. No core-level die-photo dimension is published. |
| U4 | **operating temperature / junction temperature** for any energy or power number | Neither Davies 2018 nor Frady 2020 states a temperature or a temperature corner. |
| U5 | **supply voltage of the Frady 2020 static-power measurement** | Table 1/2 report watts with no voltage. Only Davies 2018 Table 2 is voltage-pinned (0.75 V). |
| U6 | **maximum synaptic fan-in per NEURON** | Loihi's published fan-in limits are all **per core** (128 KB synaptic state; N_axin ≤ 4096 distribution lists). No per-neuron fan-in cap is specified — a single neuron may in principle consume the whole core's synaptic memory. Any per-neuron number is a mapping-policy choice, not a hardware constant. |
| U7 | **energy per synaptic op broken down by synapse format** (dense vs sparse vs population/convolutional) | Table 2 gives one number qualified "(min)". The format-resolved values used by SANA-FE (§6) are measured by Boyle et al., not published as a table. |
| U8 | **energy/latency of the x86 management cores per operation** | Frady 2020 gives aggregate x86 *power* (2.09–2.14 W over 32 chips ⇒ ~65 mW/chip) but no per-operation cost, and it is workload-specific rather than a per-unit constant. |
| U9 | **per-hop energy separated from per-router-buffer/link energy** | Table 2's "Energy per tile hop" is a lumped figure; no router/link decomposition is published. |
| U10 | **energy per neuron update WITH learning enabled** (as distinct from the 120 pJ per-synapse STDP update) | Only the per-synapse learning update is published. |
| U11 | **operating frequency** | Genuinely non-existent by design, not merely unreported — Loihi is fully asynchronous (see §1b). Any "clock frequency" in a Loihi profile is an error. |
| U12 | **leakage vs. voltage/temperature scaling curve** | The 0.50–1.25 V functional range is published, but no energy-vs-voltage or leakage-vs-V/T characterization is given for any point other than 0.75 V. |

---

## 5. Validity domain

| axis | value | source |
| --- | --- | --- |
| process node | Intel 14 nm FinFET | `davies2018loihi` p. 95: "Loihi was fabbed in Intel's 14-nm FinFET process." |
| **voltage at which all Table 2 energies/latencies hold** | **0.75 V** | `davies2018loihi` Table 2 column header, p. 96: "Value at 0.75 V" |
| functional supply range | 0.50 V – 1.25 V | `davies2018loihi` p. 95: "The device is functional over a supply voltage range of 0.50 V to 1.25 V." |
| temperature | **not stated anywhere** (U4) | — |
| basis of Table 2 | **pre-silicon SDF + SPICE simulation**, corroborated by early post-silicon characterization | `davies2018loihi` p. 95: "Table 2 provides a selection of energy and performance measurements from pre-silicon SDF and SPICE simulations, consistent with early post-silicon characterization." |
| basis of Frady 2020 power/energy | **measured** at the wall on a 32-chip instrumented board, then extrapolated per-chip to 1M-pattern scale | `frady2020neuromorphic` §5.2 (quoted in §1a); the 1M rows are *extrapolated*, i.e. **projected**, not measured |
| basis of Dey 2022 bit widths | independent (non-Intel) hardware mapping study on real Loihi | `dey2021mapping` |
| what Table 2 does NOT cover | temperature corners, voltage corners other than 0.75 V, process corners, aged/worst-case silicon | — |

**Mixing rule.** Table 2 (0.75 V, simulated) and Frady 2020 (unknown V,
measured) are two different operating points. A profile that adds the 816 µW/core
leakage to the 52 pJ inactive-update energy is combining an unpinned-voltage
measurement with a 0.75 V simulation. Record that as a known composition risk.

---

## 6. In-repo cross-check

Three places in this repo carry Loihi constants.

### 6a. `sana_fe/arch/loihi.yaml` (vendored SANA-FE arch preset) vs Davies 2018 Table 2

The file's header comment asserts:

```
# Energy and time estimates of different events, generated from SPICE
#  simulations of Loihi.  All numbers were taken from:
#  "Loihi: A Neuromorphic Manycore Processor with On-Chip Learning" (2018)
#  M. Davies et al
```

| yaml key | in-repo value | Davies 2018 Table 2 | verdict |
| --- | --- | --- | --- |
| `energy_east_hop` / `energy_west_hop` | 3.0 pJ | 3.0 pJ (E-W) | **AGREES** |
| `latency_east_hop` / `latency_west_hop` | 4.1 ns | 4.1 ns (E-W) | **AGREES** |
| `latency_north_hop` / `latency_south_hop` | 6.5 ns | 6.5 ns (N-S) | **AGREES** |
| `energy_north_hop` / `energy_south_hop` | 4.2 pJ | **4.0 pJ** (N-S) | **DISAGREES** (+5.0%) |
| `loihi_dense_synapse.energy_process_spike` | 35.5 pJ | 23.6 pJ (min) | **NOT IN PAPER** (+50% above the published min; not strictly contradictory since 23.6 is a *min*) |
| `loihi_sparse_synapse.energy_process_spike` | 33.6 pJ | — | **NOT IN PAPER** |
| `loihi_conv_synapse.energy_process_spike` | 24.0 pJ | 23.6 pJ (min) | close (+1.7%) but **not equal**; not transcribed |
| `loihi_dense_synapse.latency_process_spike` | 3.8 ns | **3.5 ns (max)** | **DISAGREES — exceeds the published MAX** |
| `loihi_sparse_synapse.latency_process_spike` | 4.7 ns | **3.5 ns (max)** | **DISAGREES — exceeds the published MAX by 34%** |
| `loihi_conv_synapse.latency_process_spike` | 3.1 ns | ≤3.5 ns (max) | consistent with the max, but not a paper value |
| `loihi_lif.energy_access_neuron` | 51.2 pJ | 52 pJ (inactive update) | near (−1.5%) but **different semantics** and not equal |
| `loihi_lif.energy_update_neuron` | 21.6 pJ | — | **NOT IN PAPER**; access+update = 72.8 pJ vs published active 81 pJ (**−10%**) |
| `loihi_lif.latency_access_neuron` | 6.0 ns | 5.3 ns (inactive) | **DISAGREES** (+13%) |
| `loihi_lif.latency_update_neuron` | 3.7 ns | — | **NOT IN PAPER**; access+update = 9.7 ns vs published active 8.4 ns (**+15%**) |
| `loihi_lif.energy_spike_out` | 69.3 pJ | within-tile spike energy 1.7 pJ | **NO CORRESPONDENCE** (~41×) |
| `loihi_lif.latency_spike_out` | 30.0 ns | within-tile spike latency 2.1 ns | **NO CORRESPONDENCE** (~14×) |
| `loihi_out.energy_message_out` | 111.0 pJ | — | **NOT IN PAPER** |
| `loihi_out.latency_message_out` | 5.1 ns | — | **NOT IN PAPER** |
| `loihi_in.energy_message_in` | 0.0 pJ | — | **NOT IN PAPER** (modelling choice) |
| `loihi_in.latency_message_in` | 16.0 ns | — | **NOT IN PAPER** |
| `latency_sync` table `{1: 0.6 µs, 2: 1.0 µs, 4: 1.4 µs, 29: 1.8 µs}` | 0.6–1.8 µs | **113–465 ns** for 1–32 tiles | **DISAGREES — 4–5× larger** |
| `max_neurons_supported` | 1024 | 1024 (N_cx) | **AGREES** |
| `width: 8`, `height: 4` (32 tiles), 4 cores/tile ⇒ 128 cores | 128 cores | 128 neuromorphic cores | core count **AGREES**; 4 cores/tile is a valid inference (§3.5); the **8×4 aspect ratio is NOT in the paper** |
| `energy_update` for both dendrite models | 0.0 | — | **NOT IN PAPER** (modelling choice: dendrite cost folded elsewhere) |

**Finding F1 (provenance defect, highest priority).** The yaml header's claim
that "All numbers were taken from" Davies 2018 is **false for most entries**.
Only 4 of ~22 numeric attributes are traceable to Davies 2018 Table 2. The claim
that they were "generated from SPICE simulations" is also wrong for the source
that actually produced them.

**Finding F2 (the true provenance).** `boyle2023sanafe` documents that the
SANA-FE Loihi model was calibrated by **micro-benchmark measurement on real
Loihi silicon**, not transcribed from Davies 2018:

> "We validated functional accuracy by comparing spike traces from SANA-FE and
> Loihi, using Intel's Nahuku platform, which was accessed through Intel's
> Neuromorphic Research Cloud. We also calibrated SANA-FE's Loihi model against
> the Nahuku platform using the methodology described in Section VI."
> (§VII, Experiments and Results)

> "The time-step latency and energy usage for varying N are measured by executing
> this micro-benchmark on a given platform, and linear regression analysis is
> used to estimate the average incremental cost per synaptic look-up (regression
> slope)." (§VI-A-1, Synapse stage calibration)

> "To first calibrate the simulator, we executed the four micro-benchmark setups
> on Loihi for 10^5 time-steps and measure energy usage and total latency."
> (§VII-A, Experimental Setup)

This explains, and largely *justifies*, the disagreements: these are
**measured-on-silicon regression fits**, which are arguably a *better* basis than
Davies 2018's pre-silicon SPICE. It also explains the 3.8 ns > 3.5 ns "max"
inversion (a measured incremental pipeline cost includes overheads the SPICE
op-level number excludes) and the 3-way synapse-format split, which Davies 2018
collapses into one "(min)".

The defect is therefore **the attribution, not the numbers**. The header should
cite `boyle2023sanafe` (measured, Nahuku) rather than `davies2018loihi` (SPICE).
Caveat on trusting them anyway: `boyle2023sanafe` Table IV reports the resulting
model's prediction error as up to **11.7% energy / 24.3% latency** across
benchmarks — that is the accuracy envelope these constants carry.

Note: `sana_fe/` is a vendored/gitlinked third-party tree; the fix belongs
upstream, not in a local edit.

### 6b. `src/mimarsinan/chip_simulation/sanafe/presets.py` → `LOIHI_PRESET`

| preset key | value | matches `loihi.yaml`? | matches Davies 2018? |
| --- | --- | --- | --- |
| `synapse_energy_j` 35.5e-12 | 35.5 pJ | **yes** (dense variant) | no (§6a) |
| `synapse_latency_s` 3.8e-9 | 3.8 ns | **yes** (dense variant) | no (§6a) |
| `soma_access_energy_j` 51.2e-12 | 51.2 pJ | **yes** | no |
| `soma_access_latency_s` 6.0e-9 | 6.0 ns | **yes** | no |
| `soma_update_energy_j` 21.6e-12 | 21.6 pJ | **yes** | no |
| `soma_update_latency_s` 3.7e-9 | 3.7 ns | **yes** | no |
| `soma_spike_out_energy_j` 69.3e-12 | 69.3 pJ | **yes** | no |
| `soma_spike_out_latency_s` 30.0e-9 | 30.0 ns | **yes** | no |
| `axon_out_energy_j` 111.0e-12 | 111.0 pJ | **yes** | no |
| `axon_out_latency_s` 5.1e-9 | 5.1 ns | **yes** | no |
| `axon_in_energy_j` 0.0 / `axon_in_latency_s` 16.0e-9 | — | **yes** | no |
| `dendrite_energy_j` 0.0 / `dendrite_latency_s` 0.0 | — | **yes** | no |
| `tile_hop_energy_j` **3.5e-12** | 3.5 pJ | **NO** — yaml has 3.0 (E-W) and 4.2 (N-S); their mean is 3.6 pJ | 3.5 pJ **is exactly** mean(3.0, 4.0) — i.e. the mean of the **paper's** two axes |
| `tile_hop_latency_s` **5.0e-9** | 5.0 ns | **NO** — yaml has 4.1 / 6.5; their mean is 5.3 ns | **NO** — paper's mean is also 5.3 ns |

**Finding F3.** The preset's comment reads "Loihi 1 reference numbers (Davies
2018; public)" — same misattribution as F1 for all 12 keys that in fact come
from the SANA-FE Nahuku calibration.

**Finding F4.** `tile_hop_energy_j = 3.5e-12` is an **undocumented directional
average** of the paper's 3.0/4.0 pJ, and is inconsistent with the yaml the preset
otherwise mirrors (whose N-S value is 4.2 pJ, giving mean 3.6 pJ). Pick one
basis and say which.

**Finding F5.** `tile_hop_latency_s = 5.0e-9` matches **neither** the paper's
mean (5.3 ns) nor the yaml's mean (5.3 ns) nor either individual axis. It is
**UNSOURCED**; it looks like a round number.

**Finding F6 (structural).** The preset collapses the direction-dependent NoC
into a single scalar hop cost, discarding the published E-W/N-S anisotropy
(3.0 vs 4.0 pJ, 4.1 vs 6.5 ns — a 1.33×/1.59× asymmetry). For a mesh whose
mapping decisions depend on hop direction this is a real modelling loss, and it
is not recorded anywhere as a deliberate simplification.

### 6c. `src/mimarsinan/mapping/platform/imc_platforms_literature.py` → `loihi_dense_equiv_128x1024`

| field | in-repo value | paper | verdict |
| --- | --- | --- | --- |
| `max_neurons` | 1024 | 1024 (N_cx) | **AGREES** |
| `count` (cores) | 128 | 128 neuromorphic cores | **AGREES** |
| `max_axons` | **128** | N_axin = **4096**; N_axout = **4096**; synaptic state = **128 KB** | **DISAGREES / UNJUSTIFIED** — 128 matches no Loihi axon constant. The quoted provenance lists 1024 / 128 KB / 4096 / 4096; "128" appears to be the *kilobyte* count of the synaptic memory reused as an axon count. Different quantity, different unit. |
| `weight_bits` | 8 | "any weight precision between one and nine bits" | within range, but the **maximum is 9** (signed). A narrowing choice, not stated as such. |
| `has_bias` | `False` | Loihi neurons **do** have a per-compartment bias — Eq. (1) "b_i is a constant bias current"; `davies2021advancing` confirms a configurable per-timestep bias | **DISAGREES** with the hardware (may be a deliberate mapping restriction, but it is not labelled as one) |
| `claim_eligibility` | `curve-only` | — | (curation field, no paper basis needed) |

**Finding F7.** `max_axons=128` is the clearest hard error in the in-repo Loihi
data: the platform name says `dense_equiv` (an explicitly synthetic dense
crossbar), but the transcribed value corresponds to none of the four constraints
quoted in its own `provenance` string. If a dense-equivalent axon count is
wanted, it must be derived and the derivation stated (e.g. from the 128 KB
budget at a chosen weight width), not silently set to 128.

**Finding F8.** `has_bias=False` contradicts the primary source. Loihi supports a
constant bias current per compartment.

### 6d. `src/mimarsinan/deployment_record/cost/coefficients.py`

| symbol | in-repo value / label | verdict |
| --- | --- | --- |
| `_REFERENCE_CORE_NEURONS_LARGE = 1024  # Loihi core (Davies 2018)` | 1024 | **AGREES** |
| `E_SYNC_BARRIER_MJ` — "~0.1 / ~1 / ~10 uJ per barrier (**Loihi-style NoC flush**)" | 0.1 / 1 / 10 µJ | **UNSOURCED attribution** — see F9 |
| `_CORE_INIT_BASIS` — per-core reset built from `soma_access + soma_update` over 256/1024 neurons | derived from the preset | inherits the §6b misattribution; the note already says "NEW - needs owner sign-off" |
| `SYNC_BARRIER_S` — built from `TRUENORTH_PRESET["tile_hop_latency_s"]` | — | not a Loihi number; out of scope here, but note that the Loihi barrier time **is** published (113–465 ns, §1b) and could ground it |

**Finding F9.** No barrier **energy** has ever been published for Loihi (U2).
Davies 2018 publishes barrier *time* only. The band's own docstring says it is
"imported from `weight_reuse_cost_model.DEFAULT_COEFFICIENT_BAND`", whose sibling
coefficients are Horowitz-45 nm / HBM2 / DDR3 figures — i.e. generic DRAM
energies with no Loihi content. Calling it a "Loihi-style NoC flush" attaches a
Loihi provenance to a number that has none.

Scale check on why the label is doubly misleading: a chip-wide, all-inactive
timestep costs 131,072 × 52 pJ = **6.82 µJ** (§3.3). A "1 µJ barrier" is
therefore ~15% of an entire idle chip timestep, and the "10 µJ" high corner
exceeds a fully active timestep (10.6 µJ). Meanwhile the only measured
whole-chip re-initialization energy we could find is 5.94 µJ/chip (§3.3), which
is a *reset*, not a barrier. Either relabel the band as generic, or replace it
with a stated Loihi derivation.

### 6e. `src/mimarsinan/chip_simulation/sanafe/arch_synth/floorplan.py`

`PRESET_CORES_PER_TILE = {"loihi": 4, ...}` — consistent with §3.5's inference
(128 cores / 32 tiles). The file correctly documents its source as the yaml's
physical wiring rather than claiming a paper citation. **No defect.**

### 6f. Cross-check summary

| # | severity | finding |
| --- | --- | --- |
| F1 | high | `sana_fe/arch/loihi.yaml` header falsely attributes ~18 of ~22 values to Davies 2018 SPICE |
| F2 | — | true provenance is `boyle2023sanafe` Nahuku silicon calibration (11.7% energy / 24.3% latency error envelope) |
| F3 | high | `presets.py` `LOIHI_PRESET` repeats the same misattribution |
| F4 | medium | `tile_hop_energy_j=3.5e-12` is an undocumented average, inconsistent with the yaml it mirrors |
| F5 | medium | `tile_hop_latency_s=5.0e-9` matches no source at all |
| F6 | medium | preset discards the published 1.33×/1.59× NoC directional anisotropy |
| F7 | **high** | `loihi_dense_equiv_128x1024.max_axons=128` matches no Loihi constant (paper: 4096 / 4096) |
| F8 | medium | `has_bias=False` contradicts Davies 2018 Eq. (1) |
| F9 | medium | `E_SYNC_BARRIER_MJ` "Loihi-style" label on a Horowitz/DRAM-derived band; no Loihi barrier energy exists |
| — | none | `floorplan.py`, `_REFERENCE_CORE_NEURONS_LARGE`, `max_neurons_supported` all correct |

---

## 7. Citations (not yet injected)

Injection into a real `.bib` is **deferred**: the only `.bib` in the tree is
`sana_fe/references.bib`, which belongs to the vendored SANA-FE subtree and must
stay untouched. The block below was produced by the `asta-papers`
`inject_papers_to_bib` tool into a scratchpad file and is transcribed here
verbatim — it was **not** hand-written. When a first-party `.bib` exists, re-run
injection with these ids rather than copying this text.

Cached ids: `davies2018loihi`, `davies2021advancing`, `frady2020neuromorphic`,
`boyle2023sanafe`, `dey2021mapping`, `blouw2018benchmarking`.

```bibtex
@Article{davies2018loihi,
 author = {Mike Davies and N. Srinivasa and Tsung-Han Lin and Gautham N. Chinya and Yongqiang Cao and S. H. Choday and Georgios Dimou and Prasad Joshi and Nabil Imam and Shweta Jain and Yuyun Liao and Chit-Kwan Lin and Andrew Lines and Ruokun Liu and D. Mathaikutty and Steve McCoy and Arnab Paul and Jonathan Tse and Guruguhanathan Venkataramanan and Y. Weng and Andreas Wild and Yoonseok Yang and Hong Wang},
 booktitle = {IEEE Micro},
 journal = {IEEE Micro},
 pages = {82-99},
 title = {Loihi: A Neuromorphic Manycore Processor with On-Chip Learning},
 volume = {38},
 year = {2018}
}

@Article{davies2021advancing,
 author = {Mike Davies and Andreas Wild and G. Orchard and Yulia Sandamirskaya and Gabriel Andres Fonseca Guerra and Prasad Joshi and Philipp Plank and Sumedh R. Risbud},
 booktitle = {Proceedings of the IEEE},
 journal = {Proceedings of the IEEE},
 pages = {911-934},
 title = {Advancing Neuromorphic Computing With Loihi: A Survey of Results and Outlook},
 volume = {109},
 year = {2021},
 doi = {10.1109/JPROC.2021.3067593},
 url = {https://doi.org/10.1109/JPROC.2021.3067593},
}

@Article{frady2020neuromorphic,
 author = {E. P. Frady and G. Orchard and David Florey and Nabil Imam and Ruokun Liu and Joyesh Mishra and Jonathan Tse and Andreas Wild and F. Sommer and Mike Davies},
 booktitle = {Neuro Inspired Computational Elements Workshop},
 journal = {Proceedings of the 2020 Annual Neuro-Inspired Computational Elements Workshop},
 title = {Neuromorphic Nearest Neighbor Search Using Intel's Pohoiki Springs},
 year = {2020},
 doi = {10.1145/3381755.3398695},
 url = {https://doi.org/10.1145/3381755.3398695},
}

@Article{boyle2023sanafe,
 author = {James A. Boyle and Mark Plagge and S. Cardwell and Frances S. Chance and A. Gerstlauer},
 booktitle = {IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems},
 journal = {IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems},
 pages = {3165-3178},
 title = {SANA-FE: Simulating Advanced Neuromorphic Architectures for Fast Exploration},
 volume = {44},
 year = {2023},
 doi = {10.1109/TCAD.2025.3537971},
 url = {https://doi.org/10.1109/TCAD.2025.3537971},
}

@Article{dey2021mapping,
 author = {Srijanie Dey and A. Dimitrov},
 booktitle = {Frontiers in Neuroscience},
 journal = {Frontiers in Neuroscience},
 title = {Mapping and Validating a Point Neuron Model on Intel's Neuromorphic Hardware Loihi},
 volume = {16},
 year = {2021},
 doi = {10.3389/fnins.2022.883360},
 url = {https://doi.org/10.3389/fnins.2022.883360},
}

@Article{blouw2018benchmarking,
 author = {Peter Blouw and Xuan Choo and Eric Hunsberger and C. Eliasmith},
 booktitle = {Neuro Inspired Computational Elements Workshop},
 journal = {ArXiv},
 title = {Benchmarking Keyword Spotting Efficiency on Neuromorphic Hardware},
 volume = {abs/1812.01739},
 year = {2018},
 doi = {10.1145/3320288.3320304},
 url = {https://doi.org/10.1145/3320288.3320304},
}
```

Note on `dey2021mapping`: the cached entry says *Front. Neurosci.* vol. 16, 2021
with DOI `10.3389/fnins.2022.883360`; the article as published is dated 2022 and
carries a 2022 corrigendum (`10.3389/fninf.2022.1023486`). Verify the year/venue
against the publisher record before it goes into a bibliography.

---

## 8. Mapping onto the `platform_physics` vocabulary

`src/mimarsinan/deployment_record/platform_physics/vocabulary.py` defines the
closed set of declarable constants. Below is what this research can and cannot
fill for a Loihi-1 profile. `evidence_kind` uses that module's enum
(`published | datasheet | derived | estimated`); `measurement` uses its
`MEASUREMENT_KINDS` (`silicon | simulation | projection | mixed`).

**Enum gap to resolve first.** Many Loihi facts are *architectural* constants
(1024 neurons/core, 9-bit weights, 8 B/synapse-word) — true by design, never
measured. `MEASUREMENT_KINDS` has no value for these; `simulation` and `silicon`
both misdescribe them. Either add a kind, or restrict such facts to the
constraints surface rather than the physics profile.

| vocabulary key | Loihi-1 value | band (low/nom/high) | evidence_kind | measurement | source |
| --- | --- | --- | --- | --- | --- |
| `e_mac` (per synaptic op) | 23.6 pJ | 23.6 / — / — (published **min** only) | published | simulation | `davies2018loihi` T2 |
| `t_array_read` (per synaptic op) | 3.5 ns | — / — / 3.5 (published **max** only) | published | simulation | `davies2018loihi` T2 |
| `e_neuron_update` | 81 pJ (active) | 52 / — / 81 (inactive→active) | published | simulation | `davies2018loihi` T2 |
| `e_leak_per_neuron_step` | 52 pJ | — | published | simulation | `davies2018loihi` T2 ("inactive" neuron update) |
| `membrane_bits` | 23 (signed; ±2²³) | 8 / 23 / 24 across state vars | published | silicon | `dey2021mapping` §3.1.2 |
| `e_intra_tile_packet` | 1.7 pJ | — | published | simulation | `davies2018loihi` T2 |
| `e_inter_tile_hop` | 3.0 (E-W) / 4.0 (N-S) pJ | 3.0 / 3.5 / 4.0 | published | simulation | `davies2018loihi` T2 — **band the anisotropy, do not average it away (F6)** |
| `t_hop` | 4.1 (E-W) / 6.5 (N-S) ns | 4.1 / 5.3 / 6.5 | published | simulation | `davies2018loihi` T2 — same anisotropy note |
| `t_sync_barrier` | 113 → 465 ns | 113 / — / 465 (1→32 tiles, single chip) | published | simulation | `davies2018loihi` T2; multi-chip is 14.3–16.2 µs (`frady2020neuromorphic` §4.3) — a *different* scale, do not conflate |
| `p_static_global` (per chip) | 104 mW | 104.3 / 104.4 / — | derived (§3.2) | silicon | `frady2020neuromorphic` T1 — **voltage unpinned (U5)** |
| `p_static_per_core` | 816 µW | — / 816 / 816 (**upper bound**) | derived (§3.2) | silicon | `frady2020neuromorphic` §5.2 — over-attributes the x86/IO share |
| `e_core_init` | 46.4 nJ/core | — | derived (§3.3) | silicon | `frady2020neuromorphic` T2 — this is a *network reset*, not weight programming |
| `bytes_per_connectivity_entry` | 8 B (= 64 b/synaptic word) | — | published | — (architectural) | `davies2018loihi` p. 89 constraint 2, "N_syn × 64b" |
| `area_global_fixed` | 60 mm² (whole die) | — | published | silicon | `davies2018loihi` p. 95 |
| `conductance_levels` | 512 (= 2⁹, max weight precision) | 2 / 256 / 512 (1b / 8b / 9b) | derived | — (architectural) | `davies2018loihi` p. 87 "between one and nine bits" |
| `t_cycle` | **MUST REMAIN UNSET** | — | — | — | Loihi has no clock (U11). Setting this to any number is a category error. |
| `e_sync_barrier` | **UNSOURCED** (U2) | — | — | — | no Loihi barrier energy has ever been published |
| `e_dma_per_byte`, `e_core_program`, `t_program_per_byte` | **UNSOURCED** (U1) | — | — | — | only a system-level programming *time* exists, and it is I/O-bound, not a hardware constant (see note) |
| `t_core_init` | **UNSOURCED** as a hardware constant | — | — | — | see note below |
| `area_per_neuron_logic`, `area_per_state_bit`, `area_per_router`, `area_per_tile_fixed` | **UNSOURCED** (U3) | — | — | — | only whole-die area is published |
| `area_per_cell`, `area_per_cell_per_weight_bit`, `write_sigma`, `read_sigma` | **N/A** | — | — | — | Loihi is digital SRAM, not an analog crossbar — these ARRAY-group cells have no Loihi meaning |
| `area_per_adc`, `adc_sharing_factor`, `e_adc_conversion`, `t_adc_conversion`, `area_per_row_driver`, `e_row_drive` | **N/A** | — | — | — | no ADCs / analog row drivers in Loihi |
| `p_host`, `host_compute_rate` | **UNSOURCED** (U8) | — | — | — | only aggregate x86 power (~65 mW/chip) exists, and it is workload-specific |

**Note on `t_core_init` / programming time.** `frady2020neuromorphic` §5.3 gives
"about 4.6 seconds per Loihi chip" (⇒ 36 ms/core), but the same sentence
disqualifies it as a hardware constant:

> "This is a very slow step due to the current unoptimized state of the Pohoiki
> Springs I/O subsystem. The programming time for a 192-chip column was measured
> to be 893 seconds, or about 4.6 seconds per Loihi chip."

Use it only as a loose upper bound with that caveat attached, or leave unset.

**Band semantics warning.** Several Loihi numbers are published as a *qualified
extremum*, not a nominal: `e_mac` is a **min**, `t_array_read` is a **max**. A
band that centres a nominal on them will silently invent an unpublished value.
Prefer one-sided bands, or take the SANA-FE measured spread (§6a: 24.0–35.5 pJ
per synaptic op by format) as the empirical high side with `boyle2023sanafe`
cited and its 11.7%/24.3% error envelope recorded.
