# Silicon correlation — reproducing published chip measurements

Research note for the **silicon-correlation** contract: given a target's declared
physics and a published workload's census, does the pricer reproduce that chip's own
published measurement? Every number below is transcribed from a published source or
derived from published numbers with the arithmetic shown.

Acceptance: every shipped reference case lands inside **(−25 %, +25 %)** of the
published value.

## 0. The defect this research found

Three chips, three papers, one shared mistake in our profiles.

A chip paper publishes two very different things that both look like "energy per
synaptic event":

- an **incremental** (marginal) energy — what one *more* event costs, and
- an **operating-point average** — measured whole-chip power divided by the event rate
  at one specific firing rate.

ODIN's paper states the distinction outright, and gives both as separate rows of its
specification table (Fig. 8b) with its own power model:

> P = P_leak + P_idle × f_clk + E_SOP × r_SOP  … (2)
> E_tot,SOP = P / r_SOP  … (3)

with `E_SOP = 8.43 pJ` (incremental) against `E_tot,SOP > 12.7 pJ` (global). The paper
is explicit that (3) is operating-point dependent:

> "the whole chip power consumption P is divided by the SOP rate r_SOP, without
> subtracting contributions from leakage and idle power"

**Our profiles filed the aggregate as if it were the constant.** `e_synaptic_event_total`
SUPERSEDES static power by design — correct for a genuine aggregate, but it turns the
model into a *point* model: exact at the operating point the number was measured at,
and wrong everywhere else. Measured against each chip's own published sweep:

| target | model | at the low-activity point | at the high-activity point |
| --- | --- | --- | --- |
| TrueNorth | aggregate 26 pJ/event | **−40.6 %** (11.58 Hz) | **+256.1 %** (95.93 Hz) |
| ODIN | aggregate 12.7 pJ/SOP | **−76.4 %** (biological time) | −0.2 % (its own calibration point) |

Both chips publish enough to build the affine model instead, and it holds:

| target | model | held-out point | independent point |
| --- | --- | --- | --- |
| TrueNorth | static + marginal | **−0.54 %** | −1.95 % / +8.61 % (other paper) |
| ODIN | published Eq. (2) | **−0.02 % / +0.27 %** across a 13.6× power range | — |

## 1. TrueNorth — the marginal energy is derivable from a published sweep

`akopyan2015truenorth` Fig. 17 (p. 1551) publishes three measured total powers at one
supply voltage and one synaptic density, differing only in firing rate:

> "Total TrueNorth chip power breakdown (@ 0.8 V) for three complex recurrent networks
> with 128 synapses per neuron average, and three different average firing rates."
> — bars printed as **68 mW @ 11.58 Hz, 71 mW @ 20.07 Hz, 94 mW @ 95.93 Hz**

The synaptic event rate is fully determined by the chip's own structure:

```
events/s = 1,048,576 neurons x rate(Hz) x 128 active synapses
```

Fitting `P = P_static + E_event x events/s` on the two **endpoints only** and holding
the middle point out:

```
E_event  = (94 - 68) mW / (1.28742e10 - 1.5541e9) events/s = 2.297 pJ/event
P_static = 68 mW - 2.297 pJ x 1.5541e9 = 64.43 mW      (both at 0.8 V)
```

| point | predicted | published | error |
| --- | --- | --- | --- |
| 11.58 Hz | 68.00 mW | 68 mW | fitted endpoint |
| **20.07 Hz** | **70.62 mW** | **71 mW** | **−0.54 % (HELD OUT)** |
| 95.93 Hz | 94.00 mW | 94 mW | fitted endpoint |
| 20 Hz @ 0.775 V (`merolla2014a`, different paper) | 70.60 mW | 72 mW | −1.95 % |
| 20 Hz @ 0.75 V (`akopyan2015truenorth` §VII) | 70.60 mW | 65 mW | +8.61 % |

Two cross-paper points at other voltages land within 9 % of a fit that never saw them.

**Cross-check that the marginal number is physical.** 2.297 pJ/event is the same order
as the published per-hop spike energy (2.3 pJ, `merolla2014a` Fig. S4), which is what
should dominate the marginal cost: the characterization networks were built to route
spikes far — "21.3 cores away on average in both x and y dimensions" (`merolla2014a`
§S7) — so each spike pays ~21 hops, amortized over its 128 synaptic events.

**Cross-check that the static number is physical.** The independently published
zero-activity corner is 42 mW at 0.70 V (`akopyan2015truenorth` §VII). Leakage scales
super-linearly with supply; 42 mW × (0.8/0.7)² = 54.9 mW and × (0.8/0.7)³ = 62.7 mW
bracket the fitted 64.4 mW at 0.8 V. The fit is consistent with the separately measured
corner without having used it.

**Consequence for the profile.** The declared corner moves to **0.8 V**, where the sweep
was measured. `e_synaptic_event_total` (26 pJ, an average at 0.775 V / 20 Hz) is no
longer declared as a priced constant — it is recorded in the profile description as the
cross-check it is. Its supersession of static power is exactly what made the point model
un-generalizable.

## 2. Loihi — the per-op constants correlate against an independent silicon measurement

Loihi's per-event energies are pre-silicon (`davies2018loihi` Table 2 is titled
"Loihi **pre-silicon** performance and energy measurements"). The independent test is
`frady2020neuromorphic`, which measures a real 32-chip Pohoiki Springs board.

**Workload, entirely from the paper's own structural statements:**

| quantity | value | source |
| --- | --- | --- |
| database patterns M (one neuron each) | 76,800 | Tables 1–2 row label |
| reduced dimensionality N_C | 500 | §3 "500 kept for dimensionality reduction, down from 3072" |
| input components | 2 N_C = 1000 | §3 "W = [E, −E] ∈ R^{2N_C × M_s}" |
| spikes actually transmitted | 667 – 750 | §3 "The used setting θ_e = 0.1 typically removes about one quarter to one third of spikes." |
| fan-out of one input spike | all M neurons | §3 "Each input spike is broadcast to all pattern match neurons" |
| window | T = 60 timesteps | §3 "the window length is T = 60 timesteps" |

```
synaptic_events/query = spikes x M            = 5.12e7 .. 5.76e7
neuron updates/query  = M x T                 = 4.608e6
```

**Published measurement** (`frady2020neuromorphic` Table 2, "Energy breakdown per query
(mJ)", row k = 1 / 76,800): **Neuro = 1.33 mJ** — the dynamic energy of the
neuromorphic cores, separate from the Static, Reset and x86 columns.

| term | predicted | source constant |
| --- | --- | --- |
| synaptic | 1.208 – 1.359 mJ | 23.6 pJ (`davies2018loihi` T2) |
| neuron updates | 0.240 mJ | 52 pJ inactive (`davies2018loihi` T2) |
| **total** | **1.448 – 1.599 mJ** | vs measured **1.33 mJ** → **+8.9 % … +20.2 %** |

Inside ±25 %, with no fitted parameter: pre-silicon per-op numbers from one paper
predicting a measured board from another. The residual sign is expected — Davies
publishes 23.6 pJ as a **minimum**, so a prediction built on it should land high.

**Two defects this exposed in `loihi.json`:**

1. `e_synaptic_event_total = 23.6 pJ` — Davies' number is a **per-operation** energy,
   not a whole-chip aggregate, so filing it in the AGGREGATE group made it suppress the
   neuron-update and static terms that Frady measures separately. It becomes `e_mac`.
2. `e_neuron_update` (band 52–81 pJ) **and** `e_leak_per_neuron_step` (52 pJ) were both
   declared, and the pricer sums both over `neurons_used × timesteps`. Davies publishes
   these as *one* quantity at two activity levels — "Energy per neuron update (active /
   inactive) | 81 pJ / 52 pJ" — so declaring both double-charges every neuron. Only
   `e_neuron_update`, carrying the published 52–81 band, survives.

Defect 2 was latent while the aggregate suppressed both terms; it would have fired the
moment defect 1 was fixed.

**A third-party calibration we did NOT adopt, and why.** The vendored
`sana_fe/arch/loihi.yaml` uses 35.5 pJ per dense synaptic operation — a value
`boyle2025sanafe` obtained by regression against real Nahuku silicon, not from Davies.
Priced against Frady's query it gives +55 % … +71 %, outside the band, where Davies'
23.6 pJ gives +8.9 % … +20.2 %. The correlation decided between two defensible sources;
the losing one is recorded here rather than deleted.

## 3. ODIN — a published power model, validated on two points 13.6× apart

`frenkel2019odin` Fig. 8b publishes each term of Eq. (2) separately at 0.55 V:
`P_leak = 27.3 µW`, `P_idle = 1.78 µW/MHz`, `E_SOP = 8.43 pJ`. The paper then reports
two *measured* operating points, each with a fully determined SOP rate:

| point | f_clk | r_SOP | published P | source |
| --- | --- | --- | --- | --- |
| accelerated | 75 MHz | 37.5 MSOP/s (= f_clk/2) | **477 µW** | §IV-A "the measured power consumption P of ODIN is 477µW at 0.55V" |
| biological | 1.3 MHz | 650 kSOP/s (256 neurons @ 10 Hz × 256 SOPs) | **35 µW** | §IV-A "the measured power consumption P of ODIN is 35µW at 0.55V" |

| point | predicted | published | error |
| --- | --- | --- | --- |
| accelerated | 476.93 µW | 477 µW | **−0.02 %** |
| biological | 35.09 µW | 35 µW | **+0.27 %** |

The event census needs no activity assumption: ODIN's architecture fixes it —
"each neuron spike event thus leads to N SOPs" with N = 256, and "the maximum SOP rate
ODIN can handle is equal to f_clk/2 as each SOP takes two clock cycles".

**Area is decomposable and independently checkable.** The paper publishes 0.68 µm² per
4-bit synapse and a 320 µm × 270 µm core:

```
65,536 synapses x 0.68 um^2 = 44,564 um^2 = 51.6 % of the 86,400 um^2 core
```

so the synapse array is about half the core, leaving the neurons, the SDSP update logic,
the scheduler and the controller — a plausible split, and the only one of the three
targets where a published per-cell area can be summed toward a published total.

**The clock-proportional idle term.** `P_idle × f_clk` is a real power component our
vocabulary has no dedicated slot for. It is declared through `p_static_per_core` at the
profile's declared clock, with the arithmetic in the constant's derivation; a reference
case at a different clock states its own override and the report prints it. ODIN's
profile therefore declares its operating point (0.55 V, 75 MHz, 24 °C) in `validity`,
which is what a Liberty corner does.

## 4. What is deliberately NOT shipped as a reference case

| candidate | why not |
| --- | --- |
| TrueNorth multi-object detection (63 mW, `merolla2014a` §S11) | Supply voltage is not stated and the firing rate is not published — the census cannot be built without inventing an activity factor. |
| TrueNorth CNN benchmarks (`esser2016convolutional` Table 3, 8 datasets) | Publishes cores/FPS/mW but no spike or synaptic-event census; the per-frame activity of a binary-activation CNN is not derivable from the paper. |
| Loihi keyword spotting (`blouw2018benchmarking` Table 1) | Network topology is fully published (390-256-256-29) but the spike activity and timesteps per inference are not, so the event census is underdetermined. |
| ODIN MNIST (15 nJ/inference) | Rank-order coding terminates on the first output spike; the paper does not state how many input events that took, so the SOP count per inference is not derivable. |
| TrueNorth 323 mW high corner | Measured at 1.05 V, outside the profile's declared 0.8 V corner; no published voltage-scaling model. |

These are the same discipline as the PRIME refusal: a case that can only be built by
fitting an unpublished parameter is not evidence, and would turn the correlation suite
into a tautology.

## 5. Sources

| key | reference |
| --- | --- |
| `merolla2014a` | P. Merolla et al., "A million spiking-neuron integrated circuit with a scalable communication network and interface", *Science* 345(6197):668–673, 2014 (+ supplementary) |
| `akopyan2015truenorth` | F. Akopyan et al., "TrueNorth: Design and Tool Flow of a 65 mW 1 Million Neuron Programmable Neurosynaptic Chip", *IEEE TCAD* 34(10):1537–1557, 2015 |
| `davies2018loihi` | M. Davies et al., "Loihi: A Neuromorphic Manycore Processor with On-Chip Learning", *IEEE Micro* 38(1):82–99, 2018 |
| `frady2020neuromorphic` | E. P. Frady et al., "Neuromorphic Nearest Neighbor Search Using Intel's Pohoiki Springs", NICE 2020 (arXiv:2004.12691) |
| `frenkel2019odin` | C. Frenkel et al., "A 0.086-mm² 12.7-pJ/SOP 64k-Synapse 256-Neuron Online-Learning Digital Spiking Neuromorphic Processor in 28-nm CMOS", *IEEE TBioCAS* 13(1):145–158, 2019 (arXiv:1804.07858) |
| `boyle2025sanafe` | J. A. Boyle et al., "SANA-FE: Simulating Advanced Neuromorphic Architectures for Fast Exploration", *IEEE TCAD* 44(8):3165–3178, 2025 |
| `esser2016convolutional` | S. K. Esser et al., "Convolutional networks for fast, energy-efficient neuromorphic computing", *PNAS* 113(41):11441–11446, 2016 |
| `blouw2018benchmarking` | P. Blouw et al., "Benchmarking Keyword Spotting Efficiency on Neuromorphic Hardware", NICE 2019 (arXiv:1812.01739) |
