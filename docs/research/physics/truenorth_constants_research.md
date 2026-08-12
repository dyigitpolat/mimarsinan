# TrueNorth physics profile — published per-unit constants

Research note. **No code was changed.** Every number below is either quoted verbatim from a
primary source or derived from quoted numbers with the arithmetic shown inline. Anything that
could not be sourced is listed in §2 as UNSOURCED rather than guessed.

## 0. Sources read

| key | source | type | how read |
|---|---|---|---|
| `merolla2014a` | P. Merolla et al., "A million spiking-neuron integrated circuit with a scalable communication network and interface", *Science* **345**(6197):668–673, 2014 — **including Supplementary Material §S1–S13, Figs. S1–S8, Tables S1–S2** | peer-reviewed paper (silicon measurements) | full PDF, 40 pp (author manuscript + SM) |
| `akopyan2015truenorth` | F. Akopyan et al., "TrueNorth: Design and Tool Flow of a 65 mW 1 Million Neuron Programmable Neurosynaptic Chip", *IEEE TCAD* **34**(10):1537–1557, 2015 | peer-reviewed paper (design + silicon measurements) | full PDF, 21 pp |

Both were located with `semantic_scholar_search` and fetched with `download_paper_pdfs`
(open-access PDFs). Page numbers below are the **journal** page numbers printed on the page.
No third-party/secondary source was needed — every quantity in §1 comes from one of these two.

**There is no TrueNorth datasheet in the public record**, so `evidence_kind = datasheet` is
used nowhere. The available kinds are `published` (verbatim in a primary source),
`derived` (arithmetic over published values, shown), and `estimated` (derived through a
modelling assumption that the source does not itself make).

---

## 1. Constants table

### 1.1 ENERGY

| quantity | value | band (low–high) | unit | evidence_kind | citation | quote |
|---|---|---|---|---|---|---|
| Energy per synaptic event (total energy: active + passive) | **26** | — | pJ/synaptic-event | published | `merolla2014a`, main text p.7 (also Fig. 4C caption, and §S12) | "At the operating point where neurons fire on average at 20Hz and have 128 active synapses, the total measured power was 72mW (at 0.775V operating voltage), corresponding to 26pJ per synaptic event (considering total energy)." |
| Energy per synaptic event (restated) | **26** | — | pJ/synaptic-event | published | `merolla2014a`, Fig. 4C caption | "For a typical network where neurons fire on average at 20Hz and have 128 active synapses (marked as ∗ on panels B and C), the total energy is 26pJ per synaptic event." |
| Energy per spike per **inter-core hop** | **2.3** | — | pJ/spike/hop | published | `merolla2014a`, Fig. S4 caption (SM p.17) | "For a power supply of 0.775V, we calculate that communicating a bit consumes 0.3fJ per µm with an intercept of 2pJ/bit. The distance between cores in x direction is 240µm, which corresponds to one hop in the router. Therefore, the energy to send a spike one hop between cores is 2.3pJ." |
| Energy per **bit** per inter-core hop | **72** | — | fJ/bit/hop | published | `merolla2014a`, §S5 (SM p.5) | "The energy to transport a bit between two adjacent cores (along x dimension) is 72fJ at 0.775V (Fig. S4)." |
| On-chip wire energy coefficient | **0.3** | — | fJ/bit/µm | published | `merolla2014a`, Fig. S4 caption | "we calculate that communicating a bit consumes 0.3fJ per µm with an intercept of 2pJ/bit" |
| Energy per bit, core → external periphery | **2** | — | pJ/bit | published | `merolla2014a`, §S5 | "The energy to transport a bit from the internal to the external periphery is 2pJ at 0.775V (Fig. S4)." |
| Energy per bit, chip → adjacent chip | **26** | — | pJ/bit @ 1.8 V | published | `merolla2014a`, §S5 | "The energy to send a bit between external peripheries of two adjacent chips is 26pJ at 1.8V." |
| Energy to move a bit, core SRAM → Controller (**intra-core**) | **≈47** | — | fJ/bit | published | `merolla2014a`, §S5 | "Within a core, the average energy required to move a bit from local memory (SRAM) to the Controller was ∼47fJ (at 1kHz and 0.775V, reading all ∼428 million bits was ∼20mW)." |
| Whole-chip memory-read power (all 428 Mb, every tick) | **≈20** | — | mW @ 0.775 V, 1 kHz | published | `merolla2014a`, §S5 | (same sentence as above) |
| Energy per spike per hop, **derived from the bit cost** | 2.304 | — | pJ/spike/hop | derived | arithmetic over `merolla2014a` §S5 + Fig. S2 caption | 72 fJ/bit × 32 bits/packet = **2304 fJ = 2.304 pJ** — reproduces the paper's own 2.3 pJ, confirming the packet is charged as 32 bits/hop |
| Energy per spike, core → external periphery | 64 | — | pJ/spike | derived | 2 pJ/bit (`merolla2014a` §S5) × 32 bits/packet (Fig. S4 caption) | 2 pJ × 32 = **64 pJ** |
| Energy per spike, chip → adjacent chip | 832 | — | pJ/spike @ 1.8 V | derived | 26 pJ/bit (`merolla2014a` §S5) × 32 bits | 26 pJ × 32 = **832 pJ** |
| Energy to read one neuron's SRAM row | 19.3 | — | pJ/neuron/tick | derived | 47 fJ/bit (`merolla2014a` §S5) × 410 bits/row (`akopyan2015truenorth` p.1546 table) | 47 fJ × 410 = **19.27 pJ**. Self-consistency check: 428×10⁶ bits × 47 fJ × 1000 ticks/s = **20.1 mW**, matching the paper's own "∼20mW" |
| Energy per **neuron update per tick** (whole-chip aggregate, incl. synapses+comms+leak) | 72 | 42 – 72 | pJ/neuron/tick | derived | `merolla2014a` p.7 (72 mW), `akopyan2015truenorth` p.1551 (65 mW, 42 mW) | 72 mW × 1 ms ÷ 10⁶ neurons = **72 pJ/neuron/tick** @0.775 V/20 Hz/128 syn. TCAD's 0.75 V point: 65 mW × 1 ms ÷ 10⁶ = **65 pJ**. Zero-activity floor: 42 mW × 1 ms ÷ 10⁶ = **42 pJ**. ⚠ This is a *system aggregate*, **not** an isolated neuron-update circuit cost — see §2. |
| Energy per **core** per tick (aggregate) | 17.6 | — | nJ/core/tick | derived | as above | 72 mW × 1 ms ÷ 4096 cores = **17.578 nJ** |
| **Static / leakage power, chip-wide** | 42 | — | mW @ 0.70 V | derived (from a published zero-activity corner) | `akopyan2015truenorth`, §VII, p.1551 | "total power consumption ranging from 42 mW in the low corner (0.70 V, 0 Hz firing rate, 0 synapses/neuron)". At 0 Hz and 0 synapses there is no spiking or synaptic activity, so 42 mW bounds static + tick-baseline power at 0.70 V. It is **not** labelled "leakage" by the authors. |
| Static / leakage power, **per core** | 10.25 | — | µW/core @ 0.70 V | derived | as above | 42 mW ÷ 4096 = **10.254 µW** |
| Total power, **full operating envelope** | — | 42 – 323 | mW | published | `akopyan2015truenorth`, §VII, p.1551 | "Overall, the TrueNorth chip is operational from 1.05 V down to 0.7 V, with total power consumption ranging from 42 mW in the low corner (0.70 V, 0 Hz firing rate, 0 synapses/neuron) to 323 mW in the high corner (1.05 V, 200 Hz firing rate, 256 synapses/neuron)." |
| Total power at the **headline benchmark** (recurrent net) | **65** | — | mW @ 0.75 V | published | `akopyan2015truenorth`, §VII, p.1551 | "While running a typical complex recurrent neural network at 0.75 V with 20 Hz average firing rate and 128 active synapses per neuron at real-time (1 kHz tick), the TrueNorth chip consumes only 65 mW and delivers 46GSOPS/W." |
| Total power at the **same benchmark, Science paper** | **72** | — | mW @ 0.775 V | published | `merolla2014a`, p.7 + Table S1 | "the total measured power was 72mW (at 0.775V operating voltage)"; Table S1 "Probabilistic Network … Power(W) 0.072". ⚠ Different voltage from the TCAD 65 mW point — see §3. |
| Total power, **multi-object detection & classification** | **63** | — | mW | published | `merolla2014a`, §S11 (SM p.9) + abstract | "We ran the multi-object detection and classification network on TrueNorth with a 400 × 240 pixel video stream (Fig. 3) in real-time—30 frames per second, where one frame used 33 time steps, 1ms each. The network used 3,406 of 4,096 cores, and we measured its total power consumption at 63mW during operation." ⚠ **supply voltage not stated for this measurement** |
| Total power, **visual filter** | 60 | — | mW | published | `merolla2014a`, Table S1 (SM p.21) | Table S1 row "TrueNorth / Power(W)": "Visual Filter … 0.060" |
| Total power, complex recurrent nets @ 0.8 V | — | 68 / 71 / 94 | mW | published | `akopyan2015truenorth`, Fig. 17, p.1551 | "Total TrueNorth chip power breakdown (@ 0.8 V) for three complex recurrent networks with 128 synapses per neuron average, and three different average firing rates." Bars printed as **68mW @ 11.58Hz, 71mW @ 20.07Hz, 94mW @ 95.93Hz** |
| Measured total power, real applications (1-chip) | — | 49 / 58 / 64 | mW @ 0.75 V | published | `akopyan2015truenorth`, Table I, p.1553 | Table I "Total Power (Watts)" column: Saccade generator **0.049W @0.75V**; Haar-like features **0.058W @0.75V**; Local Binary Patterns **0.064W @0.75V** |
| Measured total power, real applications (multi-chip) | — | 0.653 – 2.515 | W @ 1.0 V | published | `akopyan2015truenorth`, Table I, p.1553 | K-means classifier (4 chips) **0.653W @1.0V**; Grid classifier A (7 chips) **1.274W @1.0V**; Grid classifier B (16 chips) **2.515W @1.0V** |
| Active **communication** power (component), applications | — | 0.11 – 130.88 | mW | published | `akopyan2015truenorth`, Table I, p.1553 | Table I "Communication Power (Active)" / "CPLACE" column, e.g. Saccade generator **0.11mW**, Grid classifier A **130.88mW** (Default→CPLACE improvement 1.9×–5.1×) |
| Power density | 20 | — | mW/cm² | published | `merolla2014a`, p.6 | "TrueNorth's power density is 20mW per cm² while that of a typical CPU is 50 − 100W per cm²" |
| Computational efficiency, typical net, real-time | 46 | — | GSOPS/W | published | `merolla2014a` §S8; `akopyan2015truenorth` p.1551 | "TrueNorth delivers 46 billion SOPS per Watt at real-time (with 1ms time steps)". ⚠ inconsistent with 26 pJ/event — see §3. |
| Computational efficiency, 5× real-time | 70 | — | GSOPS/W | published | `merolla2014a`, §S8 | "…and 70 billion SOPS per Watt at 5× faster than real-time (with 200µs time steps)." |
| Computational efficiency, **peak** | 400 | — | GSOPS/W @ 0.75 V | published | `akopyan2015truenorth`, §VII, p.1551 | "At this supply voltage, the maximum computational speed of the chip is 58 GSOPS and the maximum computational energy efficiency is 400 GSOPS/W." Footnote 2: "Measured using a recurrent network with neuron firing rates in the range of 0–200 Hz and synaptic connectivity varying between 0–100%." |
| Peak computational speed | 58 | — | GSOPS @ 0.75 V | published | `akopyan2015truenorth`, §VII, p.1551 | (same sentence) |
| Power at the peak-efficiency point | 145 | — | mW | derived | `akopyan2015truenorth` §VII | 58 GSOPS ÷ 400 GSOPS/W = **0.145 W**. Cross-checks against `merolla2014a` Fig. 4B: "power remains low (< 150mW) for all benchmarks networks" ✓ |

### 1.2 TIME

| quantity | value | band | unit | evidence_kind | citation | quote |
|---|---|---|---|---|---|---|
| **Tick period** (nominal) | **1** | — | ms | published | `merolla2014a` p.4; `akopyan2015truenorth` §III p.1540 | "Neuron dynamics is discretized into 1ms time steps set by a global 1kHz clock."; "All the computation must finish in the current tick, which spans 1 ms." |
| Global synchronization frequency | **1** | — | kHz | published | `akopyan2015truenorth`, Fig. 1 / §III design principle 4, p.1538 | "We define real-time as evaluating every neuron once each millisecond, delineated by a 1 kHz synchronization signal." |
| Tick period, **demonstrated fastest** | 47.6 | 47.6 µs – 1 ms | µs | derived | `akopyan2015truenorth`, §VII, p.1551 | "In our experiments we measured up to 21× real-time operation, dependent on the activity rates, synaptic density, and voltage levels." → 1 ms ÷ 21 = **47.6 µs** |
| Tick period, verified 1:1 SW/HW at 5× | 200 | — | µs | published | `merolla2014a`, §S8 + Fig. S5 caption | "we ran networks with an average spike rate of 20Hz and 128 active synapses per neuron with a time step equal to 200µs, 5× faster than real-time and still observed one-to-one correspondence." |
| Internal logic clock (global) | **none — asynchronous** | — | — | published | `akopyan2015truenorth`, §IV, p.1541 | "The asynchronous control circuitry inside each TrueNorth core ensures that the core is active only when it is necessary to integrate synaptic inputs and update membrane potentials. Consequently, in our design there is no need for a high-speed global clock, and communication occurs by means of handshake protocols." |
| Neuron-block clocking | on-demand pulses from the token controller | — | — | published | `merolla2014a`, §S2 | "the neuron's clock is generated by the asynchronous Controller block, which issues on-the-fly clock pulses only when necessary" |
| Token-controller ↔ neuron handshake intervals | 7.16 / 14.32 | — | ns | published (design timing, Fig. 15) | `akopyan2015truenorth`, Fig. 15, p.1550 | Fig. 15 is annotated "δ1 = 7.16ns" and "δ0 = 14.32 ns"; caption: "Token controller/neuron detailed timing diagram: d1 and d4 are set by the first programmable delay line; d2 and d5 are set by the second programmable delay line; and d3 and d6 are the duration of an asynchronous handshake and an OR tree propagation delay (∼1 ns)." |
| Implied neuron-block operation rate | ≈140 | — | MHz | derived | as above | 1 ÷ 7.16 ns = **139.7 MHz**. ⚠ this is the δ1 interval of a design timing diagram, not a stated clock rating. |
| Neuron clock-edge placement granularity | 0.27 | — | ns | published | `akopyan2015truenorth`, §VI-B, p.1550 | "cell library) to configure the position of the rising clock edge in increments of 0.27 ns, as shown in Fig. 10." |
| Scan-chain clock (test/programming only) | 10 | — | MHz | published | `akopyan2015truenorth`, §V-H, p.1548 | "The scan chain runs at a modest speed of 10 MHz, without PLLs or global clock trees on the chip." |
| Programmable axonal delay | — | 1 – 15 | ticks | published | `merolla2014a` p.5, §S1 | "delivered after a desired axonal delay of between 1 to 15 time steps" |
| Max in-flight spike delivery window | 15 | — | ticks | published | `akopyan2015truenorth`, §V-D, p.1544 | "The delivery tick of an incoming spike needs to account for travel distance and maximum network congestion, and must be within the next 15 ticks after the spike generation (represented by 4 bits)." |
| Off-chip aggregate spike bandwidth | >160 | — | Mspikes/s (5.44 Gbit/s) | published | `merolla2014a`, §S6 | "measured the total bandwidth (transmit and receive) for all ports at more than 160 million spikes per second (5.44 Gbits/sec)" |

### 1.3 AREA

| quantity | value | band | unit | evidence_kind | citation | quote |
|---|---|---|---|---|---|---|
| **Die area** | **4.3** | — | cm² | published | `merolla2014a` p.6; `akopyan2015truenorth` §VI-C p.1550 | "With 5.4 billion transistors occupying 4.3cm² area in Samsung's 28nm process technology"; "At 4.3 cm² in die area and 5.4 billion transistors, the TrueNorth chip is larger than a typical ASIC chip" |
| **Technology node** | **28** | — | nm (Samsung LPP low-power CMOS) | published | `akopyan2015truenorth`, §VI-C, p.1550 | "The TrueNorth chip, shown in Fig. 16, was fabricated using Samsung's 28 nm LPP CMOS process technology. This silicon fabrication process was tuned for low-power devices." |
| **Transistor count, chip** | **5.4** | — | billion | published | `merolla2014a` p.6; `akopyan2015truenorth` abstract | "The fully digital 5.4 billion transistor implementation leverages existing CMOS scaling trends" |
| **Transistor count, per core** | **1.2** | — | million | published | `merolla2014a`, Fig. 2 panel J (in-figure label) | Fig. 2J is labelled "1.2 million transistors" beside the core layout |
| Transistor count per core, cross-check | 1.32 | — | million | derived | `merolla2014a` p.6 | 5.4×10⁹ ÷ 4096 = **1.318×10⁶**. Published per-core 1.2 M × 4096 = 4.92×10⁹ = **91.0 %** of 5.4 B; the balance is periphery (merge-split, I/O ring, pads). Consistent. |
| **Per-core area (measured footprint)** | **240 × 390** = **93 600 µm² = 0.0936 mm²** | — | µm | published | `akopyan2015truenorth`, Fig. 1 caption, p.1538; `merolla2014a`, Fig. 2 caption, p.14 | "one core occupies 240 × 390 μm of (d) silicon"; "(J) Physical layout of core in 28nm CMOS fits in a 240µm×390µm footprint." |
| Per-core area, **derived from die area** | 0.10498 | — | mm²/core | derived | `merolla2014a` p.6 | 4.3 cm² = 430 mm²; **430 mm² ÷ 4096 = 0.104980 mm² = 104 980 µm²**. The published 93 600 µm² footprint is **93 600 / 104 980 = 89.2 %** of that; the 10.8 % balance is chip periphery. Both numbers are therefore mutually consistent. |
| Core-grid total area | 3.834 | — | cm² | derived | as above | 4096 × 93 600 µm² = 383 385 600 µm² = **383.4 mm² = 3.834 cm²** (89.2 % of the 4.3 cm² die) |
| Core-to-core pitch, x | 240 | — | µm | published | `merolla2014a`, Fig. S4 caption | "The distance between cores in x direction is 240µm, which corresponds to one hop in the router." |
| **Neuron block area** (one 256-way time-multiplexed neuron circuit + PRNG) | **2900** | — | µm² | published | `merolla2014a`, §S3 (SM p.4) | "The neuron … uses 1,272 logic gates (924 gates for the neuron and 348 gates for the random number generator) in a 28nm process, corresponding to an area of 2900µm²" |
| Neuron-state storage | **3.0** | — | µm²/neuron | published | `merolla2014a`, §S3 | "storing neuron state (20 bits) requires an additional area of 3.0µm² per neuron" |
| **Effective area per neuron** | **14.3** | — | µm²/neuron | published | `merolla2014a`, §S3 | "By multiplexing the neuron 256 times in a time step, the effective area per neuron is: 2900µm²/256 + 3.0µm² = 14.3µm²." |
| Neuron block share of core area | 3.10 | — | % of core | derived | as above | 2900 µm² ÷ 93 600 µm² = **3.098 %** |
| Neuron-state storage, per core | 768 | — | µm² (0.82 % of core) | derived | as above | 256 × 3.0 µm² = **768 µm²**; ÷ 93 600 = 0.82 % |
| Neuron + state, per core | 3660.8 | — | µm² (3.91 % of core) | derived | as above | 256 × 14.3 µm² = **3660.8 µm²**; ÷ 93 600 = 3.91 % |
| Core-SRAM **bitcell** area | **0.152** | — | µm² (6T) | published | `akopyan2015truenorth`, §V-E, p.1545 | "The primary core memory uses a 0.152 μm² standard 6-transistor (6T) bitcell [26] [Fig. 9(b)]." |
| Core-SRAM bitcell-array area (**crossbar + neuron memory**) | ≈15 950 | 15 876 – 15 954 | µm² (≈17 % of core) | estimated | 0.152 µm²/bitcell × core-SRAM bit count | 256 rows × 410 bits = 104 960 bits × 0.152 µm² = **15 954 µm²** (TCAD row width); with `merolla2014a`'s 104 448 bits → **15 876 µm²**. Either way **≈17.0 % of the 93 600 µm² core**. ⚠ **bitcells only** — excludes decoders, sense amps, drivers, and the ten redundant columns / 16 redundant rows the TCAD paper mentions. This is a modelling estimate; the papers publish no SRAM macro area. |
| Scheduler SRAM bitcell | 12-transistor (12T) custom, **area not published** | — | — | published (topology only) | `akopyan2015truenorth`, §V-D/§V-E, p.1544–1545 | "The scheduler contains spike write/read/clear control logic and a dedicated 12-transistor (12T) 16 × 256-bit SRAM."; "The scheduler memory uses a fully custom 12T bitcell" |
| Per-component area breakdown (crossbar / router / scheduler / token controller) | — | — | — | **UNSOURCED** | — | See §2. `akopyan2015truenorth` Fig. 5 and `merolla2014a` Fig. S1A show a labelled core floorplan **image** but print **no per-block areas**. |

### 1.4 STRUCTURE

| quantity | value | unit | evidence_kind | citation | quote |
|---|---|---|---|---|---|
| Cores per chip | **4096** (64 × 64) | cores | published | `akopyan2015truenorth` abstract, §V-A p.1542 | "With 4096 neurosynaptic cores, the TrueNorth chip contains 1 million digital neurons and 256 million synapses"; "The full chip consists of a 64 × 64 array of neurosynaptic cores with associated peripheral logic." |
| **Neurons per core** | **256** | neurons | published | `merolla2014a` §S1; `akopyan2015truenorth` Fig. 1 caption | "it includes 256 neurons (computation), 256 axons (communication), and 256 × 256 synapses (memory)" |
| **Axons per core** | **256** | axons | published | (same) | (same) |
| **Crossbar size** | **256 × 256** (64k synapses/core) | — | published | `merolla2014a` p.3; `akopyan2015truenorth` Fig. 1 caption | "a core, a self-contained neural network with 256 input lines (axons) and 256 outputs (neurons) connected via 256 × 256 directed, programmable synaptic connections (Fig. 2D)"; "containing 256 input axons, 256 neurons, and a 64k synaptic crossbar" |
| Neurons per chip / synapses per chip | 1 million / 256 million | — | published | `merolla2014a` abstract | "a 5.4 billion transistor chip with 4,096 neurosynaptic cores … that integrates one million programmable spiking neurons and 256 million configurable synapses" |
| **Synapse weight bit width — crossbar cell** | **1** (binary w<sub>i,j</sub> ∈ {0,1}) | bit/synapse | published | `merolla2014a`, §S1 | "Synaptic weight from axon i to neuron j is a product of two terms: w<sub>i,j</sub> × S<sub>j</sub><sup>G<sub>i</sub></sup>, where w<sub>i,j</sub> ∈ {0, 1}. … We implement the 256 × 256 w matrix as a crossbar." |
| **Synapse weight bit width — effective value** | **9-bit signed**, one of **4** per-neuron values selected by the axon type | bits | published | `merolla2014a`, §S8 | "a synaptic operation constitutes adding a 9 bit signed integer S<sub>j</sub><sup>G<sub>i</sub></sup> to the membrane potential, which is a 20 bit signed integer, when axon A<sub>i</sub>(t) = 1 and synaptic connection w<sub>i,j</sub> = 1." |
| Axon types per core | **4** (G<sub>i</sub> ∈ {0,1,2,3}) | — | published | `merolla2014a` §S1; `akopyan2015truenorth` §III p.1540 | "each axon i is assigned one of four types G<sub>i</sub> ∈ {0, 1, 2, 3} and each neuron j can individually assign a programmable signed integer S<sub>j</sub><sup>G<sub>i</sub></sup> for axon type G<sub>i</sub>" |
| Per-neuron weight storage | 36 | bits (4 × 9) | derived | as above | 4 axon types × 9 signed bits = **36 bits** of the per-neuron parameter field |
| **Membrane potential bit width** | **20** (signed) | bits | published | `merolla2014a` §S3 and §S8 | "storing neuron state (20 bits)"; "the membrane potential, which is a 20 bit signed integer" |
| **Max fan-in per neuron** | **256** | presynaptic axons | published | `merolla2014a`, §S1 | "A core models networks with in-degree and out-degree of 256 or less, including classical neural networks as well as models of canonical cortical microcircuits; larger networks can be composed of multiple cores." |
| Max direct fan-out per neuron | **256** (one target axon; splitter neurons needed beyond that) | synapses | published | `merolla2014a`, §S4 | "a single neuron cannot target more than 256 synapses directly, but can do so by using two or more neurons in another core as 'splitters' sending spikes to more targets." |
| Core SRAM organization | **256 rows × 410 columns** | bits | published | `akopyan2015truenorth`, §V-E, p.1546 | "The SRAM is organized into 256 rows by 410 columns (not including redundant rows and columns). Each row corresponds to the information for a single neuron in the core … The neuron's 410 bits correspond to its synaptic connections, parameters and membrane potential V<sub>j</sub>(t), its core and axon targets, and the programmed delivery tick." |
| Core SRAM row fields (TCAD) | synaptic connections **256** / V<sub>j</sub>(t) & neuron parameters **124** / spike destination **26** / spike delivery tick **4** | bits | published | `akopyan2015truenorth`, unnumbered table §V-E, p.1546 | table cells: "256 bits \| 124 bits \| 26 bits \| 4 bits" |
| Core local memory (Science) | **104 448** bits/core = synapses **65 536** + neuron states/params **31 232** + destination addresses **6 656** + axonal delays **1 024** | bits | published | `merolla2014a`, p.6 | "Each core has 104,448 bits of local memory to store synapse states (65,536 bits), neuron states and parameters (31,232 bits), destination addresses (6,656 bits), and axonal delays (1,024 bits)." ⚠ **conflicts with the TCAD row width — see §3** |
| On-chip memory, chip-wide | ≈428 | Mbit | published | `merolla2014a`, p.6 | "TrueNorth has ∼428 million bits of on-chip memory" |
| Scheduler SRAM | **16 × 256** bits (16 delivery ticks × 256 axons) | bits | published | `akopyan2015truenorth` §V-A p.1542; `merolla2014a` §S1 | "The scheduler stores each spike as a binary value in an SRAM of 16 × 256-bit entries, corresponding to 16 ticks and 256 axons." |
| Spike packet width | **32** | bits | published | `merolla2014a`, Fig. S4 caption | "we injected spike packets (32 bits wide) traveling different distances, and measured the resulting power" |
| Spike packet fields | delivery tick **4** b, destination axon **8** b, dx **9** b signed, dy **9** b signed, debug **2** b | bits | published | `akopyan2015truenorth` §V-C p.1543 and §V-D p.1544 | "the router uses the information encoded in the dx (number of hops in the x direction as a 9-bit signed integer) or dy (number of hops in the y direction as a 9-bit signed integer as well) fields"; scheduler-input table: "delivery tick 4 bits \| destination axon index 8 bits \| debug bits 2 bits" |
| Router ports | **5** (north, south, east, west, local) | ports | published | `merolla2014a`, p.5 | "a two-dimensional mesh network of routers, each with five ports (north, south, east, west, and local)" |
| Routing | dimension-order (x then y), deadlock-free | — | published | `merolla2014a`, p.5 | "it is handed from core to core—first in the x dimension then in the y dimension (deadlock-free dimension-order routing)—until it arrives at its target core" |
| Routing reach | **±255 cores** in each of x and y (up to 4 chips) | cores | published | `merolla2014a` §S1 + footnote 4 | "The Router's datapath allows any neuron to transmit to any axon up to 255 cores away in both x and y directions (where a single chip is 64 × 64 cores)."; "each TrueNorth neuron can target any axon on any core up to 255 away (up to 4 chips) in both x and y dimensions, which includes 17 billion synapses, and therefore, uses 26 address bits." |
| Event-driven update sparsity, typical net | ≈640 of 65 536 possible neural updates per tick per core | updates | published | `merolla2014a`, §S1 | "For a typical network with neurons that fire at an average rate of 20Hz and make an average of 128 connections, a core will receive on average five incoming spike events in a 1ms time step, which corresponds to ∼640 neural updates. Out of the possible 256 × 256 (= 65,536) neural updates in the time step, our event-driven Controller only performs these ∼640 updates thereby eliminating 99% of the unnecessary circuit switching in the Neuron." |
| Neuron model | integrate-and-fire with leak, stochastic modes, programmable threshold/reset | — | published | `merolla2014a`, Table S2 | "V<sub>j</sub>(t) = V<sub>j</sub>(t−1) − λ<sub>j</sub> + Σ<sub>i=0</sub><sup>255</sup> A<sub>i</sub>(t) × w<sub>i,j</sub> × S<sub>j</sub><sup>G<sub>i</sub></sup>; if V<sub>j</sub>(t) ≥ α<sub>j</sub> → Spike, V<sub>j</sub>(t) = R<sub>j</sub>" |
| Core blocks | Neuron, Memory (core SRAM), Scheduler, Router, Controller (token controller) | — | published | `merolla2014a` §S1; `akopyan2015truenorth` §V-A p.1542 | "A single core consists of the scheduler block, the token controller block, the core SRAM, the neuron block, and the router block." |

### 1.5 OPERATING POINT / VALIDITY

| quantity | value | band | unit | evidence_kind | citation | quote |
|---|---|---|---|---|---|---|
| Supply voltage for the **26 pJ/synaptic-event** number | **0.775** | — | V | published | `merolla2014a`, p.7 | "the total measured power was 72mW (at 0.775V operating voltage), corresponding to 26pJ per synaptic event" |
| Supply voltage for **all `merolla2014a` communication energies** | **0.775** | — | V (chip-to-chip: 1.8 V) | published | `merolla2014a`, §S5 + Fig. S4 caption | "72fJ at 0.775V"; "2pJ at 0.775V"; "26pJ at 1.8V"; "For a power supply of 0.775V, we calculate…" |
| Supply voltage for the **65 mW / 58 GSOPS / 400 GSOPS/W** numbers | **0.75** | — | V | published | `akopyan2015truenorth`, §VII, p.1551 | "we pick an operating point of 0.75 V. At this supply voltage, the maximum computational speed of the chip is 58 GSOPS…" |
| Supply voltage for the **Fig. 17 power breakdown** | **0.8** | — | V | published | `akopyan2015truenorth`, Fig. 17 caption, p.1551 | "Total TrueNorth chip power breakdown (@ 0.8 V)" |
| Full supply range | — | 0.70 – 1.05 | V | published | `akopyan2015truenorth`, §VII, p.1551 | "the TrueNorth chip is operational from 1.05 V down to 0.7 V" |
| Voltages swept in Fig. S4 (energy vs distance) | — | 0.60 – 1.00 | V | published | `merolla2014a`, Fig. S4 legend | legend entries: "1.00 V / 0.90 V / 0.80 V / 0.70 V / 0.60 V" |
| Measured on silicon vs simulated | **measured on silicon** for every energy/power/area number above | — | published | `akopyan2015truenorth`, §VII, p.1551 | "We tested the TrueNorth chip for logical correctness and extensively characterized it for performance and power consumption [1], [3]." |
| Benchmark network for the power characterization | probabilistically generated complex recurrent networks, all 4096 cores / 1M neurons, 0–200 Hz, 0–256 active synapses/neuron, uniform-random inter-core connectivity (**21.3 cores average distance in each dimension**), all neurons in stochastic mode, axonal delays uniform 1–15 | — | published | `merolla2014a`, §S7 | "The networks use all 4,096 cores and one million neurons on the chip, and span a range of mean firing rates per neuron from 0 to 200Hz and a range of active synapses per neuron from 0 to 256. The networks were designed to push TrueNorth's power consumption as high as possible for a given spike rate and synaptic density. To this end, we selected connections between cores randomly with uniform distribution across the chip, requiring spikes to travel long distances (21.3 cores away on average in both x and y dimensions), and configured all neurons in the stochastic mode…" |
| Temperature | — | — | — | **UNSOURCED** | — | Neither paper states an ambient or junction temperature for any measurement. |
| Yield | >50 % fully functional | — | published | `merolla2014a`, §S6 | "Tested chips have a yield distribution due to physical defects, common to integrated circuit manufacturing, including fully-functional chips (more than 50%), chips with occasional faulty cores, and a minority of unusable chips." |

---

## 2. UNSOURCED list

These are quantities the brief asked for that **neither primary source publishes**. They must not
be filled in with a guess; anything downstream that needs them must either derive them with the
arithmetic stated or declare them absent.

1. **Energy per neuron update per tick, as an isolated circuit cost.** Not published. Only the
   whole-chip aggregate is derivable (72 / 65 / 42 pJ per neuron-tick, §1.1), which bundles
   synaptic integration, memory access, routing, leak and periphery into one number. There is
   no published decomposition that isolates the neuron block's update energy.
2. **Energy per spike routed *intra-core* (router local channel).** The local channel is
   described (`merolla2014a` §S1: "Spike packets generated on a core that target the same core
   use the Router's local channel") but never priced. The Fig. S4 linear model would give
   0.3 fJ/µm × 0 µm = 0, which is a model artifact, not a measurement. The nearest published
   proxy is the 47 fJ/bit core-SRAM→Controller transfer, which is a *memory* cost, not a
   routing cost.
3. **Per-component POWER breakdown (leak / memory / computation / communication).**
   `akopyan2015truenorth` Fig. 17 plots exactly this as a stacked bar at 0.8 V, but **only the
   three column totals (68 / 71 / 94 mW) are printed**; the four segment values are not
   labelled and no table restates them. Reading them off the bar heights would be pixel
   estimation, not a citation.
4. **Per-component AREA breakdown (crossbar / router / scheduler / token controller).** Not
   published. `akopyan2015truenorth` Fig. 5 and `merolla2014a` Fig. S1A both show a labelled
   core floorplan photomicrograph, but no block areas are printed and no table gives them.
   The only per-block area facts in the literature are the ones in §1.3: the 2900 µm² neuron
   block, the 3.0 µm²/neuron state store, the 14.3 µm² effective per-neuron area, and the
   0.152 µm² 6T core-SRAM bitcell.
5. **Leakage power explicitly labelled as such**, at any voltage. The 42 mW low-corner figure
   is the best available proxy and is marked `derived` for that reason. There is no published
   per-core leakage figure, and no leakage-vs-voltage curve with printed values.
6. **Spike delivery latency / per-hop latency, in seconds.** Not published, and the architecture
   makes it *deliberately* non-deterministic: `akopyan2015truenorth` §V-C p.1544 states "the
   communication time between a spiking neuron and a destination axon will fluctuate due to
   on-going traffic in the spike-routing network. However, we hide this network nondeterminism
   at the destination by ensuring that the spike delivery tick of each spike is configured to be
   longer than the maximum communication latency between its corresponding source and
   destination." The only *bound* is architectural: delivery must land within 15 ticks (≤15 ms
   at the nominal tick).
7. **A single nameable "clock frequency of the internal logic."** The chip has no global
   high-speed clock (§1.2). The published internal-timing facts are the 1 kHz tick, the 10 MHz
   scan chain, the 0.27 ns clock-edge placement granularity, and the Fig. 15 δ1 = 7.16 ns /
   δ0 = 14.32 ns handshake intervals. The ≈140 MHz figure in §1.2 is *derived from a design
   timing diagram* and is not a rated clock speed.
8. **Measurement temperature** for any reported number.
9. **Scheduler 12T SRAM bitcell area**, and any SRAM *macro* (array + periphery) area.
10. **Supply voltage of the 63 mW multi-object-detection measurement** (`merolla2014a` §S11),
    and of the 60 mW visual-filter and 72 mW probabilistic-network rows of Table S1 — the
    0.775 V attribution is stated only in the main text for the 72 mW probabilistic-network
    point, so applying it to the other two rows would be inference.
11. **Energy per spike for the merge-split / serializer path as a spike-level number** — only the
    2 pJ/bit internal→periphery figure is published; the ×32 conversion in §1.1 is derived and
    assumes the full 32-bit packet crosses that boundary, which the papers do not state
    explicitly (they note the north/south ports carry 24 bits because "the router data path in
    the vertical direction drops the dx routing information", `akopyan2015truenorth` §V-G
    footnote 1, p.1547).

---

## 3. Validity domain and internal disagreements

### 3.1 Validity domain

* All energy/power numbers are **silicon measurements on a fabricated 28 nm Samsung LPP chip**,
  not simulations. Structural and timing numbers (packet fields, SRAM organization, Fig. 15
  delays) are design specifications.
* **The 26 pJ/synaptic-event figure is a total-energy, fully-amortized number** at one specific
  operating point: 0.775 V, 20 Hz mean firing rate, 128 active synapses/neuron, 1 ms tick, all
  4096 cores active, uniform-random long-distance connectivity (21.3 cores mean hop distance in
  each dimension), all neurons in stochastic mode. It **includes leakage, memory access,
  routing and periphery**, amortized over the synaptic events. It is *not* a per-synapse
  circuit-activation energy and must not be used as one. `merolla2014a` Fig. 4C caption is
  explicit that the number falls with synaptic density "because leakage power and baseline core
  power are amortized over additional synapses" — i.e. it is a ratio, not a constant.
* Energy scales strongly with supply: the operating envelope spans 42 mW → 323 mW (7.7×) across
  0.70 V → 1.05 V and 0 Hz/0 synapses → 200 Hz/256 synapses. Any profile that quotes a single
  energy without its voltage and activity point is unusable.
* The 2.3 pJ/hop routing energy is for the **x direction** ("y direction is similar",
  Fig. S4 caption) at 0.775 V, over a 240 µm core pitch. The y pitch is 390 µm, so a y-hop is
  plausibly more expensive — the papers do not give a separate y number, and note in passing
  that north/south ports carry 24 bits rather than 32.
* All per-core derivations assume **uniform cores**, which the architecture guarantees (4096
  identical tiled instances).

### 3.2 Disagreements between the two primary sources — FLAGGED

**D1 — Headline operating point: 65 mW @ 0.75 V vs 72 mW @ 0.775 V.**
Both papers describe the *same* workload ("20 Hz average firing rate and 128 active synapses per
neuron", real-time 1 kHz tick) but report different power at different supplies:
`akopyan2015truenorth` p.1551 says **65 mW at 0.75 V**; `merolla2014a` p.7 says **72 mW at
0.775 V**. These are consistent with each other as two points on a voltage curve (higher V →
higher power), not contradictory — but a physics profile must not merge them. **Use 72 mW ⇔
0.775 V ⇔ 26 pJ/event as one self-consistent triple, and 65 mW ⇔ 0.75 V as a separate one.**

**D2 — 46 GSOPS/W and 26 pJ/synaptic-event are mutually inconsistent by ≈1.2×.**
Both are published; both describe the 20 Hz / 128-synapse real-time point.
* 26 pJ/event ⇒ 1 / 26×10⁻¹² = **38.5 GSOPS/W**, not 46.
* Synaptic events/s at the stated point = 10⁶ neurons × 20 Hz × 128 synapses = **2.56×10⁹ SOPS**.
* At `merolla2014a`'s 72 mW: 72×10⁻³ / 2.56×10⁹ = **28.1 pJ/event** (paper says 26 pJ, 8 % low),
  and 2.56×10⁹ / 72×10⁻³ = **35.6 GSOPS/W** (paper says 46, 29 % high).
* At `akopyan2015truenorth`'s 65 mW: 2.56×10⁹ / 65×10⁻³ = **39.4 GSOPS/W** (paper says 46,
  17 % high).
The 26 pJ figure is the one that reconciles with the measured 72 mW to within 8 %; the
46 GSOPS/W figure does not reconcile with either power number. **Prefer 26 pJ/event; treat
46 GSOPS/W as a headline figure of unclear derivation.**

**D3 — Core SRAM row width: 410 bits (TCAD) vs 408 bits (Science).**
* `akopyan2015truenorth` §V-E p.1546: "The SRAM is organized into 256 rows by 410 columns",
  itemized as 256 (synapses) + **124** (V<sub>j</sub> & neuron parameters) + 26 (spike
  destination) + 4 (delivery tick) = 410.
* `merolla2014a` p.6: 104 448 bits/core = 65 536 + 31 232 + 6 656 + 1 024. Per neuron this is
  65 536/256 = 256 (synapses) + 31 232/256 = **122** (neuron states & parameters) +
  6 656/256 = 26 (destination) + 1 024/256 = 4 (delay) = **408 bits/row**.
* The discrepancy is **exactly 2 bits in the neuron state/parameter field** (124 vs 122).
* Tie-break: `merolla2014a`'s own chip-wide total, "∼428 million bits", equals
  4096 × 104 448 = **427.8 Mb**, matching the 408-bit row. The 410-bit row gives
  4096 × 104 960 = **429.9 Mb**. Both round to "∼428 M", so the total does not decide it
  cleanly. **Report 408–410 bits/row as a 2-bit band; do not silently pick one.**

**D4 — Fig. S5 caption unit slip (typo in the source).**
`merolla2014a` Fig. S5 caption reads "we measured 70 SOPS/W" and "over 400 SOPS/W", where §S8
and the main text both say **70 billion** and **400 billion** SOPS/W. The caption is missing the
"billion". Use the §S8 values.

---

## 4. In-repo cross-check

Case-insensitive grep for `truenorth` over `src/` and `templates/`. `templates/` has **zero**
matches. The path the brief mentioned,
`papers/structured_elimination_aaai/research_artifacts/`, **does not exist** on this machine
(nearest is `papers/exact_structured_pruning/`, which contains no chip-geometries file) — see
finding F6 below.

### 4.1 Files carrying TrueNorth constants

| file | what it holds |
|---|---|
| `/home/yigit/repos/research_stuff/mimarsinan/src/mimarsinan/chip_simulation/sanafe/presets.py` | `TRUENORTH_PRESET` — 16 per-event energy/latency constants |
| `/home/yigit/repos/research_stuff/mimarsinan/sana_fe/arch/truenorth.yaml` | vendored SANA-FE architecture (structure + all-zero physics) |
| `/home/yigit/repos/research_stuff/mimarsinan/src/mimarsinan/mapping/platform/imc_platforms_literature.py` | `truenorth_like` IMC platform geometry + provenance quote |
| `/home/yigit/repos/research_stuff/mimarsinan/src/mimarsinan/deployment_record/cost/coefficients.py` | `_REFERENCE_CORE_NEURONS_SMALL`, and bands built on `TRUENORTH_PRESET` |
| `/home/yigit/repos/research_stuff/mimarsinan/src/mimarsinan/chip_simulation/sanafe/arch_synth/floorplan.py` | `PRESET_CORES_PER_TILE["truenorth"] = 1` |
| `/home/yigit/repos/research_stuff/mimarsinan/src/mimarsinan/config_schema/registry/entries_platform.py`, `entries_execution.py` | enum option `"truenorth"` and doc text only — **no physical constants** |

### 4.2 Value-by-value cross-check

| in-repo value | file | paper value | verdict |
|---|---|---|---|
| `max_neurons=256` | `imc_platforms_literature.py`; `truenorth.yaml` (`max_neurons_supported: 256`) | 256 neurons/core | ✅ **AGREES** |
| `max_axons=256` | `imc_platforms_literature.py` | 256 axons/core | ✅ **AGREES** |
| `count=4096` cores | `imc_platforms_literature.py` | 4096 cores | ✅ **AGREES** |
| `width: 64`, `height: 64` (4096 tiles) | `truenorth.yaml` | 64 × 64 core grid | ✅ **AGREES** |
| `PRESET_CORES_PER_TILE["truenorth"] = 1` | `floorplan.py` | one router per core, 2-D mesh of 4096 | ✅ **AGREES** (architecturally faithful) |
| `_REFERENCE_CORE_NEURONS_SMALL = 256  # TrueNorth core (Merolla 2014)` | `coefficients.py` | 256 neurons/core | ✅ **AGREES** |
| provenance quote for `truenorth_like` | `imc_platforms_literature.py` | `merolla2014a` p.3 | ✅ **VERBATIM MATCH** — I diffed it against the PDF text: "From a structural view, the basic building block is a core, a self-contained neural network with 256 input lines (axons) and 256 outputs (neurons) connected via 256 × 256 directed, programmable synaptic connections (Fig. 2D)." Exact. |
| `weight_bits=1` | `imc_platforms_literature.py` | crossbar cell w<sub>i,j</sub> ∈ {0,1} = 1 bit; **effective** weight = w<sub>i,j</sub> × a 9-bit signed S<sub>j</sub><sup>G<sub>i</sub></sup>, 1-of-4 by axon type | ⚠️ **F1 — CORRECT BUT INCOMPLETE** (see below) |
| `has_bias=True` | `imc_platforms_literature.py` | per-neuron leak λ<sub>j</sub> subtracted every tick (Table S2) | ✅ **SUPPORTED** — the leak is functionally a per-neuron bias |
| `tile_hop_energy_j = 5.0e-14` (0.05 pJ) | `presets.py` | **2.3 pJ/spike/hop** | ❌ **F2 — DISAGREES by 46×** |
| `synapse_energy_j = 2.0e-13` (0.2 pJ) | `presets.py` | 26 pJ/synaptic event (total-energy) | ❌ **F3 — DISAGREES by 130×** vs the only published figure |
| `tile_hop_latency_s = 4.0e-9` (4 ns) | `presets.py` | **no published hop latency** | ❌ **F4 — UNSOURCED** |
| `axon_in/out`, `soma_access/update/spike_out` energies (0 – 3.0e-13 J) and all latencies (1 – 5 ns) | `presets.py` | **no published per-component TrueNorth costs of any kind** | ❌ **F4 — UNSOURCED** |
| all `energy_*` / `latency_*` = `0.0` | `sana_fe/arch/truenorth.yaml` | — | ⚠️ **F5 — no physics at all** |
| docstring pointer to `papers/structured_elimination_aaai/research_artifacts/13_chip_geometries.json` | `imc_platforms_literature.py` | — | ⚠️ **F6 — dangling path** |

### 4.3 Findings

**F1 — `weight_bits=1` is right for the crossbar cell, wrong as "effective weight precision."**
TrueNorth's synapse really is a 1-bit crosspoint, so as a *crossbar geometry* fact this is
correct and the file's own framing (geometry transcription) is defensible. But the *effective*
synaptic weight is `w_ij × S_j^{G_i}` where `S` is a **9-bit signed integer** chosen from four
per-neuron values by the axon's type. Any downstream consumer that reads `weight_bits` as
"achievable weight resolution" will understate TrueNorth by ~9 bits and four programmable
levels. Recommend a provenance note rather than a value change.

**F2 — `tile_hop_energy_j = 5.0e-14` is 46× below the published per-hop spike energy.**
`merolla2014a` Fig. S4 states 2.3 pJ per spike per inter-core hop at 0.775 V, itself the product
72 fJ/bit × 32 bits. The repo value is 0.05 pJ. Ratio 2.304 / 0.05 = **46.1×**. If the repo
constant were secretly *per-bit* it would still be off (72 fJ vs 50 fJ = 1.44×), and SANA-FE's
`tile_hop_energy` is per-message, so the per-bit reading does not rescue it. The corresponding
Loihi entry (3.5 pJ) is per-message and of the right order, which suggests the TrueNorth row was
not derived from the paper at all.

**F3 — `synapse_energy_j = 2.0e-13` cannot reproduce the measured chip power.**
At the reference point there are 10⁶ × 20 Hz × 128 = 2.56×10⁹ synaptic events/s. At the repo's
0.2 pJ this is 2.56×10⁹ × 2.0×10⁻¹³ = **0.512 mW**, against a measured 72 mW — **141× low**, and
that is before adding the routing, soma and leak terms. Caveat in fairness: SANA-FE's
`synapse_energy` is an *active-only per-component* cost while the paper's 26 pJ is a
*total-energy amortized* ratio, so the two are not the same quantity and a 1:1 comparison is
unfair. The real problem is that **no published TrueNorth per-component synapse energy exists**,
so there is nothing that justifies 0.2 pJ either.

**F4 — most of `TRUENORTH_PRESET` is unsourced, and the comment overstates its provenance.**
The header comment reads `# TrueNorth reference numbers (Merolla 2014; bundled approximations).`
Of the 16 constants, exactly **zero** appear in `merolla2014a`. The paper publishes 26 pJ/event,
2.3 pJ/hop, 72 fJ/bit/hop, 47 fJ/bit SRAM read, 2 pJ/bit periphery and 26 pJ/bit chip-to-chip —
none of which is any of the 16 values. The parenthetical "bundled approximations" is honest
about their character but the "Merolla 2014" attribution is not: it credits the paper with
numbers it does not contain. **Recommend rewording the attribution.** (No code change made —
this is a research note.)

**F5 — the vendored SANA-FE TrueNorth architecture has all-zero physics.**
`sana_fe/arch/truenorth.yaml` sets every `energy_*` and `latency_*` attribute (hops in all four
directions, `axon_in`, `soma` access/update/spike-out, `synapse`, `dendrite`, `axon_out`) to
`0.0`, plus `latency_sync: 0.0`. Its structure (64 × 64 mesh, 1 core/tile,
`max_neurons_supported: 256`) is faithful, but a SANA-FE run against this preset yields
**identically zero** energy and latency. Anything reporting SANA-FE energy on the `truenorth`
preset is reporting zeros, not TrueNorth. Note this is a **vendored third-party tree** and is
out of scope to edit.

**F6 — the provenance pointer in `imc_platforms_literature.py` is dangling.**
Its module docstring says every entry is "transcribed EXACTLY … from
`papers/structured_elimination_aaai/research_artifacts/13_chip_geometries.json`" and instructs
future editors to "add a new extraction card to that JSON and re-transcribe instead." Neither
that directory nor any `*chip_geometries*` file exists under
`/home/yigit/repos/research_stuff/papers/` (the closest directory is
`papers/exact_structured_pruning/`). The stated re-transcription workflow is therefore not
executable as written. The *content* still checks out — I verified the `truenorth_like` quote
against the PDF and it is verbatim — but the audit trail it points at is missing.

**F7 — `coefficients.py` inherits the unsourced numbers.**
`_preset_core_band` and `SYNC_BARRIER_S` consume `TRUENORTH_PRESET["soma_access_energy_j"]`,
`["soma_update_energy_j"]` and `["tile_hop_latency_s"]` to form the "low" corner of the
core-init and sync-barrier bands. Those three inputs are all in the F4 unsourced set, so the low
corner of both bands is unsourced too. The file is already candid about this — both basis
strings end in `"NEW - needs owner sign-off"` — so this is a confirmation, not a new problem:
**the sign-off cannot be satisfied from the TrueNorth literature, because the literature does
not publish those quantities** (§2 items 1, 2, 6).

---

## 5. Citations (not yet injected)

Injection into a real `.bib` is **deferred**. The only `.bib` in this repo is
`sana_fe/references.bib`, which belongs to the vendored SANA-FE tree and must stay untouched;
no other `.bib` exists here and none was created.

The BibTeX below was **generated by the `asta-papers` tooling** (`semantic_scholar_search` →
`inject_papers_to_bib` into a scratchpad file, then copied verbatim). It was not hand-written.
When a real bibliography exists, re-run `inject_papers_to_bib` with ids `merolla2014a` and
`akopyan2015truenorth` rather than pasting this block.

```bibtex
@Article{merolla2014a,
 author = {P. Merolla and J. Arthur and Rodrigo Alvarez-Icaza and A. Cassidy and J. Sawada and F. Akopyan and Bryan L. Jackson and Nabil Imam and Chen Guo and Yutaka Y. Nakamura and B. Brezzo and I. Vo and S. K. Esser and R. Appuswamy and B. Taba and A. Amir and M. Flickner and W. Risk and R. Manohar and D. Modha},
 booktitle = {Science},
 journal = {Science},
 pages = {668 - 673},
 title = {A million spiking-neuron integrated circuit with a scalable communication network and interface},
 volume = {345},
 year = {2014},
 doi = {10.1126/science.1254642},
 url = {https://doi.org/10.1126/science.1254642},
}

@Article{akopyan2015truenorth,
 author = {F. Akopyan and J. Sawada and A. Cassidy and Rodrigo Alvarez-Icaza and J. Arthur and P. Merolla and Nabil Imam and Yutaka Y. Nakamura and Pallab Datta and Gi-Joon Nam and B. Taba and Michael P. Beakes and B. Brezzo and J. B. Kuang and R. Manohar and W. Risk and Bryan L. Jackson and D. Modha},
 booktitle = {IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems},
 journal = {IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems},
 pages = {1537-1557},
 title = {TrueNorth: Design and Tool Flow of a 65 mW 1 Million Neuron Programmable Neurosynaptic Chip},
 volume = {34},
 year = {2015},
 doi = {10.1109/TCAD.2015.2474396},
 url = {https://doi.org/10.1109/TCAD.2015.2474396},
}
```

Cached ids for re-injection: `merolla2014a`, `akopyan2015truenorth`.

Note on the Science entry: the tool's cached record omits `number = {6197}`. The canonical
citation is *Science* **345**(6197):668–673, 8 Aug 2014, DOI `10.1126/science.1254642`. Correct
the issue number at injection time via the tooling, not by hand-editing the `.bib`.
