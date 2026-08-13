# PRIME physics profile — published per-unit constants

Target: **PRIME** full-function (FF) ReRAM subarray, ISCA-2016 configuration.

> ## ⚠ HEADLINE FINDING, READ FIRST
>
> **The PRIME ISCA-2016 paper publishes NO absolute per-unit area, energy, power, or
> compute-mode latency numbers for PRIME itself.** Not one. It publishes *structure*
> (exact and quotable), *relative* results (speedup/energy-saving ratios vs. a CPU and vs.
> NPU baselines), and *percentage* area-overhead breakdowns — and nothing else.
>
> There is no PRIME equivalent of ISAAC's Table I. Every AREA/ENERGY/TIME row in the
> `platform_physics` vocabulary is therefore **UNSOURCED** from this paper (§2), except the
> memory-mode DRAM-style timings and the ReRAM device parameters.
>
> PRIME is also **a simulation study, not silicon** (§3.1). A PRIME physics profile cannot
> be built from published PRIME numbers alone; it must either borrow constants from a
> different source (and say so) or stay structural.

---

## 0. Sources read

| key | source | type | how read |
|---|---|---|---|
| `chi2016prime` | P. Chi, S. Li, C. Xu, T. Zhang, J. Zhao, Y. Liu, Y. Wang, Y. Xie, "PRIME: A Novel Processing-in-Memory Architecture for Neural Network Computation in ReRAM-Based Main Memory", *ISCA-43*, 2016, pp. 27–39 | peer-reviewed paper (circuit + architecture modelling) | full PDF, 13 pp, text via `pdftotext -layout`; **Figures 8/10/12 additionally read as 200–260 dpi page renders** because the figure fonts are custom-encoded and extract as mojibake |
| `chi2023retrospectiveprime` | P. Chi et al., "RETROSPECTIVE: PRIME…", 2023 | retrospective commentary | full PDF, 2 pp — **contains no hardware constants**; historical narrative only. Checked and discarded as a numeric source |
| — | SEAL-lab Technical Report 2015-001, *"Processing-in-memory in ReRAM-based main memory"* (cited by the paper as ref [75]) | technical report | **NOT RETRIEVED** — `seal.ece.ucsb.edu` is unreachable from this environment (DNS `ENOTFOUND`). It is the most likely home of any absolute PRIME numbers; a future pass with network access should try it |

---

## 1. Constants table

Vocabulary keys are those of `src/mimarsinan/deployment_record/platform_physics/vocabulary.py`.
`evidence_kind ∈ published | datasheet | derived | estimated`.

### 1.1 STRUCTURE — the well-sourced part

This is the section PRIME actually supports. Every row is a verbatim quote.

| quantity | value | evidence_kind | citation | quote |
|---|---|---|---|---|
| **Crossbar (mat) size** | **256 × 256** ReRAM cells | published | `chi2016prime`, §V-A "PRIME Configurations", p.36 | "In FF subarrays, for each mat, there are 256×256 ReRAM cells and eight 6-bit reconfigurable SAs; for each ReRAM cell, we assume 4-bit MLC for computation while SLC for memory; the input voltage has 8 levels (3-bit) for computation while 2 levels (1-bit) for memory." |
| **ADCs (sense amplifiers) per mat** | **8** | published | `chi2016prime`, §V-A, p.36 | (same sentence) "…and eight 6-bit reconfigurable SAs" |
| **`adc_sharing_factor`** | **32** columns per SA | derived | 256 bitlines ÷ 8 SAs | 256 / 8 = **32**. PRIME reuses the memory sense amplifier *as* the ADC — §III-A1: "instead of using both, we reuse SAs and write drivers to serve ADC and DAC functions by slightly modifying the circuit design." |
| **ADC (SA) resolution** | **6** bits, reconfigurable 1–6 | published | `chi2016prime`, §III-A1 "Sense Amplifier", p.31 | "We adopt a P_o-bit (P_o ≤ 8) precision reconfigurable SA design that has been tested through fabrication [64]. Second, we allow SA's precision to be configured as any value between 1-bit and P_o-bit, controlled by the counter". §V-A fixes P_o = 6: "the target output precision is 6-bit" |
| **DAC (wordline driver) resolution** | **3** bits (8 voltage levels) for compute; 1 bit (2 levels) for memory | published | `chi2016prime`, §V-A, p.36 | (same sentence) "the input voltage has 8 levels (3-bit) for computation while 2 levels (1-bit) for memory" |
| Row drivers per mat | **256** (one per wordline) | published | `chi2016prime`, §III-A1 "Decoder and Driver", p.30 | "Second, to drive the analog signals transferring on the wordlines, we employ a separate current amplifier on each wordline." |
| **Cell precision** | **4** bits MLC (16 resistance levels) for compute; SLC (1 bit) for memory | published | `chi2016prime`, §V-A + §III-D | "for each ReRAM cell, we assume 4-bit MLC for computation while SLC for memory"; §III-D: "the ReRAM cells can only represent 4-bit synaptic weights (i.e. 16 resistance levels)" |
| **`conductance_levels`** | **16** | published | §III-D, p.33 | "the ReRAM cells can only represent 4-bit synaptic weights (i.e. 16 resistance levels)" |
| **Cells per weight** | **2** (adjacent bitlines) | published | `chi2016prime`, §III-D1, p.33 | "we propose an input and synapse composing scheme, which can use two 3-bit input signals to compose one 6-bit input signal and two 4-bit cells to represent one 8-bit synaptic weight." Placement: "To implement synapse composing, the high-bit and low-bit parts of the synaptic weights are stored in adjacent bitlines of the corresponding crossbar array." |
| **Deployed weight precision** | **8** bits | published | `chi2016prime`, §V-A, p.36 | "With our input and synapse composing scheme, for computation, the input and output are 6-bit dynamic fixed point, and the weights are 8-bit." |
| **Deployed input precision** | **6** bits (2 sequential 3-bit passes) | published | §V-A + §III-D1 | (same sentence). Sequencing: "According to the control signal, the high-bit and low-bit parts of the input are fed to the corresponding crossbar array sequentially." |
| **Deployed output precision** | **6** bits | published | §V-A / §III-D | "the target output precision is 6-bit"; "P_out = 6 (enabled by 6-bit precision reconfigurable sense amplifiers)" |
| Data format | dynamic fixed point | published | §III-D | "The data format we use is dynamic fixed point [68]." |
| **Signed weights → 2 arrays** | positive and negative in **separate** crossbars sharing one input port | published | `chi2016prime`, §III-A1, p.30 | "Finally, we employ two crossbar arrays store positive and negative weights, respectively, and allow them to share the same input port." |
| Subtraction is **analog, pre-ADC** | yes | published | `chi2016prime`, §III-A2, p.31 | "in computation mode, the FF subarray fetches the input data of the NN from the Buffer subarray into the latch of the wordline decoder and driver. After the computation in the crossbar arrays that store positive and negative weights, their output signals are fed into the subtraction unit, and then the difference signal goes into the sigmoid unit. The analog output is converted to digital signal by the SA is written back to the Buffer subarray." |
| Periphery shared by the ± pair | half the column MUXes are modified | published | `chi2016prime`, §III-A1 "Column Multiplexer", p.30 | "Since a pair of crossbar arrays with positive and negative weights require one set of such peripheral circuits, we only need to modify half of the column multiplexers." |
| **FF subarrays per bank** | **2** | published | `chi2016prime`, §V-A, p.36 | "There are 2 FF subarrays and 1 Buffer subarray per bank (totally 64 subarrays)." |
| Buffer subarrays per bank | 1 | published | (same) | (same sentence) |
| **Subarrays per bank (total)** | **64** | published | §V-A + §V-D, p.36/38 | "(totally 64 subarrays)"; §V-D restates: "Given two FF subarrays and one Buffer subarray per bank (64 subarrays in total), PRIME only incurs 5.76 % area overhead." ⚠ reading: a bank *contains* 64 subarrays, of which 2 are FF and 1 Buffer — see §3.3(a) |
| **Banks per chip** | **8** | published | `chi2016prime`, Table IV "Configurations of CPU and Memory", p.36 | Table IV, ReRAM-based Main Memory row: "16GB ReRAM; 533MHz IO bus; 8 chips/rank; 8 banks/chip" |
| **Chips per rank** | **8** | published | Table IV | (same cell) |
| Total memory capacity | 16 GB | published | Table IV | (same cell) |
| **"NPUs" (= FF-equipped banks)** | **64** | published | `chi2016prime`, §IV-B2, p.35 | "Considering FF subarrays in each bank as an NPU, PRIME contains 64 NPUs in total (8 banks×8 chips) so that 64 images can be processed in parallel." |
| FF subarrays chip-set-wide | **128** | derived | 2 × 8 banks × 8 chips | 2 FF/bank × 8 banks/chip × 8 chips = **128 FF subarrays** |
| **Mats per subarray** | — | **UNSOURCED** | — | The paper never states it. Figure 4 draws a subarray as a grid of mats, but no count is given. This blocks any absolute mat population — see §5.3 finding F2 |
| Max mapped network | ~2.7 × 10⁸ synapses | published | `chi2016prime`, §IV-B1, p.35 | "If all the banks are used to implement a single NN, PRIME can handle a maximal NN with ∼2.7×10⁸ synapses, which is larger than the largest NN that have been mapped to the existing NPUs (TrueNorth [34], 1.4×10⁷ synapses)." |
| Logical NN per mat (paper's own framing) | 256 − 256 | published | `chi2016prime`, §V-C, p.37 | "since each ReRAM mat can execute as large as a 256 − 256 NN at one time"; §IV-B1: "to implement a 512−512 NN on PRIME with 256−256 mats, it is split into four 256−256 parts". ⚠ conflicts with the composing scheme — see §3.3(b) |
| Inputs per array, notation | `2^PN` (PN = 8 for 256 rows) | published | `chi2016prime`, Table II "Notation Description", p.33 | "P_N — the number of inputs to a crossbar array is 2^{P_N}" |
| Activation functions in hardware | sigmoid (analog, in col-MUX) + ReLU (digital, in SA block) | published | §III-A1 / §III-E | "The modified column multiplexer incorporates two analog processing units: an analog subtraction unit and a non-linear threshold (sigmoid) unit [63]"; "we add a hardware unit to support ReLU function… The circuit checks the sign bit of the result." |
| Pooling in hardware | 4:1 max pool; n:1 in multiple steps; mean pool via ReRAM weights | published | §III-A1 / §III-E | "a circuit to support 4-1 max pooling is included"; "we adopt 4:1 max pooling hardware in Figure 4 C, which is able to support n:1 max pooling with multiple steps for n > 4" |
| LRN support | **none** | published | `chi2016prime`, §III-E, p.34 | "Currently, PRIME does not support LRN acceleration. We did not add the hardware for LRN, because state-of-the-art CNNs do not contain LRN layers [69]. When LRN layers are applied, PRIME requires the help of CPU for LRN computation." |
| Training support | **none** (inference only) | published | `chi2016prime`, §IV-A, p.35 | "In our work, the training of NN is done off-line so that the inputs of each API are already known… we plan to further enhance PRIME with the training capability in future work." |

### 1.2 AREA — percentages only, no absolutes

| quantity | value | band | unit | evidence_kind | citation | quote |
|---|---|---|---|---|---|---|
| **Whole-chip area overhead of PRIME** | **5.76** | — | **% of the ReRAM chip** | published | `chi2016prime`, §V-D, p.38 | "Given two FF subarrays and one Buffer subarray per bank (64 subarrays in total), PRIME only incurs 5.76 % area overhead." |
| Add-on area within an FF mat | **60** | — | **% of the modified mat's area** | published | `chi2016prime`, Fig. 12 + §V-D, p.38 | Fig. 12 pie: the "Add-on" wedge is labelled **60 %**. Text: "There is 60 % area increase to support computation: the added driver takes 23 %, the subtraction and sigmoid circuits take 29 %, and the control, the multiplexer, and etc. cost 8 %." ⚠ wording ambiguity — see §3.3(c) |
| `area_per_cell` | — | — | µm² | **UNSOURCED** | — | no absolute cell area anywhere |
| `area_per_adc` | — | — | µm² | **UNSOURCED** | — | only "output (SA, etc) 15 %" of a mat whose absolute area is unknown |
| `area_per_row_driver` | — | — | µm² | **UNSOURCED** | — | only "drive (WL, BL) 11 %" + "Add-on: drivers 23 %" |
| `area_per_neuron_logic` | — | — | µm² | **UNSOURCED** | — | only "Add-on: sigmoid, SA, etc 29 %" |
| `area_per_tile_fixed`, `area_per_router`, `area_global_fixed` | — | — | — | **UNSOURCED / N-A** | — | PRIME has no NoC; it rides the existing DRAM-style bank/global-data-line hierarchy |
| **Die area** | — | — | mm² | **UNSOURCED** | — | never stated |
| **Technology node** | — | — | nm | **UNSOURCED for PRIME** | — | ⚠ **65 nm is the BASELINE's node, not PRIME's** — see §3.2 |

**Figure 12 transcribed in full** (read from a 260 dpi render; the vector text extracts as
mojibake). Caption: "Figure 12. Area Overhead of PRIME." The pie is normalised to the
**modified** FF mat = 100 %:

| wedge | share of modified mat |
|---|---|
| decoder (& mux) | 6 % |
| drive (WL, BL) | 11 % |
| output (SA, etc) | 15 % |
| misc (precharge, etc) | 8 % |
| *(baseline subtotal)* | *40 %* |
| **Add-on: drivers** | **23 %** |
| **Add-on: sigmoid, SA, etc** | **29 %** |
| **Add-on: contrl, etc** | **8 %** |
| **(add-on subtotal)** | **60 %** |
| **Σ** | **100 %** |

### 1.3 ENERGY

| quantity | value | evidence_kind | citation | note |
|---|---|---|---|---|
| `e_mac` | — | **UNSOURCED** | — | no absolute array energy anywhere in the paper |
| `e_adc_conversion` | — | **UNSOURCED** | — | no SA/ADC energy |
| `e_row_drive` | — | **UNSOURCED** | — | no driver energy |
| `e_neuron_update` | — | **UNSOURCED** | — | no sigmoid/ReLU energy |
| `e_inter_tile_hop`, `e_intra_tile_packet` | — | **UNSOURCED** | — | inter-bank movement rides the "internal data bus shared by all the banks in a chip" (§IV-B1); no energy given |
| `e_dma_per_byte` | — | **UNSOURCED** | — | no per-byte figure |
| **Total chip power** | — | **UNSOURCED** | — | never stated |
| Component power breakdown | — | **UNSOURCED** | — | **PRIME has no power-breakdown table.** (This is the single biggest asymmetry vs. ISAAC.) |

**What IS published, and only in relative form** — energy *ratios*, from Figs. 10 and 11:

| quantity | value | evidence_kind | citation | quote / reading |
|---|---|---|---|---|
| Energy saving vs. CPU (geomean, MlBench) | **10 834 ×** | published | `chi2016prime`, Fig. 10, p.37 | Fig. 10 gmean bar labels (pNPU-co / pNPU-pim-x64 / PRIME) = **12.1 / 52.6 / 10 834** |
| Energy saving vs. a state-of-the-art NPU | **~895 ×** | published | `chi2016prime`, Abstract | "compared with a state-of-the-art neural processing unit design, PRIME improves the performance by ∼2360× and the energy consumption by ∼895×, across the evaluated machine learning benchmarks." Cross-check: 10 834 / 12.1 = **895.4** ✓ |
| Speedup vs. CPU (geomean) | **11 800 ×** | published | `chi2016prime`, Fig. 8, p.37 | Fig. 8 gmean bars (pNPU-co / pNPU-pim-x1 / pNPU-pim-x64 / PRIME) = **5.0 / 45.3 / 2899 / 11 800**. Cross-check: 11 800 / 5.0 = **2360** ✓ matches the abstract |
| Per-benchmark energy saving vs CPU (PRIME) | CNN-1 **335**, CNN-2 **3801**, MLP-S **11 744**, MLP-M **23 922**, MLP-L **32 548**, VGG **138 984** | published | Fig. 10 in-bar labels | read from render |
| Per-benchmark speedup vs CPU (PRIME) | CNN-1 **5101**, CNN-2 **5824**, MLP-S **17 665**, MLP-M **44 043**, MLP-L **73 237**, VGG **1596** | published | Fig. 8 in-bar labels | read from render |
| PIM-vs-coprocessor speedup | 9.1 × | published | §V-B, p.37 | "By comparing the speedups of pNPU-co and pNPU-pim-x1, we find that the PIM solution has a 9.1× speedup on average over a co-processor solution." |
| Memory-energy saving of 3D-PIM baseline | 93.9 % | published | §V-C, p.37 | "pNPU-pim-x64 consumes almost the same energy in computation and buffer with pNUP-co, but saves the memory energy by 93.9 % on average" |

> ⚠ **These ratios are unusable as physics constants.** They are normalised to a CPU and to
> an NPU baseline whose own absolute power is likewise not stated in this paper. No
> per-unit energy can be back-solved from them.

### 1.4 TIME

| quantity | value | unit | evidence_kind | citation | quote |
|---|---|---|---|---|---|
| **`t_array_read` (compute mode)** | — | ns | **UNSOURCED** | — | The compute-mode crossbar integration time is never stated. §III-B only says qualitatively: "Benefiting from the massive parallelism of matrix-vector multiplication provided by ReRAM crossbar structures, the computation itself takes a very short time." |
| **`t_adc_conversion`** | — | ns | **UNSOURCED** | — | no SA conversion time |
| **`t_cycle`** | — | ns | **UNSOURCED** | — | no compute cycle time; PRIME has no stated compute clock |
| `t_hop` | — | ns | **UNSOURCED** | — | inter-bank transfer time not given |
| **Memory-mode tRCD** | **22.5** | ns | published | `chi2016prime`, Table IV, p.36 | Table IV: "tRCD-tCL-tRP-tWR 22.5-9.8-0.5-41.4 (ns)" |
| Memory-mode tCL | 9.8 | ns | published | Table IV | (same cell) |
| Memory-mode tRP | 0.5 | ns | published | Table IV | (same cell) |
| Memory-mode tWR | 41.4 | ns | published | Table IV | (same cell) |
| I/O bus frequency | 533 | MHz | published | Table IV | "16GB ReRAM; 533MHz IO bus" |
| Reconfiguration cost | **excluded from all results** | — | published | `chi2016prime`, §V-B, p.37 | "In our performance and energy evaluations of PRIME, we do not include the latency and energy consumption of configuring ReRAM for computation, because we assume that once the configuration is done, the NNs will be executed for tens of thousands times to process different input data." |

> ⚠ The four DRAM-style timings describe the **memory mode** of the ReRAM main memory model,
> not the analog compute path. Do **not** map `tRCD` onto `t_array_read`.

### 1.5 DEVICE / OPERATING POINT

| quantity | value | unit | evidence_kind | citation | quote |
|---|---|---|---|---|---|
| ReRAM device stack | Pt/TiO₂₋ₓ/Pt | — | published | `chi2016prime`, §V-A "Methodology", p.36 | "We adopt Pt/TiO2-x/Pt devices [65] with Ron/Roff = 1kΩ/20kΩ and 2V SET/RESET voltage." |
| **R_on** | **1** | kΩ | published | §V-A | (same sentence) |
| **R_off** | **20** | kΩ | published | §V-A | (same sentence) |
| On/off ratio | 20 | — | derived | 20 kΩ / 1 kΩ | **20 ×** |
| **SET/RESET voltage** | **2** | V | published | §V-A | (same sentence) |
| Read/compute voltage | — | V | **UNSOURCED** | — | only "8 levels (3-bit)" — the absolute levels are not given |
| **Supply voltage** | — | V | **UNSOURCED** | — | never stated |
| **Temperature** | — | °C | **UNSOURCED** | — | never stated |
| ReRAM endurance (cited, not PRIME-specific) | up to 10¹² cycles | — | published | §II-A, p.28 | "The reported endurance of ReRAM is up to 10¹² [21], [22], making the lifetime issue of ReRAM-based memory less concerned than PCM based main memory" |
| MLC precision tuning (cited) | 1 % ≈ 7-bit single cell; ~3 % in-array | published | §III-D, p.33 | "With a simple feedback algorithm, the resistance of a ReRAM device can be tuned with 1 % precision (equivalent to 7-bit precision) for a single cell and about 3 % for the cells in crossbar arrays [31], [65]." ⚠ a *cited capability*, not PRIME's operating point |
| Output-precision basis (cited) | 256×256 array: 4-bit weights → 6-bit output; 6-bit weights → 7-bit output | published | §III-D, p.33 | "The latest results of the Dot-Product Engine project from HP Labs reported that, for a 256×256 crossbar array, given full-precision inputs (e.g. usually 8-bit for image data), 4-bit synaptic weights can achieve 6-bit output precision, and 6-bit synaptic weights can achieve 7-bit output precision, when the impacts of noise on the computation precision of ReRAM crossbar arrays are considered [66]." **This is the justification for P_o = 6.** |
| Accuracy at reduced precision | 3-bit input + 3-bit weight → 99 % on MNIST/LeNet-5 | published | §III-D, p.33 + Fig. 6 | "for this NN application, 3-bit dynamic fixed point input precision and 3-bit dynamic fixed point synaptic weight precision are adequate to achieve 99 % classification accuracy, causing negligible accuracy loss compared with the result of floating point data format." |

### 1.6 UTILISATION (directly relevant to occupancy accounting)

| quantity | value | evidence_kind | citation | quote |
|---|---|---|---|---|
| FF-subarray utilisation, MlBench (excl. VGG-D), before / after replication | **39.8 % / 75.9 %** | published | `chi2016prime`, §V-D, p.38 | "Our experimental results on Mlbench (except VGG-D) show that the utilities of FF subarrays are 39.8 % and 75.9 % on average before and after replication, respectively." |
| FF-subarray utilisation, VGG-D, before / after replication | **53.9 % / 73.6 %** | published | §V-D, p.38 | "For VGG-D, the utilities of FF subarrays are 53.9 % and 73.6 % before and after replication, respectively." |

---

## 2. UNSOURCED list

Everything below is **UNSOURCED in `chi2016prime`** — reported as absent, never guessed.
This list is deliberately long: it is the honest shape of this paper.

**AREA (all absolute):** `area_per_cell`, `area_per_adc`, `area_per_row_driver`,
`area_per_neuron_logic`, `area_per_state_bit`, `area_per_tile_fixed`, `area_per_router`,
`area_global_fixed`, `area_per_core_total`, **die area**, **technology node for PRIME**.
*Only relative percentages exist (§1.2).*

**ENERGY (all):** `e_mac`, `e_adc_conversion`, `e_row_drive`, `e_neuron_update`,
`e_leak_per_neuron_step`, `e_intra_tile_packet`, `e_inter_tile_hop`, `e_dma_per_byte`,
`e_core_program`, `e_core_init`, `e_sync_barrier`, `e_synaptic_event_total`,
**total chip power**, **any component power breakdown**.
*Only CPU-normalised ratios exist (§1.3).*

**TIME (compute path):** `t_array_read`, `t_adc_conversion`, `t_cycle`, `t_hop`,
`t_program_per_byte`, `t_core_init`, `t_sync_barrier`.
*Only memory-mode tRCD/tCL/tRP/tWR exist (§1.4), which are NOT the compute path.*

**POWER:** `p_static_per_core`, `p_static_global`, and any dynamic power at all.

**OPERATING POINT:** supply voltage, temperature, compute clock frequency, crossbar read
voltage magnitude.

**DEVICE VARIABILITY:** `write_sigma`, `read_sigma` — no numeric variability model. The
paper's precision argument is entirely qualitative + delegated to refs [31], [65], [66].

**STRUCTURE:** **mats per subarray** — the single missing structural number, and the one that
blocks converting "128 FF subarrays" into an absolute crossbar population (§5.3 F2).

**PEAK THROUGHPUT:** peak GOPS is referred to but never printed — §V-D: "The choice of the
number of FF subarrays is a tradeoff between peak GOPS and area overhead," with no value.

---

## 3. Validity domain

### 3.1 What these numbers are, and what produced them

**PRIME was never fabricated.** From §V-A "Methodology" (p.36), verbatim:

> "We model the above NPU designs using Synopsys Design Compiler and PrimeTime with 65nm
> TSMC CMOS library. We also model ReRAM main memory and our PRIME system with modified
> NVSim [81], CACTI-3DD [82] and CACTI-IO [83]. We adopt Pt/TiO2-x/Pt devices [65] with
> Ron/Roff = 1kΩ/20kΩ and 2V SET/RESET voltage. The FF subarray is modeled by heavily
> modified NVSim, according to the peripheral circuit modifications, i.e., write driver [84],
> sigmoid [63], and sense amplifier [64] circuits. We built a trace-based in-house simulator
> to evaluate different systems, including CPU-only, PRIME, NPU co-processor, and NPU
> PIM-processor."

Circuit sources PRIME relies on: sigmoid = **B. Li et al., "RRAM-based analog approximate
computing", TCAD 2015** [63]; reconfigurable SA = **J. Li et al., "A novel reconfigurable
sensing scheme for variable level storage in phase change memory", IMW 2011** [64] (the SA is
"tested through fabrication"); write driver = **C. Xu et al., DAC 2013** [84]; ReRAM device =
**L. Gao et al., NVMW 2013** [65]; output-precision study = **M. Hu et al., Dot-Product
Engine, ICCAD'15 workshop** [66]; memory baseline = **C. Xu et al., HPCA 2015** [20].

### 3.2 ⚠ The 65 nm trap

"65nm TSMC CMOS library" appears in the same paragraph as the PRIME modelling sentence, but it
qualifies **"the above NPU designs"** — i.e. the *pNPU-co / pNPU-pim baselines*. PRIME itself
is modelled with NVSim/CACTI-3DD/CACTI-IO, whose node is **not stated**.

**Do not record 65 nm as PRIME's technology node.** It is the baseline's. If a profile needs a
node it must be declared `estimated` with this caveat attached.

### 3.3 Internal ambiguities — FLAGGED

**(a) "totally 64 subarrays" — per bank, not chip-wide.**
Literally read as chip-wide it is impossible: 3 modified subarrays/bank × 64 banks = 192 ≠ 64.
The consistent reading is *a bank contains 64 subarrays, of which 2 are FF and 1 is Buffer*.
§V-D's restatement ("two FF subarrays and one Buffer subarray per bank (64 subarrays in
total)") uses the same phrasing. **Adopted reading: 64 subarrays per bank.** Reconciliation
below supports it.

**(b) "each ReRAM mat can execute as large as a 256 − 256 NN" contradicts the composing scheme.**
With 2 adjacent bitlines per 8-bit weight (§III-D1), a 256-bitline mat holds **128** logical
8-bit weights per row, not 256 — and the ± split puts the negative half in a *second* mat. So
one 256×256 mat supports a 256 → 128 unsigned layer, and a mat *pair* supports a 256 → 128
signed layer. The paper's "256 − 256 mat" framing (§IV-B1, §V-C) is a coarse capacity
statement that ignores its own composing and ± scheme. **Both readings are recorded; the
composing-aware one is the physically correct one.**

**(c) "60 % area increase" vs. Fig. 12's 60 % wedge.**
Fig. 12's pie sums to 100 % with the add-ons at 60 %, i.e. the add-ons are **60 % of the
modified mat**, equivalent to a **150 % increase** over the unmodified mat. The literal phrase
"60 % area increase" would instead put the add-ons at 37.5 % of the final area, contradicting
the pie. **The pie is authoritative**, and the whole-chip number confirms it:

| reading | implied bank-level overhead from 2 FF subarrays of 64 | vs. published 5.76 % |
|---|---|---|
| add-on = 60 % of final (⇒ +150 %) | 1.50 × 2/64 = **4.69 %** | plausible: the 1.07 pp balance is the Buffer-subarray connection unit + PRIME controller ✓ |
| literal "+60 %" | 0.60 × 2/64 = **1.88 %** | cannot reach 5.76 % ✗ |

**(d) The LL partial product is computed but contributes zero bits.**
§III-D says R_target has four components "calculated one by one", yet §III-D1's worked
assumption keeps only three: "The target result should be the summation of three components:
all the 6 bits of R_full^HH output, the highest 3 bits of R_full^HL output, and the highest 2
bits of R_tar^LH output." The LL term takes `P_o − (P_in+P_w)/2 = 6 − 7 = −1` bits, i.e. none.
Whether the hardware *skips* the LL read or performs and discards it is not stated. **This is
a factor-4/3 ambiguity in the conversion count** — carried as a band in §4.

### 3.4 Validity envelope

| axis | value |
|---|---|
| Technology node | **not stated for PRIME** (65 nm is the baseline's) |
| Supply voltage | **not stated** |
| Temperature | **not stated** |
| Compute clock | **not stated** |
| Memory interface | 533 MHz I/O bus; tRCD/tCL/tRP/tWR = 22.5/9.8/0.5/41.4 ns |
| Device | Pt/TiO₂₋ₓ/Pt, R_on/R_off = 1 kΩ/20 kΩ, 2 V SET/RESET |
| Workload class | **Inference only**; MLP + CNN. No LRN. No on-chip training |
| Numeric format | 6-bit input / 8-bit weight / 6-bit output, dynamic fixed point (composed from 3-bit inputs and 4-bit cells) |
| Excluded from all results | ReRAM reconfiguration latency and energy (§1.4) |
| Host coupling | PRIME is main memory: a 4-core 3 GHz OoO CPU with 32 KB L1 / 2 MB L2 shares it (Table IV) and runs concurrently |

---

## 4. CONVERSION MODEL

### 4.1 The dataflow, from the paper

1. **Weights are bit-sliced across *adjacent bitlines*.** §III-D1: *"the high-bit and low-bit
   parts of the synaptic weights are stored in adjacent bitlines of the corresponding crossbar
   array."* → `cells_per_weight = weight_bits / cell_bits = 8/4 = 2` **columns per logical weight**.
2. **Inputs are sliced across *time*, 3 bits at a time (not bit-serial).** §III-D1: *"the
   high-bit and low-bit parts of the input are fed to the corresponding crossbar array
   sequentially."* → `input_passes = input_bits / dac_bits = 6/3 = 2`. Note PRIME's DAC carries
   3 bits *per pass* (8 voltage levels), unlike ISAAC's 1-bit DAC.
3. **The ± pair costs one conversion, not two.** §III-A2: subtraction is **analog and
   pre-SA** — *"their output signals are fed into the subtraction unit, and then the difference
   signal goes into the sigmoid unit. The analog output is converted to digital signal by the
   SA"*. The two mats double array area/energy but **not** the conversion count.
4. **Each bitline result needs one SA conversion**, with 8 SAs per 256-bitline mat →
   `adc_sharing_factor = 32`, i.e. **32 serial conversion rounds** to drain one mat read.
5. **Partial products: 4 exist, 3 carry bits.** §3.3(d).

### 4.2 The formula

For a logical layer computing `N` outputs from a length-`L` dot product (`macs = L × N`):

```
row_groups     = ceil(L / array_rows)
bitlines       = N * cells_per_weight                      (per polarity; ± share the SA path)
col_groups     = ceil(bitlines / array_cols)
partials       = number of (input-slice x weight-slice) products actually converted

adc_conversions = row_groups * col_groups * array_cols * partials / cells_per_weight
```

which, dropping the ceilings, collapses to exactly the same closed form as ISAAC:

```
adc_conversions = macs * partials / array_rows
                = macs * (weight_bits / cell_bits) * (input_bits / dac_bits) / array_rows
```

**For PRIME's ISCA-2016 configuration** (`array_rows = 256`, `cells_per_weight = 8/4 = 2`,
`input_passes = 6/3 = 2`):

| variant | partials | formula | per-MAC rate |
|---|---|---|---|
| **Composed 6b × 8b, LL skipped** (the paper's own arithmetic) | **3** | `adc_conversions = macs * 3 / 256` | **1 conversion per 85.3 MACs** |
| Composed 6b × 8b, all four parts read | 4 | `adc_conversions = macs * 4 / 256 = macs / 64` | 1 per 64 MACs |
| Native, uncomposed 3b input × 4b weight | 1 | `adc_conversions = macs / 256` | 1 per 256 MACs |

> **Recommended band: `macs × 3/256` … `macs × 4/256`**, i.e. **0.0117–0.0156 conversions per
> MAC**, with the generic (4-partial) form as the upper corner. The lower `macs/256` corner
> applies only if a workload is deployed at PRIME's *native* 3b/4b precision without composing.

Converter population:

```
adc_count = arrays * ceil(array_cols / adc_sharing_factor)
          = arrays * ceil(256 / 32) = arrays * 8              ✓ "eight 6-bit reconfigurable SAs" per mat
```

### 4.3 Reasoning

The two bit-slicing factors enter for the same reason as in ISAAC (§4 of the ISAAC report) —
one logical weight spans several *physically separate, separately converted* columns, and one
logical input spans several *sequential* array reads. What differs is the magnitude:

| | ISAAC-CE | PRIME |
|---|---|---|
| `array_rows` | 128 | 256 |
| `cells_per_weight` | 8 (16 b ÷ 2 b) | 2 (8 b ÷ 4 b) |
| `input_passes` | 16 (16 b ÷ 1 b) | 2 (6 b ÷ 3 b) |
| partials | 128 | 4 (3 useful) |
| **conversions per MAC** | **1.0** | **0.0117 – 0.0156** |
| `adc_sharing_factor` | 128 | 32 |

PRIME needs ~64–85× fewer conversions per MAC than ISAAC. The mechanism is explicit in the
design: PRIME buys it with a **multi-bit DAC** (3 b vs. ISAAC's 1 b), **denser cells** (4 b vs.
2 b), a **2× larger array**, and much **lower deployed precision** (6 b/8 b vs. 16 b/16 b). It
pays for it in accuracy, exactly as ISAAC's related-work section observes:

> `shafiee2016isaac` §IX, p.11: *"While PRIME can support positive and negative synapses,
> input vectors must be unsigned. Also, the dot-product computations in PRIME are lossy
> because the precision of the ADC does not always match the precision of the computed
> dot-product."*

⚠ **Unverifiable.** Unlike ISAAC — where the model was checked against Table I's ADC power to
1 % (see the ISAAC report §3.4) — **PRIME publishes no power or energy breakdown, so this
conversion model cannot be numerically validated against the paper.** It rests entirely on
the structural quotes in §4.1. Record it as `modeled`, and say so.

### 4.4 ⚠ Disagreement with the in-repo model — see §5.3

The repo's `bit_sliced_crossbar` model computes
`adc_conversions = ceil(macs / array_rows) * input_bits`. For PRIME this gives
`macs/256 × 6 = macs × 6/256` — wrong by using `input_bits = 6` where the physical count is
`input_passes = input_bits/dac_bits = 2`, and by omitting `cells_per_weight = 2`. Net: it
over-counts by 1.5× relative to the 4-partial form (6 vs 4), and by 2× relative to the paper's
3-partial arithmetic. `adc_count`, by contrast, is **correct**.

---

## 5. In-repo cross-check

### 5.1 Files carrying PRIME constants

| file | what it holds |
|---|---|
| `src/mimarsinan/mapping/platform/imc_platforms_literature.py` (lines 125–136) | `prime_mat_256x256` IMCPlatform geometry + provenance quote |
| `src/mimarsinan/deployment_record/platform_physics/conversion.py` (lines 96–103) | `bit_sliced_crossbar` docstring naming PRIME as one of its two motivating targets |
| `/home/yigit/backups/se_review_copy/papers/structured_elimination_aaai/research_artifacts/13_chip_geometries.json` | the extraction card the registry entry is transcribed from (outside this repo) |
| `src/mimarsinan/deployment_record/platform_physics/profiles/` | **no `prime.json` exists yet** |

### 5.2 Value-by-value cross-check of `prime_mat_256x256`

| repo field | repo value | paper value | verdict |
|---|---|---|---|
| `max_axons` (rows) | 256 | 256 wordlines | ✅ AGREES |
| `max_neurons` (cols) | 256 | 256 bitlines *physical*; **128** logical 8-bit weights | ⚠ **finding F1** |
| `count` | 128 | 2 FF subarrays × 8 banks × 8 chips = 128 **FF subarrays** | ⚠ arithmetic correct, but the unit is a *subarray*, not a *mat* — **finding F2** |
| `weight_bits` | **4** | **8** ("the weights are 8-bit"); card contract defines the field as `bits_per_cell × cells_per_weight` = 4 × 2 = 8 | ❌ **DISAGREES — finding F3** |
| `has_bias` | False | PRIME **does** implement bias in-array: "We also write b_i in ReRAM cells, and regard the corresponding input as '1'" (§III-E) | ❌ **DISAGREES — finding F4** |
| provenance quote | the §V-A "PRIME Configurations" sentence | **verbatim match**, including the trailing MLC/SLC and voltage-level clauses | ✅ quote is exact |
| provenance location | "Section V.A 'PRIME Configurations', p.36" | §V-A is "Experiment Setup"; the "PRIME Configurations" bolded run-in heading is inside it, on printed **p.36** | ✅ AGREES |
| `claim_eligibility` | `curve-only` | — | ✅ appropriate: simulated architecture, and (unlike ISAAC) with no absolute constants at all |

### 5.3 Findings

**F1 — `max_neurons = 256` over-counts logical outputs by 2×.**
A 256-bitline mat holds 128 logical 8-bit weights, because composing puts each weight on two
adjacent bitlines (§4.1). Additionally, a *signed* layer needs a second mat for the negative
half. So the true logical capacity of one mat is 256 → 128 unsigned, and of a mat pair
256 → 128 signed. The card's 256 → 256 matches the paper's own coarse "256 − 256 mat" phrasing
(§3.3(b)) but not its composing scheme. Same class of issue as ISAAC's F1, at 2× rather than 8×.

**F2 — `count = 128` counts FF *subarrays*, not mats.**
The arithmetic (2 × 8 × 8) is right, but the unit is wrong: a subarray contains **multiple
mats**, and the paper never says how many (§2). So 128 is a *lower bound* on the 256×256
crossbar population, probably by a large factor. The existing card note already concedes this
("mats-per-subarray not stated in the card, so population is approximate — treat as our
choice"); recording here that the paper genuinely does not contain the number, so no amount of
re-reading will fix it. The SEAL-lab technical report (ref [75]) is the place to look.

**F3 — ❌ `weight_bits = 4` contradicts both the paper and the card's own contract.**
The card's contract states `weight_bits` = "deployed weight precision on this platform
(bits_per_cell × cells_per_weight)". PRIME: 4-bit MLC × 2 cells = **8**. The paper is explicit:
*"With our input and synapse composing scheme, for computation, the input and output are 6-bit
dynamic fixed point, and the weights are 8-bit."* The current value 4 is the **bits per cell**,
which is a different quantity. Compare ISAAC's entry, which correctly records 16 (= 2 × 8), so
the two entries are presently inconsistent *with each other* as well as with the contract.
**Recommended: `weight_bits = 8`.**

**F4 — `has_bias = False` contradicts an explicit statement.**
§III-E, convolution layer: *"We also write b_i in ReRAM cells, and regard the corresponding
input as '1'."* PRIME implements bias as an extra in-array row driven at logical 1 — the
classic bias trick, and exactly what `has_bias=True` denotes. (Contrast ISAAC, where `False`
is defensible because the extra column is a *unit* column for encoding correction rather than
a per-neuron bias.) **Recommended: `has_bias = True`**, noting it consumes one wordline.

**F5 — ⚠ `bit_sliced_crossbar` mis-counts PRIME conversions (and its docstring over-claims).**
`conversion.py:59-62` computes `ceil(macs/array_rows) * input_bits`. Its docstring (line 99–100)
asserts this is *"The dataflow ISAAC and PRIME describe"*. For PRIME it is not: `input_bits`
must be `input_bits / dac_bits` (PRIME's DAC is 3-bit, so 2 passes not 6), and
`cells_per_weight` is missing. The model returns `macs × 6/256` where the correct value is
`macs × 3/256` (paper's arithmetic) to `macs × 4/256` (all four partials) — a 1.5–2× over-count.
Combined with the 8× *under*-count for ISAAC (see the ISAAC report, F4), the same formula errs
in **opposite directions** on the two targets it names, which is the clearest evidence that the
missing terms are real.

*Remedies (no `src/` change is in scope here, so recorded not applied):*
- **Preferred:** parameterise the model by `cells_per_weight` and `dac_bits` and use
  `adc_conversions = ceil(macs/array_rows) * cells_per_weight * (input_bits/dac_bits)`.
  This makes both targets exact and keeps the model's name honest.
- **Declaration-level workaround with the model as it stands:** declare `input_bits = 4`
  (= `cells_per_weight × input_passes` = 2 × 2) for PRIME, and `input_bits = 128` (= 8 × 16)
  for ISAAC. Arithmetically correct; semantically a lie about the parameter's name, so it
  needs a comment if used.

**F6 — `adc_count` in the repo model is CORRECT for PRIME.**
`arrays * ceil(array_cols / adc_sharing_factor)` = `arrays * ceil(256/32)` = **8 SAs per mat**,
matching "eight 6-bit reconfigurable SAs". Declare `adc_sharing_factor = 32`. No change needed.

**F7 — no PRIME physics profile exists, and one cannot honestly be built from this paper.**
`platform_physics/profiles/` has no `prime.json`. If one is created, every AREA/ENERGY/TIME
entry would have to be `estimated` with an external source named, or omitted. The only rows
this paper can populate at `published` confidence are **structural** ones plus
`conductance_levels = 16` and `adc_sharing_factor = 32`. A profile that silently borrowed
ISAAC's or a generic 22 nm node's energies and labelled them PRIME would be unsupportable.

---

## 6. Citations (not yet injected)

Injection into a real `.bib` is **deferred**. The only `.bib` in this repo is
`sana_fe/references.bib`, which belongs to the vendored SANA-FE tree and must stay untouched;
no other `.bib` exists here and none was created.

The BibTeX below was **generated by the `asta-papers` tooling** (`semantic_scholar_search` →
`inject_papers_to_bib` into a scratchpad file, then copied verbatim). It was not hand-written.
When a real bibliography exists, re-run `inject_papers_to_bib` with ids `chi2016prime` (and
`chi2023retrospectiveprime` if the retrospective is wanted) rather than pasting this block.

```bibtex
@Article{chi2016prime,
 author = {Ping Chi and Shuangchen Li and Cong Xu and Tao Zhang and Jishen Zhao and Yongpan Liu and Yu Wang and Yuan Xie},
 booktitle = {International Symposium on Computer Architecture},
 journal = {2016 ACM/IEEE 43rd Annual International Symposium on Computer Architecture (ISCA)},
 pages = {27-39},
 title = {PRIME: A Novel Processing-in-Memory Architecture for Neural Network Computation in ReRAM-Based Main Memory},
 year = {2016},
 doi = {10.1145/3007787.3001140},
 url = {https://doi.org/10.1145/3007787.3001140},
}

@Inproceedings{chi2023retrospectiveprime,
 author = {Ping Chi and Shuangchen Li and Conglei Xu and Zhang Tao and Jishen Zhao and Yongpan Liu and Yu Wang and Yuan Xie},
 title = {RETROSPECTIVE:PRIME: A Novel Processing-in-memory Architecture for Neural Network Computation in ReRAM-based Main Memory},
 year = {2023}
}
```

Cached ids for re-injection: `chi2016prime`, `chi2023retrospectiveprime`.

**Not cached / not retrieved** — the most promising lead for PRIME's missing absolutes:
P. Chi et al., *"Processing-in-memory in ReRAM-based main memory"*, SEAL-lab Technical Report
no. 2015-001, 2015 (the paper's own ref [75], cited for "a detailed NN mapping example").
Last known URL `https://seal.ece.ucsb.edu/sites/default/files/publications/prime_tr_main_2.pdf`
— unreachable from this environment (DNS failure), not paywalled as far as could be determined.

**Secondary circuit sources** cited *by* PRIME for its modified periphery (not cached; listed
so the model provenance chain is traceable):
ref [63] B. Li et al., *RRAM-based analog approximate computing*, TCAD 34(12):1905–1917, 2015 (sigmoid);
ref [64] J. Li et al., *A novel reconfigurable sensing scheme for variable level storage in phase change memory*, IMW 2011 (the fabricated reconfigurable SA);
ref [65] L. Gao et al., *A high resolution nonvolatile analog memory ionic devices*, NVMW 2013 (the Pt/TiO₂₋ₓ/Pt device);
ref [66] M. Hu et al., *Dot-product engine: Programming memristor crossbar arrays for efficient vector-matrix multiplication*, ICCAD'15 workshop (the 256×256 output-precision result);
ref [81] X. Dong et al., *NVSim*, TCAD 31(7):994–1007, 2012;
ref [82] K. Chen et al., *CACTI-3DD*, DATE 2012;
ref [83] N. P. Jouppi et al., *CACTI-IO*, ICCAD 2012;
ref [84] C. Xu et al., *Understanding the trade-offs in multi-level cell ReRAM memory design*, DAC 2013 (write driver);
ref [20] C. Xu et al., *Overcoming the challenges of crossbar resistive memory architectures*, HPCA 2015 (the ReRAM main-memory baseline).
