# NVIDIA Jetson Orin Nano 8GB as the Deployment Host

An edge host for a host-attached chip: the module runs the program's host operators on its CPU and moves every
host-chip crossing over its PCIe root port. Every constant is published or derived from published numbers; none is
measured on the machine that runs the toolchain. Classes follow the vocabulary (published, derived; no estimate is
needed). Four rows depart from the source rule (published documentation or peer-reviewed measurements); each
departure is named below, in the description's `departures` and in its row's note. The sources are listed with their
files and checksums in the research record's fetch (`paper_pdfs/fetch_20261028_host/SOURCES.md`).

## Module and Modes

- CPU: 6-core Arm Cortex-A78AE v8.2, 1.5 MB L2 + 4 MB L3. Memory: 8 GB 128-bit LPDDR5. Accelerator attach: PCIe
  Gen3 x4 root port (the module: 1 x4 + 3 x1, PCIe Gen3, root port and endpoint). Source: NVIDIA's Jetson Orin
  technical specifications.
- Power modes (NVIDIA Jetson Linux Developer Guide r36.5, "Supported Modes and Power Efficiency", table "Jetson Orin
  Nano 8GB"): 15W, the documented default (mode ID 0): 6 CPUs online at 1510.4 MHz, memory at 2133 MHz; 7W (mode ID
  1): 4 CPUs at 960 MHz, memory at 2133 MHz.
- Mode rule: `default` takes the documented default mode (15W); a mode's name takes that mode. The 7W mode has no
  published power measurement, so its block refuses `p_host` and `p_host_idle` by name.

## Constants

| row | 15W: low / nominal / high | class | source and derivation |
|---|---|---|---|
| `p_host` | 4.0 / 11.01 / 15 W | published (nominal a preprint: departure 1) | nominal cao2026vlaulap Table 4: module VDD_IN (CPU, GPU, DRAM) of the Super developer kit at its 15 W mode under an inference workload, 11.01 W; low the idle floor's low end (peer-reviewed, tsuyuki2026efficient Table 5); high the vendor's average-power budget of the mode, "VDD_IN Average Power: 15W" (Jetson Linux r36.5, TDP budget table) |
| `p_host_idle` | 4.0 / 5.685 / 5.69 W | published (nominal and high a preprint: departure 1) | low tsuyuki2026efficient (ICASSP 2026) Table 5: idle of a Jetson Orin Nano at 15 W power mode at the board's supply input, 4.0 W; nominal and high cao2026vlaulap Table 4: resident-model idle of the Super developer kit at its 15 W mode at VDD_IN (5.69 W, 5.68 W) |
| `host_macs_per_s` | 12.0832 / 12.0832 / 72.4992 G/s | derived | Cortex-A78AE SWOG Issue 6.0 sec. 3.17: FMLA 2 per cycle x 4 FP32 lanes = 8 MACs per cycle; x 1510.4 MHz for one core (the host runs one operator at a time), x 6 cores for the high rate |
| `t_host_op_overhead` | 0.273106 / 0.273106 / 112.84 us | derived | nachin2026executorch Table 1: cycles per operator of a two-operator model, ExecuTorch (75 + 360 + 390) / 2 and the PyTorch Mobile interpreter (324,399 + 7,976 + 8,493) / 2, over 1510.4 MHz |
| `t_host_mem_per_byte` | 0.0146507 / 0.0189515 / 0.0189515 ns | derived (departure 2) | 16 B x 2 x 2133 MHz = 68.256 GB/s peak; sustained at Orin AGX's measured fraction at 2.1 GHz, 103.9 / (204.8 x 2.1 / 3.2) (prashanthis2025pagoda p. 22) |
| `e_host_mem_per_byte` | 141.4 / 174.7 / 174.7 pJ | derived, a cross-module proxy (departure 2) | prashanthis2025pagoda p. 25: energy per memory operation on Orin AGX's LPDDR5, 174.7 pJ/byte at 2.1 GHz (this module's memory clock), 141.4 at 3.2 GHz, read as this module's |
| `e_link_per_byte` | 17.8571 / 18.6567 / 40.5117 pJ | derived | bae2021a: 1.93 pJ per line bit of a PCIe Gen4 PHY transmitter; line bits per payload byte 8 x 130/128 over the TLP share 57.88/62.96 and MPS / (MPS + 24 B) (neugebauer2018understanding); the host's end at MPS 512 and 256, both ends at MPS 128 |
| `t_link_per_byte` | 0.289137 / 0.302084 / 0.327977 ns | derived | PCIe Gen3 x4: 8 GT/s x 128/130 x 4 lanes, x the TLP share, x MPS / (MPS + 24 B) for MPS 512, 256, 128 (neugebauer2018understanding pp. 3-4) |
| `t_link_transfer` | 0.529 / 0.9 / 1.8 us | derived | neugebauer2018understanding: median 128 B DMA read 529 ns; PCIe share of a 128 B round trip about 900 ns; two round trips (descriptor, then payload) |

The 7W mode's compute rows follow the same derivations at 960 MHz and 4 cores (7.68 / 7.68 / 30.72 G/s; 0.429688 /
0.429688 / 177.535 us); its memory row equals the 15W mode's (same memory clock).

## Departures from the Source Rule

The owner's rule of 2026-10-28 asks for constants from the module's published documentation or peer-reviewed
measurements. Two departures remain, named here, in `departures` and in each row's note:

1. `p_host` and `p_host_idle`: each end rests on its own source (U-J2, 2026-10-29).
   - Vendor: the mode's average-power budget, "VDD_IN Average Power: 15W" (OC2; "VDD_IN Instantaneous Power: 20W",
     OC3), Jetson Linux r36.5, TDP budget table, row "Jetson Orin Nano 8GB": the high end of `p_host`. It bounds the
     module's average power while it works and states no idle or active figure.
   - Peer-reviewed: tsuyuki2026efficient (ICASSP 2026, read in its arXiv version 2603.26145v1), Table 5: a Jetson
     Orin Nano at 15 W power mode, power at the board's supply input: idle 4.0 W, the low end of `p_host_idle` and of
     `p_host`. Its active readings (ResNet 19.9 W, MobileViT-XXS 14.0 W) are the board's under GPU inference, above
     the module's 15 W budget, so they are not the module's active power.
   - Preprint, class stated: the nominals (11.01 W, 5.685 W) and `p_host_idle`'s high end (5.69 W) stay
     cao2026vlaulap's, an arXiv preprint (arXiv:2609.18663v2; every page prints "Preprint"). The deployment's work is
     a difference of two readings (`p_host` - `p_host_idle`), and only the preprint states both in one setup (module
     VDD_IN, the 15 W mode), so the nominal pair is never mixed across measurement boundaries. The work's band ends
     are the cost ends of the difference (U-J2 targeted pass, M1), each a bound across these sources and never one
     setup: its high end `p_host`'s high (the vendor's budget, 15 W) less `p_host_idle`'s low (the peer-reviewed
     idle, 4.0 W), 11 W; its low end `p_host`'s low less `p_host_idle`'s high, never below zero (0 W). No vendor or
     peer-reviewed figure of the module's own active power in this mode was reachable: the module data sheet
     (DS-11105-001 v1.7) timed out at its distributor's host; tek2026measurement, szegedi2026a and
     giedra2026benchmarking refuse this host; gamage2026realtime (Neurocomputing 2026, Table 6) prints total
     6.01-6.71 W and dynamic 2.09-2.55 W in an unstated mode, so it is not used.
   - The measurement is on the Jetson Orin Nano 8GB Super developer kit at its 15 W mode. Jetson Linux r36.5, table
     "Jetson Orin Nano 8GB Super": that 15W mode runs 6 CPUs at 1497.6 MHz, the GPU at 612 MHz and the memory at
     2133 MHz, and the Super configuration's default mode is 25W.
   - This description's mode facts and default rule are the module's own (table "Jetson Orin Nano 8GB": 15W, the
     documented default, 6 CPUs at 1510.4 MHz, GPU 624.75 MHz, memory 2133 MHz). The power rows are read as that
     mode's: same memory clock and CPU count, CPU 0.85% and GPU 2.1% faster.
2. `e_host_mem_per_byte` and `t_host_mem_per_byte` read prashanthis2025pagoda in its arXiv version
   (arXiv:2509.20189v1). Its peer-reviewed version (k2026pagoda, ICPP 2026, doi 10.1145/3832810.3832912) is paywalled
   and was not read, so the numbers are not checked against it. The measurement is of Jetson Orin AGX's LPDDR5, read
   as this module's: the energy row is therefore classed derived, a cross-module proxy.

## Energy Attribution

Owner answer of 2026-10-29, "Host's work only (Recommended)": the main energy charges what the host spends because of
the deployment. The description states it per energy row (`energy_attribution`; every block row carries its
`energy_plane`): `p_host`, `e_link_per_byte` and `e_host_mem_per_byte` are attributable; `p_host_idle` is
system-level, the module's own draw whether or not the deployment runs. `p_host` is module power (VDD_IN) with the
idle draw inside it, so the idle power is a floor: the work is charged above it, (`p_host` - `p_host_idle`) x host
time, at the cost ends of the difference (0 / 5.325 / 11 W above the floor: the high end the budget less the lowest
idle, the low end never below zero, the nominal one setup's; corner by corner the high end read 9.31 W). The floor
under the work and over the wait goes to the whole-system line (`energy_system_mj`), printed beside the main energy
and never in it; the system line is the chip plus the whole host at its own corners (`p_host` over the host's time,
`p_host_idle` over the wait, the link and host memory). Host time always adds to latency.

## Composition and Limits

- The pricer (`cost/host_terms.py`): host time = host MACs at the rate + one dispatch per host operator + a MAC-free
  operator's bytes at the memory time; the host draws `p_host` over that time and `p_host_idle` over the rest of the
  inference (waiting on the chip and the link); every crossing byte costs `e_link_per_byte` + `e_host_mem_per_byte`
  and the slower of `t_link_per_byte` and `t_host_mem_per_byte`, each transfer `t_link_transfer`. The main energy
  takes the attributable terms: the work above the idle floor, the link and host memory per byte, each its own term
  (`energy_host_work_mj`, `energy_host_link_mj`, `energy_host_mem_mj`); the idle floor under the work and over the
  wait (`energy_host_floor_mj`, `energy_host_idle_mj`) are the system-level terms, and `energy_host_system_mj` is
  the whole host at its corners (at the nominal, the attributable host plus those two).
- Upper-side or unmodelled, stated: the active power was measured with the GPU active (a CPU-only operator draws
  less); the MAC rate is the core's peak issue rate; a polling host draws more than idle; the transfer latency was
  measured on Xeon root complexes; the PHY energy is a Gen4 transmitter's, read at Gen3, with the receiver end at the
  transmitter's figure; idle link power is inside the module's idle power only.
