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
| `p_host` | 5.68 / 11.01 / 15 W | published (a preprint: departure 1) | cao2026vlaulap Table 4: module VDD_IN (CPU, GPU, DRAM) of the Super developer kit at its 15 W mode under an inference workload, 11.01 W; low the measured idle floor; high the mode's 15 W budget (Jetson Linux r36.5) |
| `p_host_idle` | 5.68 / 5.685 / 5.69 W | published (a preprint: departure 1) | cao2026vlaulap Table 4: resident-model idle of the Super developer kit at its 15 W mode (5.69 W, 5.68 W) |
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

1. `p_host` and `p_host_idle` (about 99% of the main reading's energy) come from cao2026vlaulap, an arXiv preprint
   (arXiv:2609.18663v2; every page prints "Preprint"). No peer-reviewed or vendor measurement of this module's active
   and idle power in a stated mode was reachable: tek2026measurement and azadi2025towards refuse this host, and
   gamage2026realtime (Neurocomputing 2026, Table 6) prints the 8 GB developer kit's total power 6.01-6.71 W and
   dynamic power 2.09-2.55 W in an unstated mode with no idle row, so it is not used.
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

## Composition and Limits

- The pricer (`cost/host_terms.py`): host time = host MACs at the rate + one dispatch per host operator + a MAC-free
  operator's bytes at the memory time; the host draws `p_host` over that time and `p_host_idle` over the rest of the
  inference (waiting on the chip and the link); every crossing byte costs `e_link_per_byte` + `e_host_mem_per_byte`
  and the slower of `t_link_per_byte` and `t_host_mem_per_byte`, each transfer `t_link_transfer`.
- Upper-side or unmodelled, stated: the active power was measured with the GPU active (a CPU-only operator draws
  less); the MAC rate is the core's peak issue rate; a polling host draws more than idle; the transfer latency was
  measured on Xeon root complexes; the PHY energy is a Gen4 transmitter's, read at Gen3, with the receiver end at the
  transmitter's figure; idle link power is inside the module's idle power only.
