# The 2026-08-25 hacchead transcripts, as the stubs replay them

These files are what the local gate harness answers `sinfo`, `scontrol`,
`groups` and `sbatch` with, and what it builds the stub `/tools` and
`/opt/xilinx/platforms` trees from. They carry the facts observed in the live
sessions on **2026-08-25**, in slurm's and Vitis's own output shapes, so
`run_all.sh`'s partition auto-pick and `cards.sh`'s card gate are exercised
against the cluster they will actually meet.

## Session 1 — the partitions

| Fact | File |
|---|---|
| `cpu_only` maps to **hacc-gpu0**, which is DOWN (`NO NETWORK ADDRESS FOUND`) — the docs' `hacc-node2` is stale | `sinfo_a_N.txt`, `scontrol_cpu_only.txt` |
| `vck5000_compile`: `AllowGroups=ALL`, `MaxTime=7-00:00:00`, **hacc-node0 idle** — the living build venue | `scontrol_vck5000_compile.txt` |
| the account is in `yigit video render hgpu gpgpu fpga_u280 fpga_u250 fpga_vck5000` — **not** `fpga_u55c`, **not** `lab` | `groups.txt` |
| the U55C board partition allows `lab,hgpu,fpga_u55c` and is reachable **through `hgpu`**, 1 h cap | `scontrol_xilinx_u55c_gen3x16_xdma_3_202210_1.txt` |
| `mi210_vck_u55c` is absent from `scontrol show partition` and refuses submission | `scontrol_missing.txt`, `sbatch_not_permitted.txt` |
| `mi210_u280_u55c` allows `lab,hgpu,fpga_u280`, 12 h (5 d on `_long_reservation`), on hacc-gpu1 = MI210 + U280 + U55C | `scontrol_mi210_u280_u55c*.txt` |

## Session 2 (2026-08-25, later) — the U55C deadlock and the U250 route

| Fact | File |
|---|---|
| The ONLY U55C platform anywhere — hacc-node0 **and** hacc-gpu3 — is `xilinx_u55c_gen3x16_xdma_3_202210_1`. The build node also carries the two U250 shells. | `platforms_installed.txt` |
| Vitis installed: `2020.1 2020.2 2021.2 2022.1 2023.2 2024.2`. There is **no 2022.2**. | `vitis_installed.txt` |
| The U55C shell's SmartConnect IP is locked to the 2022.2 release; `vpl` refused under Vitis **2022.1, 2023.2 and 2024.2** — three separate logs, same VPL 60-704 / 60-732 pair. **U55C is deadlocked pending admins.** | `vpl_60_704_u55c.txt` |
| hacc-gpu3 answered `0 devices found` under `xbutil` from a NON-FPGA partition: the cards are VM-passed at allocation time, so only a board-partition job sees one. This confirms doc/1's board-VM model in the field. | `xbutil_no_devices_non_board_partition.txt` |
| `xilinx_u250_gen3x16_xdma_3_1_202020_1`: `AllowGroups=lab,fpga_u250`, 1 h, six idle nodes — and the account **is** in `fpga_u250`. Its 202020 shell pairs with Vitis **2020.2**, which is installed. | `scontrol_xilinx_u250_gen3x16_xdma_3_1_202020_1.txt` |
| `u250_standard_reservation_pool` 12 h, `u250_long_reservation_pool` 2 d, same `lab,fpga_u250` | `scontrol_u250_*_reservation_pool.txt` |
| `xilinx_u250_gen3x16_xdma_4_1_202210_1` exists on the build node but is a 2022.2-era shell and would hit the SAME lock — `cards.sh` refuses it by name and says why | `platforms_installed.txt` |

PROVENANCE, HONESTLY. The observations are the field sessions'; the byte-level
formatting of these tables was reconstructed into stock output shapes
(`sinfo -a -N`'s four columns, `scontrol show partition`'s key=value dump,
`xbutil examine`'s section headings, VPL's error wording) because the sessions
were relayed as findings rather than as captured files. What the harness proves
is therefore the DECISION — which partition and which card/toolchain pairing the
package picks from these facts, and which it refuses — not the exact column
widths.

Two reconstructions are worth naming individually, because they are the only
places where a hostname or a count was filled in rather than observed:

* **U250 node names and the six-node roster.** The field reported "1 h, 6 nodes
  idle" for `xilinx_u250_gen3x16_xdma_3_1_202020_1`. The vendor doc
  (`hacc_demo/doc/1-FPGA-allocation.md`) names `hacc-u250-[1-4]` plus
  `hacc-u250-frp`, i.e. five. The fixture extends the range to
  `hacc-u250-[1-6]` to match the observed count and keeps the doc's rosters for
  the two pools. Nothing in the pick depends on a hostname — only on group
  admission and on at least one node not being down.
* **The VPL excerpt.** The three logs were relayed as the error pair and the
  "customized with software release 2022.1 ... different revision" wording; the
  surrounding lines are stock VPL framing. It is evidence of the *reason*, and
  the card gate never parses it — the gate reads the installed platform list and
  the installed Vitis list, which are facts a machine can re-check.
