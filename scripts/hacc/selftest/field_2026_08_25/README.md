# The 2026-08-25 hacchead transcripts, as the stubs replay them

These files are what the local gate harness answers `sinfo`, `scontrol`,
`groups` and `sbatch` with. They carry the facts observed in the live session on
**2026-08-25**, in slurm's own output shapes, so `run_all.sh`'s partition
auto-pick is exercised against the cluster it will actually meet:

| Fact | File |
|---|---|
| `cpu_only` maps to **hacc-gpu0**, which is DOWN (`NO NETWORK ADDRESS FOUND`) — the docs' `hacc-node2` is stale | `sinfo_a_N.txt`, `scontrol_cpu_only.txt` |
| `vck5000_compile`: `AllowGroups=ALL`, `MaxTime=7-00:00:00`, **hacc-node0 idle** — the living build venue | `scontrol_vck5000_compile.txt` |
| the account is in `yigit video render hgpu gpgpu fpga_u280 fpga_u250 fpga_vck5000` — **not** `fpga_u55c`, **not** `lab` | `groups.txt` |
| the U55C board partition allows `lab,hgpu,fpga_u55c` and is reachable **through `hgpu`**, 1 h cap | `scontrol_xilinx_u55c_gen3x16_xdma_3_202210_1.txt` |
| `mi210_vck_u55c` is absent from `scontrol show partition` and refuses submission | `scontrol_missing.txt`, `sbatch_not_permitted.txt` |
| `mi210_u280_u55c` allows `lab,hgpu,fpga_u280`, 12 h (5 d on `_long_reservation`), on hacc-gpu1 = MI210 + U280 + U55C | `scontrol_mi210_u280_u55c*.txt` |

PROVENANCE, HONESTLY. The observations are the field session's; the byte-level
formatting of these tables was reconstructed into stock slurm output shapes
(`sinfo -a -N`'s four columns, `scontrol show partition`'s key=value dump)
because the session was relayed as findings rather than as captured files. What
the harness proves is therefore the DECISION — which partition run_all.sh picks
from these facts and why — not slurm's exact column widths.
