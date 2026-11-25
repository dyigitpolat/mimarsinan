# Clock Records of Read-Only Probes (Fixtures of the SCK Clock-Definition Check)

Each folder is the record set a read-only Vivado 2022.2.2 job wrote on the cluster's compile partition after
opening S1-NC1's routed checkpoint (the S1-NC1 build, NC 1, `prj.runs/impl_1/level0_wrapper_routed.dcp`, sha256
`04dfee1b3184d03464d5d4b3d5bee89783d47d0573e2d5bf572c8064089a736c` before and after each job) and, except the own-build folder, reading one SPI
clock-domain text with `read_xdc -cells` on the kernel top, in memory only (nothing placed, routed or saved). The
probe is U-SP4's `usp4_probe.tcl` (unit records `11_wave2/final_units/usp4/`; the rev3 probe ran its revision `9d3fb8ba`, the fix probe its revision `82e8aaa6`, which adds diagnostic lines to `probe.txt` only). Files copied unchanged, except
`timing_summary.head.rpt`, the summary's part before "| Timing Details" (`head -n <that line - 2>`), and except that the build
host's name in Vivado's headers reads `build-node` and the per-user data directory in the recorded paths reads `/data/user/`.

| folder | job | text | what it shows |
|---|---|---|---|
| `s1nc1_rev3_probe` | the rev3 probe | `spi_fix` revision 3, as built (`build_xclbn.sh` `9df0e01e`, NC 1 text `cadca689`) | SOURCE `sck_reg/C`, whose fan-in holds `core_en_q_reg` and the ULP reset register; the feedback's setup launch clock path through the reset register, -1.573 ns at SCD 13.862 ns, the build's own figures |
| `s1nc1_fix_probe` | the fix probe | U-SP4's text (NC 1 text `74d4e37c`, the one `build_xclbn.sh` `22c68ef8` writes) | SOURCE the gate LUT3's I0 on the global clock net; feedback setup +5.077 ns at SCD 7.741 ns, hold +2.526 ns; no register in SOURCE's fan-in; `generated_clocks (0)` |
| `s1nc1_own_probe` | the own-build probe | none read: U-SP4's targeted pass, `usp4_probe.tcl` `b3ec7240` in mode own on a sha256-checked copy of the checkpoint; `probe_mode.txt` names the build's packaged `hw_nc1/odin_spi_domain.xdc` (`cadca689`, revision 3's NC 1 text), resolved only | the build's OWN clock as the checkpoint restored it ("Restoring constraints from binary archive", no `[Constraints 18-619]`): no FILE_NAME, no LINE_NUMBER, SOURCE `sck_reg/C`, records otherwise equal to the control's; red by D-1 exactly as the rev3 probe |

The own-build fixture also keeps `probe_mode.txt` and the job's `vivado.log` as `vivado.log.gz` (`gzip -9 -n`;
the tree ignores `*.log`; uncompressed sha256 `297e031fd06df4d2f24d171df695eb3889cc16ec2a1a2cf79c2cbc7357b64056`), the two records
`check_sck_clock_definition.py --own-build` reads beside the clock records).
