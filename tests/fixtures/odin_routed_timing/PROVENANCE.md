# Routed Timing Summaries of ODIN Builds (Fixtures of the SPI Timing Gate)

Each file is the part of a Vivado routed timing summary before its "Timing Details" section (header,
`check_timing`, Design Timing Summary, Clock Summary, Intra and Inter Clock Tables, Other Path Groups),
cut byte for byte with `head -n <line before "| Timing Details">` on the build host.

| fixture | build | source report (sha256 of the whole report, lines) |
|---|---|---|
| `b71035_A_impl_1.summary.rpt` | route D revision 5, candidate A (Slurm 71035, `impl_1`, 7 units, NC 1, UUID `8c8ff5ce`) | `prj.runs/impl_1/hw_bb_locked_timing_summary_routed.rpt`, `1d65b044943baffe2515ac31b132c90be15f62b6b416738afd008872b1d9b487`, 101,012 lines |
| `b71018_stage1_nc1.summary.rpt` | multi-core Stage 1 fix, NC 1 (Slurm 71018, 1 unit, UUID `458da03b`) | `prj.runs/impl_1/hw_bb_locked_timing_summary_routed.rpt`, `390db98ed95d2834cb74d39fea80723d5e012cb97c7dcbd4da53c94e06327a9d`, 101,000 lines |

Both were built without any constraint on the SPI clock: `check_timing` lists 381 unclocked register pins and
790 unconstrained endpoints per core, rooted at `.../core_gen[0].spi_i/sck_reg/Q`.

The two `*_spi_xdc_in_memory.summary.rpt` files are the same cut of a `report_timing_summary -max_paths 10
-warn_on_violation` that a read-only Vivado job wrote after opening the same routed checkpoint and reading the
SPI clock-domain XDC that `scripts/hacc/build_xclbn.sh` writes at NC 1 (sha256
`8ed9f91189c9a3c7a964ace7026835891b63153994307a6b103748c3e1cccb5f`) with `read_xdc -cells` on every
`level0_i/ulp/odin_*/inst`, the scope an IP-level XDC gets, in memory only (nothing placed, routed or saved):

| fixture | checkpoint (`prj.runs/impl_1/level0_wrapper_routed.dcp`, sha256) | what it shows |
|---|---|---|
| `b71018_stage1_nc1_spi_xdc_in_memory.summary.rpt` | 71018, `fc00ef9f3c5b0e1cbfca146bd16dfe4972ef1b68d85b94e150d2040a63a34401` (Slurm 71054) | the domain timed and met: SCK on a global buffer, intra-SCK hold +0.031 ns |
| `b71035_A_impl_1_spi_xdc_in_memory.summary.rpt` | 71035 `impl_1`, `b8f053a4eb759e2c888f713c671becc921d532599fdef47fc8b37509b73c0ae6` (Slurm 71053) | the domain timed but routed without the constraint: same-edge hold races on the six fabric-routed SCK domains (`odin_1` to `odin_6`, worst -0.116 to -0.803 ns) |

`b71070_fixed_cu7_impl_1.summary.rpt` is the same cut of a build that carried the constraint: the fixed N = 7 build
(Slurm 71070, `impl_1`, 7 units, NC 1, UUID `a58f67f7`), built with `scripts/hacc/build_xclbn.sh` sha256
`9df0e01ec46d83937a65df001c5764d83ca979fa312fbcf936a2a9752ad85f3f`. Source report
`prj.runs/impl_1/hw_bb_locked_timing_summary_routed.rpt`,
`153f89884d0ccb453d9107b9c852106eb1ae658f79c4d86b0978d2969fa9cf8f`, 155,210 lines. `check_timing` is clear, and
every SCK domain is timed and met: 7 SCK clocks, each with its Intra Clock Table row and both crossings with the
core clock, every crossing with a number for WNS and WHS.

The three `*.rpt.gz` files are the fixtures of the SCK clock-path check (`scripts/hacc/check_sck_clock_paths.py`, U-SP4
L1), gzip -9 -n of the survey diagnosis records (survey build lock revision 2, `clock_path_read`; the diagnosis
`hardware/multicore/survey/builds/S1-NC1/diagnose/`, called D, `D/SHA256SUMS`):

| fixture | uncompressed sha256 | source |
|---|---|---|
| `s1nc1_71211_sck_sections.rpt.gz` | `c8523936b03b7d77bedb118116294f8716edf46139462ef9a34023add79159fe` | every section of the Timing Details of S1-NC1's (Slurm 71211, NC 1, `spi_fix` revision 3) routed summary whose launch or capture clock is an SCK clock (8 sections, 71 printed paths), cut unchanged from `D/vivado/timing_summary.rpt.gz` (uncompressed `ea105a33220a9af090036439745332d808c5a75eab9a626639fd5e2f18ab9128`, written by the read-only diagnosis job 71219 in the build's form) |
| `s1nc4_71213_sck_to_core_sections.rpt.gz` | `811269ddb4219c918d429a0eca9a0ce0d816c9707813120bc847185452d2d0cc` | `D/comparison/S1-NC4/sck_to_core_sections.rpt`: the SCK-to-core sections of S1-NC4's (71213, NC 4) routed summary |
| `b71070_sck_to_core_sections.rpt.gz` | `fc2f559493b0f8c235d8ca99f82f57cf649223714cd4beeb10756572d73e07c3` | `D/comparison/b71070/sck_to_core_sections.rpt`: the SCK-to-core sections of build 71070's (7 units, NC 1) `impl_1` routed summary |
