# Frontier Phase 2 — certification campaign report

Per-cell freeze (controller floor) + Fix-B fast candidate + E6 verdict.

| cell | floor acc | floor wall | fast acc | fast wall | speedup | verdict |
|------|-----------|-----------|----------|-----------|---------|---------|
| lif@nevresim#rate | 0.972 | 1703.19 | 0.968 | 1009.4 | 1.7x | pass |
| lif@nevresim#novena_offload | 0.974 | 974.53 | 0.97 | 646.13 | 1.5x | pass |
| lif@nevresim#pruned_scheduled | 0.94 | 1205.18 | 0.924 | 983.54 | 1.2x | fail |
| ttfs@nevresim#analytical | 0.978 | 1207.58 | 0.978 | 1207.58 | 1.0x | pass |
| ttfs_quantized@nevresim#offload | 0.972 | 617.08 | 0.972 | 617.08 | 1.0x | pass |
| ttfs_cycle_based/cascaded@nevresim#plain | 0.96 | 762.84 | 0.96 | 466.56 | 1.6x | pass |
| ttfs_cycle_based/synchronized@nevresim#plain | 0.9635 | 234.89 | 0.9618 | 232.78 | 1.0x | pass |
| ttfs_cycle_based/cascaded@nevresim#offload_scheduled_nobias | 0.932 | 780.44 | 0.924 | 578.34 | 1.3x | pass |
| ttfs@nevresim#vanilla_noWQ | 0.984 | 703.27 | 0.984 | 703.27 | 1.0x | pass |

Floor book: `docs/certification/regression_floor.json`
Results: `docs/certification/campaign_results.json`

- **matrix_1_lif_rate**: pass — certified: accuracy 0.9680 >= 0.9670 and wall-clock 1009.4s <= 2129.0s
- **matrix_2_lif_novena_offload_loihi**: pass — certified: accuracy 0.9700 >= 0.9690 and wall-clock 646.1s <= 1218.2s
- **matrix_3_lif_pruned_scheduled**: fail — regression: accuracy 0.9240 < floor 0.9350 (deployed 0.9400 − eps 0.0050)
- **matrix_4_ttfs_analytical**: pass — certified: accuracy 0.9780 >= 0.9730 and wall-clock 1207.6s <= 1509.5s
- **matrix_5_ttfs_quantized_offload**: pass — certified: accuracy 0.9720 >= 0.9670 and wall-clock 617.1s <= 771.4s
- **matrix_6_ttfs_cycle_cascaded**: pass — certified: accuracy 0.9600 >= 0.9500 and wall-clock 466.6s <= 953.6s
- **matrix_7_ttfs_cycle_synchronized**: pass — certified: accuracy 0.9618 >= 0.9585 and wall-clock 232.8s <= 293.6s
- **matrix_8_ttfs_cycle_offload_scheduled_nobias**: pass — certified: accuracy 0.9240 >= 0.9220 and wall-clock 578.3s <= 975.6s
- **matrix_9_ttfs_vanilla_noWQ**: pass — certified: accuracy 0.9840 >= 0.9790 and wall-clock 703.3s <= 879.1s
