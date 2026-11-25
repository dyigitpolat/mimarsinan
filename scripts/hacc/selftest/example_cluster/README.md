# Invented example cluster for the local gate harness

Every file in this folder is an INVENTED example, not a capture of any real
cluster: the partitions, nodes, groups and limits are made up so that each
decision the package makes has a case to decide. Each data file says so in its
first line, and the stubs skip lines that begin with `#`. The local gate harness
(`run_local_gates.sh`) answers `sinfo`, `scontrol`, `groups` and `sbatch` with
these files, and builds the stub `/tools` and `/opt/xilinx/platforms` trees from
them, so `run_all.sh`'s partition auto-pick and `cards.sh`'s card gate are
exercised off-cluster. The harness sets the candidate lists the way a site
would, through `ODIN_BUILD_CANDIDATES`, `ODIN_BOARD_CANDIDATES` and
`ODIN_JOINT_CANDIDATES`.

## The partitions

| Case | File |
|---|---|
| `build_down` admits everyone but its only node is down: the build pick must reject it | `sinfo_a_N.txt`, `scontrol_build_down.txt` |
| `build_long`: `AllowGroups=ALL`, `MaxTime=7-00:00:00`, one idle node, the build venue | `scontrol_build_long.txt` |
| the account is in `user1 users gpu_users u280_users u250_users`, not in `staff` and not in `u55c_users` | `groups.txt` |
| `u55c_board_1h` allows `staff,gpu_users,u55c_users` with a 1 h cap and is reachable through `gpu_users`, a group in the middle of the list | `scontrol_u55c_board_1h.txt` |
| `joint_hidden` is absent from `scontrol show partition` and refuses a submission | `scontrol_missing.txt`, `sbatch_not_permitted.txt` |
| `joint_gpu` allows `staff,gpu_users,u280_users`, 12 h (5 d on `joint_gpu_long`), on a node that holds the card and a GPU | `scontrol_joint_gpu*.txt` |
| `u250_board_1h`: `AllowGroups=staff,u250_users`, 1 h, six idle nodes, admitted through the LAST group of its list | `scontrol_u250_board_1h.txt` |
| `u250_pool_12h` 12 h and `u250_pool_2d` 2 d, same groups | `scontrol_u250_pool_*.txt` |

## The toolchain

| Case | File |
|---|---|
| The only U55C platform is `xilinx_u55c_gen3x16_xdma_3_202210_1`, whose IP is locked to the 2022.2 release, beside the two U250 shells | `platforms_installed.txt` |
| Vitis `2020.1 2020.2 2021.2 2022.1 2023.2 2024.2` is installed and 2022.2 is not, so the U55C refuses and the U250 pairs with 2020.2 | `vitis_installed.txt` |
| `xilinx_u250_gen3x16_xdma_4_1_202210_1` is a 2022.2-era shell and would hit the same lock, so `cards.sh` refuses it by name | `platforms_installed.txt` |

What the harness proves is the DECISION: which partition and which card and
toolchain pairing the package picks from these facts, and which it refuses.
Nothing in the pick depends on a node name, only on group admission and on at
least one node not being down.
