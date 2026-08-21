# ODIN on HACC@NUS — the owner's runbook (P7b)

Everything in P7a is local and already gated: the physical backend is complete
software, the RTL cosimulation is the first device behind the transport seam,
and `scripts/hw_tests/run_hw_tests.sh` proves the deployed counts equal the HCM
reference and nevresim at zero difference. **This runbook is the part that
needs you**: HACC login is 2FA, so nothing here can be automated from the repo.

What comes back from it is one number nothing local can produce — the same
counts, produced by an Alveo card, with measured programming and execution
walls.

---

## 0. What you are doing, in one paragraph

Build the ODIN RTL kernel into a `.xclbin` on a **compile** node, stage it plus
this repository under `/data/${USER}`, allocate a **U55C** board through Slurm,
and run the same deployment config you already ran locally with
`odin_fpga_transport` flipped from `rtl_cosim` to `xrt`. The program bytes the
board receives are byte-identical to the ones the cosimulation received — that
identity is a committed test — so a divergence on the board is a *silicon or
shell* finding, never a payload one.

---

## 1. Log in (2FA — only you can do this)

```bash
ssh <username>@xacchead.d2.comp.nus.edu.sg     # 2FA per the registration email
source /home/hacc_env                          # or add it to ~/.bashrc
```

You land on `hacchead`. Check what is free before planning anything:

```bash
sinfo      # partitions, time limits, states
squeue     # who is holding what
```

The rows that matter:

| Partition | Nodes | Time limit | Use it for |
|---|---|---|---|
| `cpu_only` | `hacc-node2` | 7 days | **building the xclbin** |
| `xilinx_u55c_gen3x16_xdma_3_202210_1` | `hacc-gpu1/2/3` | **1 hour** | a bring-up run, a small campaign |
| `mi210_vck_u55c` | `hacc-gpu2`, `hacc-gpu3` | 7 days | U55C **and** MI210 in one job |
| `mi210_u250_u55c` | `hacc-gpu1` | 7 days | U55C, but its GPU pair is U250/U280 |

Only **hacc-gpu2** and **hacc-gpu3** are true U55C + MI210 nodes. Both were
busy in the sampled `sinfo`: expect to queue for the 7-day pools.

## 2. Build the kernel (on `cpu_only` or `hacchead`, NEVER on a board)

The board partitions are capped at one hour; a place-and-route is hours. The
build needs no board at all.

```bash
git clone <this repo> /data/${USER}/odin/mimarsinan
cd /data/${USER}/odin/mimarsinan

srun -p cpu_only -n 1 --pty bash -i          # or just run on hacchead

scripts/hacc/build_xclbn.sh hw_emu           # FIRST: ~15 min, functional
scripts/hacc/build_xclbn.sh hw               # THEN: the real bitstream
```

Toolchain: Vitis **2022.2** under `/tools/xilinx`, XRT 2.14.384, shell
`xilinx_u55c_gen3x16_xdma_3_202210_1`. The script refuses loudly if it cannot
find them, which is what happens if you run it on your laptop.

**How long:** `hw_emu` ~15 minutes. `hw` is dominated by place-and-route of the
ODIN cores; the local yosys census (`hw/fpga/synth_resources.json`) puts one
stock core at 5,659 LUT-equivalents, 4,362 FFs and 10 RAMB36E2, so a 1-core
kernel is small and a multi-core one scales linearly — budget **2–6 hours** and
run it inside a `screen`/`tmux` on the compile node.

Outputs land in `build/hacc/<target>_nc<N>/`:
`odin_fpga_<target>.xclbin`, plus `reports/` (timing, utilization) and `logs/`.
**Bring the reports back** — they are P8's implementation-closure evidence and
the real-shell half of gate row 19.

## 3. What to do if the build fails

* **`package_xo` errors on a port** — the kernel.xml and the Verilog port list
  disagree. `scripts/hacc/gen_kernel_xml.py` derives the register map from the
  host-side SSOT (`odin_fpga/xrt_transport.py`), so fix it there, not in the XML.
* **Elaboration errors** — reproduce them locally in seconds:
  `scripts/hw_tests/run_hw_tests.sh -k kernel_elaborates`. That gate exists so
  a syntax error never costs you a cluster hour.
* **Timing not met** — lower the kernel clock in `scripts/hacc/odin_u55c.cfg`
  (`kernel_frequency`) before touching the design. The ODIN core is a slow,
  event-serial machine; it does not need a fast clock.

## 4. Stage the run

`/data` is the only path shared between the head node and the board VMs.

```bash
mkdir -p /data/${USER}/odin
cp build/hacc/hw_nc1/odin_fpga_hw.xclbin /data/${USER}/odin/
cp <your deployment config>.json         /data/${USER}/odin/odin_u55c.json
# the repository itself must be under the stage dir (run_board.sh expects
# ${STAGE}/mimarsinan, or set ODIN_REPO)
```

In the config, flip the transport and point it at the staged image:

```json
"enable_odin_fpga_simulation": true,
"odin_fpga_transport": "xrt",
"odin_fpga_xclbin_path": "/tmp/<user>_odin/odin_fpga_hw.xclbin",
"odin_fpga_device_index": 0,
"odin_fpga_sample_count": 8
```

Everything else stays exactly as the local run: same mapping, same soma law,
same exporter. That is the point of the transport seam.

## 5. Run the campaign

Interactive, for the first bring-up (watch it, reset the card if it wedges):

```bash
srun -p xilinx_u55c_gen3x16_xdma_3_202210_1 -n 1 --pty bash -i
source /opt/xilinx/xrt/setup.sh
xbutil examine                      # the card must be listed and healthy
cd /data/${USER}/odin/mimarsinan
scripts/hacc/run_board.sh /data/${USER}/odin odin_u55c.json
```

Batched, for the campaign (the 7-day pool if you need more than an hour):

```bash
sbatch -p mi210_vck_u55c scripts/hacc/odin_u55c.sbatch
squeue                              # watch it
cat /data/${USER}/log/slurm-*.out
```

**Bring-up order is not negotiable (plan R11b): the STOCK core first.** A
generated variant is only worth a board hour after the stock one has reproduced
its counts there.

## 6. What comes back

Under `/data/${USER}/log/odin_<timestamp>/`:

* `scan.log` — `xbutil examine`: which card, which shell, which XRT.
* `exec.log` — the run, including the two lines that matter:
  * `[SpikeCountCertificate] spike-count certificate [odin_fpga/exact]: PASS
    exact=1.000000 max|dcount|=0 over N neuron-windows` — the board reproduced
    the reference **exactly**. Anything else is a finding, not a tolerance.
  * `ODIN FPGA deployment: N sample(s) on 'xrt', programming X s, execution Y s`
    — the two MEASURED walls. `programming` is the per-pass reprogramming
    physics (~236 ms/core over SPI is the derived figure to check against); it
    is never folded into `execution`.
* `generated/` — the deployment record, whose timing fragment carries those
  same walls per segment, and whose accuracy read is `kind="measured"`,
  `backend="odin_fpga"`.

Copy the whole log directory back and attach the build `reports/` to it. That
bundle is the R11b evidence.

## 7. If the board disagrees with the cosimulation

Do not adjust a tolerance — the backend is classified `exact` in
`BACKEND_CLASSES` and a hardware read that needs slack is a different chip.
Work the difference:

1. `xbutil examine` — is the shell/XRT what the xclbin was built against?
2. Re-run the SAME config with `odin_fpga_transport: "rtl_cosim"` on the node.
   Same program bytes, no board. If that passes and the board does not, the
   divergence is in the kernel or the shell, not in the export.
3. Re-run the board with `odin_fpga_sample_count: 1`. A first-sample-only
   divergence points at the CLEAR stage; an every-sample one at programming.
4. Hot-reset the card (`reset_fpga` in `run_board.sh`) and repeat: a wedged
   AER link after a previous job looks exactly like a physics bug.

## 8. Things that will bite you

* Submitting a build to a board partition — killed at one hour, every time.
* Forgetting `--account=slurm` — the sbatch template has it; `srun` needs it too.
* Staging outside `/data` — the VM cannot see it.
* `hacc-gpu1` when you wanted an MI210 next to the U55C — it has U250/U280.
* Assuming `.slurmech.toml` applies: it targets a different cluster entirely
  (xlog1/H100). HACC submission is hand-written `sbatch`, which is what this
  directory is.
