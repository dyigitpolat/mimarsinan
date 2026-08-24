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

## 0. What you are doing, and what is already proven

Build the ODIN RTL kernel into a `.xclbin` on a **compile** node, stage it plus
this repository under `/data/${USER}`, allocate a **U55C** board through Slurm,
and run the same deployment config you already ran locally with
`odin_fpga_transport` flipped from `rtl_cosim` to `xrt`.

**What is proven before you log in.** The kernel's AXI4 master is implemented
and simulation-proven: `tests/integration/test_odin_fpga_kernel.py` drives the
packaged WRAPPER (`odin_fpga_kernel_top`) through a behavioural AXI4 memory
model — program buffer in, stimulus buffer in, run, capture buffer out — and
the per-neuron counts that come back out of that memory equal the host
testbench's exactly. The program bytes the board receives are byte-identical to
the ones the cosimulation received (also a committed test).

**What is NOT proven before you log in.** Silicon. The AXI model is a
testbench, not an HBM controller behind an XDMA shell: nothing local can
exercise real memory ordering, the shell's address translation, XRT's buffer
allocation, or clock closure. That is what phase **B0** below is for, and it is
the reason B0 comes before the campaign rather than after it.

Two phases, in this order:

| Phase | What it is | What it costs | Gate to leave it |
|---|---|---|---|
| **B0** | build → load → CSR/status smoke on one board | one build (2–6 h) + <1 board hour | the card enumerates, the kernel's capacity registers read back the geometry you built, and one tiny run returns `err=0` |
| **B1** | the parity campaign + measured walls | board hours | `PASS exact=1.000000 max\|dcount\|=0`, plus the two measured walls |

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
| `xilinx_u55c_gen3x16_xdma_3_202210_1` | `hacc-gpu1/2/3` | **1 hour** | B0, or a small campaign |
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

Toolchain: Vitis **2022.2** under `/tools/xilinx`, shell
`xilinx_u55c_gen3x16_xdma_3_202210_1`. The script refuses loudly if it cannot
find them, which is what happens if you run it on your laptop.

**XRT version — the two cluster docs disagree, and the newer one wins.**
`Xtra-Computing/hacc_demo/README.md`'s current cluster table lists the U55C
cluster at **XRT 2.18.179** (Vitis 2022.2). `hacc_demo/doc/0-login.md` still
carries an older table listing 2.14.384 for the same shell; that page is stale.
Take 2.18.179 as the expected runtime and **confirm it on the node** with
`xbutil examine` (§5) before blaming anything else — a shell/XRT mismatch
against what the xclbin was linked with is triage step 1, not a footnote.

**Kernel geometry is compile-time, and it is load-bearing.** The wrapper's
`PROG_WORDS` defaults to `NC * 262144` words and `CAP_WORDS` to `16384`;
the v1 packaging flow builds **NC = 1 only** (`build_xclbn.sh` refuses more:
the RTL parameter exists, the `package_xo` plumbing for it is a P7b follow-up)
(≈147.6k program words are needed to SPI-program ONE stock core, and a
16384-word capture RAM holds 4095 event records — it is a BLOCK RAM, 16
RAMB36E2 tiles, which is what makes that depth shippable at all; see
`docs/odin_fpga_compile_limits_study.md`). The host does not assume
these: it reads them back from the read-only registers at `0x5C` and `0x54` and
REFUSES a run that would overrun either. If a run refuses with
`OdinFpgaProgramTooLarge` or `OdinFpgaCaptureTruncated`, raise the parameter in
`hw/fpga/kernel/odin_fpga_kernel_top.v` and rebuild. Lowering the sample count
until it fits is a way to get a number, not a way to get a result.

**How long:** `hw_emu` ~15 minutes. `hw` is dominated by place-and-route of the
ODIN cores; the local yosys census (`hw/fpga/synth_resources.json`) puts one
stock core at 5,659 LUT-equivalents, 4,362 FFs and 10 RAMB36E2, so a 1-core
kernel is small — budget **2–6 hours** and
run it inside a `screen`/`tmux` on the compile node. On top of the cores, the
program and capture RAMs are ≈8.4 Mbit (per core) and 0.5 Mbit of on-chip
memory, and both INFER BLOCK RAM in the local census (one RAMB36E2 per 1,024
32-bit words); nothing local has placed them, so read the build's
`utilization` report for where they actually landed and at what cost.

Outputs land in `build/hacc/<target>_nc<N>/`:
`odin_fpga_<target>.xclbin`, plus `reports/` (timing, utilization) and `logs/`.
**Bring the reports back** — they are P8's implementation-closure evidence and
the real-shell half of gate row 19.

## 3. What to do if the build fails

* **`package_xo` errors on a port** — the kernel.xml and the Verilog port list
  disagree. `scripts/hacc/gen_kernel_xml.py` derives the register map from the
  host-side SSOT (`odin_fpga/kernel_registers.py`), so fix it there, not in the XML.
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
mkdir -p /data/${USER}/log            # optional: run_board.sh and the sbatch both create it
cp build/hacc/hw_nc1/odin_fpga_hw.xclbin /data/${USER}/odin/
cp <your deployment config>.json         /data/${USER}/odin/odin_u55c.json
# the repository itself must be under the stage dir (run_board.sh expects
# ${STAGE}/mimarsinan, or set ODIN_REPO)
```

`/data/${USER}/log` is where `run_board.sh` puts each run's artifact directory.
Create it now: slurm's own job logs are written relative to `--chdir` (`/tmp`
on the node) precisely so a missing directory can never stop a submission from
launching, but the job's copy-back step still needs this one to exist.

In the config, flip the transport and point it at the staged image:

```json
"enable_odin_fpga_simulation": true,
"odin_fpga_transport": "xrt",
"odin_fpga_xclbin_path": "/tmp/<user>_odin/odin_fpga_hw.xclbin",
"odin_fpga_device_index": 0,
"odin_fpga_sample_count": 8
```

`odin_fpga_xclbin_path` is the ONLY place the bitstream is named — there is no
environment override, and `run_board.sh` copies the stage directory to
`/tmp/${USER}_odin` on the node, so the path above is what that copy produces.

Everything else stays exactly as the local run: same mapping, same soma law,
same exporter. That is the point of the transport seam.

## 5. Phase B0 — build, load, CSR/status smoke

This is the first thing you do with a board, and it is short. It answers
"does this bitstream load and does the kernel answer its own register map",
which is everything the local gates cannot answer.

```bash
srun -p xilinx_u55c_gen3x16_xdma_3_202210_1 -n 1 --pty bash -i
source /opt/xilinx/xrt/setup.sh
xbutil examine                      # the card must be listed and healthy;
                                    # READ THE XRT VERSION HERE (§2)
cd /data/${USER}/odin/mimarsinan
scripts/hacc/run_board.sh /data/${USER}/odin odin_u55c.json
```

Run it first with `"odin_fpga_sample_count": 1`. What you are checking, in
order:

1. `xbutil examine` lists the card, the shell is
   `xilinx_u55c_gen3x16_xdma_3_202210_1`, and the XRT version matches §2.
2. The xclbin loads and the kernel resolves by name
   (`odin_fpga_kernel_top`) — a failure here is packaging, not physics.
3. The session's opening register reads succeed: `capture_capacity` and
   `program_capacity` in the run's `detail` must equal the geometry you built
   (16383 and `NC * 262144` by default). A zero there refuses at `open()` by
   design — it means the host is talking to something that is not this kernel.
4. The run finishes with `err = 0` on the status register at `0x4C`. An `err`
   raises `OdinFpgaKernelError` naming how many events the fabric saw.

**CU access mode.** The transport opens the kernel with **exclusive** access —
register reads (`0x4C`/`0x54`/`0x5C`) require it, and the deployment owns the
board for the whole reservation. If a future setup must share the CU, set
`rw_shared=true` under `[Runtime]` in `xrt.ini` and change the open mode
deliberately; do not weaken it by default.

**Known simulation-coverage limits** (the AXI model proves the payload path,
not these): the datapath is 32-bit-beat only (any other `C_M_AXI_DATA_WIDTH`
now fails at elaboration by design); the 4 KiB burst-boundary clamp branch is
never exercised by the testbench (all tb buffers are 4 KiB-aligned — correct
by inspection only); RRESP/BRESP error paths are untested (the tb slave always
answers OKAY). A board-side AXI anomaly can therefore live in exactly these
three shadows — check them before blaming the ODIN core, which R11a proved.

If all four hold, B0 is passed: the bitstream loads, the register map answers,
and the DMA has moved real bytes across a real shell. Only then is a campaign
worth board hours.

## 6. Phase B1 — the parity campaign

Batched, for the campaign (the 7-day pool if you need more than an hour):

```bash
sbatch -p mi210_vck_u55c scripts/hacc/odin_u55c.sbatch
squeue                              # watch it
ls -t /data/${USER}/log/odin_*/     # the newest run's artifact directory
tail -f /data/${USER}/log/odin_*/run_board.log
```

Slurm's own `--output`/`--error` files go to the node's `/tmp` (see §8), so
`run_board.sh` mirrors everything it prints into `run_board.log` under the
run's `/data` directory — that mirror, not the slurm file, is what you read
and what you bring back.

**Bring-up order is not negotiable (plan R11b): the STOCK core first.** A
generated variant is only worth a board hour after the stock one has reproduced
its counts there.

Under `/data/${USER}/log/odin_<timestamp>/`:

* `run_board.log` — everything the node-side script printed, slurm file or not.
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

The payload path is no longer a candidate first: the program bytes are a
committed identity test, and the DMA that delivers them is proven against an
AXI4 memory model in simulation. So work the difference in this order, cheapest
and most likely first:

1. **Shell / XRT version.** `xbutil examine` on the node against §2's table and
   against what the xclbin was linked with. This is the single most common
   cause and it costs nothing to check.
2. **The CSR smoke (B0 step 3).** Do the capacity registers read back the
   geometry you built? A wrong `capture_capacity`/`program_capacity` means the
   loaded xclbin is not the one you think it is.
3. **The capture status (B0 step 4).** `err = 1` is the fabric REFUSING, not
   disagreeing: an opcode it does not implement, an AER timeout, or a payload
   that did not fit. `OdinFpgaCaptureTruncated` means the capture RAM filled —
   raise `CAP_WORDS` and rebuild; the counts of a truncated run are not a
   result.
4. **Re-run the SAME config with `odin_fpga_transport: "rtl_cosim"`** on the
   node. Same program bytes, no board. If that passes and the board does not,
   the divergence is in the shell, the silicon, or timing closure — not in the
   export and not in the delivery.
5. **Re-run the board with `odin_fpga_sample_count: 1`.** A first-sample-only
   divergence points at the CLEAR stage; an every-sample one at programming.
6. **Hot-reset the card** (`reset_fpga` in `run_board.sh`) and repeat: a wedged
   AER link after a previous job looks exactly like a physics bug.
7. **Then, and only then, silicon.** Check the build's `reports/` for timing
   violations at the kernel clock before concluding anything about the design.

## 8. Things that will bite you

* Submitting a build to a board partition — killed at one hour, every time.
* `--account=slurm` is required in the **sbatch** preamble
  (`hacc_demo/doc/1-FPGA-allocation.md`'s `example.sh` carries it, and
  `scripts/hacc/odin_u55c.sbatch` does too). The **`srun`** examples in that
  same document pass no `--account` at all — do not add one to an `srun` on the
  strength of the sbatch template.
* Slurm opens `--output`/`--error` before the job body runs. Ours are relative
  to `--chdir` (`/tmp`) for exactly that reason; if you point them at `/data`,
  create the directory first or the submission never launches.
* Staging outside `/data` — the VM cannot see it.
* `hacc-gpu1` when you wanted an MI210 next to the U55C — it has U250/U280.
* Assuming `.slurmech.toml` applies: it targets a different cluster entirely
  (xlog1/H100). HACC submission is hand-written `sbatch`, which is what this
  directory is.
