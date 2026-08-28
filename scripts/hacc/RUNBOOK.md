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

## READ FIRST — UPDATE (admin-disclosed 2026-08-25): the U55C block is LIFTED

The admins answered the deadlock report the same day:

* **Vitis 2022.2 exists** — at `/tools/Xilinx/new/Vitis/2022.2`, a root the
  conventional trees do not reveal. `toolchain.sh` now probes it first, so the
  u55c profile self-selects 2022.2 and the deadlock guard stands down on its
  own evidence. Nothing needs pinning.
* **U55C cards on `hacc-gpu3` are reachable via `amd_gpu_rocm_6_0_0`**
  (12 h cap, allowed via `gpgpu`) — admin-granted; the earlier probe that saw
  "0 devices" predates the grant. It is now a u55c board candidate after the
  1-hour platform partition. Verify with `xbutil examine` inside an
  allocation before trusting it.

The section below records the deadlock as it stood, because its evidence
pattern (locked platform IP vs installed toolchains) is the reusable lesson.

## Superseded (v3, field-observed 2026-08-25): the U55C was admin-blocked

The U55C path does not build on this cluster, and no amount of scripting fixes
it. The facts, each checked in the live session:

* The **only** U55C platform installed anywhere — `/opt/xilinx/platforms` on
  `hacc-node0` **and** on `hacc-gpu3` — is
  `xilinx_u55c_gen3x16_xdma_3_202210_1`.
* Its SmartConnect IP is **customized with the 2022.2 release**. `vpl` refuses
  to link it under anything else: three separate logs, under Vitis **2022.1,
  2023.2 and 2024.2**, all ending in `VPL 60-704` / `60-732` — *"IP ... is
  customized with software release 2022.1 and is being used with a different
  revision."*
* Vitis installed: `2020.1 2020.2 2021.2 2022.1 2023.2 2024.2`. **There is no
  2022.2.**

So U55C is **deadlocked pending admins**, and there are exactly two fixes, both
theirs: install Vitis/Vivado 2022.2, or install a U55C shell built for a
release that is already here.

**The field-viable route is the U250.** The build node also carries
`xilinx_u250_gen3x16_xdma_3_1_202020_1` — a 2020.2-era shell — and **Vitis
2020.2 IS installed**. This account is in `fpga_u250`, and all three U250
partitions admit it:

| Partition | AllowGroups | MaxTime | Nodes |
|---|---|---|---|
| `xilinx_u250_gen3x16_xdma_3_1_202020_1` | `lab,fpga_u250` | **1 h** | 6 idle |
| `u250_standard_reservation_pool` | `lab,fpga_u250` | 12 h | `hacc-u250-[1-4]`, `hacc-u250-frp` |
| `u250_long_reservation_pool` | `lab,fpga_u250` | 2 d | `hacc-u250-[1-4]` |

The build node ALSO carries `xilinx_u250_gen3x16_xdma_4_1_202210_1`. **Do not
use it** — it is a 2022.2-era shell and hits the identical lock. The package
refuses it by name and says so.

**A card is allocated, not ambient.** `xbutil` on `hacc-gpu3` answered
`0 devices found` from a non-FPGA partition on 2026-08-25. The cards are passed
into the job's VM at allocation time — `doc/1-FPGA-allocation.md`'s board-VM
model, now field-confirmed — so only a board-partition job sees a card. An
empty `xbutil examine` is the wrong partition, not a dead board.

### Therefore: the target card is a parameter (v3)

`scripts/hacc/cards.sh` is the ONE place that knows what a card implies, and
everything downstream reads it:

| | `u55c` (default) | `u250` (use this today) |
|---|---|---|
| platform | `xilinx_u55c_gen3x16_xdma_3_202210_1` | `xilinx_u250_gen3x16_xdma_3_1_202020_1` |
| Vivado part | `xcu55c-fsvh2892-2L-e` | `xcu250-figd2104-2L-e` |
| v++ config | `scripts/hacc/odin_u55c.cfg` | `scripts/hacc/odin_u250.cfg` |
| AXI master bank | `HBM[0]` | `DDR[0]` (no HBM on a U250) |
| preferred Vitis | 2022.2 — **absent** | **2020.2** |
| board partitions | the U55C shell partition | the three above, 1 h first |
| joint partitions | `mi210_vck_u55c`, `mi210_u280_u55c`, `…_long_reservation` | the 12 h pool, then 2 d, then 1 h |

```bash
./bootstrap_hacc.sh --card u250            # the packaged path
ODIN_CARD=u250 scripts/hacc/build_xclbn.sh hw     # the by-hand path
```

The `u55c` default refuses **at startup**, before it queues anything, printing
the deadlock and naming the admin fix. It is evidence-gated, not a hard-coded
verdict: it re-reads `/opt/xilinx/platforms` and the installed Vitis list on
every run and clears itself the moment either admin fix lands. Off-cluster,
where there is no platform root to read, it says nothing at all — no evidence
is never treated as evidence of a problem.

---

**There is now a shorter path: one zip and one `sh` (v2, carded in v3).**
`scripts/hacc/make_package.py` builds `dist/odin_hacc_package.zip` **and**
`dist/bootstrap_hacc.sh` beside it, with the zip's own sha256 baked into that
bootstrap. The zip carries the RTL the build compiles, the fixtures with their
expected per-neuron counts already frozen from the committed cosimulation, a
host driver that needs nothing but Python 3 and XRT, and a `run_all.sh` that
walks phases 0–7 (env probe, no-hardware selftest, `hw_emu` build + a bounded
emulation smoke, `hw` build, staging, B0, B1, and a two-component
board-plus-reference job).

Upload **both files into the same directory** on `hacchead` and run
`./bootstrap_hacc.sh`: it verifies the upload, archives any previous install
to `odin_prev_<utc>.tar.gz`, unpacks under `/data/${USER}`, and launches
`run_all.sh` **detached** (`setsid nohup`, log at `results/run_all.log`). It
prints the tail command, the status command, and that you may log out.
`scripts/status.sh` inside the package answers "how far has it got" at any time.
Its `README_HACC.md` is the condensed form of this document. Everything below
stays true and remains the reference for what the phases are doing and why —
read it if a phase refuses. The two paths differ in one thing worth knowing up
front: the package does **not** clone this repository onto the cluster, so it
runs the frozen fixtures rather than a full `run.py` deployment config.

**Why v2 exists.** The 2026-08-25 live session found three things that made v1
need a human at the keyboard: the documented build venue is down, the
documented joint venue refuses this account, and the documented Vitis is not
the installed one. v2 stops trusting any of the three and asks the cluster
instead — see §1 and §2, where every field-observed fact is tagged as such.

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

Three phases, in this order:

| Phase | What it is | What it costs | Gate to leave it |
|---|---|---|---|
| **B0** | build → introspect → load → **null-program round trip** on one board | one build (2–6 h) + <1 board hour | the xclbin's metadata declares `odin_fpga_kernel_top` with its six arguments, the card takes the bitstream, the CU opens EXCLUSIVE, and a one-token null program comes back with a capture header the fabric wrote |
| **B1** | the parity campaign + measured walls | board hours | `PASS exact=1.000000 max\|dcount\|=0`, plus the two measured walls |
| **HACC NUS - ODIN Deployment** (§6b) | a whole multi-core NETWORK, run as one host-mediated pass per core, certified per pass and end to end | minutes on a board, after B1 | every per-pass certificate green, every sample's readout equal to its frozen one, and an ACCURACY that came from counting |

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

### The venues, as the cluster actually answered (field-observed 2026-08-25)

Read this table, not the vendor doc's. Every row on the left was checked in the
live session; the right column is what `Xtra-Computing/hacc_demo` still says.

| Partition | Field-observed 2026-08-25 | Stale doc claim |
|---|---|---|
| `cpu_only` | node is **hacc-gpu0**, state `down*`, reason `NO NETWORK ADDRESS FOUND`. Unusable. | `hacc-node2`, 7 days, the build venue |
| `vck5000_compile` | `AllowGroups=ALL`, `MaxTime=7-00:00:00`, **hacc-node0 idle** — **the build venue** | not mentioned |
| `xilinx_u55c_gen3x16_xdma_3_202210_1` | `AllowGroups=lab,hgpu,fpga_u55c`, 1 h cap — this account gets in through **`hgpu`**, without `fpga_u55c` | hacc-gpu1/2/3, one hour |
| `mi210_vck_u55c` | **absent** from `scontrol show partition`; `sbatch` answers `User's group not permitted to use this partition` | hacc-gpu2/3, 7 days, U55C + MI210 |
| `mi210_u280_u55c` | `AllowGroups=lab,hgpu,fpga_u280`, 12 h, node **hacc-gpu1** which hosts MI210 **and** U280 **and** U55C — **the joint venue** | doc names `mi210_u250_u55c` and warns its pair is U250/U280 |
| `mi210_u280_u55c_long_reservation` | same groups, **5 days** | not mentioned |
| `xilinx_u250_gen3x16_xdma_3_1_202020_1` | `AllowGroups=lab,fpga_u250`, 1 h, **6 nodes idle** — the board venue that actually builds | 1 h, `hacc-u250-[1-4]` |
| `u250_standard_reservation_pool` | `AllowGroups=lab,fpga_u250`, **12 h** | 12 h, `hacc-u250-[1-4],hacc-u250-frp` |
| `u250_long_reservation_pool` | `AllowGroups=lab,fpga_u250`, **2 days** | 2 d, `hacc-u250-[1-4]` |

The account's groups, verbatim: `yigit video render hgpu gpgpu fpga_u280
fpga_u250 fpga_vck5000`. Note what is NOT there — `fpga_u55c` and `lab` — and
that `hgpu` is nevertheless enough for both U55C partitions, while `fpga_u250`
opens all three U250 ones outright.

`run_all.sh` no longer takes any of this on faith: it re-derives the pick at
phase time from `scontrol show partition` (groups) and `sinfo` (a node that is
up), prints why each rejected candidate lost, and yields to
`ODIN_BUILD_PARTITION` / `ODIN_BOARD_PARTITION` / `ODIN_JOINT_PARTITION`. When
the cluster changes again, the script follows it and this table is the record of
what it looked like on 2026-08-25.

## 2. Build the kernel (on a compile partition or `hacchead`, NEVER on a board)

The board partitions are capped at one hour; a place-and-route is hours. The
build needs no board at all.

```bash
git clone <this repo> /data/${USER}/odin/mimarsinan
cd /data/${USER}/odin/mimarsinan

srun -p vck5000_compile -n 1 --pty bash -i   # field-observed 2026-08-25:
                                             # cpu_only's node is DOWN

export ODIN_CARD=u250                        # see READ FIRST: u55c is blocked
scripts/hacc/build_xclbn.sh hw_emu           # FIRST: ~15 min, functional
scripts/hacc/build_xclbn.sh hw               # THEN: the real bitstream
```

Build venues are **card-independent** — a place-and-route needs a compile
partition, not a card — so `vck5000_compile` is right for either card.

Toolchain (field-observed 2026-08-25): Vitis lives under **`/tools/Xilinx`** —
capital X; the vendor docs' `/tools/xilinx` is stale and cost the first live
session a refusal at the door — and the installed set is
`2020.1 2020.2 2021.2 2022.1 2023.2 2024.2`.
`scripts/hacc/toolchain.sh` is the single place that resolves this: it probes
`/tools/Xilinx`, `/tools/xilinx`, `/opt/Xilinx`, `/opt/xilinx` and picks, in
order, `VITIS_VERSION` if you set it, then **the card profile's preferred
release** if installed, then the newest. Newest alone is wrong: a platform's IP
is locked to the release it was built with, which is the whole U55C story, so
`ODIN_CARD=u250` builds under **2020.2** on purpose. It refuses loudly if there
is no Vitis anywhere, which is what happens if you run it on your laptop.

`hacc_demo/README.md` pairs the U250 202020 shell with Vitis **2021.2** rather
than 2020.2. Both are plausible for a 202020 shell; the profile pins 2020.2
because it is the release the shell was built against and it is installed. If a
2020.2 link ever fails on this shell, `VITIS_VERSION=2021.2` is the first thing
to try, and it is one environment variable, not a code change.

That file also owns the `set +u` around the vendor setup scripts: Vitis 2024.x's
`.settings64-Vitis.sh` reads `$PYTHONPATH`, and under `set -u` an unset
`PYTHONPATH` in a fresh slurm shell kills the build before it starts.

**XRT version — the two cluster docs disagree, and the newer one wins.**
`Xtra-Computing/hacc_demo/README.md`'s current cluster table lists the U55C
cluster at **XRT 2.18.179** (Vitis 2022.2). `hacc_demo/doc/0-login.md` still
carries an older table listing 2.14.384 for the same shell; that page is stale.
Take 2.18.179 as the expected runtime and **confirm it on the node** with
`xbutil examine` (§5) before blaming anything else — a shell/XRT mismatch
against what the xclbin was linked with is triage step 1, not a footnote.

**Kernel geometry is compile-time, and it is load-bearing.** The wrapper's
`FIFO_WORDS` defaults to `1024` words and `CAP_WORDS` to `16384`;
the v1 packaging flow builds **NC = 1 only** (`build_xclbn.sh` refuses more:
the RTL parameter exists, the `package_xo` plumbing for it is a P7b follow-up)
(≈147.6k program words are needed to SPI-program ONE stock core — and the
fabric holds NONE of them: the op stream arrives live from the host through the
FIFO, so a run's length is bounded only by the word-count arguments and there is
no program capacity to declare. The 16384-word capture RAM holds 4095 event
records — it is a BLOCK RAM, 16 RAMB36E2 tiles, which is what makes that depth
shippable at all; see `docs/odin_fpga_compile_limits_study.md`).
**The host cannot read these back.**
The kernel does implement a read-only register at `0x54`, and the RTL
testbench reads it over AXI-Lite — but the XRT Python binding binds no
register access at all (`github.com/Xilinx/XRT@2024.2`,
`src/python/pybind11/src/pyxrt.cpp`; a driver that tried died on a U250 on
2026-08-25). So the host DECLARES them from the sources the xclbin was compiled
from — `kernel_registers.SHIPPED_*`, mirrored in the shipped driver — and every
refusal that spends a capacity prints that provenance in its own text. A
bitstream that never drains is caught the one way memory allows: the host's
no-verdict sentinel comes home untouched (`OdinFpgaKernelError`).
If a run refuses with `OdinFpgaCaptureTruncated`, raise `CAP_WORDS` in
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
* **`VPL 60-704` / `60-732`, "customized with software release ... different
  revision"** — you are linking a shell against a Vitis it was not built with.
  That is the U55C deadlock (READ FIRST); on any other card it means the
  version pin is wrong, and `VITIS_VERSION=<release>` is the lever.
* **Timing not met** — lower the kernel clock (`kernel_frequency`) in **your
  card's** config, `scripts/hacc/odin_u250.cfg` or `odin_u55c.cfg`, before
  touching the design. The ODIN core is a slow, event-serial machine; it does
  not need a fast clock.

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

## 5. Phase B0 — build, introspect, load, and run a null program

This is the first thing you do with a board, and it is short. It answers "does
this bitstream reach a card and does the whole host↔fabric round trip close",
which is everything the local gates cannot answer.

B0 used to be a CSR read. It cannot be: **pyxrt binds no `read_register`**, and
the driver that assumed otherwise crashed on a U250 at its first CSR touch on
2026-08-25. So B0 is now the round trip instead, which proves strictly more.

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
2. **Introspection.** `pyxrt.xclbin(path).get_kernels()` declares
   `odin_fpga_kernel_top` with its six arguments, and `get_mems()` names the
   bank `gmem` is mapped onto. This happens BEFORE the card is touched, so a
   wrong bitstream costs no load — a failure here is packaging, not physics.
3. **Load and open.** The device takes the xclbin and the CU resolves by name
   with **EXCLUSIVE** access.
4. **The null-program round trip.** The driver sends a one-word program (a
   single `END` token), a one-word stimulus, and a capture buffer whose two
   header words carry the no-verdict sentinel `0xFFFFFFFF`. It waits on the
   run's `ert_cmd_state`, syncs the capture back, and reads the header. Passing
   means: the six arguments reached the CU, its AXI master read host memory,
   the sequencer executed, the capture engine wrote its header, and the master
   wrote that header back into host memory. `events_seen` must be `0` — a
   fabric that spikes with no network loaded is not this design.

`results/board_probe/probe.json` carries the whole report plus two explicit
lists, `proves` and `does_not_prove`. Read the second one: B0 does not prove any
spike count, and it cannot see the fabric's `err` bit, because a sequencer that
refuses an opcode MID-run still drains a header. What catches that is B1's
certificate.

**CU access mode.** The driver opens the kernel with **exclusive** access: the
deployment owns the board for the whole reservation, and a shared CU would let
another job's run interleave with this one's capture buffer. (It is no longer
about register reads — there are none.)

**Known simulation-coverage limits** (the AXI model proves the payload path,
not these): the datapath is 32-bit-beat only (any other `C_M_AXI_DATA_WIDTH`
now fails at elaboration by design); the 4 KiB burst-boundary clamp branch is
never exercised by the testbench (all tb buffers are 4 KiB-aligned — correct
by inspection only); RRESP/BRESP error paths are untested (the tb slave always
answers OKAY). A board-side AXI anomaly can therefore live in exactly these
three shadows — check them before blaming the ODIN core, which R11a proved.

If all four hold, B0 is passed: the bitstream loads, the arguments land, and
the DMA has moved real bytes across a real shell in BOTH directions. Only then
is a campaign worth board hours.

## 6. Phase B1 — the parity campaign

Batched, for the campaign (the 7-day pool if you need more than an hour):

```bash
sbatch -p mi210_u280_u55c scripts/hacc/odin_u55c.sbatch   # field-observed 2026-08-25
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

## 6b. HACC NUS - ODIN Deployment

B1 certifies FIXTURES — frozen programs whose counts the cosimulation recorded.
This phase deploys a NETWORK, and everything below is about the one thing that
makes it different: the shipped bitstream holds ONE ODIN core, and the chip
routes nothing between cores (`SPI_OPEN_LOOP` — v1 routing is host-mediated by
design), so a multi-core network runs as one PASS PER CORE with the host doing
the wiring in between.

```bash
./run_all.sh --only 8                        # after 3-4 staged an xclbin
ODIN_DEPLOY_SAMPLES=8 ./run_all.sh --only 8  # bound the campaign
```

What one pass costs, and what is deliberately NOT paid per sample:

* the core is programmed ONCE. Fabric memories persist across sequencer runs,
  so the ~236 ms/core SPI shift is paid once per core per campaign, not once
  per sample. The per-sample membrane CLEAR is an OP inside the stimulus.
* each sample is: rewrite the stimulus buffer, re-poison the capture header,
  start, wait, sync back, decode, fold, transcode. All seven are timed
  separately with `time.perf_counter`, because a deployment that reports one
  number cannot say which stage a slow campaign spent its time in.
* the buffers are allocated once per core and rewritten — a fresh `xrt::bo` per
  sample would put an allocator in the middle of the measurement.

Read, under `results/board_deploy/`:

* `deployment_report.json` — per-pass certificates on the certification subset
  (same house line as B1: `[odin_fpga/exact] PASS exact=1.000000
  max|dcount|=0`), the final readout of EVERY shipped sample against its frozen
  scores and label, the accumulated ACCURACY, and wall aggregates with
  percentiles for `bo_write_s`, `sync_s`, `run_s`, `readback_s`, `decode_s`,
  `transcode_s`, `pass_total_s`, plus per-sample totals and per-core
  programming;
* `deployment_samples.tsv` — one row per pass.

**A red certificate here is not a tolerance question.** Read which PASS
diverged first: a divergence on the first pass is the card or the bitstream and
belongs in §7; a divergence that starts on a LATER pass with the first one
green is the transcode or the counts it was fed, and the first pass's stimulus
self-check (below) will already have ruled out the stimulus builder.

**The self-check that runs before any of it is trusted.** The host builds every
consumer pass's stimulus itself. Before it does, it rebuilds the FIRST pass's
stimulus and requires byte-identity with the one the repository's own encoder
froze into the bundle. If those disagree the run REFUSES
(`OdinTranscodeDiverged`) rather than stimulating a network nobody assembled —
so a stimulus-arithmetic drift can never be mistaken for a hardware finding.

**Do not hand-edit a bundle.** It carries its own sha256; an edited or damaged
one refuses as `OdinBundleCorrupt` before the card is touched. Regenerate it
with `scripts/hacc/make_deployment_bundle.py`, which needs an RTL simulator and
re-measures every count.

### Which bundle, and where a REAL network's bundle comes from

The bring-up package (`odin_hacc_package.zip`) carries the committed two-core
witness bundle and phase 8 runs that. A DEPLOYMENT package
(`odin_hacc_deployment.zip`) carries whatever bundle the export step produced,
named by `deployment/DEPLOYMENT.json`; phase 8 reads that index. TODAY that
bundle is **`odin_narrowconv_mnist_wb4_s4`** — a REAL TRAINED NETWORK, exported
2026-08-28 from `scripts/hacc/odin_deployment_cell.json`: a `narrow_conv` MNIST
vehicle as **2 host-mediated NC=1 passes**, 300 shipped samples, 50 of them
carrying frozen per-pass counts, **frozen accuracy 0.873333 (262/300)**. It
replaces the synthetic `odin_hacc_micro` witness (a transport proof whose
1.000000 over 3 one-hot stimuli must never be quoted as deployed accuracy);
that witness still ships inside the bring-up package and phase 8 runs it there.
Same bootstrap, same `run_all.sh`, and `ODIN_BUNDLE=<path>` overrides either.

What the network had to be, and why nothing off the shelf was: the stock
crossbar is 256 PHYSICAL rows and the per-axon sign expansion spends two per
logical slot, so a core holds 128 logical axons, 127 effective with no bias
lane. `lenet5`'s flatten→FC junction is one **785-axon** soft core — 1570
physical rows, 6.1x the crossbar — which the mapper refuses outright with no
inter-core membrane partial-sum transfer to coalesce into, and its two MaxPool
stages split the program into TWO neural segments where a bundle freezes one.
On top of the geometry, `export_odin` deploys only `firing_granularity=per_event`,
and the event-serial training twin folds a hop only when its effective weight
spans its WHOLE input and its upstream is another hop: a 3x3 conv over a 7x7
map is refused by the TWIN (that core fits the crossbar fine at 127 axons), and
so is a `Flatten` sitting between two hops. `narrow_conv` states both
conditions in the architecture. The measured ladder — pretrain 0.9820, LIF
0.8972, wb=4 weight quantization **0.8814** (the platform-J MLP lost 0.32
here; this vehicle loses 0.016), NF↔SCM parity **0.9883** over 256 samples,
HCM 0.8830 — is recorded in the cell's own `_note`.

That bundle is produced by the pipeline itself, not by hand. The step is
`"HACC NUS - ODIN Deployment"`; a tier cell (or any deployment document) turns
it on with `enable_odin_hacc_export` and sizes the campaign with
`odin_hacc_bundle_samples` (how many test-set samples the bundle can execute at
all) and `odin_hacc_certification_samples` (how many carry frozen PER-PASS
counts rather than just a readout). The step writes
`<run>/odin_hacc/deployment_bundle.json` and its `_capture.json`; package them:

```bash
env/bin/python scripts/hacc/make_package.py \
    --deployment generated/<run>/odin_hacc/deployment_bundle.json
```

Its expectations are the cycle-accurate twin's, gated EXACT against the HCM
torch reference sample by sample at export time (a mismatch refuses and writes
nothing) — not an RTL measurement, because cosimulating hundreds of samples is
not affordable. `provenance.derivation` says exactly that, in the file.

**A bundle can only execute the samples it ships**, because each one carries the
entry raster this program's HOST compute stages produced for it. Raising
`ODIN_DEPLOY_SAMPLES` past that count runs every shipped sample and no more;
running the full test set means exporting with a larger
`odin_hacc_bundle_samples` and paying for the larger upload.

## 6c. The chip cache, and mining a routed checkpoint

Phase 3 is 2-6 hours. Do not pay for it twice:

```bash
./scripts/chip_cache.sh key hw     # the key and every input that made it
./scripts/chip_cache.sh list       # what is already paid for
./scripts/chip_cache.sh adopt /data/${USER}/odin_hacc_package --alias v4
```

Phases 2 and 3 consult the cache before submitting and publish on success. The
key covers the RTL digest, card, platform, part, NC, `FIFO_WORDS`, `CAP_WORDS`,
the kernel clock from the card's v++ config, the Vitis release, the target and
the sha256 of `build_xclbn.sh` — one function computes it for both the build
path and `adopt`, so an adopted install lands exactly where a rebuild looks.
Publishing claims a key with `mkdir`, stages beside it and renames in, and never
clobbers an existing entry. `ODIN_CHIP_CACHE_DISABLE=1` turns it off.

After a `hw` build closes, mine the checkpoint while the temp dir still exists:

```bash
scripts/hacc/mine_checkpoint.sh hw
```

It files `report_utilization` (flat and hierarchical), congestion, post-route
timing, route status and a per-primitive placement CSV into that build's cache
entry, and draws a die map from the CSV — matplotlib to PNG if the node has it,
otherwise an SVG written out of the standard library alone. `collect_results.sh`
brings the reports and the maps home and leaves the bitstream where it is.

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
2. **The B0 round trip (B0 step 4).** Did the null program come back with a
   header the fabric wrote? `OdinFpgaKernelError: ... NO-VERDICT sentinel` means
   the kernel never drained — almost always an xclbin built with a smaller
   `CAP_WORDS` than the package declares, or built from a different kernel
   entirely, i.e. the loaded bitstream is not the one you think it is. Nothing can read the
   geometry back off the card, so compare `probe.json`'s declared capacity
   against the `.built_with` sidecar of the xclbin you staged.
3. **The capture header (B1).** `OdinFpgaCaptureTruncated` means the capture RAM
   filled — raise `CAP_WORDS` and rebuild; the counts of a truncated run are not
   a result. A sequencer that REFUSED an opcode (`err = 1` on `0x4C`) is
   invisible to the host and shows up as a FAILING certificate with missing
   counts, not as a typed refusal.
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
* Trusting the vendor doc's partition table: **field-observed 2026-08-25**,
  `cpu_only`'s node is down, `mi210_vck_u55c` refuses this account outright, and
  `hacc-gpu1` — which that doc warns off as "U250/U280" — is in fact the node
  carrying MI210 + U280 + **U55C**, i.e. the one joint venue that works. Ask
  `scontrol show partition` and `sinfo -a -N` before believing any of it; that
  is exactly what `run_all.sh` now does at phase time.
* Assuming a card is present because the cluster owns one. `xbutil` answered
  `0 devices found` on `hacc-gpu3` from a non-FPGA partition: cards are
  VM-passed at allocation time. Allocate a board partition first.
* Reaching for `xilinx_u250_gen3x16_xdma_4_1_202210_1` because it is newer than
  the 202020 one. It is 2022.2-era and hits the same lock as the U55C shell;
  `cards.sh` refuses it by name.
* Spending a queue slot to rediscover the U55C deadlock. `run_all.sh` refuses
  before submitting, from `/opt/xilinx/platforms` and the installed Vitis list.
* Running the bring-up in the foreground: v1's `sbatch --wait` chain meant every
  hiccup needed you at the keyboard. Use `bootstrap_hacc.sh`, which detaches it,
  and `scripts/status.sh` to look in.
* Assuming `.slurmech.toml` applies: it targets a different cluster entirely
  (xlog1/H100). HACC submission is hand-written `sbatch`, which is what this
  directory is.
