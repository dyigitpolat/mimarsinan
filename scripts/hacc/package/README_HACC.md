# ODIN on HACC@NUS — the whole bring-up in one zip (v3)

You are logged in to HACC. The machine this package was built on cannot reach
the cluster, so everything the bring-up needs travels here: the RTL the build
compiles, the fixtures with their expected counts already frozen, and one host
driver that needs nothing but Python 3 and XRT.

## Read this before you run anything: the U55C is blocked, use the U250

**Field-observed 2026-08-25.** The only U55C platform installed anywhere on
this cluster — checked on `hacc-node0` **and** `hacc-gpu3` under
`/opt/xilinx/platforms` — is `xilinx_u55c_gen3x16_xdma_3_202210_1`. Its
SmartConnect IP is customized with the **2022.2** release, and 2022.2 is the
one Vitis this cluster does **not** carry:

```
/tools/Xilinx/Vitis:  2020.1  2020.2  2021.2  2022.1  2023.2  2024.2
```

Three `v++ --link` attempts — under **2022.1, 2023.2 and 2024.2** — died the
same way, in `vpl`, with `VPL 60-704` / `60-732`: *"IP ... is customized with
software release 2022.1 and is being used with a different revision."* There is
no flag, no config and no version of this package that gets past a locked IP
revision. **The U55C is deadlocked pending an admin action**, and there are
exactly two of those: install Vitis/Vivado 2022.2 beside the others, or install
a U55C shell built for a release that is already here.

**So the card became a parameter.** The route that builds on HACC@NUS today is
the **U250**: `xilinx_u250_gen3x16_xdma_3_1_202020_1` (a 2020.2-era shell) with
**Vitis 2020.2**, which is installed. This account is in `fpga_u250`, and all
three U250 partitions admit it.

**One command. Upload `odin_hacc_package.zip` and `bootstrap_hacc.sh` into the
same directory, then:**

```bash
./bootstrap_hacc.sh --card u250
```

`--card u55c` is still the default and still works the moment the admins act —
until then it refuses at startup, before submitting anything, and prints the
paragraph above with the installed lists it actually read. That refusal is
**evidence-gated, not a hard-coded verdict**: it re-reads
`/opt/xilinx/platforms` and the installed Vitis versions every run, and
disappears on its own when either of the two fixes lands.

It checks the zip against the sha256 it was cut for, archives any previous
install to `odin_prev_<utc>.tar.gz` (it never deletes one silently), unpacks a
fresh one under `/data/${USER}/odin_hacc_package`, and launches `run_all.sh`
**detached** — `setsid nohup`, output to `results/run_all.log`. It prints the
three lines you need: how to watch it, how to ask for status, and that it is
safe to log out. Nothing about this bring-up needs your shell to stay open.

Everything unpacks **under `/data`**: that is the only path shared between the
head node and the board VMs, so a package anywhere else is invisible to the jobs
that have to read it. (`ODIN_DATA_ROOT` moves it, for rehearsals off-cluster.)

While it runs:

```bash
tail -f /data/${USER}/odin_hacc_package/results/run_all.log
/data/${USER}/odin_hacc_package/scripts/status.sh    # phases, squeue, last lines
```

Re-running `./run_all.sh` by hand is always safe: a live run holds a lock and a
second one refuses by pid, and anything already finished is kept, not redone.
`./run_all.sh --dry-run` prints every command and executes nothing.

---

## What the phases do, and what each one costs

| # | Phase | Where | Wall |
|---|---|---|---|
| 0 | env probe → `results/env.txt` | hacchead | seconds |
| 1 | driver selftest vs a fake `pyxrt` | anywhere, no hardware | seconds |
| 2 | build `hw_emu` + a bounded emulation smoke | a compile partition | ~15 min + smoke |
| 3 | build `hw` (the real bitstream) | a compile partition | **2–6 h** |
| 4 | stage the xclbin under `/data/${USER}/odin` | hacchead | seconds |
| 5 | **B0**: introspect, load, open EXCLUSIVE, run a null program, read its header back | a board partition for your card | < 1 board hour |
| 6 | **B1**: every fixture, certified | a board partition for your card | < 1 board hour |
| 7 | board + independent reference in one job | a joint partition | queue-bound |
| 8 | **HACC NUS - ODIN Deployment**: the sealed bundle run as one host-mediated pass per core | a board partition for your card | minutes |

Phase 3 dominates — and the **chip cache** below is what stops you paying for
it twice. `--only N` runs one phase, `--from N` starts at one, `--force` redoes
one whose artifact is already there.

### Resume is artifact-based — there are no stamps

A phase is done when the thing it was supposed to produce EXISTS: the xclbin
plus a `.built_with` sidecar naming the sha256 of the `build_xclbn.sh` that
produced it, the staged bitstream, the driver's own result JSONs. So:

* kill the run, log out, come back, run `./run_all.sh` — it picks up exactly
  where the artifacts stop and never re-pays for a finished build;
* edit `scripts/hacc/build_xclbn.sh` and the xclbin is stale by definition; the
  build runs again and says which sha256 it was expecting;
* nothing has to be stamped, un-stamped or `--force`d by hand to make that
  work. Every failure path ends with the same line —
  `fix, then re-run ./run_all.sh — completed work is kept`.

### Partitions are chosen from the cluster, not from a document

`run_all.sh` picks the partition **at phase time** out of a candidate list, and
keeps a candidate only if BOTH gates hold: `scontrol show partition` shows it
and admits one of your groups, and `sinfo` shows it a node that is not down or
drained. It prints the winner and why each earlier candidate lost.
`ODIN_BUILD_PARTITION` / `ODIN_BOARD_PARTITION` / `ODIN_JOINT_PARTITION` still
win outright if you know better.

| Partition | Field-observed 2026-08-25 | The stale doc claim |
|---|---|---|
| `cpu_only` | maps to **hacc-gpu0**, `down*` — `NO NETWORK ADDRESS FOUND` | `hacc-node2`, 7 days, the build venue |
| `vck5000_compile` | `AllowGroups=ALL`, `MaxTime=7-00:00:00`, **hacc-node0 idle** — the living build venue | not mentioned |
| `xilinx_u55c_gen3x16_xdma_3_202210_1` | `AllowGroups=lab,hgpu,fpga_u55c`, 1 h — reachable **through `hgpu`**, without `fpga_u55c` | hacc-gpu1/2/3, one hour |
| `mi210_vck_u55c` | **absent** from `scontrol show partition`; a submission answers `User's group not permitted to use this partition` | hacc-gpu2/3, 7 days, the joint venue |
| `mi210_u280_u55c` | `AllowGroups=lab,hgpu,fpga_u280`, 12 h, on **hacc-gpu1** which carries MI210 + U280 + U55C — the living joint venue | listed as `mi210_u250_u55c`, "its GPU pair is U250/U280" |
| `mi210_u280_u55c_long_reservation` | same groups, **5 days** | not mentioned |

The **U250** venues, from the same session — these are the ones `--card u250`
uses, and this account is in `fpga_u250`:

| Partition | AllowGroups | MaxTime | Role |
|---|---|---|---|
| `xilinx_u250_gen3x16_xdma_3_1_202020_1` | `lab,fpga_u250` | **1 h** | the bare shell partition; first choice for B0/B1, which fit inside an hour |
| `u250_standard_reservation_pool` | `lab,fpga_u250` | 12 h | for anything longer, and the first choice for the joint phase |
| `u250_long_reservation_pool` | `lab,fpga_u250` | 2 d | the fallback when the pool is busy |

The account this was observed from is in `yigit video render hgpu gpgpu
fpga_u280 fpga_u250 fpga_vck5000` — **not** `fpga_u55c`, **not** `lab`. `hgpu`
is what opens both U55C partitions; `fpga_u250` opens all three U250 ones.

**A card is allocated, not ambient.** On 2026-08-25 `xbutil` on `hacc-gpu3`
answered `0 devices found` — from a non-FPGA partition. The cards are passed
into the job's VM at allocation time, so only a board-partition job sees one.
An empty `xbutil examine` means the wrong partition, not a dead card.

**Never submit a build to a board partition** — it is killed at one hour, every
time. The candidate lists encode that: builds only ever consider compile-class
partitions, and those are card-independent (a place-and-route needs a compile
venue, not a card).

### The card is a parameter, and `cards.sh` is the only place it lives

`scripts/hacc/cards.sh` holds one profile per card, and nothing card-shaped is
written down anywhere else:

| | `u55c` (default) | `u250` (the field-viable route) |
|---|---|---|
| platform | `xilinx_u55c_gen3x16_xdma_3_202210_1` | `xilinx_u250_gen3x16_xdma_3_1_202020_1` |
| Vivado part | `xcu55c-fsvh2892-2L-e` | `xcu250-figd2104-2L-e` |
| v++ config | `scripts/hacc/odin_u55c.cfg` | `scripts/hacc/odin_u250.cfg` |
| AXI master lands on | `HBM[0]` | `DDR[0]` — the U250 has no HBM |
| preferred Vitis | 2022.2 (**absent**: the deadlock) | **2020.2** |
| board partitions | the U55C shell partition | the three above, 1 h first |
| joint partitions | `mi210_vck_u55c`, `mi210_u280_u55c`, `…_long_reservation` | the 12 h pool, then 2 d, then 1 h |

`xilinx_u250_gen3x16_xdma_4_1_202210_1` is also on the build node and is
**refused by name**: it is a 2022.2-era shell and would hit exactly the same
lock the U55C does.

Overrides still win where you need them: `ODIN_PLATFORM`, `ODIN_FPGA_PART`,
`ODIN_VXX_CONFIG`, `VITIS_VERSION`, and the three
`ODIN_{BUILD,BOARD,JOINT}_PARTITION`.

### The toolchain is discovered, not assumed

FIELD-OBSERVED 2026-08-25: Vitis lives under **`/tools/Xilinx`** (capital X) —
the vendor docs' `/tools/xilinx` is stale, and v1 refused at phase 0 because of
it — and the installed set is `2020.1 2020.2 2021.2 2022.1 2023.2 2024.2`.

`scripts/hacc/toolchain.sh` probes `/tools/Xilinx`, `/tools/xilinx`,
`/opt/Xilinx`, `/opt/xilinx` and picks, in this order: `VITIS_VERSION` if you
set it, else **the card profile's preferred release** if it is installed, else
the newest. "Newest" alone is not right: a platform's IP is locked to the
release it was built with, which is the entire U55C story, so `--card u250`
deliberately builds under **2020.2** and not under 2024.2.

---

## What phase 2 (`hw_emu`) can prove, and what only the card can

`XCL_EMULATION_MODE=hw_emu` binds XRT to the emulation model `v++` packaged
into the xclbin, so phase 2's smoke needs **no board** and runs on the build
node.

**It is BOUNDED.** An xsim-backed emulation has no published wall, so the
default smoke runs the SMALLEST shipped fixture (`nc1_single_core_ceiling`)
under `ODIN_EMU_SMOKE_TIMEOUT` (5400 s). If it runs out of time the build still
stands and `results/hw_emu/EMU_SMOKE_SKIPPED.txt` says so honestly: the wall is
unknown, not slow-but-fine, and B0/B1 on silicon supersede it either way.
`ODIN_EMU_SMOKE=all` runs all five fixtures; `ODIN_EMU_SMOKE=off` skips it.

**hw_emu proves:** the packaged kernel opens by its name
(`odin_fpga_kernel_top`) with the six arguments the frozen `kernel.xml`
declares; the AXI master moves the program and stimulus payloads and the
sequencer executes them; the capture buffer comes home carrying a header the
fabric wrote, and decodes into the per-neuron counts this package froze from the
RTL cosimulation. (It cannot prove anything about the `s_axilite` status or
capacity registers: pyxrt binds no register access, so no host reads them —
see `host/odin_board_driver.py`.) A packaging error — a port mismatch, a wrong offset, a kernel that
does not resolve — dies here instead of costing a board hour.

**Only silicon proves:** real memory ordering behind a real XDMA shell — HBM on
the U55C, **DDR on the U250**, and the connectivity config is the only line
that differs — the shell's address translation, XRT's buffer allocation on a
device, and timing closure at the kernel clock. That is exactly why **B0 comes
before B1** rather than after it.

Three simulation-coverage shadows are known and are the first place to look at
a board-side AXI anomaly: the datapath is 32-bit-beat only; the 4 KiB
burst-boundary clamp is never exercised by the testbench (its buffers are all
4 KiB-aligned, so it is correct by inspection only); and the RRESP/BRESP error
paths are untested because the testbench slave always answers OKAY.

---

## The fixtures

`fixtures/INDEX.json` lists what shipped. Each fixture carries its program
word-stream, its stimulus word-stream, its run parameters, the expected
per-neuron counts, the commit that generated it, and its own SHA-256; the
driver refuses a fixture whose bytes do not hash to what it claims, so a damaged
upload never certifies.

Those expected counts came out of the **committed RTL cosimulation** — the
byte-identical vendored ODIN core executing the exporter's own sequencer
program — and packaging asserted them against the golden gates cycle by cycle
before writing this zip.

**NC = 1.** The v1 packaging flow builds one ODIN core only
(`build_xclbn.sh` refuses more; the RTL parameter exists but the `package_xo`
plumbing for it does not yet). The package DECLARES that geometry, so the
driver **skips** fixtures that program more cores, naming the reason. On an NC=1 bitstream the
`nc1_*` fixtures run and the rest are skipped; that is expected, not a failure.

---

## Reading the result

Each fixture prints one line in the house format:

```
[SpikeCountCertificate] spike-count certificate [odin_fpga/exact]: PASS exact=1.000000 max|dcount|=0 over N neuron-windows, M sample(s)
```

`odin_fpga` is classified **exact**. A hardware read that needs slack is a
different chip, not a looser tolerance — anything other than
`exact=1.000000 max|dcount|=0` is a finding.

Results land as JSON under `results/`: `probe.json` (B0's round trip, with its
own `proves` / `does_not_prove` lists),
`fixture_<name>.json` per fixture, `summary_board.json`, and for phase 7
`summary_join.json`. The measured walls are in each fixture's `walls`:
programming (the payload DMA) is reported separately from execution and is
never folded into it.

---

## If the board disagrees with the frozen counts

Do not adjust a tolerance. Work it in this order — cheapest and most likely
first.

1. **Shell / XRT version.** `xbutil examine` on the node. The cluster's current
   table (`hacc_demo/README.md`) lists the U55C cluster at **XRT 2.18.179**
   with Vitis 2022.2; `hacc_demo/doc/0-login.md` still shows 2.14.384 for the
   same shell and is stale. A mismatch against what the xclbin was linked with
   is triage step 1, not a footnote.
2. **The B0 round trip (phase 5).** Did the null program come home with a
   header the fabric wrote? `probe.json` also records the capacity this package
   DECLARES — nothing can read it back off a card — so compare it against the
   `.built_with` sidecar of the xclbin you staged. A mismatch means the loaded
   xclbin is not the one you think it is.
3. **`OdinFpgaKernelError`.** The capture header came back still carrying the
   host's no-verdict sentinel: the fabric refused at `ap_start` and never
   drained, almost always because the built geometry is smaller than the
   declared one. A sequencer that refuses an opcode MID-run is invisible to the
   host (its `err` bit lives on `0x4C`, which pyxrt cannot read) and shows up as
   a FAILING certificate with missing counts instead.
4. **`OdinFpgaCaptureTruncated`.** The capture RAM filled. Raise `CAP_WORDS` in
   `hw/fpga/kernel/odin_fpga_kernel_top.v` and rebuild; the counts of a
   truncated run are not a result. Lowering the sample count until it fits is a
   way to get a number, not a way to get a result.
5. **Re-run phase 2's `hw_emu` smoke** on the same fixtures. Same program bytes,
   no board. If emulation passes and the card does not, the divergence is in the
   shell, the silicon or timing closure — not in the export and not in the
   delivery.
7. **Hot-reset the card** and repeat. A wedged AER link after a previous job
   looks exactly like a physics bug:
   `xbutil reset -d <board_id> --force`, with the id from the bracketed column
   of `xbutil examine`.
8. **Then, and only then, silicon.** Read the build's `reports/` for timing
   violations at the kernel clock before concluding anything about the design.
   If timing is the problem, add a `kernel_frequency` to **your card's** config
   (`scripts/hacc/odin_u250.cfg` or `odin_u55c.cfg` — the `.built_with` sidecar
   next to the xclbin names which one built it) before touching the design: the
   ODIN core is a slow, event-serial machine and does not need a fast clock.

If a build fails instead, `package_xo` port errors mean the kernel.xml and the
Verilog port list disagree — but this package's `kernel.xml` was frozen from the
host-side register SSOT at packaging time, so that would be a packaging bug to
report home, not something to hand-edit here.

---

## HACC NUS - ODIN Deployment

Phase 6 certifies FIXTURES: frozen programs whose counts the RTL cosimulation
recorded. Phase 8 deploys a NETWORK, and the difference is the whole point.

The shipped bitstream instantiates ONE ODIN core (`NC = 1`), and the chip
routes nothing between cores in any case — `SPI_OPEN_LOOP`
is asserted, so v1 routing is host-mediated by design. A multi-core network
therefore runs as one **pass per core**:

1. program the core — ONCE; fabric memories persist across sequencer runs, so
   every later sample costs a stimulus, not a reprogram;
2. run one sample: the stimulus opens with the membrane CLEAR (an op, not a
   reprogram), then one TAG / AER burst / TREF / BARRIER per cycle;
3. read the capture back and fold it into per-cycle, per-neuron counts;
4. **transcode**: walk the next core's axon-source table and turn the producer's
   counts into that core's per-slot counts — the producer's counts from the
   cycle BEFORE, the entry raster from this one;
5. build that core's stimulus on the host from those slot counts, and run again.

```bash
./run_all.sh --only 8                        # after phases 3-4 have staged an xclbin
ODIN_DEPLOY_SAMPLES=8 ./run_all.sh --only 8  # bound the campaign
```

**What it certifies.** Per-pass spike counts against the frozen cosimulation on
a *certification subset* of samples, and the final readout — the class scores
and the predicted label — on **every** shipped sample. Both in the house
format; anything other than `PASS exact=1.000000 max|dcount|=0` is a finding,
not a tolerance. It also accumulates ACCURACY against the bundle's labels.

**Where the numbers come from.** `deployment/*.json` is a sealed bundle: a
manifest with the generating commit and the chip-config inputs, per-core PROGRAM
streams, per-sample core-0 STIMULUS streams, the routing plan, and the frozen
expectations. The file carries its own hash: a damaged upload refuses instead of
certifying, and its `provenance.derivation` says in words WHAT produced its
expectations — either the RTL cosimulation on the vendored core (the committed
witness bundle) or the cycle-accurate twin gated sample by sample against the
HCM torch reference (a bundle exported from a trained network, where a
cosimulation of hundreds of samples is not affordable). Neither is a silicon
measurement; the run you are about to start is.

**Which bundle phase 8 runs.** A **bring-up** package carries the committed
two-core witness bundle and runs that. A **deployment** package
(`odin_hacc_deployment.zip`, built by
`scripts/hacc/make_package.py --deployment BUNDLE`) additionally carries an
exported network and `deployment/DEPLOYMENT.json`, the index naming the default.
The bootstrap flow is the same one either way — `./bootstrap_hacc.sh --card
u55c` — and phase 8 reads the index when it is there. `ODIN_BUNDLE=<path>`
always wins, so one install can run either network:

```bash
ODIN_BUNDLE=deployment/<other>.json ./run_all.sh --only 8
```

**How many samples.** A bundle can only execute the samples it SHIPS: each one
carries the entry raster this program's host stages produced for it. The
campaign default runs the first 64; `ODIN_DEPLOY_SAMPLES=N` widens or narrows
it, and `ODIN_DEPLOY_SAMPLES=100000` simply runs every shipped sample. Shipping
the whole test set is a matter of raising `odin_hacc_bundle_samples` at export
time and paying for the larger upload.

**The one self-check worth knowing about.** The host builds every consumer
pass's stimulus itself. Before it trusts that builder, it rebuilds the FIRST
pass's stimulus and requires it to be byte-identical to the frozen one the
repository's own encoder produced. If those two ever disagree, the run refuses
rather than stimulating a network nobody assembled.

**What comes out**, under `results/board_deploy/`:

* `deployment_report.json` — the certificates, the readout of every sample, the
  accuracy, and wall aggregates with percentiles for **every distinct stage**:
  `bo_write_s`, `sync_s`, `run_s`, `readback_s`, `decode_s`, `transcode_s`,
  `pass_total_s`, plus per-sample totals and per-core programming;
* `deployment_samples.tsv` — one row per pass, capped so it stays readable.

---

## The chip cache — never pay for the same bitstream twice

A place-and-routed xclbin is 2–6 hours. `scripts/chip_cache.sh` keeps it under
`${ODIN_CHIP_CACHE:-/data/${USER}/odin_chip_cache}/<key>/`, and phases 2 and 3
consult it before they spend a compile slot:

```bash
./scripts/chip_cache.sh key hw       # the key, and every input that made it
./scripts/chip_cache.sh list         # what is already paid for
./scripts/chip_cache.sh adopt /data/${USER}/odin_hacc_package --alias v4
```

The key is a sha256 over everything that could change the bitstream: the RTL
digest the MANIFEST carries, the card, the platform, the part, NC, `FIFO_WORDS`,
`CAP_WORDS`, the kernel clock from the card's v++ config, the Vitis release, the
target, and the sha256 of `build_xclbn.sh` itself — because the build script IS
the recipe, and without it a cache hit could resurrect a bitstream that
run_all.sh's own artifact-resume rule had just called stale.

`adopt` takes an install that ALREADY paid — the v4 install is the first entry —
and derives its key from that install's own sidecars and manifest through the
same function the build path uses, so the entry lands exactly where a rebuild
will look for it. `--alias NAME` leaves a readable symlink beside it.

Publishing claims a key with `mkdir` (the one portable atomic test-and-set on a
shared filesystem), stages beside the entry and renames in, and **never**
clobbers an entry that is already there: two jobs that computed the same key
built the same bitstream, and the one on disk may already have been read.
`ODIN_CHIP_CACHE_DISABLE=1` turns the whole thing off.

### Mining a routed checkpoint

`scripts/hacc/mine_checkpoint.sh` opens the routed checkpoint v++ left under
`--temp_dir` and files what only the real shell can say — `report_utilization`
(flat and hierarchical), `report_design_analysis -congestion`,
`report_timing_summary`, `report_route_status`, and a per-primitive placement
CSV — into the chip-cache entry for that build, then draws a die map from the
CSV with `host/render_die_map.py` (matplotlib if the node has it, otherwise an
SVG written straight out of the standard library). The kernel's sub-blocks get
ink; the shell stays grey.

```bash
scripts/hacc/mine_checkpoint.sh hw            # after phase 3
```

---

## Bringing the evidence home

```bash
./collect_results.sh          # -> odin_hacc_results_$(hostname).tar.gz
```

It gathers `results/` (including `phase_journal.tsv` and
`partition_picks.txt`, so the evidence names the partition that produced it),
the `.built_with` sidecars, the package `MANIFEST.json`, and the
build's `reports/` and `logs/` — the timing and utilization reports are the
real-shell half of the implementation-closure evidence and nothing off-cluster
can produce them. It also brings the deployment report and its per-pass TSV, and
— for every target whose bitstream is in the chip cache — that entry's key
inputs, mined reports and die maps. The xclbin itself is deliberately left
behind: the pictures and the tables are kilobytes, the bitstream is hundreds of
megabytes and stays where it is.
