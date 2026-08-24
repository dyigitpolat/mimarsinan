# ODIN on HACC@NUS — the whole bring-up in one zip (v2)

You are logged in to HACC. The machine this package was built on cannot reach
the cluster, so everything the bring-up needs travels here: the RTL the build
compiles, the fixtures with their expected counts already frozen, and one host
driver that needs nothing but Python 3 and XRT.

**One command. Upload `odin_hacc_package.zip` and `bootstrap_hacc.sh` into the
same directory, then:**

```bash
./bootstrap_hacc.sh
```

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
| 5 | **B0**: load, resolve, read the CSRs | U55C partition | < 1 board hour |
| 6 | **B1**: every fixture, certified | U55C partition | < 1 board hour |
| 7 | board + independent reference in one job | a joint partition | queue-bound |

Phase 3 dominates. `--only N` runs one phase, `--from N` starts at one,
`--force` redoes one whose artifact is already there.

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

The account this was observed from is in `yigit video render hgpu gpgpu
fpga_u280 fpga_u250 fpga_vck5000` — **not** `fpga_u55c`, **not** `lab`. `hgpu`
is what opens both board partitions.

**Never submit a build to a board partition** — it is killed at one hour, every
time. The candidate lists encode that: builds only ever consider compile-class
partitions.

### The toolchain is discovered, not assumed

FIELD-OBSERVED 2026-08-25: Vitis is **2024.2** under **`/tools/Xilinx`** (capital
X). The vendor docs' 2022.2 under `/tools/xilinx` is stale, and v1 refused at
phase 0 because of it. `scripts/hacc/toolchain.sh` now probes `/tools/Xilinx`,
`/tools/xilinx`, `/opt/Xilinx`, `/opt/xilinx` and takes the newest Vitis it
finds; `XILINX_ROOT` and `VITIS_VERSION` pin it if you need a specific one.

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
(`odin_fpga_kernel_top`); the `s_axilite` register map answers — the two
read-only capacity registers report the geometry the bitstream was compiled
with, and the status register reports `err`; the AXI master moves the program
and stimulus payloads and the sequencer executes them; the capture buffer
decodes into the per-neuron counts this package froze from the RTL
cosimulation. A packaging error — a port mismatch, a wrong offset, a kernel that
does not resolve — dies here instead of costing a board hour.

**Only silicon proves:** real HBM ordering behind a real XDMA shell, the
shell's address translation, XRT's buffer allocation on a device, and timing
closure at the kernel clock. That is exactly why **B0 comes before B1** rather
than after it.

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
plumbing for it does not yet). The kernel reports `PROG_WORDS = NC * 262144`, so
the driver can see how many cores the loaded bitstream holds and **skips**
fixtures that program more, naming the reason. On an NC=1 bitstream the
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

Results land as JSON under `results/`: `probe.json` (B0's CSR read),
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
2. **The capacity registers (phase 5).** Do `capture_capacity` and
   `program_capacity` read back the geometry you built? A wrong pair means the
   loaded xclbin is not the one you think it is.
3. **`err` on the status register.** `err = 1` is the fabric REFUSING, not
   disagreeing: an opcode it does not implement, an AER handshake that timed
   out, or a payload that did not fit. The driver names it
   `OdinFpgaKernelError` and decodes nothing.
4. **`OdinFpgaCaptureTruncated`.** The capture RAM filled. Raise `CAP_WORDS` in
   `hw/fpga/kernel/odin_fpga_kernel_top.v` and rebuild; the counts of a
   truncated run are not a result. Lowering the sample count until it fits is a
   way to get a number, not a way to get a result.
5. **`OdinFpgaProgramTooLarge`.** The token stream does not fit the fabric's
   program RAM. Raise `PROG_WORDS` and rebuild — same rule.
6. **Re-run phase 2's `hw_emu` smoke** on the same fixtures. Same program bytes,
   no board. If emulation passes and the card does not, the divergence is in the
   shell, the silicon or timing closure — not in the export and not in the
   delivery.
7. **Hot-reset the card** and repeat. A wedged AER link after a previous job
   looks exactly like a physics bug:
   `xbutil reset -d <board_id> --force`, with the id from the bracketed column
   of `xbutil examine`.
8. **Then, and only then, silicon.** Read the build's `reports/` for timing
   violations at the kernel clock before concluding anything about the design.
   If timing is the problem, lower `kernel_frequency` in
   `scripts/hacc/odin_u55c.cfg` before touching the design: the ODIN core is a
   slow, event-serial machine and does not need a fast clock.

If a build fails instead, `package_xo` port errors mean the kernel.xml and the
Verilog port list disagree — but this package's `kernel.xml` was frozen from the
host-side register SSOT at packaging time, so that would be a packaging bug to
report home, not something to hand-edit here.

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
can produce them. The xclbin itself is deliberately left behind.
