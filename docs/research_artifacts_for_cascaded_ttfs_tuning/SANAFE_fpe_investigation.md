# SANA-FE SIGFPE investigation (mmixcore, 109-core mapping)

## RESOLVED (2026-06-20): unpinned-dependency regression

**Root cause: an unpinned `pip install sanafe` in `bootstrap_sanafe.sh` upgraded SANA-FE
from 2.1.1 → 2.2.6 on 2026-06-17.** The integration targets 2.1.1; 2.2.x SIGFPEs on arch
load. SANA-FE last succeeded Jun 8–9 (2.1.1); the Jun 19 batch failed (2.2.6) with a
**byte-identical mapping** — only the sanafe version changed. **Fix (commit 51dfdf4):**
pin `sanafe==2.1.1` in the bootstrap + a `_check_sanafe_version` guard that fails loud
instead of core-dumping. **Confirmed:** with 2.1.1 the previously-crashing mmixcore
SANA-FE step COMPLETES — Parity 1.0, 594140 spikes, 0.043 mJ. **Full fresh
end-to-end run (cascaded, two-stage, all backends) also completes**: SCM 0.9514,
HCM 0.9514, nevresim Simulation OK, SANA-FE Simulation Parity 1.0 / 0.0405 mJ /
475925 spikes — AC6 (zero crashes, all backends) met on this venv (Lava absent).

The "needs a debugger" conclusion below was SUPERSEDED — the regression was found by
date-bisecting the runs (last-good Jun 8 vs broken Jun 19) to the sole change: the sanafe
version. The phantom-tile mesh fix (1341f57) is an independent correctness improvement
(the Jun 8 good run used the old 4×3 mesh and succeeded — phantom tiles were never the
trigger). Original (now-historical) investigation follows.

---


**Symptom.** The `SANA-FE Simulation` step crashes the whole process with **SIGFPE
(exit 136, core dump)** on the mmixcore mapping — `sanafe_sample_count=1`, so it's not
slowness. It kills Python directly (the C++ raises the FP exception), so the run status
is never updated → the 2026-06-19 batch showed a stale `running` (it had crashed). The
`sanafe` C++ extension itself is fine (`test_sanafe_runner` + `test_sanafe_simulation_step`,
26 tests that drive the real `chip.sim()`, all pass on small nets).

## What was ruled OUT (each by experiment)

| hypothesis | experiment | result |
|---|---|---|
| arch energy/latency preset | re-ran with `sanafe_arch_preset="truenorth"` | **also SIGFPE** → not preset/energy |
| `threshold == 0` core | scanned cached HCM (109 cores) | min threshold 14.84, none ≤ 0 |
| `active_length == 0` gating | LIF path sets `active_length = T` (=4), never 0 | not the cause |
| phantom-tile mesh (`width*height > n_tiles`) | fixed to exact `5×2` (commit 1341f57) | **still SIGFPE** with the exact mesh |
| FPE in `chip.sim` / `chip.load` | monkeypatched both (reassignable; verified) | neither fired → FPE is **earlier** |

## Localization

The FPE is in the C++ **before** `SpikingChip.load`/`.sim` — i.e. in
`sanafe.load_arch(yaml)` or the `SpikingChip(arch)` constructor (both run while building
the chip from the synthesized arch). The rendered arch YAML (10257 lines, 109 cores)
shows no anomalous fields (mesh `5×2`, `axons/neurons_per_core=512`, all dims > 0). The
arch references the custom mimarsinan soma/dendrite plugin `.so`s (present, built
2026-06-17) — a plugin registration/init during `load_arch` is a candidate.

## Blocker

Pinpointing a C++ divide-by-zero in an **installed** extension needs a debugger:
**no gdb / lldb / valgrind / catchsegv on this host, non-root (uid 1005), no network** →
cannot install one and cannot get the C++ backtrace. Net/arch bisection on a 109-core
mapping without a debugger is the only remaining route and is open-ended.

## Next step (for a host with a debugger or a sanafe debug build)

1. `gdb --args python run.py --headless <resume-sanafe-config>`, `catch signal SIGFPE`,
   `run`, `bt` → the exact C++ frame (load_arch vs constructor vs plugin init).
2. Or build `sanafe` from source with FP-trap symbols and step the arch load.
3. Then guard the degenerate value on OUR side (the plugins-not-patches rule):
   `experiments/_sanafe_loihi_probe.json` (num_workers=0) is the minimal repro;
   `experiments/sanafe_fpe_probe.py` is the load/sim monkeypatch harness used here.

Until then, SANA-FE is the only backend that does not run end-to-end on mmixcore; the
deployment accuracy comes from nevresim (works) and the SCM identity sim (works).
