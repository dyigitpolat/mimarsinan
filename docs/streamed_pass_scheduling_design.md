# Streamed LIF × multi-pass scheduling — design

2026-08-13 · status: **design of record, implementing** · owner decisions recorded in §7

## 1. The question

`allow_scheduling` is locked off for streamed LIF:

```python
doc="... Streamed lif needs the whole span resident in ONE program: locks false."
legal_values=lambda cfg: (False,) if _is_streamed_lif(cfg) else (False, True)
```

The stated reason is real, but the conclusion is too strong. A segment that does not
fit the grid *can* be cut into passes under streamed semantics — provided the cut
carries what streaming actually depends on.

## 2. What actually breaks today

A **pass is implemented as a segment**. `_flush_scheduled_segment` gives every pass its
own `HybridStage`, and every stage hands off through

```python
state_buffer: Dict[node_id, ndarray[samples, size]]     # NO time axis
```

The splitter says so itself:

> "adjacent sub-segments communicate through the state buffer at segment-level rates
> (same semantics as a real ComputeOp sync barrier)."

That is the **windowed** transcode: collapse to a count, re-emit as an even train.
For `spiking_variant=synchronized` it costs nothing, because timing is normalized at
every boundary anyway. For **streamed** it is a lossy interior transcode, and the
`atol=0` NF↔SCM gate forbids exactly that. The lock protects a real invariant — from
the wrong side.

## 3. The two concepts that were fused

| | meaning | boundary transfer |
|---|---|---|
| **Neural Segment** | a SEMANTIC unit: one continuous streamed window between host ComputeOps | **collapse** — host ops compute on counts |
| **Pass** | a PHYSICAL unit: one chip program, one residency of the core pool | **verbatim** — the spike raster crosses unchanged |

Separating them is the whole change. A pass boundary is **semantically invisible**;
that is a test, not a claim (§6).

## 4. The invariant that makes it rigorous

> **A pass is a spatial cut of the core DAG, never a temporal one.**

Every core runs its entire cycle window inside one pass, so **membrane state never
crosses a pass boundary** — only spike rasters do. The segment is feedforward and
causal, so replaying the exact raster reproduces the fused execution bit-for-bit.

Two consequences that do the real work:

**(a) Absolute cycle time.** A pass does **not** renumber its cores' latencies. All
passes of a segment share one absolute cycle grid `[0, latency_max + T)`, and a carried
raster is recorded and replayed in that grid. Nothing re-times, so no alignment
argument is needed — including for skip connections, whose fused skew is reproduced
because it is the *same* grid.

**(b) A carried wire is a `buf` wire whose producer lives in an earlier pass.** In the
packed executor a core gathers its inputs from three places:

```python
if bucket.on_dst  is not None: ...            # always-on bias
if bucket.inp_dst is not None: src = train[cycle - bucket.latency]   # segment input, LOCAL time
if bucket.buf_dst is not None: fires.index_select(1, bucket.buf_src) # same pass, previous cycle
```

The change adds exactly one more, with identical semantics to `buf` and different
storage:

```python
if bucket.carry_dst is not None: carried[cycle - 1].index_select(1, bucket.carry_src)
```

`fires` is the live previous-cycle vector; `carried` is a replayed one. Bit-exactness
is therefore **structural**, not argued.

## 5. The true graph cut

The cut between passes is **every edge from a core in pass ≤ p to a core in pass > p** —
not "the last latency group". Two cases the naive view misses, both required by the
owner's "properly routing the potentially incomplete latency groups":

- **A split latency group.** If group *g* is halved across passes p and p+1 (its cores
  are mutually independent, so this is legal), then *g*'s own **inputs** must also be
  carried into p+1, because the second half still needs them.
- **A long live range.** A wire produced in pass p and last consumed in pass q > p+1
  stays live across every pass between. That is a liveness interval, exactly as in
  register allocation; the buffer requirement is the peak live set, not the widest
  single cut.

Sizing: a carried wire costs `ceil(T/8)` bytes. A 4096-wire cut at T=32 is **16 KB** —
which is why this is physically realistic rather than a simulation-only trick, and why
it must be *charged* rather than hidden (§7.3).

## 6. Acceptance

The parity ladder already in the repo gives the criterion for free:

```
HCM(k passes)  ==  HCM(1 pass)  ==  SCM identity  ==  NF        all at atol=0
```

The load-bearing test is **fused-vs-passed equivalence**: the same model, same weights,
run once on a grid that fits and once on a grid that forces ≥2 passes, must produce
bit-identical per-neuron window counts. Any backend that cannot carry a raster must
**refuse by name**, never silently collapse.

## 7. Owner decisions (asked and answered 2026-08-13)

1. **Cut objective: fewest passes.** Keep today's greedy latency-group accumulation —
   reprogramming energy and time dominate, and `max_schedule_passes` already bounds it.
   The carry width falls where it falls, and §7.3 makes it visible.
2. **Backend scope: all of them** — the HCM torch executor (the parity anchor),
   SANA-FE, nevresim (C++ program-format seam) and lava.
3. **The carry is a first-class costed quantity** — `carried_raster_bytes` and
   `carry_peak_live_bytes` enter the quantity catalog, are sealed in the deployment
   record, and are priced through `e_dma_per_byte`, so a pass that halves cores but
   doubles carry stops looking like a free win.

## 8. What already exists (and why the change is contained)

The architecture anticipated this. `encode_segment_input` **already** prefers a cached
train over a re-encode, and already refuses a partial cache under streamed semantics:

> "Every non-raw input slice must have a cached train (cycle-accurate parity)."

`state_buffer_spikes: Dict[node_id, train]` **already** exists beside the count buffer,
and `decref_consumers` already frees trains on the same refcount. SANA-FE's
`set_input_spike_trains(..., encoded)` already takes an arbitrary `(1, size, T)` binary
raster.

The one thing missing: **a neural stage never publishes its output raster.** Only host
ComputeOps write `state_buffer_spikes` (via `encode_compute_boundary`). Neural stages
write counts and stop. So the core of the change is symmetric and small — *a neural
segment publishes its output train exactly as a ComputeOp does*, but only for the wires
a later pass of the same segment consumes.

## 9. Stages

| stage | content |
|---|---|
| **SP1** | The pass-cut graph core as pure functions — monotone pass assignment, the true graph cut, live ranges, peak carry bytes — plus the one predicate deciding verbatim-vs-collapse from activation semantics. No behavior change. |
| **SP2** | Raster carry in the HCM executor: the `carry_dst/carry_src` fill path, output-raster publication for carried wires, absolute-cycle discipline. Unlock `allow_scheduling` for streamed. Fused-vs-passed equivalence gate. |
| **SP3** | Carry through SANA-FE, nevresim and lava; a backend that cannot carry refuses by name. |
| **SP4** | `carried_raster_bytes` / `carry_peak_live_bytes` in the quantity catalog, sealed in the record, priced via `e_dma_per_byte`. |

## 10. Deliberately unchanged

- **The windowed disciplines.** `synchronized` and the TTFS family normalize at every
  boundary, so a pass boundary is already free for them. They keep the collapse
  transfer, and `is_retimed_level` stages keep their explicit count re-encode (their
  input IS the re-encode by definition — the t0_04 s32 catch).
- **The cutter's policy.** Greedy latency-group accumulation with within-group halving
  stays; only the *transfer* at the resulting boundaries changes.
- **Segment boundaries.** Host ops still compute on counts. Nothing about the
  end-to-end (`segments == 1`) property changes.

## 11. SP2 — as landed (2026-08-14)

**Implemented.** The carry seam is `models/spiking/hybrid/carry.py`: the packed cycle
executor records the segment's output raster in PRODUCER-LOCAL time from the same
`fires` the counts accumulate, and a neural stage publishes it into
`state_buffer_spikes` for exactly the wires a later pass of its own segment reads.
`encode_segment_input` already preferred a cached train, so the consuming side needed
no change. An execution path that cannot record a raster (synchronized / recording /
single-spike) REFUSES by name rather than handing the next pass a re-encoded count.

**Proven.**

- `raster.sum(0) == counts` — the carry is recorded from the same fires the counts
  accumulate, so a carry that invented spikes would fail this.
- Fused-vs-passed equivalence at bit level on a 4-hop LIF vehicle, at 2 and 3 passes.
- The carry is LOAD-BEARING: with publication disabled the scheduled run diverges from
  the fused one, which is the whole reason the lock existed.

**A trap worth recording.** The first version of the equivalence test passed with the
carry disabled. Untrained random weights saturate every neuron, and a saturated raster
IS its own uniform re-encode — so the vehicle could not witness the carry at all. The
weights are now set explicitly for mid-range firing, and
`test_the_carry_is_load_bearing` exists precisely so a future vehicle that drifts back
into saturation fails loudly instead of passing vacuously.

**Two mutations still survive, both from vehicle simplicity, not from the code:**

| mutation | why it is not witnessed | vehicle needed |
| --- | --- | --- |
| clamp `local` into `[0, T)` instead of skipping outside the producer window | every pass in the vehicle holds ONE latency group, so producer latency is 0 within its pass and the window never overhangs | a pass holding >= 2 latency groups |
| publish wires crossing to a later SEGMENT too | the vehicle has one segment, so no host boundary exists to be wrongly upgraded | a multi-segment vehicle (a host op between two neural segments) |

Both are coverage gaps in the test vehicle and are the first work of SP3.

**The lock stayed ON through SP2** and was lifted in SP3, once every backend either
carries or refuses.

## 12. SP3 — carry-or-refuse, and the unlock (2026-08-14)

**Both surviving mutations are dead**, and neither needed a bigger vehicle:

- The over-publish mutation was already caught — by `test_pass_cut.py`, not by the flow
  suite. It survived only because that round ran the wrong file.
- The producer-window mutation is killed by unit-testing the pure recorder directly.
  A segment whose outputs all sit at the deepest latency can never exhibit the
  overhang, so no end-to-end vehicle could witness it; `record_carry` with
  `latency < max` can, in three lines.

**Carry-or-refuse — superseded, see §14.** SP3 first shipped a refusal for backends
that could not replay a raster. That was the wrong conclusion from a right premise, and
§14 replaces it.

**The unlock.** `allow_scheduling` no longer carries a `legal_values` lambda: it is
legal under every execution semantics, because a pass is a spatial cut and the carry is
what keeps a cut exact. The DEFAULT is still `False`, so nothing about existing runs
changes — scheduling became choosable, not automatic.

## 13. What remains

| item | why it is not done |
| --- | --- |
| **SANA-FE carry** | The pieces are in place — `_pack_spike_trace_matrix` already produces a `(neurons, T)` raster and `set_input_spike_trains` already accepts an arbitrary one — but mapping segment-output SLICES to raster rows per cycle is real work, and the refusal is correct in the meantime. This is the highest-value next step: it turns a refusal into a measurement. |
| **nevresim / lava carry** | nevresim needs the carried raster to cross the C++ program-format seam; lava needs Loihi channel semantics. Both are larger than the SANA-FE case. |
| **SP4 — the carry census** | `carried_raster_bytes` / `carry_peak_live_bytes` as quantities, sealed and priced through `e_dma_per_byte`. `PassCut` already computes both; nothing consumes them yet, so a scheduled run still looks cheaper than it is. |

## 14. The correction: a pass boundary is one we INTRODUCE (owner, 2026-08-14)

SP3's refusal rested on an unexamined premise: that collapsing a pass boundary to
counts is a *lie*. It is not, and the owner's correction is the load-bearing one:

> "this is not a function requirement since we are introducing pass boundaries,
> decode/re-encode at boundary becomes available and it can operate on spike counts"

A pass boundary does not pre-exist the decision to schedule — **we create it**. A chip
that reprograms between passes must buffer the intermediate signal either way, and the
only question is *what* it buffers:

| discipline | buffered per wire | fidelity |
|---|---|---|
| `VERBATIM` | the raster, `ceil(T/8)` B | reproduces the fused execution bit-for-bit |
| `COLLAPSE` | window counts, `ceil(log2(T+1)/8)` B | re-emits an even train; rhythm normalized |

Both are honest deployments of a scheduled segment. `COLLAPSE` is the same
decode/re-encode a host boundary already performs, applied at a boundary that now
genuinely exists — and it is *cheaper*, which is exactly the trade a CAD tool should
expose rather than forbid. I had already written the `VERBATIM`/`COLLAPSE` vocabulary in
SP1 and then implemented only one arm, which is how the refusal crept in.

**As landed.** `require_backend_carry` is gone. `pass_transfer_for_backend(backend)`
returns the discipline that backend will execute — `VERBATIM` for the HCM executor,
`COLLAPSE` for SANA-FE, nevresim and lava — and `carried_wire_bytes(width, T, transfer)`
prices the choice. No backend refuses a scheduled deployment; each reports which of the
two computations produced its numbers.

## 15. Remaining, in value order

| item | note |
| --- | --- |
| **Record the discipline per run** | The record must carry which discipline each backend executed, since they are different computations. Nothing consumes `pass_transfer_for_backend` yet. |
| **SANA-FE `VERBATIM`** | Both primitives exist (`_pack_spike_trace_matrix` out, `set_input_spike_trains` in), and SANA-FE is the cost-measuring backend, so carrying there raises fidelity where it counts most. |
| **nevresim spike-train extraction** | Per the owner: NOT a functional gap, but a missing feature worth having on its own merits. `SPKREC` currently prints per-core COUNTS (`SPKREC <core> IN … OUT …`); a per-timestep variant would give nevresim segment-output rasters generally, not only for this program. Clean if done as a separate record line so the existing parser is untouched. |
| **SP4 — the carry census** | `PassCut` computes `carried_bytes`/`peak_live_bytes`; with `carried_wire_bytes` the census can now be priced under EITHER discipline. |

## 16. The degenerate vehicle, found and fixed (2026-08-14)

A probe of the test vehicle's own outputs found it **near-degenerate**: every output
neuron in a row carried the SAME count (`7,7,7,7` / `8,8,8,8`) and eight samples
produced only two distinct rows. The cause was my own earlier "fix" for saturation —
all-POSITIVE uniform weights with zero bias make every output neuron see essentially
the same sum, and at T=8 the counts sat at 7–8, one step from the ceiling.

Nothing was *wrong* in the implementation; the WITNESS was weak. Every equivalence
claim in the file rested on a network that barely discriminated, which is the same
class of error as the original saturated vehicle — and is exactly how the first
version of these tests passed with the carry disabled.

Fixed with signed weights around a small positive bias (`(rand - 0.5) * 2.0`, bias
0.3): eight samples now give 7 distinct rows, a mean per-row spread of 5.75, and a
maximum of 7 against a window of 8, so nothing saturates.

**Pinned so it cannot drift back.** `TestTheVehicleDiscriminates` asserts the three
properties directly — distinct answers across inputs, output neurons that do not all
agree, and counts that never reach the window. A vehicle that degenerates again fails
loudly instead of quietly weakening every other test in the file.

**What the stronger witness bought.** Re-running the carry mutations on the fixed
vehicle: disabling the carry now fails 2 tests (was 1), dropping the producer-latency
origin fails 4, and clamping the producer window — previously invisible at flow level
and killable only on the pure recorder — now fails at flow level too. The one remaining
survivor was the `input`-kind output span, which a linear chain never produces; it is
now pinned directly on `record_carry`, along with the always-on span.

## 17. SANA-FE carries verbatim (2026-08-14)

SANA-FE is the **cost-measuring** backend — it feeds the physics and energy path — so a
collapsed pass boundary there would model a different computation from the one the
record claims, and its energy number would be for the wrong chip. It now carries.

Both primitives already existed, so this is a gather, not a new capability:
`_pack_spike_trace_matrix` gives a `(neurons, T)` trace, and `set_input_spike_trains`
accepts an arbitrary `(1, size, T)` train.

Two conversions do the work, and both are places a plausible implementation goes wrong:

- **Produce.** `_compute_seg_output_raster` is the per-cycle twin of
  `_compute_seg_output_spike_count` — same spans, same sources, gathering the trace row
  instead of the accumulated count, and **shifted by each source core's own latency** so
  index *k* is that producer's k-th emission. SANA-FE's trace is in ABSOLUTE cycle time
  and net-group row order; a carried raster is PRODUCER-LOCAL and in segment-output
  order.
- **Consume.** `apply_carried_input` overwrites only the input slices whose producer
  published a train, transposing `(T, size)` into the `(1, size, T)` SANA-FE reads. A
  host boundary sharing the same input map keeps its uniform re-encode.

`publish_carried_trains` now slices with `[..., a:b]` rather than a fixed rank, which is
what lets the torch flow's `(T, B, size)` and a per-sample numpy `(T, size)` share one
publication path — the canonical carried layout is TIME-FIRST with the feature axis
LAST.

`VERBATIM_BACKENDS` is now `{hcm, sanafe}`. nevresim and lava still take COLLAPSE, for
the reason recorded in §14: neither records a rhythm it could replay, and the cheaper
discipline is a legitimate deployment rather than a refusal.

**Mutation-checked.** Replaying without the transpose, replaying host boundaries too,
dropping the producer latency, and dropping always-on spans all fail. The last two
needed direct tests of the gather: the earlier tests covered only the consume side, so
the produce side was passing unexamined.

## 18. What the first end-to-end scheduled run found (2026-08-14)

The unit tier was green and every gate passed, but **no tier cell combines streamed lif
with scheduling** — 18 scheduling cells exist and all are lifsync / ttfs / mvm / casc.
So the newly unlocked combination had zero end-to-end coverage. Running one found three
defects in three successive attempts, none of which any unit test could have caught.

**1. The census read the wrong source.** `simulation_steps` is not in
`platform_constraints_resolved`; it lives on `pipeline.config`. Reading it from the
platform dict yielded `None`, and the carry census refused to seal — the fail-loud guard
working exactly as intended, on a bug the unit tests could not see because they passed
`timesteps` explicitly. Now `simulation_steps_of(step)` reads the config SSOT and raises
rather than returning `None`.

**2. Backends ran different disciplines.** HCM carried while nevresim collapsed:

```
nevresim<->HCM window-count exactness violated: exact=0.955584 max|dcount|=1 over 788 windows
```

4.4% of neuron windows differed by one spike — precisely the rhythm a collapse
normalizes away. Both computations were *correct*; comparing them was not. The fix is
the rule this stage should have had from the start: **one discipline per RUN**, the
weakest of the enabled backends (`run_pass_transfer`). Enabling a backend that cannot
replay a raster costs the whole run its verbatim boundaries — and teaching that backend
to record one upgrades the run, which is the incentive pointing at nevresim's `SPKREC`.

**3. Half the fix is worse than none.** Gating only the HCM flow produced the mirror
image — HCM collapsing while SANA-FE still carried:

```
spike parity FAIL @ stage 2 ('neural_segment_final_cap1') 9/64 differ, sum exp=34 act=34
```

Totals equal, distribution different: the signature of a rhythm difference. The
`SanafeRunner` now takes the run's discipline too.

**As landed.** A 3-pass streamed MLP deploys clean on every backend:

```
Hybrid program (scheduled): 1 neural segment(s) (seg 0: 3 passes)
hcm/exact    PASS exact=1.000000 max|dcount|=0
loihi        PASS exact=1.000000 max|dcount|=0   (3 segments, 7 cores)
sanafe/exact PASS exact=1.000000 max|dcount|=0
carry: 6 wires, 384 B, 256 B peak live, T=4, transfer=collapse
```

`collapse` because this config enables nevresim and lava. Disabling them makes the same
run verbatim.

**The coverage gap is the real lesson**, and it remains open: a tier cell for
streamed × scheduling belongs in tier 0, so this path is exercised by the matrix rather
than by hand.
