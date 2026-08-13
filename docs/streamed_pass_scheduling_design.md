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

**The lock is deliberately still ON.** `allow_scheduling` stays `(False,)` for streamed
until SP3: SANA-FE, nevresim and lava do not carry yet, and unlocking now would let a
run reach a backend that silently collapses the boundary — the exact failure the lock
exists to prevent. The unlock belongs with the backends' carry-or-refuse.
