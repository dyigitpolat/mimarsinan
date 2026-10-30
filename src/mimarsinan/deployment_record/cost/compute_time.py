"""[U-C2] The compute time of one inference by the chip's declared law, and the per-core programming wall.

Two compute-time laws (``platform_physics/vocabulary.py``), one per chip class:

- per timestep, ``t_cycle x latency_steps``: a chip whose every timestep takes the declared wall (a timestep-
  synchronous chip, or a measured sustained per-timestep wall);
- per synaptic operation, ``t_synaptic_op x synaptic_events`` (survey gap B6.3): an event-driven core that runs its
  operations one at a time and has no fixed timestep, priced on the event census the energy's ``e_mac`` reads. The
  census total is exact where one core runs every event of the inference. On several declared cores the cores of one
  (pass, segment) row run concurrently and the rows in turn, so the wall is the sum over the rows of each row's
  busiest core (``synaptic_events_busiest_core``). [targeted M2] The candidate plane states it from the energy's own
  model: per physical tile its as-mapped cells x timesteps x the declared activity_factor (``busiest_core_events``),
  formed only where the resolved physics prices compute per event (``prices_per_event``), so no other chip's
  quantities change. A view that states the census as one total (a sealed record) cannot split it per core: refused by
  name, as an absent census is. The same per-row events price the overlapped write exposure (``pass_compute_walls``)
  and the layer-pipelined bound (``busiest_stage``). Inter-core spike delivery inside a row is not priced (the
  timestep-synchronous discipline that keeps ``t_hop`` out of the step law).

A declared ``t_cycle`` takes precedence (a timestep's wall holds the events of the step), so a profile declaring both,
the capability probe among them, keeps the step law. ``t_core_program`` (survey gap B6.2) is the time twin of
``e_core_program``: the wall of programming one core beyond its payload bytes, which a whole-image interface (ODIN's
SPI) declares in place of a per-byte wall; ``core_program_walls`` is the write law's per-pass share of it.

[U-J5 L-5] A host-sourced payload's transfer over the host-chip link is timed with U-J3's ``store_walls`` as the
precedent (``producer_walls``): where the physics prices the link and the operand payloads come from the host, a pass's
payload bytes stream at the link's per-byte stage, so the pass's producer wall is the slower of its producer write
steps and that link wall (the pipelined bound), plus one ``t_link_transfer`` per pass carrying a payload. The producer
writes stay the write law's per-inference term, serial in both overlap modes, so on a host-owned store the period's
backing-store payload (``store_walls``) and the host-sourced payload take the one link in turn, each in its own write
phase (``payload_link_note`` states it in the exposed write's basis). Without link rows nothing changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple

from mimarsinan.deployment_record.build.write_census import pass_programmed_cores
from mimarsinan.deployment_record.cost.absolute.formulas import add_bands, scale_band
from mimarsinan.deployment_record.platform_physics.hosts.store import host_owned, link_stage, prices_host_link
from mimarsinan.deployment_record.platform_physics.profile import PlatformPhysics
from mimarsinan.deployment_record.schema.provenance import Band

STEP_LAW = "t_cycle"
EVENT_LAW = "t_synaptic_op"
CORE_PROGRAM = "t_core_program"
#: [targeted M2] The per-core event census: per (pass, segment) row its busiest physical core's events, summed.
BUSIEST_EVENTS = "synaptic_events_busiest_core"
BUSIEST_BASIS = "t_synaptic_op x the busiest core's synaptic events of each pass (synaptic_events_busiest_core)"
#: The basis of a pass's share of the per-core wall (the write law's multiplicand, ``programmed_cores`` per pass).
CORE_PROGRAM_BASIS = "t_core_program x rewritten cores holding a static program"
#: The producer writes' basis without the link: every pass's producer write steps at one step's wall.
PRODUCER_BASIS = "producer write steps, per pass"


@dataclass(frozen=True)
class ComputeTime:
    """The priced compute wall of one inference with its evidence note, or the refusal by name."""

    band: Optional[Band] = None
    evidence: str = ""
    refusal: Optional[str] = None


def compute_law(physics: Any, priced: Optional[Callable[[str], bool]] = None) -> Optional[str]:
    """The law the physics prices compute time by: ``t_cycle`` where declared, else ``t_synaptic_op``, else None."""
    has = priced or physics.has
    return STEP_LAW if has(STEP_LAW) else EVENT_LAW if has(EVENT_LAW) else None


def prices_per_event(physics_payload: Optional[Mapping[str, Any]]) -> bool:
    """Whether a resolved physics payload (``platform_physics_resolved``) prices compute per synaptic operation."""
    return bool(physics_payload) and compute_law(PlatformPhysics.from_dict(dict(physics_payload or {}))) == EVENT_LAW


def busiest_core_events(passes: Any, events_per_cell: Callable[[int], float]) -> Tuple[float, ...]:
    """Per (pass, segment) row of a plane's ``ProgramPasses``, the events of its busiest physical tile: its as-mapped
    cells (static cells and operand cells, ``PassRow.core_cells`` and ``producer_core_cells``) priced by the plane's
    own event model (``events_per_cell``: cells -> events, the energy's ``synaptic_events`` rule per core)."""
    out: List[float] = []
    for row in passes.rows:
        operand = row.producer_core_cells or (0,) * len(row.core_cells)
        out.append(max((events_per_cell(int(c) + int(o)) for c, o in zip(row.core_cells, operand)), default=0.0))
    return tuple(out)


def _several_cores_refusal(quantities: Any) -> Optional[str]:
    """Why the view's event census does not state the compute wall on several concurrent cores, or None (one core,
    or the per-core census is stated)."""
    cores = quantities.get("cores_physical").value if quantities.has("cores_physical") else None
    if (cores is not None and cores <= 1) or quantities.has(BUSIEST_EVENTS):
        return None
    count = "an undeclared number of" if cores is None else f"{cores:g}"
    held = quantities.withheld_reason(BUSIEST_EVENTS)
    return (f"t_synaptic_op prices the synaptic operations of one core, which runs them one at a time, and the chip "
            f"declares {count} cores, whose events in one pass run concurrently; this view states the "
            f"synaptic_events census as one total, with no per-core split ({BUSIEST_EVENTS} is absent"
            + (f": {held}" if held else "; a sealed event census is one total per inference") + "), so the compute "
            "time lies between the busiest core's share and the serial sum and is not determined")


def compute_time(physics: Any, quantities: Any, priced: Optional[Callable[[str], bool]] = None) -> ComputeTime:
    """The compute wall of one inference under the physics' law (the step law's refusals are the latency headline's
    as it has always stated them)."""
    law = compute_law(physics, priced)
    if law is None:
        return ComputeTime(refusal="t_cycle is undeclared — the one constant that converts latency_steps to seconds")
    if law == STEP_LAW:
        if not quantities.has("latency_steps"):
            return ComputeTime(refusal="t_cycle is declared but the latency_steps census is absent")
        band = physics.band(STEP_LAW)
        return ComputeTime(scale_band(band, quantities.get("latency_steps").value, "t_cycle x latency_steps"),
                           f"compute: t_cycle [{band.basis}]")
    if not quantities.has("synaptic_events"):
        held = quantities.withheld_reason("synaptic_events") or (
            "at candidate time it needs the declared activity_factor; a record carries it from a sealed event census")
        return ComputeTime(refusal=f"t_synaptic_op is declared but the synaptic_events census is absent ({held})")
    several = _several_cores_refusal(quantities)
    if several is not None:
        return ComputeTime(refusal=several)
    band = physics.band(EVENT_LAW)
    cores = quantities.get("cores_physical").value if quantities.has("cores_physical") else None
    if cores is not None and cores <= 1:
        return ComputeTime(scale_band(band, quantities.get("synaptic_events").value, "t_synaptic_op x synaptic_events"),
                           f"compute: t_synaptic_op [{band.basis}]")
    return ComputeTime(scale_band(band, quantities.get(BUSIEST_EVENTS).value, BUSIEST_BASIS),  # [targeted M2]
                       f"compute: t_synaptic_op [{band.basis}] x {BUSIEST_EVENTS}")


def pass_compute_walls(physics: Any, census: Any, pass_sops: Optional[Sequence[float]] = None
                       ) -> Tuple[Optional[List[Band]], str]:
    """(per pass of a period census, the compute an overlapped write hides behind, or None with the refusal by name):
    ``t_cycle`` x the pass's compute steps under the step law; [targeted M2] under the event law ``t_synaptic_op`` x
    the pass's busiest-core events (``pass_sops``, one per census pass, the candidate plane's
    ``busiest_core_events``), refused by name on a plane that states none (a sealed event census is one total)."""
    law = compute_law(physics)
    if law == STEP_LAW:
        cycle = physics.band(STEP_LAW)
        return [scale_band(cycle, p.compute_steps, "t_c") for p in census.passes], ""
    if law == EVENT_LAW:
        if pass_sops is None or len(pass_sops) != len(census.passes):
            return None, ("overlapped writes hide behind compute, which this chip prices per synaptic operation "
                          "(t_synaptic_op), and this plane states no per-pass event census (a sealed event census is "
                          "one total per inference)")
        sop = physics.band(EVENT_LAW)
        return [scale_band(sop, float(e), "t_c") for e in pass_sops], ""
    return None, "overlapped writes hide behind compute, which needs t_cycle"


def busiest_stage(physics: Any, pass_sops: Optional[Sequence[float]]) -> Optional[Band]:
    """[targeted M2] The layer-pipelined stage floor of a resident program on several event-serial cores: a core runs
    its own synaptic operations one at a time, so no stage is shorter than the most loaded core's events of any pass
    (``t_synaptic_op`` x the largest of ``pass_sops``); None off the event law or without the per-pass census."""
    if compute_law(physics) != EVENT_LAW or not pass_sops:
        return None
    return scale_band(physics.band(EVENT_LAW), max(float(e) for e in pass_sops),
                      "t_synaptic_op x the most loaded core's synaptic events of one pass")


def exposed_beyond(write: Band, k: int, compute: Band) -> Band:
    """The write law's overlapped exposure of one pass: max(0, t_w / K - t_c) corner-wise (monotone up in the write,
    down in the compute it hides behind)."""
    return Band(max(0.0, write.low / k - compute.high), max(0.0, write.nominal / k - compute.nominal),
                max(0.0, write.high / k - compute.low), "max(0, t_w(q+1) / K - t_c(q))")


def core_program_walls(census: Any, physics: Any) -> Optional[List[Band]]:
    """Per pass of a period census, ``t_core_program`` x its rewritten cores holding a static program; None where the
    physics declares no per-core wall."""
    if not physics.has(CORE_PROGRAM):
        return None
    wall = physics.band(CORE_PROGRAM)
    return [scale_band(wall, pass_programmed_cores(p), CORE_PROGRAM_BASIS) for p in census.passes]


def store_walls(walls: Optional[List[Band]], census: Any, physics: Any) -> Optional[List[Band]]:
    """[U-J3] Per pass, the write wall on the declared store: on a host-owned store (``hosts.store``) a written pass's
    payload streams from host memory over the link, so its wall is the slower of the chip's port wall and the payload
    at the store's per-byte stage, plus one transfer latency (``t_link_transfer``); ``walls`` elsewhere."""
    stage = link_stage(physics) if host_owned(physics) else None
    if stage is None:
        return walls
    transfer = physics.band("t_link_transfer") if physics.has("t_link_transfer") else None
    out: List[Band] = []
    for index, census_pass in enumerate(census.passes):
        own = None if walls is None else walls[index]
        if not census_pass.payload_bytes:
            out.append(own if own is not None else Band(0.0, 0.0, 0.0, "nothing streams"))
            continue
        link = scale_band(stage, census_pass.payload_bytes, f"payload_bytes x {stage.basis}")
        slower = f"the slower of [{own.basis}] and [{link.basis}]" if own is not None else ""
        wall = link if own is None else Band(max(own.low, link.low), max(own.nominal, link.nominal),
                                             max(own.high, link.high), slower)
        out.append(wall if transfer is None else add_bands([wall, transfer], f"{wall.basis} + t_link_transfer"))
    return out


def plus_core_walls(walls: Optional[List[Band]], census: Any, physics: Any) -> List[Band]:
    """Per pass, a write wall (``walls``, None where the physics prices none) plus the per-core wall where declared
    (``core_program_walls``); ``walls`` as they are where it is not. [U-J3] The wall on the declared store first."""
    walls = store_walls(walls, census, physics)
    cores = core_program_walls(census, physics)
    if cores is None:
        return list(walls or ())
    if walls is None:
        return cores
    return [add_bands([w, c], f"{w.basis} + {c.basis}") for w, c in zip(walls, cores)]


def payload_stage(physics: Any, policy: Any) -> Optional[Band]:
    """[U-J5 L-5] The per-byte stage a host-sourced payload streams at (``link_stage``): where the physics prices the
    host-chip link and the operand payloads come from the host; None elsewhere, or where no stage is declared."""
    if not (prices_host_link(physics) and policy.producer_write_source == "host" and policy.payload_bits is not None):
        return None
    return link_stage(physics, "the host-sourced payload's stage")


def payload_link_note(census: Any, physics: Any, policy: Any) -> str:
    """[U-J5 L-5] What the exposed write's basis appends where ``producer_walls`` times a payload over the link, ""
    where it does not."""
    stage = payload_stage(physics, policy)
    if stage is None or not any(p.written_cells_producer for p in census.passes):
        return ""
    transfer = " + t_link_transfer per pass carrying it" if physics.has("t_link_transfer") else ""
    shared = ("; the backing-store payload takes the same link in the period's write phase (store_walls), so the link "
              "carries both, each in its own phase") if host_owned(physics) else ""
    return (f"; producer writes per pass: the slower of the write steps and the host-sourced payload bytes x "
            f"{stage.basis}{transfer} (the payload streams over the host-chip link in the producer's own write phase, "
            f"per inference){shared}")


def producer_walls(step: Band, steps: Sequence[int], census: Any, physics: Any, policy: Any) -> Band:
    """[U-J5 L-5] The producer writes per inference: per pass its write steps (``steps``) at one step's wall; where a
    host-sourced payload crosses the priced link (``payload_stage``), per pass the slower of those and its payload
    bytes (producer cells x ``payload_bits`` / 8) at the link's stage, plus one ``t_link_transfer`` per pass carrying a
    payload. Without the link (or without a payload) the band is the steps' sum, as before."""
    stage = payload_stage(physics, policy)
    if stage is None or not any(p.written_cells_producer for p in census.passes):
        return scale_band(step, sum(steps), PRODUCER_BASIS)
    transfer = physics.band("t_link_transfer") if physics.has("t_link_transfer") else None
    walls: List[Band] = []
    for census_pass, count in zip(census.passes, steps):
        if not census_pass.written_cells_producer:
            continue
        own = scale_band(step, count, "producer write steps")
        link = scale_band(stage, census_pass.written_cells_producer * int(policy.payload_bits) / 8.0,
                          f"host-sourced payload bytes x {stage.basis}")
        wall = Band(max(own.low, link.low), max(own.nominal, link.nominal), max(own.high, link.high),
                    f"the slower of [{own.basis}] and [{link.basis}]")
        walls.append(wall if transfer is None else add_bands([wall, transfer], f"{wall.basis} + t_link_transfer"))
    return walls[0] if len(walls) == 1 else add_bands(walls, "per pass: " + " + ".join(w.basis for w in walls))
