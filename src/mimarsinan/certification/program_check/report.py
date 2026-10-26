"""[W15] The static check's report: one result per condition of the condition map, Theorem 1 and Proposition 2 read
separately, the executor code clauses and reference assumptions each verdict rests on (``ASSUMED``), the walls, and
the two certificate records the deployment record carries beside the sampled run's.

A passing verdict reads ``established on the program against the IR at the declared geometry; assumed: ...``: every
program condition and every statically checkable executor condition holds for every input, relative to the identity
program of the IR at the declared geometry; the clauses named after ``assumed`` are properties of the executors' code
(or of the reference) that the check does not read, so the sampled run tests them."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from mimarsinan.deployment_record.schema import CertificateRecord

#: A condition's status. ``not_required``: the semantics does not need it (Lemma 5 binds timing only under timed
#: semantics); ``vacuous``: the program holds no element it constrains; ``not_static``: no input-independent
#: predicate implies it, so it stays with the sampled run, named.
ESTABLISHED, FAILED, NOT_REQUIRED, VACUOUS, NOT_STATIC = "established", "failed", "not_required", "vacuous", "not_static"
PASSING = (ESTABLISHED, NOT_REQUIRED, VACUOUS)
ESTABLISHED_ON_PROGRAM = "established on the program"
REFERENCE = "the IR at the declared geometry"
#: How many violations a result keeps in full (the count is always exact).
KEPT_VIOLATIONS = 8
T1, P2 = "theorem1", "proposition2"
ASSUMED_EXECUTOR, ASSUMED_REFERENCE = "assumed_executor", "assumed_reference"


@dataclass(frozen=True)
class Assumption:
    """A clause a verdict rests on that the check does not read: an executor code property (``assumed_executor``) or
    a property of the reference (``assumed_reference``), the condition it is part of, its code site, and what tests it."""

    name: str
    status: str
    theorems: Tuple[str, ...]
    condition: str
    site: str
    statement: str
    tested_by: str = "the sampled run (512 inputs at atol 0 against the int64 golden or the identity program)"


ASSUMED: Tuple[Assumption, ...] = (
    Assumption("H0 dispatch", ASSUMED_EXECUTOR, (T1, P2), "H0",
               "models/spiking/hybrid/flow.py (SpikingHybridCoreFlow), chip_simulation/spiking_mode_policy.py",
               "the executor built from the declaration runs the declared semantics"),
    Assumption("H8 operand", ASSUMED_EXECUTOR, (T1, P2), "H8",
               "models/spiking/wire_semantics.py (lif_window_count), chip_simulation/ttfs/membrane_lattice.py",
               "every count or level a core reads lies on its lattice"),
    Assumption("R6 walk", ASSUMED_EXECUTOR, (T1, P2), "R6",
               "models/spiking/hybrid/executors/sync_counts.py, chip_simulation/ttfs/ttfs_segment.py, "
               "chip_simulation/value_run/value_execution.py (dependency_order)",
               "the count, TTFS and value walks visit a segment's cores in dependency order"),
    Assumption("stage-input local time", ASSUMED_EXECUTOR, (T1, P2), "R1",
               "models/spiking/hybrid/lif_step.py:167, models/spiking/hybrid/executors/packed_cycle.py:219",
               "every core reads a stage-input row in its own local time (cycle t at global cycle latency + t)"),
    Assumption("producer-local carry", ASSUMED_EXECUTOR, (T1,), "R2",
               "models/spiking/hybrid/executors/carry_record.py:8-38",
               "a carried raster is recorded in its producer's local time"),
    Assumption("per-core output windows", ASSUMED_EXECUTOR, (T1, P2), "R1",
               "models/spiking/hybrid/executors/single_spike.py:35-38",
               "every core's outputs are read over its own window [latency, latency + T)"),
    Assumption("H8-staircase kernel", ASSUMED_EXECUTOR, (P2,), "H8-staircase",
               "models/spiking/wire_semantics.py, chip_simulation/ttfs/ttfs_segment.py (closure fix MJ4)",
               "the staircase kernels take the integer charge"),
    Assumption("IR tiling", ASSUMED_REFERENCE, (T1, P2), "tiles",
               "pipelining/pipeline_steps/mapping/soft_core_mapping_step.py:216-220 (the IR mapped at the declared "
               "per-core limits)", "the IR at the declared geometry computes what the IR at unbounded geometry "
               "computes (core geometry is cost-only); the reference is the identity program of the former",
               "the X3a screen and K7 (sampled); measured departure under a dithered encode, 26 of 100"),
    Assumption("identity builder", ASSUMED_REFERENCE, (T1, P2), "tiles",
               "mapping/packing/hybrid_build_pool.py (build_identity_hybrid_mapping)",
               "the identity builder shares code with the deployment builder, so a defect of the shared code is in "
               "both programs", "the sampled run against the int64 golden"),
)


@dataclass(frozen=True)
class Violation:
    """One failed predicate instance: the condition, the named rule, where in the program, and what differs."""

    condition: str
    rule: str
    where: str
    detail: str

    def text(self) -> str:
        return f"[{self.condition} {self.rule}] {self.where}: {self.detail}"


@dataclass
class ConditionResult:
    """One condition's outcome over a program: its status, a note (why not required, vacuous or not static, or what
    was established), the elements checked, and its violations (kept up to ``KEPT_VIOLATIONS``)."""

    condition: str
    status: str
    note: str = ""
    checked: int = 0
    violations: List[Violation] = field(default_factory=list)
    violation_count: int = 0

    def add(self, violation: Violation) -> None:
        self.violation_count += 1
        if len(self.violations) < KEPT_VIOLATIONS:
            self.violations.append(violation)

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["violations"] = [asdict(v) for v in self.violations]
        return out


@dataclass(frozen=True)
class ConditionSpec:
    """A row of the condition map: the theorem it serves, its kind (deployment, program, executor, network), and,
    when no static predicate implies it, why (it then stays with the sampled run)."""

    condition: str
    theorem: str
    kind: str
    statement: str
    not_static_reason: str = ""


@dataclass
class StaticCheckReport:
    """Every condition's result, the verdicts, the unknown and unread program forms and the walls."""

    semantics: str
    in_scope: bool
    scope_note: str
    results: Dict[str, ConditionResult]
    specs: Tuple[ConditionSpec, ...]
    unknown_forms: List[Violation] = field(default_factory=list)
    unread_forms: List[Violation] = field(default_factory=list)
    wall_s: float = 0.0
    compile_wall_s: Optional[float] = None
    counts: Dict[str, Any] = field(default_factory=dict)
    proposition2_note: str = ""

    def assumed(self, theorem: str) -> List[Assumption]:
        return [a for a in ASSUMED if theorem in a.theorems]

    def _verdict(self, theorem: str) -> str:
        if not self.in_scope:
            return f"outside_scope: {self.scope_note}"
        if self.unknown_forms:
            return "failed: unknown program form " + ", ".join(sorted({v.rule for v in self.unknown_forms}))
        if self.unread_forms:
            return "failed: unread program form " + ", ".join(sorted({v.rule for v in self.unread_forms}))
        mine = [s.condition for s in self.specs if s.theorem == theorem or (theorem == P2 and s.kind == "executor")]
        failed = [c for c in mine if self.results[c].status == FAILED]
        if failed:
            return "failed: " + ", ".join(failed)
        assumed = "assumed: " + ", ".join(a.name for a in self.assumed(theorem)) + " (tested by the sampled run)"
        open_ = [c for c in mine if self.results[c].status == NOT_STATIC]
        if open_:
            return "conditional: " + ", ".join(open_) + " stay with the sampled run; " + assumed
        return f"{ESTABLISHED_ON_PROGRAM} against {REFERENCE}; {assumed}"

    @property
    def theorem1(self) -> str:
        return self._verdict(T1)

    @property
    def proposition2(self) -> str:
        if self.in_scope and self.proposition2_note:
            return self.proposition2_note
        return self._verdict(P2)

    def established(self, theorem: str) -> bool:
        """Every condition of ``theorem`` established, vacuous or not required on the program (``passed``)."""
        return (self.theorem1 if theorem == T1 else self.proposition2).startswith(ESTABLISHED_ON_PROGRAM)

    def failures_by_condition(self) -> Dict[str, int]:
        out = {c: r.violation_count for c, r in self.results.items() if r.status == FAILED}
        if self.unknown_forms:
            out["unknown_form"] = len(self.unknown_forms)
        if self.unread_forms:
            out["unread_form"] = len(self.unread_forms)
        return out

    def summary(self) -> str:
        walls = f"check {self.wall_s:.3f} s" + (
            f" against compile {self.compile_wall_s:.3f} s" if self.compile_wall_s is not None else
            f" (mapping {self.counts.get('compile_source', 'unpriced')})")
        statuses = ", ".join(f"{c} {r.status}" for c, r in self.results.items())
        return (f"[StaticProgramCheck] {self.semantics}: Theorem 1 {self.theorem1}; Proposition 2 "
                f"{self.proposition2}; {walls}; {statuses}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "semantics": self.semantics, "in_scope": self.in_scope, "scope_note": self.scope_note,
            "theorem1": self.theorem1, "proposition2": self.proposition2, "reference": REFERENCE,
            "failures_by_condition": self.failures_by_condition(),
            "conditions": {c: r.to_dict() for c, r in self.results.items()},
            "condition_map": [asdict(s) for s in self.specs],
            "assumed": [dict(asdict(a), theorems=list(a.theorems)) for a in ASSUMED],
            "unknown_forms": [asdict(v) for v in self.unknown_forms],
            "unread_forms": [asdict(v) for v in self.unread_forms],
            "wall_s": self.wall_s, "compile_wall_s": self.compile_wall_s, "counts": dict(self.counts),
        }

    def certificate_records(self) -> List[Dict[str, Any]]:
        """Two records beside the sampled run's, Theorem 1 and Proposition 2 apart: input-independent, so no
        neuron window is compared; ``passed`` is ``established``; ``exact_match_fraction`` is the share of the
        theorem's conditions established, vacuous or not required; ``detail`` names the clauses assumed."""
        out = []
        for theorem, verdict in ((T1, self.theorem1), (P2, self.proposition2)):
            mine = [s.condition for s in self.specs if s.theorem == theorem]
            held = sum(self.results[c].status in PASSING for c in mine)
            detail = json.dumps({
                "verdict": verdict, "semantics": self.semantics, "wall_s": round(self.wall_s, 6),
                "compile_wall_s": None if self.compile_wall_s is None else round(self.compile_wall_s, 6),
                "compile_source": self.counts.get("compile_source"), "reference": REFERENCE,
                "assumed": [a.name for a in self.assumed(theorem)],
                "conditions": {c: self.results[c].status for c in mine},
                "failures": {c: n for c, n in self.failures_by_condition().items()
                             if c in mine or c in ("unknown_form", "unread_form")},
            }, sort_keys=True)
            out.append(CertificateRecord(
                name=f"static_program_check_{theorem}", backend="static", passed=self.established(theorem),
                neuron_windows_compared=0, exact_match_fraction=held / max(1, len(mine)), max_abs_delta=0.0,
                detail=detail).to_dict())
        return out


def spec_table(specs: Sequence[ConditionSpec]) -> Mapping[str, ConditionSpec]:
    return {s.condition: s for s in specs}
