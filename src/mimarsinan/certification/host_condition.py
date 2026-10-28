"""[W49] Theorem 1's host condition, checked by the sampled run.

The condition: the host kernels are deterministic and batch-invariant on one device and dtype, so a sample's host
values are a function of that sample alone. While ``host_kernel_replay()`` is armed, every call of the deployment seam
of a host op (``ComputeOp.execute_on_gathered``, which the one host evaluator ``execute_compute_op_torch`` calls) is
replayed by the same kernel, in the same context, on every sampled input alone and once more on the whole batch, and
both are compared at atol 0 with what the kernel gave in the batch (a NaN equals a NaN; anything else must be equal).
Every replayed call runs on a copy of the input as it came in, so a kernel that writes its input in place is replayed
on its own input and the deployment's buffers are never touched. A kernel that differs alone is batch-dependent, one
that differs on the repeat is nondeterministic, one seen on more than one device or dtype breaks "one device and
dtype"; each is named, and its deployment is outside the per-sample certified scope by name. A kernel that only ever ran
on a batch of one cannot show batch dependence (alone is the batch call): its batch invariance is not checked, by name,
and the condition is not established. Nothing is refused, and the replay leaves the executor's state as it found it
(the random streams, a training-mode module's buffers)."""

from __future__ import annotations

import contextlib
import json
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import torch

from mimarsinan.deployment_record.schema import CertificateRecord
from mimarsinan.mapping.capability.op_kinds import UnknownOpKindError, op_kind
from mimarsinan.mapping.ir.types import HOST_KERNEL_REPLAY

HOST_CONDITION = "deterministic, batch-invariant host kernels on one device and dtype"
RECORD_NAME, RECORD_BACKEND = "host_condition", "host"
ESTABLISHED, OUTSIDE_SCOPE, NOT_CHECKED = "established", "outside_scope", "not_checked"


#: What a kernel replayed on one sample alone may raise: such a kernel is no per-sample function (counted as
#: batch-dependent, the error named); anything else (memory, interrupts) propagates.
REPLAY_ERRORS = (RuntimeError, ValueError, TypeError, IndexError, AssertionError, NotImplementedError)


def _kind(op) -> str:
    try:
        return op_kind(op)
    except (UnknownOpKindError, AttributeError, KeyError, TypeError):  # unnamed by the table: its display label
        return str(getattr(op, "op_type", "host"))


def _differ(got: torch.Tensor, want: torch.Tensor) -> torch.Tensor:
    """Elementwise inequality at atol 0, a NaN equal to a NaN (a shape mismatch differs everywhere)."""
    if got.shape != want.shape:
        return torch.ones(want.shape, dtype=torch.bool, device=want.device)
    same = got == want
    if want.is_floating_point():
        same = same | (torch.isnan(got) & torch.isnan(want))
    return ~same


@contextlib.contextmanager
def _executor_state_kept(op, device: torch.device) -> Iterator[None]:
    """The random streams and, for a module holding a training-mode submodule, its buffers, restored after the replay."""
    cpu = torch.random.get_rng_state()
    cuda = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    module = (getattr(op, "params", None) or {}).get("module")
    kept: List[Tuple[torch.Tensor, torch.Tensor]] = []
    if isinstance(module, torch.nn.Module) and any(m.training for m in module.modules()):
        kept = [(b, b.detach().clone()) for b in module.buffers()]
    try:
        yield
    finally:
        torch.random.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state(cuda, device)
        with torch.no_grad():
            for buffer, value in kept:
                buffer.copy_(value)


@dataclass
class _Tally:
    name: str
    kind: str
    calls: int = 0
    samples: int = 0
    invariance_samples: int = 0
    elements: int = 0
    batch_dependent: int = 0
    nondeterministic: int = 0
    max_abs_delta: float = 0.0
    placements: set = field(default_factory=set)
    replay_errors: set = field(default_factory=set)


@dataclass(frozen=True)
class HostKernelReading:
    """One host kernel (by IR op id) over every call the replay saw: samples replayed alone, the ones of them replayed
    out of a batch of several (``invariance_samples``: only those can show batch dependence), output elements
    compared, the elements that differed alone (``batch_dependent``) and on the repeat (``nondeterministic``), and
    every ``device dtype`` it ran on."""

    op_id: int
    name: str
    kind: str
    calls: int
    samples: int
    elements: int
    batch_dependent: int
    nondeterministic: int
    max_abs_delta: float
    placements: Tuple[str, ...]
    replay_errors: Tuple[str, ...] = ()
    invariance_samples: int = 0

    @property
    def breaks(self) -> bool:
        """Whether the replay saw this kernel break the condition (batch-dependent, nondeterministic, or on several
        devices or dtypes)."""
        return bool(self.batch_dependent or self.nondeterministic or len(self.placements) > 1)

    @property
    def invariance_checked(self) -> bool:
        """Whether a call of a batch of several was replayed: at a batch of one, alone is the batch call."""
        return self.invariance_samples > 0

    @property
    def holds(self) -> bool:
        return not self.breaks and self.invariance_checked

    def label(self) -> str:
        return f"{self.name} ({self.kind})"


@dataclass(frozen=True)
class HostConditionCertificate:
    """The host condition over the kernels the sampled run evaluated; ``program_host_ops`` (the deployed program's
    host ops, when known) tells a program without host kernels from one the sampled run did not exercise."""

    kernels: Tuple[HostKernelReading, ...]
    samples: int
    program_host_ops: Optional[int] = None

    @property
    def batch_dependent(self) -> Tuple[HostKernelReading, ...]:
        return tuple(k for k in self.kernels if k.batch_dependent)

    @property
    def nondeterministic(self) -> Tuple[HostKernelReading, ...]:
        return tuple(k for k in self.kernels if k.nondeterministic)

    @property
    def several_placements(self) -> Tuple[HostKernelReading, ...]:
        return tuple(k for k in self.kernels if len(k.placements) > 1)

    @property
    def invariance_unchecked(self) -> Tuple[HostKernelReading, ...]:
        return tuple(k for k in self.kernels if not k.invariance_checked)

    @property
    def status(self) -> str:
        """``outside_scope`` where a kernel breaks the condition; else ``not_checked`` where no sampled certificate
        evaluated a host kernel of a program that has some, or a kernel only ran on batches of one; else
        ``established``."""
        if not self.kernels:
            return ESTABLISHED if self.program_host_ops == 0 else NOT_CHECKED
        if any(k.breaks for k in self.kernels):
            return OUTSIDE_SCOPE
        return NOT_CHECKED if self.invariance_unchecked else ESTABLISHED

    @property
    def passed(self) -> bool:
        return self.status == ESTABLISHED

    def placements(self) -> Tuple[str, ...]:
        return tuple(sorted({p for k in self.kernels for p in k.placements}))

    def verdict(self) -> str:
        status = self.status
        if status == NOT_CHECKED and not self.kernels:
            return (f"{NOT_CHECKED}: no sampled certificate evaluated a host kernel of the program's "
                    f"{self.program_host_ops if self.program_host_ops is not None else 'unknown number of'} host ops")
        if status == NOT_CHECKED:
            return (f"{NOT_CHECKED}: batch invariance not checked for host kernels that only ran on a batch of one ("
                    + ", ".join(k.label() for k in self.invariance_unchecked)
                    + f"; alone is the batch call), so {HOST_CONDITION} is not established over {self.samples} "
                    f"sampled input(s); determinism and one device and dtype hold for all {len(self.kernels)} "
                    f"kernel(s), on {', '.join(self.placements())}")
        if not self.kernels:
            return f"{ESTABLISHED}: the program has no host kernel ({HOST_CONDITION} holds vacuously)"
        if status == ESTABLISHED:
            return (f"{ESTABLISHED}: {len(self.kernels)} host kernel(s) {HOST_CONDITION}: every one of "
                    f"{self.samples} sampled input(s) alone and the repeated batch equal at atol 0, on "
                    f"{', '.join(self.placements())}")
        parts = []
        for what, kernels, count in (("batch-dependent", self.batch_dependent, lambda k: k.batch_dependent),
                                     ("nondeterministic", self.nondeterministic, lambda k: k.nondeterministic)):
            if kernels:
                parts.append(f"{what} host kernels " + ", ".join(
                    f"{k.label()}: {count(k)} of {k.elements} elements"
                    + (f" (alone: {'; '.join(k.replay_errors)})" if k.replay_errors else "") for k in kernels))
        if self.several_placements:
            parts.append("host kernels not on one device and dtype " + ", ".join(
                f"{k.label()}: {' / '.join(k.placements)}" for k in self.several_placements))
        return (f"{OUTSIDE_SCOPE}: the per-sample certified scope excludes this deployment (Theorem 1's host "
                f"condition, {HOST_CONDITION}, does not hold over {self.samples} sampled input(s)): "
                + "; ".join(parts))

    def summary(self) -> str:
        return f"host condition {self.status.upper()} ({len(self.kernels)} kernel(s), {self.samples} sample(s))"

    def record(self) -> Dict[str, object]:
        """The deployment record's certificate row (backend ``host``): passed only where the condition is established."""
        elements = sum(k.elements for k in self.kernels)
        differing = sum(max(k.batch_dependent, k.nondeterministic) for k in self.kernels)
        detail = {"verdict": self.verdict(), "condition": HOST_CONDITION, "samples": self.samples,
                  "program_host_ops": self.program_host_ops, "kernels": [asdict(k) for k in self.kernels]}
        return CertificateRecord(
            name=RECORD_NAME, backend=RECORD_BACKEND, passed=self.passed, neuron_windows_compared=elements,
            exact_match_fraction=(1.0 - differing / elements) if elements else 0.0,
            max_abs_delta=max((k.max_abs_delta for k in self.kernels), default=0.0),
            detail=json.dumps(detail, sort_keys=True)).to_dict()


class HostKernelReplay:
    """The armed replay: each evaluator call replayed sample by sample and once on the whole batch, then handed on to
    the replay armed around this one (``parent``), so a nested certificate leaves the outer reading whole."""

    def __init__(self, parent: Optional["HostKernelReplay"] = None) -> None:
        self._tallies: Dict[int, _Tally] = {}
        self._samples = 0
        self._parent = parent

    def __call__(self, op, gathered: torch.Tensor, result: torch.Tensor,
                 kernel: Optional[Callable[[torch.Tensor], torch.Tensor]] = None) -> None:
        batch = int(gathered.shape[0]) if gathered.dim() else 0
        want = result.detach()
        tally = self._tallies.setdefault(int(op.id), _Tally(str(getattr(op, "name", op.id)), _kind(op)))
        again: Optional[torch.Tensor] = None
        replayed: Optional[torch.Tensor] = None
        run = kernel if kernel is not None else op._exec_module
        with _executor_state_kept(op, want.device), torch.no_grad():
            try:  # every call on its own copy: a kernel that writes its input in place never reaches ``gathered``
                again = run(gathered.clone()).detach()
                alone = [run(gathered[i:i + 1].clone()).detach() for i in range(batch)]
                replayed = torch.cat(alone) if batch else want[:0]
            except REPLAY_ERRORS as exc:  # a kernel that cannot run on one sample alone is not a per-sample function
                tally.replay_errors.add(f"{type(exc).__name__}: {str(exc)[:120]}")
        tally.calls += 1
        tally.samples += batch
        tally.invariance_samples += batch if batch > 1 else 0
        tally.elements += int(want.numel())
        tally.placements.add(f"{want.device} {want.dtype}")
        tally.nondeterministic += int(_differ(again, want).sum()) if again is not None else 0
        if batch:
            tally.batch_dependent += int(_differ(replayed, want).sum()) if replayed is not None else int(want.numel())
            if replayed is not None and replayed.shape == want.shape and want.is_floating_point() and want.numel():
                delta = (replayed.double() - want.double()).abs().nan_to_num(0.0)
                tally.max_abs_delta = max(tally.max_abs_delta, float(delta.max()))
        self._samples = max(self._samples, batch)
        if self._parent is not None:
            self._parent(op, gathered, result, kernel)

    def certificate(self, *, program_host_ops: Optional[int] = None) -> HostConditionCertificate:
        kernels = tuple(HostKernelReading(op_id=i, name=t.name, kind=t.kind, calls=t.calls, samples=t.samples,
                                          elements=t.elements, batch_dependent=t.batch_dependent,
                                          nondeterministic=t.nondeterministic, max_abs_delta=t.max_abs_delta,
                                          placements=tuple(sorted(t.placements)),
                                          replay_errors=tuple(sorted(t.replay_errors)),
                                          invariance_samples=t.invariance_samples)
                        for i, t in sorted(self._tallies.items()))
        return HostConditionCertificate(kernels=kernels, samples=self._samples, program_host_ops=program_host_ops)


@contextlib.contextmanager
def host_kernel_replay() -> Iterator[HostKernelReplay]:
    """Arm the replay for the block; a nested block's replay feeds the outer one too, and the outer one is armed
    again on exit."""
    replay = HostKernelReplay(parent=HOST_KERNEL_REPLAY.get())
    token = HOST_KERNEL_REPLAY.set(replay)
    try:
        yield replay
    finally:
        HOST_KERNEL_REPLAY.reset(token)


def host_condition_of(run: Callable[[torch.Tensor], object], samples: torch.Tensor, *,
                      program_host_ops: Optional[int] = None) -> HostConditionCertificate:
    """``run(samples)`` (a deployed program's flow) under the replay, and the host condition it gives."""
    with host_kernel_replay() as replay, torch.no_grad():
        run(samples)
    return replay.certificate(program_host_ops=program_host_ops)
