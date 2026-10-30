"""[U-H] Which integration policies the nevresim tree a run compiles against executes.

The default policy (``WholeVectorIntegrate``) is the template default of every nevresim tree, so it is never probed
and no default program changes. The event-serial fold (``EventSerialIntegrate``) and the sync-fire register
(``WholeVectorSaturatingSigned``) are the ODIN program's change to nevresim's sources; nevresim is a nested repository
whose commit of that change waits for its owner, so a checkout can hold a tree without it, where a program naming
the policy fails only after a full build and with no name. The driver therefore asks this probe before it emits such
a program: the policy's declared sites, then the tree's own self-test constant-evaluated by the compiler nevresim
builds with (``source_probe``, the two checks of W33's probe), with a verdict kept per include-tree fingerprint."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Tuple

from mimarsinan.chip_simulation.nevresim.compile_cache import simulator_source_fingerprint
from mimarsinan.chip_simulation.nevresim.compile_nevresim import constexpr_options
from mimarsinan.chip_simulation.nevresim.source_probe import ReadSite, self_test_problem, site_problems
from mimarsinan.chip_simulation.nevresim_policy_types import (
    EVENT_SERIAL_INTEGRATE,
    WHOLE_VECTOR_INTEGRATE,
    WHOLE_VECTOR_SATURATING_SIGNED,
    NevresimPolicyTypeError,
    integration_policy_base,
)
from mimarsinan.common.refusal import ToolchainRefusal

_POLICIES = "simulator/compute_policy/integration_policy/"
#: Both laws are template arguments of the spiking compute policy and of the spiking execution.
_TEMPLATE_SITES: Tuple[ReadSite, ...] = (
    ("simulator/compute_policy/spiking_compute.hpp", "typename IntegrationPolicy", ()),
    ("simulator/execution/spiking_execution.hpp", "IntegrationPolicy::counted_wire", ()),
)
#: Per policy template: its sites under ``include/`` and its self-test (header, constexpr function). The event-serial
#: fold also needs the counted wire (the recorder's ``SPKTRN2`` line, the generator's multiplicity-preserving load)
#: and the 16-bit count currency.
POLICY_SITES: Mapping[str, Tuple[Tuple[ReadSite, ...], Tuple[str, str]]] = {
    EVENT_SERIAL_INTEGRATE: (
        ((_POLICIES + "event_serial_integrate.hpp", "struct EventSerialIntegrate", ()), *_TEMPLATE_SITES,
         ("simulator/recording/spike_train_recorder.hpp", "SPKTRN2", ()),
         ("simulator/spike_generation/spike_train_spike_generator.hpp", "PreserveMultiplicity", ()),
         ("common/types.hpp", "k_count_currency_floor_bits", ()),
         ("_tests/event_serial_integrate.hpp", "test_event_serial_integrate", ())),
        ("_tests/event_serial_integrate.hpp", "nevresim::tests::test_event_serial_integrate"),
    ),
    WHOLE_VECTOR_SATURATING_SIGNED: (
        ((_POLICIES + "whole_vector_saturating_signed.hpp", "struct WholeVectorSaturatingSigned", ()),
         *_TEMPLATE_SITES,
         ("_tests/whole_vector_saturating_signed.hpp", "test_whole_vector_saturating_signed", ())),
        ("_tests/whole_vector_saturating_signed.hpp", "nevresim::tests::test_whole_vector_saturating_signed"),
    ),
}
#: (policy template, include-tree fingerprint) -> (established, what the probe found).
_VERDICTS: Dict[Tuple[str, str], Tuple[bool, str]] = {}


class IntegrationPolicyRefusal(ToolchainRefusal, NevresimPolicyTypeError):
    """A soma point whose integration policy the nevresim tree a run compiles against does not carry."""


@dataclass(frozen=True)
class PolicyRead:
    """Whether the tree executes one integration policy, and what says so."""

    established: bool
    evidence: str


def integration_policy_read(integration_policy: str, root) -> PolicyRead:
    """Whether the nevresim tree at ``root`` executes ``integration_policy`` (its template, any width)."""
    policy = integration_policy_base(integration_policy)
    if policy == WHOLE_VECTOR_INTEGRATE:
        return PolicyRead(True, f"{policy} is the template default of every nevresim tree")
    declared = POLICY_SITES.get(policy)
    if declared is None:
        return PolicyRead(False, f"{policy} is no integration policy nevresim declares")
    include = Path(root) / "include"
    if not include.is_dir():
        return PolicyRead(False, f"no nevresim include tree at {include}")
    key = (policy, simulator_source_fingerprint(root))
    if key not in _VERDICTS:
        sites, self_test = declared
        missing = site_problems(include, sites)
        problem = None if missing else self_test_problem(include, self_test, constexpr_options)
        _VERDICTS[key] = (
            (False, f"lack {policy}: " + "; ".join(missing)) if missing
            else (False, f"hold every {policy} site, but {problem}") if problem
            else (True, f"hold every {policy} site, and the self-test {self_test[1]} holds"))
    established, found = _VERDICTS[key]
    return PolicyRead(established, f"the nevresim sources at {include} {found}")


def require_integration_policy(integration_policy: str, root) -> None:
    """Refuse by name a policy the tree at ``root`` does not execute, before a program naming it is emitted."""
    read = integration_policy_read(integration_policy, root)
    if not read.established:
        raise IntegrationPolicyRefusal(
            f"nevresim cannot execute integration_policy={integration_policy!r} on this tree: {read.evidence}. "
            f"The policy is a change to nevresim's sources that this tree does not carry; run the soma point on a "
            f"nevresim tree that does, or on an executor that implements it (the torch flow).")
