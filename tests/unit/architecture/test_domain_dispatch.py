"""[D2] Deployment steps ask CAPABILITIES, never the domain name.

The program's design is domain-first dispatch through policies and the
packaging contract. A step that tests ``plan.is_mvm`` re-derives a decision
some policy already owns, so a third core family (RRAM, stochastic-rounding
cores) would mean auditing every such site instead of implementing one
method. Domain resolution itself stays in the SSOTs listed below.
"""

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[3] / "src" / "mimarsinan"

# Where deciding on the domain IS the job.
_DOMAIN_SSOTS = {
    "chip_simulation/core_semantics.py",       # the axis itself
    "pipelining/core/deployment_plan.py",      # config -> plan resolution
    "config_schema/deployment_derivation.py",  # config -> derived flags
    "config_schema/registry/domain_rules.py",  # document legality
    "mapping/platform/packaging_contract.py",  # plan -> contract dispatch
}

_IS_MVM = re.compile(r"\bis_mvm\b")


def _offenders() -> list[str]:
    hits = []
    for path in SRC.rglob("*.py"):
        rel = path.relative_to(SRC).as_posix()
        if rel in _DOMAIN_SSOTS:
            continue
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            if _IS_MVM.search(line) and not line.lstrip().startswith("#"):
                hits.append(f"{rel}:{number}")
    return hits


def test_no_domain_predicates_outside_the_ssots():
    assert not _offenders(), (
        "steps must ask a capability (mode_policy().observes_values(), "
        ".requires_activation_alignment(), packaging_contract_for(plan)"
        ".boundary_is_gridded / .boundary.signed) instead of testing the "
        f"domain: {_offenders()}"
    )
