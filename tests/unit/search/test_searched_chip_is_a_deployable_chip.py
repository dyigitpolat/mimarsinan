"""The searched chip must be a chip a FIXED deployment could have declared.

W5.1 made every candidate platform the deployment's own resolution of the
declared platform plus the searched overlay. One knob still stood outside that
promise: the search resolver dropped ``allow_neuron_splitting``, so a searched
chip's resolved surface was a strict SUBSET of what the same chip gets when it
is declared by hand — and ``ChipCapabilities`` reads a missing permission as
DENIED. The search therefore scored candidates on a chip that could not split
neurons while the run deploying them could, and the promoted platform was not
the fixed-mode resolution of the winner.

Found end to end: the W5.3 tier-0 search cell deploys a discovered chip whose
64-neuron cores host a 256-neuron layer, and its resolved platform differed from
the hand-declared twin by exactly this key.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from mimarsinan.mapping.platform.mapping_structure import ChipCapabilities
from mimarsinan.pipelining.core.platform_constraints_resolver import (
    build_platform_constraints_resolved,
)
from mimarsinan.pipelining.pipeline_steps.config.architecture_search_helpers import (
    build_fixed_platform_constraints,
    make_platform_resolver,
)

SPLITTING_KEY = "allow_neuron_splitting"


def declared_config(*, allow_splitting: bool) -> Dict[str, Any]:
    return {
        "cores": [{"max_axons": 784, "max_neurons": 512, "count": 60}],
        "target_tq": 8,
        "weight_bits": 5,
        "allow_scheduling": False,
        "allow_neuron_splitting": allow_splitting,
    }


CANDIDATE_OVERLAY = {
    "cores": [{"max_axons": 864, "max_neurons": 64, "count": 8, "has_bias": True}],
    "target_tq": 8,
}


@pytest.mark.parametrize("allow_splitting", [True, False])
class TestOneResolutionForSearchedAndDeclaredChips:
    def test_a_candidate_platform_is_the_fixed_resolution_of_that_chip(
        self, allow_splitting,
    ):
        cfg = declared_config(allow_splitting=allow_splitting)
        candidate = make_platform_resolver(cfg)(CANDIDATE_OVERLAY)
        hand_declared = build_platform_constraints_resolved({**cfg, **CANDIDATE_OVERLAY})
        assert candidate == hand_declared, (
            "a searched chip must resolve exactly as the same chip declared by hand"
        )

    def test_the_run_base_resolves_the_same_either_way(self, allow_splitting):
        cfg = declared_config(allow_splitting=allow_splitting)
        assert build_fixed_platform_constraints(cfg) == (
            build_platform_constraints_resolved(cfg)
        )

    def test_the_searched_chip_carries_the_runs_splitting_permission(
        self, allow_splitting,
    ):
        cfg = declared_config(allow_splitting=allow_splitting)
        candidate = make_platform_resolver(cfg)(CANDIDATE_OVERLAY)
        assert candidate[SPLITTING_KEY] is allow_splitting

    def test_candidate_capabilities_match_the_deployed_ones(self, allow_splitting):
        # ChipCapabilities is what the layout scoring and the deployed mapping
        # both read; a permission missing from the dict reads as DENIED, so the
        # search would score a chip the run does not deploy.
        cfg = declared_config(allow_splitting=allow_splitting)
        candidate = make_platform_resolver(cfg)(CANDIDATE_OVERLAY)
        deployed = build_platform_constraints_resolved({**cfg, **CANDIDATE_OVERLAY})
        assert (
            ChipCapabilities.from_platform_constraints(candidate).capability_bits()
            == ChipCapabilities.from_platform_constraints(deployed).capability_bits()
        )
        assert (
            ChipCapabilities.from_platform_constraints(candidate).allow_neuron_splitting
            is allow_splitting
        )


class TestTheResolverHasNoSecondMode:
    def test_resolution_takes_no_flag_that_could_drop_a_permission(self):
        # The one resolver had a keyword that omitted a permission for the
        # search only; a re-added mode switch is how the two chips drift apart.
        import inspect

        params = inspect.signature(build_platform_constraints_resolved).parameters
        assert list(params) == ["pipeline_config"], (
            "platform resolution is ONE function of the config, with no mode "
            f"switch; found extra parameters: {list(params)[1:]}"
        )
