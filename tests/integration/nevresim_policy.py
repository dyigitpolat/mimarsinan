"""[U-H] A nevresim arm runs where the checkout's nevresim tree carries the integration policy it compiles.

The event-serial fold and the sync-fire register are the ODIN program's change to nevresim, a nested repository whose
commit waits for its owner; on a tree without it the driver refuses the soma point by name
(``nevresim/integration_policy_read``). A test whose nevresim arm needs such a policy is skipped by name with the
probe's evidence on such a tree, and runs (and must pass) on a tree that carries it."""

from __future__ import annotations

import pytest

from integration.parity_harness import mimarsinan_root
from mimarsinan.chip_simulation.nevresim.integration_policy_read import integration_policy_read


def requires_nevresim_policy(policy: str):
    """``skipif`` the tree the parity harness points nevresim at (``<repo>/nevresim``) lacks ``policy``."""
    read = integration_policy_read(policy, mimarsinan_root() / "nevresim")
    return pytest.mark.skipif(not read.established, reason=f"nevresim tree lacks {policy}: {read.evidence}")
