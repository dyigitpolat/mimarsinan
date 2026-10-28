"""Shared fixtures for the chip-simulation tests."""

from __future__ import annotations

import fcntl

import pytest


@pytest.fixture
def lava_graph_slot(tmp_path_factory):
    """Run at most one real Lava process graph at a time across every xdist worker of the session.

    A graph holds about 35 tasks (Lava's actor processes and their channel threads; 40 measured
    for one worker running ``test_lava_subtractive_lif.py``). Four graphs at once reached a
    150-task process-tree cap, where a thread that cannot start makes the run fail (and, before the
    executor released an unstarted runtime's actors, hang). The slot is a file lock under the
    session's shared temporary root, so the workers of one session serialize their graphs.
    """
    lock = tmp_path_factory.getbasetemp().parent / "mimarsinan_lava_graph.lock"
    with open(lock, "a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
