"""Locks the minimal GPU leasing logic (no real GPU needed: snapshots are injected)."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts", "gpu"))

import gpu_lease as gl  # noqa: E402


def stat(idx, total=100000, free=100000, util=0):
    return gl.GpuStat(idx, total, free, util)


# --------------------------------------------------------------------------- #
# parsing + freeness predicate
# --------------------------------------------------------------------------- #

def test_parse_nvidia_smi_skips_garbage_and_reads_rows():
    text = "0, 97887, 96536, 0\n1, 97887, 8312, 0\nfoo,bar\n"
    stats = gl.parse_nvidia_smi(text)
    assert [s.index for s in stats] == [0, 1]
    assert stats[0].mem_free == 96536 and stats[1].util == 0


def test_is_free_requires_both_memory_and_util():
    assert stat(0, 100000, 96000, 0).is_free()            # 96% idle, 0% util
    assert not stat(0, 100000, 96000, 50).is_free()       # busy util
    assert not stat(0, 100000, 40000, 0).is_free()        # <80% mem idle


# --------------------------------------------------------------------------- #
# choose(): the pure pick decision
# --------------------------------------------------------------------------- #

def test_choose_free_picks_an_exclusively_free_unleased_gpu():
    stats = [stat(0, free=8000, util=100), stat(1, free=96000, util=0)]
    assert gl.choose("free", 0, stats, []) == 1   # only GPU1 is idle+free-mem


def test_choose_free_skips_a_gpu_already_leased_by_the_pool():
    stats = [stat(0, free=96000, util=0), stat(1, free=96000, util=0)]
    leases = [gl.Lease(gpu=0, pid=os.getpid(), mode="free", mb=0, path="/x")]
    assert gl.choose("free", 0, stats, leases) == 1


def test_choose_free_returns_none_when_nothing_is_free():
    stats = [stat(0, free=8000, util=100), stat(1, free=32000, util=100)]
    assert gl.choose("free", 0, stats, []) is None


def test_choose_fit_ignores_util_and_respects_memory_only():
    # GPU2 pinned at 100% util by another user but has room -> fit is fine there.
    stats = [stat(0, free=2000, util=0), stat(2, free=40000, util=100)]
    assert gl.choose("fit", need_mb=8000, stats=stats, leases=[]) == 2


def test_choose_fit_subtracts_pool_reservations():
    stats = [stat(0, free=20000, util=0)]
    leases = [gl.Lease(gpu=0, pid=os.getpid(), mode="fit", mb=15000, path="/a")]
    # 20000 - 15000 = 5000 remaining < 8000 needed -> no fit.
    assert gl.choose("fit", 8000, stats, leases) is None
    assert gl.choose("fit", 4000, stats, leases) == 0


def test_choose_fit_avoids_a_gpu_held_exclusively_by_free():
    stats = [stat(0, free=96000, util=0)]
    leases = [gl.Lease(gpu=0, pid=os.getpid(), mode="free", mb=0, path="/a")]
    assert gl.choose("fit", 4000, stats, leases) is None


def test_choose_load_balances_to_most_headroom():
    stats = [stat(0, free=20000, util=0), stat(1, free=90000, util=0)]
    assert gl.choose("fit", 8000, stats, leases=[]) == 1


def test_choose_fit_respects_max_per_gpu_cap():
    # One big GPU with room for many by memory, but capped at 2 concurrent fit jobs.
    stats = [stat(0, free=96000, util=0)]
    two = [gl.Lease(0, os.getpid(), "fit", 8000, "/a"),
           gl.Lease(0, os.getpid(), "fit", 8000, "/b")]
    assert gl.choose("fit", 8000, stats, [], max_per_gpu=2) == 0     # 0 leases < 2
    assert gl.choose("fit", 8000, stats, two, max_per_gpu=2) is None  # at cap
    assert gl.choose("fit", 8000, stats, two, max_per_gpu=None) == 0  # uncapped ok


# --------------------------------------------------------------------------- #
# lease lifecycle: dead-pid pruning + acquire/release round-trip
# --------------------------------------------------------------------------- #

def test_read_leases_prunes_dead_pids(tmp_path):
    live = tmp_path / "live.lease"
    dead = tmp_path / "dead.lease"
    live.write_text(json.dumps({"gpu": 0, "pid": os.getpid(), "mode": "fit", "mb": 1}))
    dead.write_text(json.dumps({"gpu": 1, "pid": 999999999, "mode": "fit", "mb": 1}))
    leases = gl.read_leases(str(tmp_path))
    assert [l.gpu for l in leases] == [0]
    assert not dead.exists()   # the dead lease was reaped
    assert live.exists()


def test_acquire_writes_a_lease_and_release_removes_it(tmp_path):
    snap = lambda: [stat(0, free=96000, util=0)]
    lease = gl.acquire("free", 0, directory=str(tmp_path), snapshot=snap, cmd="x")
    assert lease is not None and lease.gpu == 0
    assert os.path.exists(lease.path)
    # a second 'free' acquire now sees GPU0 held -> None
    assert gl.acquire("free", 0, directory=str(tmp_path), snapshot=snap) is None
    gl.release(lease)
    assert not os.path.exists(lease.path)
    # released -> acquirable again
    assert gl.acquire("free", 0, directory=str(tmp_path), snapshot=snap) is not None


def test_acquire_blocking_times_out_when_never_free(tmp_path):
    snap = lambda: [stat(0, free=1000, util=100)]
    slept = []
    lease = gl.acquire_blocking(
        "free", 0, timeout=10, poll=4, directory=str(tmp_path), snapshot=snap,
        sleep=slept.append, clock=lambda: slept and sum(slept) or 0.0,
    )
    assert lease is None
    assert slept  # it actually waited/polled before giving up
