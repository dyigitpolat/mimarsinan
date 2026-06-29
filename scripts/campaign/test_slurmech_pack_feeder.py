"""Hermetic tests for the slurmech pack feeder (no real submissions)."""

from __future__ import annotations

import os
import sys

import pytest
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import slurmech_pack_feeder as spf  # noqa: E402


def _jobs(n: int, prefix: str = "job") -> list[dict]:
    return [{"id": f"{prefix}_{i}"} for i in range(n)]


# --- chunk_manifest -------------------------------------------------------


def test_chunk_six_packs_balanced_and_complete():
    jobs = _jobs(30)
    chunks = spf.chunk_manifest(jobs, 6)
    assert len(chunks) == 6
    assert [len(c) for c in chunks] == [5, 5, 5, 5, 5, 5]
    # order-preserving, no loss, no overlap
    flat = [j for c in chunks for j in c]
    assert flat == jobs


def test_chunk_uneven_sizes_differ_by_at_most_one():
    chunks = spf.chunk_manifest(_jobs(10), 3)
    sizes = [len(c) for c in chunks]
    assert sizes == [4, 3, 3]
    assert max(sizes) - min(sizes) <= 1


def test_chunk_more_packs_than_jobs_yields_no_empty_chunks():
    chunks = spf.chunk_manifest(_jobs(3), 5)
    assert len(chunks) == 3
    assert all(len(c) == 1 for c in chunks)


def test_chunk_empty_jobs_returns_empty():
    assert spf.chunk_manifest([], 6) == []


def test_chunk_rejects_zero_packs():
    with pytest.raises(ValueError):
        spf.chunk_manifest(_jobs(4), 0)


# --- normalize_job_cmd / build_pack_yaml ----------------------------------


def test_normalize_job_cmd_exact_form():
    cmd = spf.normalize_job_cmd({"id": "abc_s0"})
    assert cmd == (
        "export PYTHONPATH=src:spikingjelly:sana_fe:${PYTHONPATH:-}; "
        "python run.py --headless experiments/campaign/abc_s0.json"
    )


def test_config_path_for_job_falls_back_to_flat():
    # No cmd field: reconstruct the canonical flat path from the id.
    assert spf.config_path_for_job({"id": "abc_s0"}) == "experiments/campaign/abc_s0.json"


def test_config_path_for_job_honors_manifest_cmd_subdir():
    # A manifest job carrying its own --headless config path (e.g. a --config-dir
    # subdirectory) must be honored verbatim, not flattened back to the default dir.
    job = {
        "id": "abc_s0",
        "cmd": ["env/bin/python", "run.py", "--headless",
                "experiments/campaign/round2/abc_s0.json"],
    }
    assert spf.config_path_for_job(job) == "experiments/campaign/round2/abc_s0.json"


def test_normalize_job_cmd_honors_manifest_cmd_subdir():
    job = {
        "id": "abc_s0",
        "cmd": ["env/bin/python", "run.py", "--headless",
                "experiments/campaign/round2/abc_s0.json"],
    }
    cmd = spf.normalize_job_cmd(job)
    assert cmd.endswith("experiments/campaign/round2/abc_s0.json")
    assert "experiments/campaign/abc_s0.json" not in cmd


def test_build_pack_yaml_schema_and_cmd():
    pack = spf.build_pack_yaml(_jobs(2), parallelism=5)
    assert pack["parallelism"] == 5
    assert pack["fail_fast"] is False
    assert pack["kill_on_failure"] is False
    assert [j["name"] for j in pack["jobs"]] == ["job_0", "job_1"]
    for job in pack["jobs"]:
        assert job["cmd"] == spf.normalize_job_cmd({"id": job["name"]})
        assert job["cmd"].startswith("export PYTHONPATH=src:spikingjelly:sana_fe:")
        assert job["cmd"].endswith(f"experiments/campaign/{job['name']}.json")


# --- threads_per_job / thread_cap_prefix / thread-capped commands ---------


def test_threads_per_job_even_split():
    assert spf.threads_per_job(40, 6) == 6
    assert spf.threads_per_job(40, 5) == 8
    assert spf.threads_per_job(40, 1) == 40


def test_threads_per_job_floor_one():
    # More concurrent jobs than cores still leaves each job at least one thread.
    assert spf.threads_per_job(4, 8) == 1


def test_threads_per_job_rejects_zero_parallelism():
    with pytest.raises(ValueError):
        spf.threads_per_job(40, 0)


def test_thread_cap_prefix_none_is_empty():
    assert spf.thread_cap_prefix(None) == ""


def test_thread_cap_prefix_sets_every_pool():
    prefix = spf.thread_cap_prefix(6)
    assert prefix == (
        "export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 "
        "OPENBLAS_NUM_THREADS=6 NUMEXPR_NUM_THREADS=6;"
    )


def test_thread_cap_prefix_rejects_zero():
    with pytest.raises(ValueError):
        spf.thread_cap_prefix(0)


def test_normalize_job_cmd_with_thread_cap_prepends_pools():
    cmd = spf.normalize_job_cmd({"id": "abc_s0"}, thread_cap=6)
    assert cmd == (
        "export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 "
        "OPENBLAS_NUM_THREADS=6 NUMEXPR_NUM_THREADS=6; "
        "export PYTHONPATH=src:spikingjelly:sana_fe:${PYTHONPATH:-}; "
        "python run.py --headless experiments/campaign/abc_s0.json"
    )
    # The PYTHONPATH export and config tail are unchanged by the cap.
    assert cmd.endswith("experiments/campaign/abc_s0.json")
    assert "export PYTHONPATH=src:spikingjelly:sana_fe:" in cmd


def test_build_pack_yaml_threads_each_job():
    pack = spf.build_pack_yaml(_jobs(2), parallelism=6, thread_cap=6)
    for job in pack["jobs"]:
        assert job["cmd"].startswith("export OMP_NUM_THREADS=6 ")


def test_write_pack_yaml_roundtrips(tmp_path):
    pack = spf.build_pack_yaml(_jobs(2), parallelism=3)
    path = tmp_path / "nested" / "pack_1.yaml"
    written = spf.write_pack_yaml(pack, path)
    assert os.path.exists(written)
    loaded = yaml.safe_load(open(written))
    assert loaded == pack
    assert list(loaded.keys()) == ["parallelism", "fail_fast", "kill_on_failure", "jobs"]


# --- default_parallelism --------------------------------------------------


def test_default_parallelism_limited_by_job_count():
    assert spf.default_parallelism(5) == 5


def test_default_parallelism_capped():
    assert spf.default_parallelism(20) == 8
    assert spf.default_parallelism(20, cap=3) == 3


def test_default_parallelism_limited_by_memory():
    # floor(81000 / 40000) == 2 beats both the job count and the cap
    assert spf.default_parallelism(10, gpu_mem_mb=81000, per_job_mb=40000) == 2


def test_default_parallelism_floor_one():
    assert spf.default_parallelism(1) == 1


def test_default_parallelism_rejects_zero_per_job():
    with pytest.raises(ValueError):
        spf.default_parallelism(4, per_job_mb=0)


# --- pack_command / plan_packs --------------------------------------------


def test_pack_command_exact():
    cmd = spf.pack_command("runs/campaign/slurmech_packs_v2/p1.yaml", 5)
    assert cmd == [
        "env/bin/slurmech", "pack",
        "runs/campaign/slurmech_packs_v2/p1.yaml",
        "--parallelism", "5", "--detach",
    ]


def test_pack_prefix_from_manifest():
    assert spf.pack_prefix_from_manifest("a/b/mnist_mixer_queue_manifest.json") == "mnist_mixer"
    assert spf.pack_prefix_from_manifest("foo_manifest.json") == "foo"


def test_plan_packs_names_paths_parallelism(tmp_path):
    plans = spf.plan_packs(_jobs(30), 6, str(tmp_path), pack_prefix="mnist_mixer")
    assert [p.name for p in plans] == [f"mnist_mixer_pack_{i}" for i in range(1, 7)]
    assert all(p.parallelism == 5 for p in plans)
    assert all(p.path == os.path.join(str(tmp_path), f"{p.name}.yaml") for p in plans)
    assert plans[0].command[:3] == ["env/bin/slurmech", "pack", plans[0].path]


def test_plan_packs_splits_node_cores_into_thread_cap(tmp_path):
    # 30 jobs / 6 packs -> parallelism 5; 40 cores / 5 jobs -> 8 threads each.
    plans = spf.plan_packs(_jobs(30), 6, str(tmp_path), pack_prefix="t", cpus_per_node=40)
    assert all(p.thread_cap == 8 for p in plans)
    yaml_cmd = spf.build_pack_yaml(plans[0].jobs, plans[0].parallelism, plans[0].thread_cap)
    assert all(j["cmd"].startswith("export OMP_NUM_THREADS=8 ") for j in yaml_cmd["jobs"])


def test_plan_packs_thread_cap_disabled_leaves_defaults(tmp_path):
    plans = spf.plan_packs(_jobs(6), 3, str(tmp_path), pack_prefix="t", cpus_per_node=None)
    assert all(p.thread_cap is None for p in plans)
    yaml_cmd = spf.build_pack_yaml(plans[0].jobs, plans[0].parallelism, plans[0].thread_cap)
    assert all(j["cmd"].startswith("export PYTHONPATH=") for j in yaml_cmd["jobs"])


# --- submit_packs ---------------------------------------------------------


def test_submit_packs_writes_yaml_and_runs_each(tmp_path):
    plans = spf.plan_packs(_jobs(6), 3, str(tmp_path), pack_prefix="t")
    calls: list[list[str]] = []
    logs: list[str] = []

    spf.submit_packs(plans, runner=calls.append, logger=logs.append)

    assert len(calls) == 3
    for plan, cmd in zip(plans, calls):
        assert cmd == plan.command
        assert os.path.exists(plan.path)
    assert all(line.startswith("[submit]") for line in logs)


def test_submit_packs_dry_run_does_not_call_runner(tmp_path):
    plans = spf.plan_packs(_jobs(4), 2, str(tmp_path), pack_prefix="t")
    calls: list[list[str]] = []
    logs: list[str] = []

    spf.submit_packs(plans, runner=calls.append, dry_run=True, logger=logs.append)

    assert calls == []
    assert all(os.path.exists(p.path) for p in plans)
    assert all(line.startswith("[dry-run]") for line in logs)


def test_submit_requires_runner_when_not_dry_run(tmp_path):
    plans = spf.plan_packs(_jobs(2), 1, str(tmp_path), pack_prefix="t")
    with pytest.raises(ValueError):
        spf.submit_packs(plans, runner=None, dry_run=False, logger=lambda *_: None)


# --- parse_active_count ---------------------------------------------------


def test_parse_active_count_counts_active_states():
    text = (
        "run_a  RUNNING  node01\n"
        "run_b  PENDING  (Resources)\n"
        "run_c  COMPLETED  node02\n"
        "header line with no state\n"
    )
    assert spf.parse_active_count(text) == 2


def test_parse_active_count_empty():
    assert spf.parse_active_count("") == 0


# --- feed_packs -----------------------------------------------------------


def test_feed_packs_throttles_to_n_nodes(tmp_path):
    plans = spf.plan_packs(_jobs(8), 4, str(tmp_path), pack_prefix="t")
    calls: list[list[str]] = []
    sleeps: list[float] = []

    # Scripted active-allocation counts seen on each status poll.
    # n_nodes=2: poll0 -> 0 free=2 (submit p1,p2); poll1 -> 2 free=0 (wait);
    # poll2 -> 1 free=1 (submit p3); poll3 -> 1 free=1 (submit p4); done.
    active_seq = iter([0, 2, 1, 1])
    statuses: list[int] = []

    def status_fn() -> str:
        statuses.append(1)
        return "scripted"

    def active_count_fn(_text: str) -> int:
        return next(active_seq)

    submitted = spf.feed_packs(
        plans,
        n_nodes=2,
        runner=calls.append,
        status_fn=status_fn,
        active_count_fn=active_count_fn,
        poll_interval=7.0,
        sleep_fn=sleeps.append,
        logger=lambda *_: None,
    )

    assert [p.name for p in submitted] == [p.name for p in plans]
    assert len(calls) == 4
    # Never submitted more than the free capacity in any wave:
    # waves were [2, 0, 1, 1] -> at most n_nodes concurrent newly submitted.
    assert all(s == 7.0 for s in sleeps)
    # Slept only while work remained pending (3 waits before the final drain).
    assert len(sleeps) == 3


def test_feed_packs_dry_run_skips_polling(tmp_path):
    plans = spf.plan_packs(_jobs(6), 3, str(tmp_path), pack_prefix="t")
    calls: list[list[str]] = []
    logs: list[str] = []
    polled = {"n": 0}

    def status_fn() -> str:
        polled["n"] += 1
        return ""

    submitted = spf.feed_packs(
        plans,
        n_nodes=2,
        runner=calls.append,
        status_fn=status_fn,
        dry_run=True,
        sleep_fn=lambda *_: None,
        logger=logs.append,
    )

    assert polled["n"] == 0
    assert calls == []
    assert len(submitted) == 3
    assert all(os.path.exists(p.path) for p in plans)
    assert all(line.startswith("[dry-run]") for line in logs)


def test_feed_packs_rejects_zero_nodes(tmp_path):
    plans = spf.plan_packs(_jobs(2), 1, str(tmp_path), pack_prefix="t")
    with pytest.raises(ValueError):
        spf.feed_packs(plans, n_nodes=0, runner=lambda *_: None)


# --- load_manifest --------------------------------------------------------


def test_load_manifest_rejects_non_list(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"id": "x"}')
    with pytest.raises(ValueError):
        spf.load_manifest(str(bad))


def test_load_manifest_reads_list(tmp_path):
    good = tmp_path / "good.json"
    good.write_text('[{"id": "a"}, {"id": "b"}]')
    assert spf.load_manifest(str(good)) == [{"id": "a"}, {"id": "b"}]


# --- main (CLI integration, hermetic) -------------------------------------


def _write_manifest(tmp_path, n: int):
    import json
    path = tmp_path / "queue_manifest.json"
    path.write_text(json.dumps([{"id": f"cell_{i}"} for i in range(n)]))
    return path


def test_main_dry_run_writes_yamls_no_submit(tmp_path, capsys):
    manifest = _write_manifest(tmp_path, 30)
    packs_dir = tmp_path / "packs_v2"
    calls: list[list[str]] = []

    rc = spf.main(
        [str(manifest), "--nodes", "6", "--packs-dir", str(packs_dir), "--dry-run"],
        runner=calls.append,
    )

    assert rc == 0
    assert calls == []
    written = sorted(os.listdir(packs_dir))
    assert written == [f"queue_pack_{i}.yaml" for i in range(1, 7)]
    out = capsys.readouterr().out
    assert "6 packs" in out
    assert "env/bin/slurmech pack" in out


def test_main_max_parallelism_cap(tmp_path):
    manifest = _write_manifest(tmp_path, 30)
    packs_dir = tmp_path / "packs_v2"
    calls: list[list[str]] = []

    spf.main(
        [str(manifest), "--nodes", "1", "--packs", "1",
         "--max-parallelism", "3", "--packs-dir", str(packs_dir)],
        runner=calls.append,
    )

    # one pack of 30 jobs, parallelism capped at 3
    assert len(calls) == 1
    assert calls[0] == [
        "env/bin/slurmech", "pack",
        os.path.join(str(packs_dir), "queue_pack_1.yaml"),
        "--parallelism", "3", "--detach",
    ]


def test_main_keep_feeding_throttles_to_nodes(tmp_path):
    manifest = _write_manifest(tmp_path, 6)
    packs_dir = tmp_path / "packs_v2"
    calls: list[list[str]] = []
    sleeps: list[float] = []
    # nodes=2, 3 packs of 2: wave0 active=0 -> submit 2; wave1 active=2 -> wait;
    # wave2 active=0 -> submit the last. Exercises real feeding under throttle.
    active_seq = iter([0, 2, 0])

    # feed_packs resolves parse_active_count at call time, so patching the module
    # attribute scripts the active-allocation count main feeds through.
    orig = spf.parse_active_count
    spf.parse_active_count = lambda _t: next(active_seq)  # noqa: E731
    try:
        rc = spf.main(
            [str(manifest), "--nodes", "2", "--packs", "3",
             "--packs-dir", str(packs_dir), "--keep-feeding", "--poll-interval", "0.01"],
            runner=calls.append,
            status_fn=lambda: "x",
            sleep_fn=sleeps.append,
        )
    finally:
        spf.parse_active_count = orig

    assert rc == 0
    assert len(calls) == 3
    # waited on wave1 (free=0) and wave2's pre-poll; last drain needs no wait.
    assert sleeps == [0.01, 0.01]
