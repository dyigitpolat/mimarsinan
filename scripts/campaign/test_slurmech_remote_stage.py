"""Hermetic tests for the remote slurmech-pack artifact stager (no real SSH)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import slurmech_remote_stage as s  # noqa: E402


class FakeConn:
    """Records get_file calls; serves canned ls listings and an existence set."""

    def __init__(self, listing: str = "", present: set[str] | None = None):
        self._listing = listing
        self._present = present if present is not None else None
        self.get_file_calls: list[tuple[str, str]] = []
        self.streamed: list[str] = []

    def run_with_streaming(self, command: str, cb) -> int:
        self.streamed.append(command)
        cb(self._listing)
        return 0

    def exists(self, path: str) -> bool:
        return True if self._present is None else path in self._present

    def get_file(self, remote: str, local: str) -> None:
        self.get_file_calls.append((remote, local))
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        Path(local).write_text("0.95\n")


class FakeRegistry:
    def __init__(self, mapping: dict[str, str]):
        self._mapping = mapping

    def find_run(self, run_id=None, job_id=None):
        if run_id in self._mapping:
            return {"remote_run_dir": self._mapping[run_id], "run_id": run_id}
        return None


# --- pure path logic ------------------------------------------------------


def test_phased_run_dirname_appends_suffix_idempotently():
    assert s.phased_run_dirname("foo") == "foo" + s.RUN_SUFFIX
    assert s.phased_run_dirname("foo" + s.RUN_SUFFIX) == "foo" + s.RUN_SUFFIX


def test_run_name_from_dirname_inverts_and_rejects_nonmatching():
    assert s.run_name_from_dirname("foo" + s.RUN_SUFFIX) == "foo"
    assert s.run_name_from_dirname("foo") is None


def test_remote_artifact_paths_compose_under_workspace_generated():
    paths = s.remote_artifact_paths("/runs/abc", "myrun")
    assert paths["metric"] == f"/runs/abc/workspace/generated/myrun{s.RUN_SUFFIX}/__target_metric.json"
    assert paths["run_info"] == (
        f"/runs/abc/workspace/generated/myrun{s.RUN_SUFFIX}/_GUI_STATE/run_info.json"
    )
    assert paths["cost_record"] == (
        f"/runs/abc/workspace/generated/myrun{s.RUN_SUFFIX}/cost_record.json"
    )


def test_remote_generated_root_strips_trailing_slash():
    assert s.remote_generated_root("/runs/abc/") == "/runs/abc/workspace/generated"


def test_local_artifact_paths_mirror_local_generated_tree(tmp_path):
    paths = s.local_artifact_paths(tmp_path, "myrun")
    assert paths["metric"] == tmp_path / f"myrun{s.RUN_SUFFIX}" / "__target_metric.json"
    assert paths["run_info"] == (
        tmp_path / f"myrun{s.RUN_SUFFIX}" / "_GUI_STATE" / "run_info.json"
    )


def test_parse_generated_listing_keeps_only_phased_dirs_sorted_unique():
    listing = "\n".join([
        f"b_run{s.RUN_SUFFIX}/",
        f"a_run{s.RUN_SUFFIX}",
        "status.tsv",
        "",
        f"a_run{s.RUN_SUFFIX}",  # dup
        "some_other_dir",
    ])
    assert s.parse_generated_listing(listing) == ["a_run", "b_run"]


# --- enumerate / stage (FakeConn) -----------------------------------------


def test_enumerate_runs_parses_remote_listing():
    conn = FakeConn(listing=f"x{s.RUN_SUFFIX}\ny{s.RUN_SUFFIX}\nstatus.tsv\n")
    assert s.enumerate_runs(conn, "/runs/abc") == ["x", "y"]
    assert "workspace/generated" in conn.streamed[0]


def test_stage_run_pulls_all_three_jsons_when_present(tmp_path):
    conn = FakeConn()  # present=None -> everything exists
    result = s.stage_run(conn, "/runs/abc", "myrun", tmp_path)
    assert result.complete
    assert result.have_metric and result.have_run_info and result.have_cost_record
    assert len(conn.get_file_calls) == 3
    metric_local = tmp_path / f"myrun{s.RUN_SUFFIX}" / "__target_metric.json"
    assert metric_local.is_file()


def test_stage_run_tolerates_missing_cost_record_still_complete(tmp_path):
    base = f"/runs/abc/workspace/generated/myrun{s.RUN_SUFFIX}"
    present = {f"{base}/__target_metric.json", f"{base}/_GUI_STATE/run_info.json"}
    conn = FakeConn(present=present)
    result = s.stage_run(conn, "/runs/abc", "myrun", tmp_path)
    assert result.complete  # cost_record is optional
    assert not result.have_cost_record
    assert len(conn.get_file_calls) == 2


def test_stage_run_tolerates_missing_metric_orphan(tmp_path):
    present = {
        f"/runs/abc/workspace/generated/myrun{s.RUN_SUFFIX}/_GUI_STATE/run_info.json",
    }
    conn = FakeConn(present=present)
    result = s.stage_run(conn, "/runs/abc", "myrun", tmp_path)
    assert not result.complete
    assert result.have_run_info and not result.have_metric
    assert len(conn.get_file_calls) == 1


def test_stage_pack_stages_every_listed_run(tmp_path):
    conn = FakeConn(listing=f"r0{s.RUN_SUFFIX}\nr1{s.RUN_SUFFIX}\n")
    pack = s.stage_pack(conn, "/runs/abc", tmp_path, run_id="RID")
    assert pack.run_id == "RID"
    assert [r.run_name for r in pack.results] == ["r0", "r1"]
    assert all(r.complete for r in pack.results)


def test_stage_packs_resolves_run_dirs_via_registry(tmp_path):
    conn = FakeConn(listing=f"r0{s.RUN_SUFFIX}\n")
    registry = FakeRegistry({"RID1": "/runs/one", "RID2": "/runs/two"})
    logs: list[str] = []
    packs = s.stage_packs(conn, registry, ["RID1", "RID2"], tmp_path, log=logs.append)
    assert [p.remote_run_dir for p in packs] == ["/runs/one", "/runs/two"]
    assert len(logs) == 2
    assert "RID1" in logs[0]


def test_resolve_remote_run_dir_raises_on_unknown():
    registry = FakeRegistry({})
    try:
        s.resolve_remote_run_dir(registry, "nope")
    except KeyError as exc:
        assert "nope" in str(exc)
    else:
        raise AssertionError("expected KeyError")
