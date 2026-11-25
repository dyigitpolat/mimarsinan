""

from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))

import cl6_common as K
import cl6_head as CH
import w34_points as P

RA = P.RA
E2E = RA / "12_tcad_readiness" / "e2e"
RB = E2E / "routeb"
UNIT = RA / "11_wave2" / "final_units" / "uw34"
HEAD_FORMAT = "e2e-routeb-head-amendment/1"
TZ = datetime.timezone(datetime.timedelta(hours=8))
Git = Callable[[List[str]], Tuple[int, str]]


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, path
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _git_at(worktree: Path) -> Git:
    def run(args: List[str]) -> Tuple[int, str]:
        done = subprocess.run(["git", "-C", str(worktree), *args], capture_output=True, text=True)
        return done.returncode, done.stdout.strip()
    return run


def head_block(head_record: Path, *, plan_lock: Path, queue: Path, ro_worktree: Path, code_patch: Path,
               revision: int, git: Optional[Git] = None) -> Tuple[Dict[str, Any], List[str]]:
    ""
    problems: List[str] = []
    rec = json.loads(Path(head_record).read_text())
    head, export = str(rec.get("head") or ""), dict(rec.get("export") or {})
    lock = json.loads(Path(plan_lock).read_text())
    bad = []
    if rec.get("format") != HEAD_FORMAT:
        bad.append(f"format {rec.get('format')!r} is not {HEAD_FORMAT}")
    amends = rec.get("amends") or {}
    if amends.get("file") != "routeb/plan_lock.json" or amends.get("sha256") != P.sha256_file(plan_lock):
        bad.append(f"it amends {amends.get('file')} {str(amends.get('sha256'))[:12]}, not {plan_lock} "
                   f"{P.sha256_file(plan_lock)[:12]}")
    if lock.get("revision") != revision:
        bad.append(f"the plan lock in force is revision {lock.get('revision')}, not {revision}")
    if export.get("patch_state") != "applied":
        bad.append(f"export patch_state {export.get('patch_state')!r} is not 'applied'")
    if export.get("patch_sha256") != P.sha256_file(code_patch):
        bad.append(f"export patch {str(export.get('patch_sha256'))[:12]} is not {code_patch} "
                   f"{P.sha256_file(code_patch)[:12]}")
    pin_head = _module("pin_head_uw34", RB / "pin_head.py")
    section = str((lock.get("head_rule") or {}).get("queue_section") or "")
    entry = pin_head.queue_entry(Path(queue), head, section) if section else {"found": False}
    if not entry.get("found") or not entry.get("names_head"):
        bad.append(f"the queue section {section!r} of {queue} does not name {head[:12]}")
    problems += [f"W34_HEAD_RECORD: {b}" for b in bad]
    git = git or _git_at(Path(ro_worktree))
    reads = {"head": git(["rev-parse", "HEAD"]), "tree": git(["rev-parse", "HEAD^{tree}"]),
             "status": git(["status", "--porcelain", "--ignore-submodules=all"])}
    if reads["head"] != (0, head):
        problems.append(f"W34_HEAD_GIT: {ro_worktree} HEAD reads {reads['head']}, not {head}")
    if reads["tree"] != (0, export.get("tree")):
        problems.append(f"W34_HEAD_GIT: {ro_worktree} HEAD^{{tree}} reads {reads['tree']}, not {export.get('tree')}")
    if reads["status"] != (0, ""):
        problems.append(f"W34_HEAD_GIT: {ro_worktree} status (exit {reads['status'][0]}) is not clean: "
                        f"{reads['status'][1][:300]!r}")
    problems += export_problems(Path(str(export.get("path"))), head, export)
    block = {"commit": head, "tree": export.get("tree"), "worktree": export.get("path"),
             "ro_worktree": str(ro_worktree), "export_record_sha256": export.get("record_sha256"),
             "code_patch_sha256": export.get("patch_sha256"), "src_manifest_sha256": export.get("src_manifest_sha256"),
             "head_record": {"file": str(head_record), "sha256": P.sha256_file(head_record)},
             "route_b_revision": revision}
    if not problems:
        block["src_digest"] = K.src_digest(Path(str(export["path"])))
    return block, problems


def is_form_head(head: Any) -> bool:
    ""
    return isinstance(head, dict) and bool(head.get("form_sha256"))


def form_head_block(head: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    ""
    block, problems = dict(head), []
    manifest = CH.tr_path(str(head.get("form_manifest") or ""))
    now = P.sha256_file(manifest) if manifest.is_file() else None
    if now != head.get("form_sha256"):
        problems.append(f"W34_HEAD_FORM: the form's manifest {manifest} is {str(now)[:12]}, not the head's "
                        f"{str(head.get('form_sha256'))[:12]}")
    export = {"tree": head.get("tree"), "src_manifest_sha256": head.get("src_manifest_sha256"),
              "record_sha256": head.get("export_record_sha256")}
    problems += export_problems(Path(str(head.get("worktree"))), str(head.get("commit")), export)
    if not problems:
        block["src_digest"] = K.src_digest(Path(str(head["worktree"])))
    return block, problems


def export_problems(path: Path, head: str, export: Dict[str, Any]) -> List[str]:
    ""
    if not (path / "E2E_EXPORT.json").is_file():
        return [f"W34_EXPORT_MOVED: {path} holds no E2E_EXPORT.json"]
    et = _module("export_tree_uw34", E2E / "driver" / "export_tree.py")
    now = et.verify(path)
    bad = [k for k, want in (("commit", head), ("tree", export.get("tree")),
                             ("src_manifest_sha256", export.get("src_manifest_sha256")),
                             ("export_record_sha256", export.get("record_sha256"))) if now.get(k) != want]
    if not now.get("export_verified"):
        bad.append("export_verified False")
    return [f"W34_EXPORT_MOVED: {path}: {', '.join(bad)}"] if bad else []


def frozen_digests(path: Path) -> Dict[str, str]:
    out = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            digest, name = line.split(None, 1)
            out[name.strip()] = digest
    return out


def fill_files(files: Dict[str, str], frozen: Optional[Dict[str, str]]) -> Tuple[Dict[str, str], List[str]]:
    filled, problems = {}, []
    for rel, want in files.items():
        path = P.ra_path(rel)
        now = P.sha256_file(path) if path.is_file() else None
        if str(want).startswith("<U-W34 frozen digest"):
            unit = (frozen or {}).get(rel)
            if now is None or unit is None or now != unit:
                problems.append(f"W34_NOT_INSTALLED: {rel} is {str(now)[:12]}, not U-W34's frozen "
                                f"{str(unit)[:12]}: install U-W34 first")
            filled[rel] = now
        elif str(want).startswith("<at write"):
            if now is None:
                problems.append(f"W34_FILE_MOVED: {rel} is absent")
            filled[rel] = now
        else:
            if now != want:
                problems.append(f"W34_FILE_MOVED: {rel} is {str(now)[:12]}, not the pinned {str(want)[:12]}")
            filled[rel] = want
    return filled, problems


def source_problems(lock: Dict[str, Any]) -> List[str]:
    ""
    problems = []
    try:
        order: Optional[List[str]] = P.precut_order(lock)
    except P.LockError as error:
        order, problems = None, [str(error)]
    for key, m in P.members(lock).items():
        for budget in ("B0", "B1"):
            pinned = m.get(f"{budget.lower()}_sources_sha256")
            if not isinstance(pinned, dict):
                continue
            d = P.row_dir(m, budget)
            for name, want in pinned.items():
                now = P.sha256_file(d / name) if (d / name).is_file() else None
                if now != want:
                    problems.append(f"W34_SOURCE_MOVED: {key} {budget} {d / name} is {str(now)[:12]}, not {want[:12]}")
        files = m.get("precut_files_sha256")
        if isinstance(files, dict) and order is not None:
            if sorted(order) != sorted(files):
                problems.append(f"W34_SOURCE_MOVED: {key}: precut_order {order} names other payloads than {sorted(files)}")
                continue
            digest, now = P.precut_digest(Path(m["b0_working_directory"]), order)
            moved = sorted(n for n in files if now[n] != files[n])
            if moved or digest != m.get("precut_digest"):
                problems.append(f"W34_SOURCE_MOVED: {key} B0 working directory {m['b0_working_directory']}: payloads "
                                f"{moved} moved, pre-cut digest {digest[:12]} against {str(m.get('precut_digest'))[:12]}")
    return problems


def split_problems(lock: Dict[str, Any]) -> List[str]:
    ""
    root = Path(str(lock["datasets"])) / "cifar-100-python"
    entries = sorted(p.name for p in root.iterdir()) if root.is_dir() else None
    if entries is None or not set(entries) <= {"meta", "train"} or "train" not in entries:
        return [f"W34_DATASETS_TEST_BATCH: {root} holds {entries}, not meta and train only"]
    return []


def bib_problems(lock: Dict[str, Any], paper_root: Optional[Path] = None) -> List[str]:
    ""
    bib = (lock.get("pricing") or {}).get("sources_bib") or {}
    path = Path(P.template(bib.get("file", "references.bib")))
    path = path if path.is_absolute() else Path(paper_root or RA.parent) / path
    text = path.read_text() if path.is_file() else ""
    missing = [k for k in bib.get("keys") or [] if f"{{{k}," not in text]
    return [f"W34_BIB_KEY_ABSENT: {missing} not in {path}"] if missing else []


def _walk(x: Any, path: Tuple = ()):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from _walk(v, path + (k,))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from _walk(v, path + (i,))
    else:
        yield path, x


def placeholder_problems(lock: Dict[str, Any]) -> List[str]:
    ""
    at_cut = {i for i, m in enumerate(lock["members"]) if not isinstance(m.get("b0_sources_sha256"), dict)}
    allowed = {"b0_sources_sha256", "b0_working_directory", "precut_digest"}
    left = [".".join(map(str, p)) for p, v in _walk(lock) if isinstance(v, str) and v.startswith("<")
            and not (len(p) == 3 and p[0] == "members" and p[1] in at_cut and p[2] in allowed)]
    return [f"W34_PLACEHOLDER_UNFILLED: {left}"] if left else []


def lock_name(revision: int) -> str:
    return f"lock.r{int(revision)}.json"


def write(args, git: Optional[Git] = None) -> int:
    draft_path = Path(args.draft)
    lock = json.loads(draft_path.read_text())
    out = Path(args.dir) / lock_name(lock["revision"])
    if out.exists():
        print(f"REFUSED: W34_LOCK_EXISTS: {out} exists (a lock revision is written once; a change is a new revision)",
              file=sys.stderr)
        return 1
    problems = []
    if int(lock["revision"]) > 0:
        sup = lock.get("supersedes") or {}
        prior = Path(args.dir) / str(sup.get("file"))
        if not prior.is_file() or P.sha256_file(prior) != sup.get("sha256") or not lock.get("revision_reason"):
            problems.append(f"W34_REVISION: revision {lock['revision']} must supersede an existing lock by file and "
                            f"sha256 and state its reason")
    form = is_form_head(lock.get("head"))
    if form != bool(getattr(args, "form_head", False)):
        print(f"REFUSED: W34_HEAD_FORM: the draft's head {'declares' if form else 'declares no'} form: write it "
              f"{'with --form-head (a form has no git head)' if form else 'with --name-head (the git form)'}",
              file=sys.stderr)
        print("nothing written", file=sys.stderr)
        return 1
    if form:
        head, head_problems = form_head_block(lock["head"])
        lock["head_function"] = CH.head_function_pin(lock)
    else:
        head, head_problems = head_block(Path(args.head_record), plan_lock=Path(args.plan_lock),
                                         queue=Path(args.queue), ro_worktree=Path(args.ro_worktree),
                                         code_patch=Path(args.code_patch), revision=args.route_b_revision, git=git)
    problems += head_problems
    lock["head"] = {**{k: v for k, v in lock["head"].items() if not str(v).startswith("<")}, **head}
    frozen = frozen_digests(Path(args.frozen_sums)) if Path(args.frozen_sums).is_file() else None
    lock["files_sha256"], file_problems = fill_files(lock["files_sha256"], frozen)
    problems += file_problems + source_problems(lock) + split_problems(lock) + bib_problems(lock, args.paper_root)
    problems += P.prefix_problems(lock)
    stamp = datetime.datetime.now(TZ).strftime("%Y-%m-%dT%H:%M:%S%z")
    lock["locked_at"] = stamp
    lock["drafted_from"] = {"file": str(draft_path), "sha256": P.sha256_file(draft_path)}
    lock["status"] = (f"LOCK revision {lock['revision']}, written once by make_lock.py --write --name-head at {stamp} "
                      f"from its draft ({draft_path.name}, sha256 {lock['drafted_from']['sha256'][:12]}); never edited")
    problems += placeholder_problems(lock)
    if problems:
        for p in problems:
            print(f"REFUSED: {p}", file=sys.stderr)
        print("nothing written", file=sys.stderr)
        return 1
    names = CH.head_names_of(lock)
    out.write_text(json.dumps(lock, indent=1, sort_keys=True) + "\n")
    digest = P.sha256_file(out)
    freeze = out.with_name(out.name[: -len(".json")] + ".freeze.json")
    freeze.write_text(json.dumps({"file": out.name, "sha256": digest, "written_at": stamp, "head": names["head"],
                                  "base_commit": names["base_commit"]}, indent=1, sort_keys=True) + "\n")
    print(f"locked {out} sha256 {digest} head {names['head']} base {names['base_commit']}")
    return 0


def verify_problems(lock_path: Path) -> List[str]:
    lock_path = Path(lock_path)
    freeze = lock_path.with_name(lock_path.name[: -len(".json")] + ".freeze.json")
    if not freeze.is_file():
        return [f"W34_LOCK_SHA256: {freeze} is absent"]
    pinned = json.loads(freeze.read_text())
    if pinned.get("sha256") != P.sha256_file(lock_path) or pinned.get("file") != lock_path.name:
        return [f"W34_LOCK_SHA256: {lock_path} is not the file its freeze record names ({str(pinned.get('sha256'))[:12]})"]
    lock = json.loads(lock_path.read_text())
    _, problems = fill_files(lock["files_sha256"], None)
    problems += source_problems(lock) + split_problems(lock) + P.prefix_problems(lock)
    head = lock["head"]
    if is_form_head(head):
        block, head_problems = form_head_block({k: v for k, v in head.items() if k != "src_digest"})
        problems += head_problems
        try:
            CH.head_function_pin(lock)
        except SystemExit as refusal:
            problems.append(str(refusal).replace("REFUSED: ", "", 1))
        if not head_problems and block.get("src_digest") != head.get("src_digest"):
            problems.append(f"W34_SRC_DIGEST: {head['worktree']} src_digest {block.get('src_digest')}, not the lock's "
                            f"{head.get('src_digest')}")
        return problems
    rec = json.loads(Path(head["head_record"]["file"]).read_text()) if Path(head["head_record"]["file"]).is_file() else {}
    export = {"tree": head["tree"], "src_manifest_sha256": head["src_manifest_sha256"],
              "record_sha256": head["export_record_sha256"]}
    problems += export_problems(Path(head["worktree"]), head["commit"], export)
    if rec.get("head") != head["commit"]:
        problems.append(f"W34_HEAD_RECORD: {head['head_record']['file']} names {str(rec.get('head'))[:12]}, not the "
                        f"lock's {head['commit'][:12]}")
    now = K.src_digest(Path(head["worktree"])) if Path(head["worktree"], "src").is_dir() else None
    if now != head["src_digest"]:
        problems.append(f"W34_SRC_DIGEST: {head['worktree']} src_digest {now}, not the lock's {head['src_digest']}")
    return problems


def main(argv: Optional[List[str]] = None, git: Optional[Git] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--name-head", action="store_true")
    ap.add_argument("--form-head", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--draft", default=str(RA / "12_tcad_readiness/tiers/family_locks/w34/lock.r0.draft.json"))
    ap.add_argument("--dir", default=str(HERE))
    ap.add_argument("--lock", default=str(HERE / "lock.r0.json"))
    ap.add_argument("--head-record", default=str(RB / "head_of_record.json"))
    ap.add_argument("--plan-lock", default=str(RB / "plan_lock.json"))
    ap.add_argument("--queue", default=str(RA / "11_wave2" / "integration_queue.md"))
    ap.add_argument("--ro-worktree", default=None)
    ap.add_argument("--frozen-sums", default=str(UNIT / "frozen" / "SHA256SUMS"))
    ap.add_argument("--code-patch", default=str(E2E / "code.patch"))
    ap.add_argument("--route-b-revision", type=int, default=11)
    ap.add_argument("--paper-root", default=str(RA.parent))
    args = ap.parse_args(argv)
    if args.verify == args.write:
        ap.error("give exactly one of --write and --verify")
    if args.verify:
        problems = verify_problems(Path(args.lock))
        for p in problems:
            print(f"REFUSED: {p}", file=sys.stderr)
        if problems:
            return 1
        print(f"lock verified: {args.lock} sha256 {P.sha256_file(Path(args.lock))}")
        return 0
    if args.form_head and args.name_head:
        ap.error("give one of --form-head and --name-head")
    if not args.form_head and (not args.name_head or not args.ro_worktree):
        ap.error("--write needs --name-head and --ro-worktree (the head is named at the write, never typed), or "
                 "--form-head (a head that declares its form)")
    return write(args, git=git)


if __name__ == "__main__":
    raise SystemExit(main())
