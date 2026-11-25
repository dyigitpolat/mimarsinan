""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Iterable, List, Optional

HERE = Path(__file__).resolve().parent
SCHEMA = "cl_cov.census_record/1"

CONVERTS = "converts"
CONVERTS_CPU = "converts with CPU share"
STAGE1_AGREES = "stage 1 agrees"
REJECTED = "rejected"
NOT_CHECKED = "not checked within the cap"
VALUE_MISMATCH = "value mismatch"
LOUD = "loud failure"
UNDECIDABLE = "not decidable at the line"
VERDICT_CLASSES = (CONVERTS, CONVERTS_CPU, REJECTED, UNDECIDABLE, NOT_CHECKED, VALUE_MISMATCH, LOUD)
SUPPLEMENT_CLASSES = (STAGE1_AGREES, REJECTED, UNDECIDABLE, NOT_CHECKED, VALUE_MISMATCH, LOUD)

BLOCKING = (VALUE_MISMATCH, LOUD)


STAGE_NATIVE, STAGE_CONVERTED, STAGE_VALUE_PROGRAM = "native reference", "converted model", "identity value program"

STAGE_MARKERS = (("identity_value_program_stage", STAGE_VALUE_PROGRAM), ("prepare_witness", STAGE_CONVERTED),
                 ("judge", STAGE_CONVERTED))


class RecordRefused(RuntimeError):
    ""


def source_digest(root: Path) -> str:
    ""
    base = Path(root)
    digest = hashlib.sha256()
    for rel in sorted(p.relative_to(base).as_posix() for p in base.rglob("*.py") if "__pycache__" not in p.parts):
        digest.update(rel.encode() + b"\0" + (base / rel).read_bytes() + b"\0")
    return digest.hexdigest()[:16]


def harness_digest() -> str:
    ""
    digest = hashlib.sha256()
    for path in sorted(list(HERE.glob("cov_*.py")) + [HERE / "run_census.sh"]):
        digest.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()[:16]


def code_state(worktree: Path, git: dict) -> dict:
    src = Path(worktree) / "src" / "mimarsinan"
    return {"worktree": str(Path(worktree).resolve()), **git, "mimarsinan_src_sha16": source_digest(src),
            "harness_sha16": harness_digest()}


def verify_import(state: dict) -> dict:
    ""
    import mimarsinan

    package = Path(mimarsinan.__file__).resolve().parent
    planned = Path(state["worktree"]) / "src" / "mimarsinan"
    found = {"imported_from": str(package), "imported_sha16": source_digest(package),
             "harness_sha16": harness_digest()}
    if package != planned.resolve():
        raise RecordRefused(f"mimarsinan imported from {package}, the plan names {planned}")
    if found["imported_sha16"] != state["mimarsinan_src_sha16"]:
        raise RecordRefused(f"the worktree changed since the plan: source digest {found['imported_sha16']}, "
                            f"planned {state['mimarsinan_src_sha16']}")
    if found["harness_sha16"] != state["harness_sha16"]:
        raise RecordRefused(f"the harness changed since the plan: {found['harness_sha16']}, planned "
                            f"{state['harness_sha16']}")
    return found


def _simple_attributes(exc: BaseException) -> dict:
    out = {}
    for key, value in vars(exc).items():
        if key.startswith("_") or key in ("result", "args"):
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[key] = value if not isinstance(value, str) else value[:300]
        elif isinstance(value, (tuple, list)) and all(isinstance(v, (str, int, float)) for v in value):
            out[key] = [v if not isinstance(v, str) else v[:200] for v in value][:12]
    return out


def _first_line(exc: BaseException, limit: int = 700) -> str:
    text = str(exc).strip()
    return text.splitlines()[0][:limit] if text else ""


def is_a(exc: BaseException, class_name: str) -> bool:
    ""
    return any(c.__name__ == class_name for c in type(exc).__mro__)


def _is_refusal(exc: BaseException) -> bool:
    return is_a(exc, "ToolchainRefusal")


def chain(exc: BaseException) -> List[dict]:
    ""
    out, seen, current = [], set(), exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        out.append({"type": type(current).__name__, "module": type(current).__module__,
                    "toolchain_refusal": _is_refusal(current), "message": _first_line(current),
                    "attributes": _simple_attributes(current)})
        inner = getattr(getattr(current, "result", None), "error", None)
        current = inner if isinstance(inner, BaseException) else (current.__cause__ or current.__context__)
    return out


def refusal_in_chain(exc: Optional[BaseException]) -> Optional[BaseException]:
    seen, current = set(), exc
    while current is not None and id(current) not in seen:
        if _is_refusal(current):
            return current
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return None


def root_refusal(refusal: BaseException) -> BaseException:
    ""
    seen, current = set(), refusal
    while id(current) not in seen:
        seen.add(id(current))
        error = getattr(getattr(current, "result", None), "error", None)
        inner = refusal_in_chain(error) if isinstance(error, BaseException) else None
        if inner is None:
            return current
        current = inner
    return current


def raised_at(exc: BaseException) -> dict:
    ""
    frames = traceback.extract_tb(exc.__traceback__) if exc.__traceback__ is not None else []
    names = [f.name for f in frames]
    stage = next((s for marker, s in STAGE_MARKERS if marker in names), None)
    inner = [f"{Path(f.filename).name}:{f.name}:{f.lineno}" for f in frames if "mimarsinan" in f.filename][-4:]
    return {"stage_from_traceback": stage, "innermost_toolchain_frames": inner}


def traceback_tail(exc: BaseException, lines: int = 18) -> List[str]:
    text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return text.rstrip().splitlines()[-lines:]


def read_records(path: Path) -> List[dict]:
    if not Path(path).exists():
        return []
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_record(record: dict, *, table: dict, table_sha16: str, plan: dict, records_path: Path) -> None:
    ""
    member, check = record["member"], record["check"]
    entry = table["members"].get(member, {}).get("checks", {}).get(check)
    if entry is None:
        raise RecordRefused(f"{member} {check}: no side table entry; the side table fixes every checked input")
    if list(record.get("input_shape") or []) != list(entry["input_shape"]) or record.get("side") != entry["side"]:
        raise RecordRefused(f"{member} {check}: input {record.get('input_shape')} (side {record.get('side')}) "
                            f"differs from the side table's {entry['input_shape']} (side {entry['side']})")
    if record.get("side_table_sha16") != table_sha16 or table_sha16 != plan["side_table_sha16"]:
        raise RecordRefused(f"{member} {check}: side table {record.get('side_table_sha16')}, the plan's "
                            f"{plan['side_table_sha16']}")
    state = record.get("code_state") or {}
    for key in ("head", "index_diff_sha16", "unstaged_diff_sha16", "untracked_sha16", "mimarsinan_src_sha16",
                "harness_sha16"):
        if state.get(key) != plan["code_state"].get(key):
            raise RecordRefused(f"{member} {check}: code state {key} {state.get(key)}, the plan's "
                                f"{plan['code_state'].get(key)}")
    allowed = VERDICT_CLASSES if check == "verdict" else SUPPLEMENT_CLASSES
    if record.get("outcome", {}).get("class") not in allowed:
        raise RecordRefused(f"{member} {check}: outcome class {record.get('outcome', {}).get('class')!r}")
    record = {"schema": SCHEMA, **record}
    line = json.dumps(record, sort_keys=True, default=str)
    lock = Path(records_path).with_suffix(".lock")
    with open(lock, "a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        try:
            if any((r["member"], r["check"]) == (member, check) for r in read_records(records_path)):
                raise RecordRefused(f"{member} {check}: a record exists already")
            with open(records_path, "a") as out:
                out.write(line + "\n")
                out.flush()
                os.fsync(out.fileno())
        finally:
            fcntl.flock(guard, fcntl.LOCK_UN)


def attempts_of(outdir: Path, job_id: str) -> List[dict]:
    path = Path(outdir) / "attempts.jsonl"
    if not path.exists():
        return []
    return [a for a in (json.loads(line) for line in open(path) if line.strip()) if a.get("job_id") == job_id]


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("code-state",))
    parser.add_argument("--worktree", required=True)
    parser.add_argument("--git-json", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    state = code_state(Path(args.worktree), load_json(Path(args.git_json)))
    state.update({k: v for k, v in verify_import(state).items() if k == "imported_from"})
    out = Path(args.out)
    if out.exists() and load_json(out) != state:
        print(f"REFUSED: {out} holds another code state; one output directory holds one code state", file=sys.stderr)
        return 3
    out.write_text(json.dumps(state, indent=1, sort_keys=True) + "\n")
    print(json.dumps(state, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
