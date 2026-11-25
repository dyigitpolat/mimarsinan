""

from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
B = HERE.parents[3] / "06_build"
LISTING = HERE / "COPIES.sha256"
HARNESSES = ("screen_v42p", "headline_v46")
SNAPSHOT = Path("screen_v42") / "src_snapshot"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _code_files(harness: str) -> tuple[str, ...]:
    ""
    import ast

    tree = ast.parse((B / harness / "screen_common.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "CODE_FILES" for t in node.targets):
            return tuple(ast.literal_eval(node.value))
    raise SystemExit(f"REFUSED: {harness}/screen_common.py declares no CODE_FILES")


def planned() -> list[Path]:
    ""
    rels = [Path(h) / name for h in HARNESSES for name in _code_files(h)]
    manifest = B / SNAPSHOT / "MANIFEST.sha256"
    rels.append(SNAPSHOT / "MANIFEST.sha256")
    rels += [SNAPSHOT / line.split("  ", 1)[1] for line in manifest.read_text().splitlines() if line.strip()]
    return rels


def make() -> int:
    rels = planned()
    lines = []
    for rel in rels:
        dst = HERE / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(B / rel, dst)
        lines.append(f"{_sha256(dst)}  {rel}")
    LISTING.write_text("\n".join(lines) + "\n")
    print(f"{len(rels)} files copied")
    return verify_count()


def verify() -> dict:
    ""
    listed = {}
    for line in LISTING.read_text().splitlines():
        digest, rel = line.split("  ", 1)
        listed[rel] = digest
    if sorted(listed) != sorted(str(r) for r in planned()):
        raise SystemExit("REFUSED: the copy listing differs from the locked code set")
    bad = [rel for rel, d in listed.items() if not (_sha256(HERE / rel) == _sha256(B / rel) == d)]
    if bad:
        raise SystemExit(f"REFUSED: {len(bad)} copies differ from their locked originals: {bad[:5]}")
    return listed


def verify_count() -> int:
    print(f"{len(verify())} copies equal their locked originals")
    return 0


if __name__ == "__main__":
    sys.exit(make() if sys.argv[1:] == ["make"] else verify_count())
