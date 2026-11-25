""
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER_FIGURES = HERE.parent / "paper" / "figures"
FIGURES = ["fig_capacity", "fig_toolchain", "fig_places", "fig_classes", "fig_invariants", "fig_replay",
           "fig_exact", "fig_invariance", "fig_compare", "fig_card", "fig_area", "fig_space", "fig_block",
           "fig_regime", "fig_fronts", "fig_chipprops"]


def check_data() -> None:
    bad = []
    for line in (HERE / "data" / "MANIFEST.sha256").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        got = hashlib.sha256((HERE / "data" / name).read_bytes()).hexdigest()
        if got != digest:
            bad.append(name)
    if bad:
        raise SystemExit("data files differ from data/MANIFEST.sha256: " + ", ".join(bad))


def build(name: str, copy: bool = True) -> bool:
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", SOURCE_DATE_EPOCH="0")
    run = subprocess.run([sys.executable, str(HERE / "scripts" / f"{name}.py")], cwd=HERE, env=env,
                         capture_output=True, text=True)
    report = [ln for ln in (run.stdout + run.stderr).splitlines() if ln.startswith(("OK", "FAIL", "Traceback"))]
    print("\n".join(report) if report else f"{name}: no report")
    if run.returncode != 0:
        print(run.stderr[-2000:])
        return False
    if not copy:
        return True
    PAPER_FIGURES.mkdir(exist_ok=True)
    shutil.copyfile(HERE / "out" / f"{name}.pdf", PAPER_FIGURES / f"{name}.pdf")
    return True


def main(argv) -> int:
    copy = "--no-copy" not in argv
    names = [a for a in argv if a != "--no-copy"] or FIGURES
    unknown = [n for n in names if n not in FIGURES]
    if unknown:
        raise SystemExit("not a figure of the paper: " + ", ".join(unknown))
    check_data()
    failed = [n for n in names if not build(n, copy)]
    print(f"{len(names) - len(failed)} of {len(names)} figures built"
          + (" and written to ../paper/figures" if copy else " into out/ only")
          + (f"; failed: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
