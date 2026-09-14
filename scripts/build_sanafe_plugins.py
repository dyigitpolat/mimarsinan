#!/usr/bin/env python3
"""Build the six mimarsinan-owned SANA-FE plugins into build/mimarsinan_sanafe_plugins/.

The plugins are mimarsinan source (``src/mimarsinan/chip_simulation/sanafe/
plugins/*.cpp``) compiled against SANA-FE's header tree. The PyPI wheel ships
no headers, so CMake fetches the ``v2.1.1`` tag archive -- pinned by SHA256 --
into the build directory. Nothing GPL-licensed lands in the source tree, which
is the posture the submodule had.

``MIMARSINAN_SANAFE_SRC`` (or ``--sanafe-src``) points the build at an existing
SANA-FE checkout instead, for from-source work.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PLUGIN_SRC = REPO_ROOT / "src" / "mimarsinan" / "chip_simulation" / "sanafe" / "plugins"
PLUGIN_BUILD = REPO_ROOT / "build" / "mimarsinan_sanafe_plugins"
SANAFE_SRC_VAR = "MIMARSINAN_SANAFE_SRC"

PLUGIN_NAMES = (
    "mimarsinan_dendrite",
    "mimarsinan_soma",
    "mimarsinan_ttfs_continuous_soma",
    "mimarsinan_ttfs_quantized_soma",
    "mimarsinan_ttfs_cycle_soma",
    "mimarsinan_ttfs_cascade_soma",
)


def expected_libraries() -> list[Path]:
    """The full plugin set the SANA-FE step resolves at run time."""
    return [PLUGIN_BUILD / f"lib{name}.so" for name in PLUGIN_NAMES]


def _run(cmd: list[str]) -> None:
    print("==> " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def build(sanafe_src: str | None = None, clean: bool = True) -> list[Path]:
    if shutil.which("cmake") is None:
        raise RuntimeError(
            "cmake is required to build the SANA-FE plugins "
            "(build deps: CMake >= 3.16, a C++17 compiler)."
        )
    if clean and PLUGIN_BUILD.exists():
        shutil.rmtree(PLUGIN_BUILD)
    PLUGIN_BUILD.mkdir(parents=True, exist_ok=True)

    configure = ["cmake", "-S", str(PLUGIN_SRC), "-B", str(PLUGIN_BUILD)]
    override = sanafe_src or os.environ.get(SANAFE_SRC_VAR, "").strip()
    if override:
        configure.append(f"-DSANAFE_SRC={Path(override).expanduser().resolve()}")
    _run(configure)
    _run(["cmake", "--build", str(PLUGIN_BUILD), "--parallel"])

    built = expected_libraries()
    missing = [p.name for p in built if not p.is_file()]
    if missing:
        raise RuntimeError(f"plugin build produced no {', '.join(missing)}")
    return built


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--sanafe-src",
        default=None,
        help=f"path to a SANA-FE src/ tree (also {SANAFE_SRC_VAR}); "
        "default is the v2.1.1 tag archive CMake fetches by SHA256",
    )
    parser.add_argument(
        "--no-clean", action="store_true", help="reuse the existing build directory"
    )
    args = parser.parse_args(argv)

    built = build(sanafe_src=args.sanafe_src, clean=not args.no_clean)
    print(f"==> {len(built)} plugins in {PLUGIN_BUILD}:")
    for path in built:
        print(f"    {path.name}")
    print(
        '==> Set "enable_sanafe_simulation": true in deployment_parameters '
        "(or in the wizard) to run the step."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
