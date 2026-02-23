import subprocess
import shutil


def _try_compiler(cmd: str) -> bool:
    """Return True if *cmd* is a usable C++ compiler."""
    try:
        subprocess.check_output(f"{cmd} --version", shell=True, stderr=subprocess.STDOUT)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def find_cpp20_compiler():
    """
    Return ``(compiler_command, compiler_family)`` for the best available
    C++20-capable compiler.

    Search order
    ------------
    1. ``clang++-N`` for N in 20 … 17  (fully supports ``<ranges>`` with libc++)
    2. ``g++-N``     for N in 14 … 11  (libstdc++ ships ``<ranges>`` since GCC 10)
    3. ``clang++-N`` for N in 16 … 14  (may work if libstdc++ headers are visible)
    4. ``g++``       (system default, if new enough)

    Returns ``None`` when nothing suitable is found.
    """
    # Prefer modern Clang (>= 17 → full libc++ ranges support)
    for v in range(20, 16, -1):
        cmd = f"clang++-{v}"
        if _try_compiler(cmd):
            return cmd, "clang"

    # GCC >= 11 has good C++20 support with libstdc++
    for v in range(14, 10, -1):
        cmd = f"g++-{v}"
        if _try_compiler(cmd):
            return cmd, "gcc"

    # Older Clang *without* -stdlib=libc++ (uses system libstdc++)
    for v in range(16, 13, -1):
        cmd = f"clang++-{v}"
        if _try_compiler(cmd):
            return cmd, "clang-libstdcxx"

    # Bare g++
    if _try_compiler("g++"):
        return "g++", "gcc"

    return None, None


# ── legacy wrappers (kept for backward compat) ──────────────────────────

def find_latest_clang_version():
    """Legacy: returns the best C++20 compiler command (not necessarily Clang)."""
    cmd, _ = find_cpp20_compiler()
    return cmd


def verify_clang_version(cc_command):
    """Legacy: returns True when *cc_command* is usable."""
    return cc_command is not None
