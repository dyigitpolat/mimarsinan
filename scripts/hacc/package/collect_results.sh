#!/usr/bin/env bash
# Bundle everything this bring-up measured into one tarball to bring home.
#
#   ./collect_results.sh            -> odin_hacc_results_$(hostname).tar.gz
#   ./collect_results.sh /some/dir  -> writes it there instead
#
# What goes in: the env probe, every phase log, every result JSON the driver
# wrote, the stamps that say which phases actually ran, the package MANIFEST
# (so the evidence names the bytes it came from), and the BUILD REPORTS —
# timing and utilization — because those are the real-shell half of the
# implementation-closure evidence and nothing local can produce them.
#
# What stays behind: the xclbin (hundreds of MB) and the v++ temp directories.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${1:-${HERE}}"
NAME="odin_hacc_results_$(hostname)"
ARCHIVE="${OUT_DIR}/${NAME}.tar.gz"

STAGING="$(mktemp -d)"
trap 'rm -rf "${STAGING}"' EXIT
ROOT="${STAGING}/${NAME}"
mkdir -p "${ROOT}"

copy_if_present() {
    local source="$1"
    local relative="$2"
    if [ -e "${source}" ]; then
        mkdir -p "${ROOT}/$(dirname "${relative}")"
        cp -r "${source}" "${ROOT}/${relative}"
        printf '  + %s\n' "${relative}"
    else
        printf '  - %s (absent)\n' "${relative}"
    fi
}

printf 'Collecting from %s\n' "${HERE}"
copy_if_present "${HERE}/results" "results"
copy_if_present "${HERE}/.state" "state"
copy_if_present "${HERE}/MANIFEST.json" "MANIFEST.json"
copy_if_present "${HERE}/fixtures/INDEX.json" "fixtures_INDEX.json"
for target in hw_emu hw; do
    copy_if_present "${HERE}/build/hacc/${target}_nc1/reports" \
        "build_reports/${target}_nc1"
    copy_if_present "${HERE}/build/hacc/${target}_nc1/logs" \
        "build_logs/${target}_nc1"
done

{
    printf 'collected_utc : %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'host          : %s\n' "$(hostname)"
    printf 'user          : %s\n' "${USER}"
    printf 'package       : %s\n' "${HERE}"
} > "${ROOT}/COLLECTED.txt"

mkdir -p "${OUT_DIR}"
tar -czf "${ARCHIVE}" -C "${STAGING}" "${NAME}"
printf 'Wrote %s (%s)\n' "${ARCHIVE}" "$(du -h "${ARCHIVE}" | cut -f1)"
