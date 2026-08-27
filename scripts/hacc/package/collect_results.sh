#!/usr/bin/env bash
# Bundle everything this bring-up measured into one tarball to bring home.
#
#   ./collect_results.sh            -> odin_hacc_results_$(hostname).tar.gz
#   ./collect_results.sh /some/dir  -> writes it there instead
#
# What goes in: the env probe, every phase log, every result JSON the driver
# wrote, the phase journal and partition picks (results/ already holds both, so
# the evidence says WHICH partition produced it), the .built_with sidecars that
# name the build script each xclbin came from, the package MANIFEST (so the
# evidence names the bytes it came from), and the BUILD REPORTS —
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
copy_if_present "${HERE}/MANIFEST.json" "MANIFEST.json"
copy_if_present "${HERE}/fixtures/INDEX.json" "fixtures_INDEX.json"
for target in hw_emu hw; do
    copy_if_present "${HERE}/build/hacc/${target}_nc1/odin_fpga_${target}.xclbin.built_with" \
        "built_with/${target}_nc1.txt"
    copy_if_present "${HERE}/build/hacc/${target}_nc1/reports" \
        "build_reports/${target}_nc1"
    copy_if_present "${HERE}/build/hacc/${target}_nc1/logs" \
        "build_logs/${target}_nc1"
done

CHIP_CACHE="${HERE}/scripts/chip_cache.sh"
if [ -x "${CHIP_CACHE}" ]; then
    for target in hw_emu hw; do
        if entry="$("${CHIP_CACHE}" path "${target}" 2>/dev/null)" \
           && [ -d "${entry}" ]; then
            copy_if_present "${entry}/key_inputs.txt" \
                "chip_cache/${target}/key_inputs.txt"
            copy_if_present "${entry}/published.txt" \
                "chip_cache/${target}/published.txt"
            copy_if_present "${entry}/reports" "chip_cache/${target}/reports"
            copy_if_present "${entry}/maps" "chip_cache/${target}/maps"
        else
            printf '  - chip_cache/%s (no entry for this key)\n' "${target}"
        fi
    done
fi

{
    printf 'collected_utc : %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'host          : %s\n' "$(hostname)"
    printf 'user          : %s\n' "${USER}"
    printf 'package       : %s\n' "${HERE}"
} > "${ROOT}/COLLECTED.txt"

mkdir -p "${OUT_DIR}"
tar -czf "${ARCHIVE}" -C "${STAGING}" "${NAME}"
printf 'Wrote %s (%s)\n' "${ARCHIVE}" "$(du -h "${ARCHIVE}" | cut -f1)"
