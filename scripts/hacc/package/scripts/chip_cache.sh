#!/usr/bin/env bash
# The CHIP CACHE: a place-and-routed xclbin is worth hours, so it is kept.
#
#   scripts/chip_cache.sh key <target>            print the key, and its inputs
#   scripts/chip_cache.sh path <target>           print the entry directory
#   scripts/chip_cache.sh lookup <target>         exit 0 if a complete entry is there
#   scripts/chip_cache.sh restore <target>        copy an entry's artifacts INTO the build
#   scripts/chip_cache.sh publish <target>        copy this build's artifacts INTO the cache
#   scripts/chip_cache.sh adopt <install> [--alias NAME]
#                                                 take an EXISTING install's xclbin in
#   scripts/chip_cache.sh list                    what is cached, newest first
#
# THE KEY is a sha256 over everything that could change the bitstream, and it is
# computed by ONE function (cache_key_inputs) that the build path and the
# adoption path both call — a second recipe would let an adopted install land
# under a key a build would never look up:
#
#   rtl_sha256   the package MANIFEST's digest over every RTL source v++ compiles
#                FOR THIS CHIP -- the stock fabric reads the manifest's own
#                `rtl_sha256`, any other reads its entry in `chip_rtl_sha256`
#   chip_config  the fabric profile (scripts/hacc/chips.sh): which core the
#                kernel instantiates. Two fabrics compiled from disjoint source
#                sets must never share a key, and rtl_sha256 alone would not
#                separate a chip whose RTL happened to be unchanged
#   card         the card profile (scripts/hacc/cards.sh)
#   platform     the shell v++ links against
#   part         the Vivado part
#   NC           ODIN cores in the kernel (the packaging flow builds 1)
#   FIFO_WORDS   from hw/fpga/kernel/odin_fpga_kernel_top.v, as built
#   CAP_WORDS    likewise
#   clock        defaultFreqHz from the card's v++ config
#   vitis        the Vitis release that ran the link
#   target       hw_emu or hw — the same sources at two targets are two bitstreams
#   build_script the sha256 of build_xclbn.sh, because it IS the recipe: without
#                it a cache hit would resurrect a bitstream that run_all.sh's own
#                artifact-resume rule (a changed build script makes the xclbin
#                stale) had just declared out of date, and the two protections
#                would quietly contradict each other
#
# THE ROOT is ${ODIN_CHIP_CACHE:-${DATA_ROOT}/${USER}/odin_chip_cache}, because
# /data is the only path the head node and the board VMs share.
#
# PUBLISHING IS ATOMIC AND NEVER CLOBBERS. A key is claimed with mkdir (the one
# portable atomic test-and-set on a shared filesystem), the artifacts are staged
# beside the entry and renamed into place, and an entry that already exists is
# LEFT ALONE — two jobs that built the same key built the same bitstream, and
# the one already on disk may already have been read.

set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="${ODIN_DATA_ROOT:-/data}"
CACHE_ROOT="${ODIN_CHIP_CACHE:-${DATA_ROOT}/${USER}/odin_chip_cache}"
BUILD_DIR="${ODIN_BUILD_DIR:-${HERE}/build/hacc}"
MANIFEST="${HERE}/MANIFEST.json"
KERNEL_TOP="${HERE}/hw/fpga/kernel/odin_fpga_kernel_top.v"

# shellcheck source=../scripts/hacc/toolchain.sh
source "${HERE}/scripts/hacc/toolchain.sh"
# shellcheck source=../scripts/hacc/cards.sh
source "${HERE}/scripts/hacc/cards.sh"
# shellcheck source=../scripts/hacc/chips.sh
source "${HERE}/scripts/hacc/chips.sh"

say() { printf '%s\n' "$*"; }
die() { printf 'REFUSING: %s\n' "$*" >&2; exit 2; }

# --------------------------------------------------------------------------
# The inputs, read from the package itself — never from a shell's memory
# --------------------------------------------------------------------------

# A verilog `parameter NAME = <expr>` from the shipped kernel top. A depth may
# be written as an expression in NC, so NC is substituted and the product
# evaluated: the key must name the geometry the fabric was BUILT at, not the
# expression for it.
verilog_parameter() {
    local name="$1" raw
    [ -f "${KERNEL_TOP}" ] || die "${KERNEL_TOP} is missing; re-unzip the package"
    raw="$(sed -n "s/^[[:space:]]*parameter[[:space:]]\+${name}[[:space:]]*=[[:space:]]*\([^,]*\),.*/\1/p" \
        "${KERNEL_TOP}" | head -1 | tr -d '[:space:]')"
    case "${raw}" in
        *NC*)
            local nc
            nc="$(sed -n "s/^[[:space:]]*parameter[[:space:]]\+NC[[:space:]]*=[[:space:]]*\([^,]*\),.*/\1/p" \
                "${KERNEL_TOP}" | head -1 | tr -d '[:space:]')"
            printf '%s' "$(( ${raw//NC/${nc}} ))"
            ;;
        *) printf '%s' "${raw}" ;;
    esac
}

# The RTL identity of ONE fabric. The default chip reads the manifest's own
# top-level `rtl_sha256` -- the field every published cache entry was keyed on --
# and any other reads its entry in the `chip_rtl_sha256` table.
manifest_rtl_sha256() {
    local chip="${1:-$(odin_chip)}" digest
    [ -f "${MANIFEST}" ] || die "${MANIFEST} is missing; re-unzip the package"
    if [ "${chip}" = "${ODIN_CHIP_DEFAULT}" ]; then
        sed -n 's/.*"rtl_sha256"[[:space:]]*:[[:space:]]*"\([0-9a-f]*\)".*/\1/p' \
            "${MANIFEST}" | head -1
        return 0
    fi
    digest="$(sed -n '/"chip_rtl_sha256"/,/}/p' "${MANIFEST}" \
        | sed -n 's/.*"'"${chip}"'"[[:space:]]*:[[:space:]]*"\([0-9a-f]*\)".*/\1/p' \
        | head -1)"
    [ -n "${digest}" ] || die \
        "the package MANIFEST carries no chip_rtl_sha256 for '${chip}'; it was \
built before that fabric existed, and a key derived from a guess would resurrect \
someone else's bitstream"
    printf '%s' "${digest}"
}

clock_hz() {
    local cfg="$1"
    if [ -f "${cfg}" ]; then
        sed -n 's/^[[:space:]]*defaultFreqHz[[:space:]]*=[[:space:]]*\([0-9]*\).*/\1/p' \
            "${cfg}" | head -1
    fi
}

build_script_sha() {
    sha256sum "${HERE}/scripts/hacc/build_xclbn.sh" 2>/dev/null | cut -d' ' -f1
}

vitis_release() {
    local vitis
    if vitis="$(odin_vitis_settings 2>/dev/null)"; then
        vitis="${vitis#*|}"
        printf '%s' "${vitis%%|*}"
    else
        printf 'none'
    fi
}

# THE ONE KEY RECIPE. Prints one `name=value` per line, in a fixed order; the
# key is the sha256 of exactly this text. Both the build path and `adopt` call
# it, which is what makes an adopted install findable by a later build.
cache_key_inputs() {
    local target="$1" rtl="$2" card="$3" platform="$4" part="$5" cfg="$6"
    local nc="${7:-$(verilog_parameter NC)}"
    local script="${9:-$(build_script_sha)}"
    local chip="${10:-$(odin_chip)}"
    printf 'target=%s\n' "${target}"
    printf 'rtl_sha256=%s\n' "${rtl}"
    printf 'chip_config=%s\n' "${chip}"
    printf 'card=%s\n' "${card}"
    printf 'platform=%s\n' "${platform}"
    printf 'part=%s\n' "${part}"
    printf 'nc=%s\n' "${nc}"
    printf 'fifo_words=%s\n' "$(verilog_parameter FIFO_WORDS)"
    printf 'cap_words=%s\n' "$(verilog_parameter CAP_WORDS)"
    printf 'clock_hz=%s\n' "$(clock_hz "${cfg}")"
    printf 'vitis=%s\n' "${8:-$(vitis_release)}"
    printf 'build_script_sha256=%s\n' "${script}"
}

digest_of() { sha256sum | cut -d' ' -f1; }

# The key for a target built by THIS install, from THIS package's card profile.
cache_key_for_target() {
    local target="$1"
    odin_card_resolve > /dev/null 2>&1 || true
    cache_key_inputs "${target}" "$(manifest_rtl_sha256)" "$(odin_card)" \
        "${PLATFORM}" "${PART}" "${HERE}/${CFG}" | digest_of
}

cache_entry() { printf '%s/%s\n' "${CACHE_ROOT}" "$1"; }

# The build directory the chip's artifact lands in. The DEFAULT chip's suffix is
# empty, so every path this printed before the chip axis existed is unchanged.
xclbin_of() {
    printf '%s/%s_nc1%s/odin_fpga_%s.xclbin\n' \
        "${BUILD_DIR}" "$1" "$(odin_chip_field build_suffix "${2:-$(odin_chip)}")" "$1"
}

# An entry is COMPLETE when it holds the bitstream and the sidecar that says
# what produced it. A half-copied entry must never satisfy a lookup.
entry_complete() {
    local entry="$1"
    [ -f "${entry}/odin_fpga.xclbin" ] && [ -f "${entry}/built_with.txt" ] \
        && [ -f "${entry}/key_inputs.txt" ]
}

# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------

cmd_key() {
    local target="${1:?key needs a target (hw_emu|hw)}"
    odin_card_resolve > /dev/null 2>&1 || true
    cache_key_inputs "${target}" "$(manifest_rtl_sha256)" "$(odin_card)" \
        "${PLATFORM}" "${PART}" "${HERE}/${CFG}"
    printf 'key=%s\n' "$(cache_key_for_target "${target}")"
}

cmd_path() { cache_entry "$(cache_key_for_target "${1:?path needs a target}")"; }

cmd_lookup() {
    local target="${1:?lookup needs a target}" entry
    entry="$(cache_entry "$(cache_key_for_target "${target}")")"
    entry_complete "${entry}"
}

cmd_restore() {
    local target="${1:?restore needs a target}" key entry dest
    key="$(cache_key_for_target "${target}")"
    entry="$(cache_entry "${key}")"
    entry_complete "${entry}" || die "no complete cache entry at ${entry}"
    dest="$(dirname "$(xclbin_of "${target}")")"
    mkdir -p "${dest}"
    cp "${entry}/odin_fpga.xclbin" "$(xclbin_of "${target}")"
    cp "${entry}/built_with.txt" "$(xclbin_of "${target}").built_with"
    for extra in reports logs maps; do
        if [ -d "${entry}/${extra}" ]; then
            rm -rf "${dest:?}/${extra}"
            cp -r "${entry}/${extra}" "${dest}/${extra}"
        fi
    done
    say "[cache] restored ${target} from ${key}"
    say "[cache] entry    ${entry}"
    return 0
}

# Claim the key with mkdir — the one portable atomic test-and-set on a shared
# filesystem — then stage beside it and rename in.
cmd_publish() {
    local target="${1:?publish needs a target}" key entry stage xclbin sidecar
    key="$(cache_key_for_target "${target}")"
    entry="$(cache_entry "${key}")"
    xclbin="$(xclbin_of "${target}")"
    sidecar="${xclbin}.built_with"
    [ -f "${xclbin}" ] || die "nothing to publish: ${xclbin} does not exist"
    mkdir -p "${CACHE_ROOT}"
    if entry_complete "${entry}"; then
        say "[cache] ${key} is already published; leaving it alone."
        return 0
    fi
    if ! mkdir "${entry}.lock" 2>/dev/null; then
        say "[cache] another job holds the lock on ${key}; not publishing."
        return 0
    fi
    # shellcheck disable=SC2064  # the entry path must be expanded NOW
    trap "rm -rf '${entry}.lock'" EXIT
    stage="${entry}.staging.$$"
    rm -rf "${stage}"
    mkdir -p "${stage}"
    cp "${xclbin}" "${stage}/odin_fpga.xclbin"
    if [ -f "${sidecar}" ]; then
        cp "${sidecar}" "${stage}/built_with.txt"
    else
        printf 'target=%s\nnote=published without a .built_with sidecar\n' \
            "${target}" > "${stage}/built_with.txt"
    fi
    odin_card_resolve > /dev/null 2>&1 || true
    cache_key_inputs "${target}" "$(manifest_rtl_sha256)" "$(odin_card)" \
        "${PLATFORM}" "${PART}" "${HERE}/${CFG}" > "${stage}/key_inputs.txt"
    printf 'published_utc=%s\nhost=%s\nfrom=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(hostname)" "${xclbin}" \
        > "${stage}/published.txt"
    for extra in reports logs; do
        if [ -d "$(dirname "${xclbin}")/${extra}" ]; then
            cp -r "$(dirname "${xclbin}")/${extra}" "${stage}/${extra}"
        fi
    done
    if [ -d "${entry}" ]; then
        rm -rf "${stage}"
        say "[cache] ${key} appeared while staging; leaving the published one."
        return 0
    fi
    mv "${stage}" "${entry}"
    say "[cache] published ${target} as ${key}"
    say "[cache] entry     ${entry}"
    return 0
}

# ADOPTION: an install that already paid for a bitstream hands it to the cache.
# The key is derived from THAT install's own sidecars and manifest, through the
# same cache_key_inputs, so the entry is the one a rebuild would look up.
cmd_adopt() {
    local install="" alias_name=""
    while [ $# -gt 0 ]; do
        case "$1" in
            --alias) alias_name="${2:?--alias needs a name}"; shift ;;
            -*) die "unknown adopt option '$1'" ;;
            *) install="$1" ;;
        esac
        shift
    done
    [ -n "${install}" ] || die "adopt needs an install path"
    install="$(cd "${install}" && pwd)" || die "no such install"
    local target xclbin sidecar key entry stage
    local adopt_chip adopt_suffix
    # The install's OWN chip, when its sidecar records one. Sidecars written
    # before the chip axis existed name no fabric and are the default one.
    for target in hw hw_emu; do
        adopt_chip="$(sed -n 's/^chip_config=//p' \
            "${install}/build/hacc/${target}_nc1/odin_fpga_${target}.xclbin.built_with" \
            2>/dev/null | head -1)"
        [ -n "${adopt_chip}" ] || adopt_chip="$(odin_chip)"
        adopt_suffix="$(odin_chip_field build_suffix "${adopt_chip}")" || die \
            "the install names chip '${adopt_chip}', which this package does not carry"
        xclbin="${install}/build/hacc/${target}_nc1${adopt_suffix}/odin_fpga_${target}.xclbin"
        [ -f "${xclbin}" ] || continue
        sidecar="${xclbin}.built_with"
        [ -f "${sidecar}" ] || die \
            "${xclbin} has no .built_with sidecar, so nothing says which card, \
platform and build script produced it; a cache entry keyed on a guess is worse \
than no entry"
        local rtl card platform part cfg
        rtl="$(MANIFEST="${install}/MANIFEST.json" manifest_rtl_sha256 "${adopt_chip}")"
        card="$(sed -n 's/^card=//p' "${sidecar}" | head -1)"
        platform="$(sed -n 's/^platform=//p' "${sidecar}" | head -1)"
        cfg="${install}/$(sed -n 's/^vxx_config=//p' "${sidecar}" | head -1)"
        part="$(ODIN_CARD="${card}" odin_card_field part "${card}")"
        # The install's OWN build-script digest, read from its sidecar: an
        # adopted entry must land under the key that install would look up.
        local script
        script="$(sed -n 's/^build_script_sha256=//p' "${sidecar}" | head -1)"
        key="$(cache_key_inputs "${target}" "${rtl}" "${card}" "${platform}" \
            "${part}" "${cfg}" "" "" "${script}" "${adopt_chip}" | digest_of)"
        entry="$(cache_entry "${key}")"
        mkdir -p "${CACHE_ROOT}"
        if entry_complete "${entry}"; then
            say "[cache] ${target}: ${key} already adopted; leaving it alone."
        elif ! mkdir "${entry}.lock" 2>/dev/null; then
            say "[cache] ${target}: another job holds ${key}; not adopting."
        else
            stage="${entry}.staging.$$"
            rm -rf "${stage}"; mkdir -p "${stage}"
            if [ "${ODIN_CACHE_ADOPT_MOVE:-0}" = "1" ]; then
                mv "${xclbin}" "${stage}/odin_fpga.xclbin"
            else
                cp "${xclbin}" "${stage}/odin_fpga.xclbin"
            fi
            cp "${sidecar}" "${stage}/built_with.txt"
            cache_key_inputs "${target}" "${rtl}" "${card}" "${platform}" \
                "${part}" "${cfg}" "" "" "${script}" "${adopt_chip}" \
                > "${stage}/key_inputs.txt"
            printf 'adopted_utc=%s\nhost=%s\nfrom=%s\n' \
                "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(hostname)" "${xclbin}" \
                > "${stage}/published.txt"
            for extra in reports logs; do
                if [ -d "$(dirname "${xclbin}")/${extra}" ]; then
                    cp -r "$(dirname "${xclbin}")/${extra}" "${stage}/${extra}"
                fi
            done
            mv "${stage}" "${entry}"
            rm -rf "${entry}.lock"
            say "[cache] adopted ${target} from ${install} as ${key}"
        fi
        if [ -n "${alias_name}" ]; then
            ln -sfn "${entry}" "${CACHE_ROOT}/${alias_name}_${target}"
            say "[cache] alias    ${CACHE_ROOT}/${alias_name}_${target} -> ${key}"
        fi
    done
    return 0
}

cmd_list() {
    [ -d "${CACHE_ROOT}" ] || { say "[cache] nothing cached under ${CACHE_ROOT}"; return 0; }
    local entry
    say "key                                                               target   card    published"
    for entry in "${CACHE_ROOT}"/*/; do
        [ -f "${entry}/key_inputs.txt" ] || continue
        printf '%-64s  %-7s  %-6s  %s\n' \
            "$(basename "${entry}")" \
            "$(sed -n 's/^target=//p' "${entry}/key_inputs.txt" | head -1)" \
            "$(sed -n 's/^card=//p' "${entry}/key_inputs.txt" | head -1)" \
            "$(sed -n 's/^published_utc=//p;s/^adopted_utc=//p' \
                "${entry}/published.txt" 2>/dev/null | head -1)"
    done
}

case "${1:-}" in
    key) shift; cmd_key "$@" ;;
    path) shift; cmd_path "$@" ;;
    lookup) shift; cmd_lookup "$@" ;;
    restore) shift; cmd_restore "$@" ;;
    publish) shift; cmd_publish "$@" ;;
    adopt) shift; cmd_adopt "$@" ;;
    list) shift; cmd_list "$@" ;;
    *) sed -n '2,35p' "$0"; exit 2 ;;
esac
