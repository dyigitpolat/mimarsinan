# shellcheck shell=bash
# Where the cluster's Vitis and XRT actually are — discovered, never assumed.
#
# FIELD LESSON (2026-08-25, hacchead). The v1 scripts hard-coded Vitis 2022.2
# under /tools/xilinx because that is what Xtra-Computing/hacc_demo documents.
# The live cluster carries Vitis **2024.2** under **/tools/Xilinx** (capital X),
# so every hard-coded probe refused before it ever reached a compiler. This file
# is the one place that resolves it: newest version first, env overrides win,
# and both the build script and run_all.sh source it so they can never disagree.
#
# SECOND FIELD LESSON (2026-08-25, hacc-node0). "Newest" is not always right.
# The installed set is 2020.1 2020.2 2021.2 2022.1 2023.2 2024.2, and a
# platform's IP is locked to the release it was built with: the U250
# 3_1_202020_1 shell wants 2020.2, and linking it under 2024.2 is how you find
# out the hard way. So the CARD PROFILE (scripts/hacc/cards.sh) names a
# preferred version, exported here as ODIN_VITIS_PREFER, and it is used when
# that version is installed. Precedence, highest first:
#     VITIS_VERSION (explicit, and it may fail)  >  ODIN_VITIS_PREFER  >  newest

# Every root that may carry a Vitis install, in probe order.
odin_vitis_roots() {
    if [ -n "${XILINX_ROOT:-}" ]; then
        printf '%s\n' "${XILINX_ROOT}"
    else
        printf '%s\n' /tools/Xilinx /tools/xilinx /opt/Xilinx /opt/xilinx
    fi
}

# Every installed Vitis, as "ROOT|VERSION|SETTINGS", newest first within a
# root and roots in probe order. Returns 1 if there is none anywhere.
odin_vitis_versions() {
    local root settings version found=1
    while IFS= read -r root; do
        [ -n "${root}" ] || continue
        [ -d "${root}/Vitis" ] || continue
        while IFS= read -r settings; do
            [ -n "${settings}" ] || continue
            version="$(basename "$(dirname "${settings}")")"
            printf '%s|%s|%s\n' "${root}" "${version}" "${settings}"
            found=0
        done < <(find "${root}/Vitis" -mindepth 2 -maxdepth 2 -name settings64.sh \
                     2>/dev/null | sort -Vr)
    done < <(odin_vitis_roots)
    return "${found}"
}

odin_vitis_has() {
    local want="$1" line version
    while IFS= read -r line; do
        [ -n "${line}" ] || continue
        version="${line#*|}"
        version="${version%%|*}"
        if [ "${version}" = "${want}" ]; then
            return 0
        fi
    done < <(odin_vitis_versions 2>/dev/null || true)
    return 1
}

# Prints "ROOT|VERSION|SETTINGS" for the Vitis this cluster has, or returns 1.
odin_vitis_settings() {
    local want="${VITIS_VERSION:-}" prefer="${ODIN_VITIS_PREFER:-}"
    local all line version first="" preferred=""
    all="$(odin_vitis_versions 2>/dev/null || true)"
    if [ -z "${all}" ]; then
        return 1
    fi
    while IFS= read -r line; do
        [ -n "${line}" ] || continue
        version="${line#*|}"
        version="${version%%|*}"
        if [ -n "${want}" ]; then
            if [ "${version}" = "${want}" ]; then
                printf '%s\n' "${line}"
                return 0
            fi
            continue
        fi
        if [ -z "${first}" ]; then
            first="${line}"
        fi
        if [ -n "${prefer}" ] && [ -z "${preferred}" ] && [ "${version}" = "${prefer}" ]; then
            preferred="${line}"
        fi
    done <<< "${all}"
    if [ -n "${want}" ]; then
        return 1
    fi
    if [ -n "${preferred}" ]; then
        printf '%s\n' "${preferred}"
        return 0
    fi
    printf '%s\n' "${first}"
}

# The XRT the cluster keeps at /opt/xilinx/xrt (hacc_demo/doc/0-login.md line 55).
odin_xrt_setup() {
    printf '%s/setup.sh\n' "${XRT_ROOT:-/opt/xilinx/xrt}"
}

# Source both, with nounset relaxed for exactly that: Vitis 2024.x's
# .settings64-Vitis.sh reads $PYTHONPATH, which `set -u` treats as fatal.
odin_source_toolchain() {
    local settings="$1"
    set +u
    # shellcheck disable=SC1090  # cluster-side script, absent in this repo
    source "${settings}"
    # shellcheck disable=SC1090
    source "$(odin_xrt_setup)"
    set -u
}
