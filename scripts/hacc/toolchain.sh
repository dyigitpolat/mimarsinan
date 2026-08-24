# shellcheck shell=bash
# Where the cluster's Vitis and XRT actually are — discovered, never assumed.
#
# FIELD LESSON (2026-08-25, hacchead). The v1 scripts hard-coded Vitis 2022.2
# under /tools/xilinx because that is what Xtra-Computing/hacc_demo documents.
# The live cluster carries Vitis **2024.2** under **/tools/Xilinx** (capital X),
# so every hard-coded probe refused before it ever reached a compiler. This file
# is the one place that resolves it: newest version first, env overrides win,
# and both the build script and run_all.sh source it so they can never disagree.

# Prints "ROOT|VERSION|SETTINGS" for the Vitis this cluster has, or returns 1.
odin_vitis_settings() {
    local roots=() root version
    if [ -n "${XILINX_ROOT:-}" ]; then
        roots=("${XILINX_ROOT}")
    else
        roots=(/tools/Xilinx /tools/xilinx /opt/Xilinx /opt/xilinx)
    fi
    for root in "${roots[@]}"; do
        [ -d "${root}/Vitis" ] || continue
        if [ -n "${VITIS_VERSION:-}" ]; then
            if [ -f "${root}/Vitis/${VITIS_VERSION}/settings64.sh" ]; then
                printf '%s|%s|%s\n' "${root}" "${VITIS_VERSION}" \
                    "${root}/Vitis/${VITIS_VERSION}/settings64.sh"
                return 0
            fi
            continue
        fi
        local settings
        while IFS= read -r settings; do
            [ -n "${settings}" ] || continue
            version="$(basename "$(dirname "${settings}")")"
            printf '%s|%s|%s\n' "${root}" "${version}" "${settings}"
            return 0
        done < <(find "${root}/Vitis" -mindepth 2 -maxdepth 2 -name settings64.sh \
                     2>/dev/null | sort -Vr)
    done
    return 1
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
