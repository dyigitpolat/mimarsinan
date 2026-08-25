# shellcheck shell=bash
# The TARGET CARD is a parameter, and this file is the only place that knows
# what one implies.
#
# FIELD LESSON (2026-08-25, hacchead + hacc-node0 + hacc-gpu3). v2 hard-wired
# the U55C everywhere — platform name, Vivado part, v++ config, board
# partition. Then the U55C path turned out to be ADMIN-BLOCKED: the only U55C
# platform installed anywhere on the cluster is
# xilinx_u55c_gen3x16_xdma_3_202210_1, whose SmartConnect IP is locked to the
# 2022.2 release, and 2022.2 is the one Vitis this cluster does NOT carry
# (2020.1 2020.2 2021.2 2022.1 2023.2 2024.2 are installed). Three vpl links
# under 2022.1, 2023.2 and 2024.2 all died the same way (VPL 60-704 / 60-732,
# "customized with software release 2022.1 ... different revision").
#
# So the card became a parameter. ODIN_CARD=u250 selects the
# xilinx_u250_gen3x16_xdma_3_1_202020_1 shell and Vitis 2020.2 — a pairing this
# cluster can actually build today — and every card-shaped fact below travels
# with it: the platform, the part, the v++ connectivity config (and therefore
# whether the AXI master lands on HBM or DDR), the preferred Vitis, and the
# partitions the board and joint phases may use.
#
#   odin_card                      the selected card (ODIN_CARD, default u55c)
#   odin_card_profile [card]       KEY=VALUE lines, or 1 for an unknown card
#   odin_card_field KEY [card]     one field of that profile
#   odin_card_refusal [card]       prints WHY this card cannot build here, and
#                                  returns 0 — i.e. 0 means REFUSE. Returns 1
#                                  when there is nothing against the card.
#
# The refusal is EVIDENCE-GATED, never a hard-coded verdict: it reads the
# installed platforms and the installed Vitis versions and only refuses when
# they still show the deadlock. Install 2022.2, or install a U55C shell built
# for a release that is here, and the refusal disappears on its own.

ODIN_CARD_DEFAULT="u55c"

odin_card() { printf '%s' "${ODIN_CARD:-${ODIN_CARD_DEFAULT}}"; }

odin_card_profile() {
    local card="${1:-}"
    [ -n "${card}" ] || card="$(odin_card)"
    case "${card}" in
        u55c)
            cat <<'PROFILE'
card=u55c
platform=xilinx_u55c_gen3x16_xdma_3_202210_1
part=xcu55c-fsvh2892-2L-e
cfg=scripts/hacc/odin_u55c.cfg
memory=HBM[0]
vitis_prefer=2022.2
board_candidates=xilinx_u55c_gen3x16_xdma_3_202210_1 amd_gpu_rocm_6_0_0
joint_candidates=mi210_vck_u55c mi210_u280_u55c mi210_u280_u55c_long_reservation
joint_model=board_and_gpu_in_one_chassis
PROFILE
            ;;
        u250)
            cat <<'PROFILE'
card=u250
platform=xilinx_u250_gen3x16_xdma_3_1_202020_1
part=xcu250-figd2104-2L-e
cfg=scripts/hacc/odin_u250.cfg
memory=DDR[0]
vitis_prefer=2020.2
board_candidates=xilinx_u250_gen3x16_xdma_3_1_202020_1 u250_standard_reservation_pool u250_long_reservation_pool
joint_candidates=u250_standard_reservation_pool u250_long_reservation_pool xilinx_u250_gen3x16_xdma_3_1_202020_1
joint_model=board_and_cpu_reference_in_one_allocation
PROFILE
            ;;
        *)
            return 1
            ;;
    esac
}

odin_card_field() {
    local key="$1" card="${2:-}" profile
    profile="$(odin_card_profile "${card}")" || return 1
    printf '%s' "${profile}" | sed -n "s/^${key}=//p" | head -1
}

odin_card_names() { printf 'u55c u250'; }

# ---------------------------------------------------------------------------
# Evidence: what this machine actually has
# ---------------------------------------------------------------------------

odin_platform_root() { printf '%s' "${ODIN_PLATFORM_ROOT:-/opt/xilinx/platforms}"; }

# Every platform directory name under the platform root, or 1 if there is no
# root to read — off-cluster that is the normal case, and "no evidence" must
# never be mistaken for "evidence of a problem".
odin_platforms_available() {
    local root entry found=1
    root="$(odin_platform_root)"
    [ -d "${root}" ] || return 1
    for entry in "${root}"/*; do
        [ -e "${entry}" ] || continue
        printf '%s\n' "$(basename "${entry}")"
        found=0
    done
    return "${found}"
}

odin_effective_platform() {
    local card="${1:-}"
    if [ -n "${ODIN_PLATFORM:-}" ]; then
        printf '%s' "${ODIN_PLATFORM}"
        return 0
    fi
    odin_card_field platform "${card}"
}

#: Shells whose IP is locked to the 2022.2 release this cluster does not carry.
ODIN_DEAD_2022_2_PLATFORMS="xilinx_u250_gen3x16_xdma_4_1_202210_1"

_odin_vitis_list() {
    if command -v odin_vitis_versions > /dev/null 2>&1; then
        odin_vitis_versions 2>/dev/null | cut -d'|' -f2 | tr '\n' ' '
    else
        printf 'unknown (toolchain.sh not sourced)'
    fi
}

_odin_has_2022_2() {
    command -v odin_vitis_has > /dev/null 2>&1 || return 1
    odin_vitis_has 2022.2
}

_odin_deadlock_body() {
    cat <<BODY
  The shell's SmartConnect IP is customized with the 2022.2 software release.
  Vitis installed here: $(_odin_vitis_list)— there is no 2022.2, and vpl
  refuses across every one that is (field-observed 2026-08-25: three logs, VPL
  60-704 / 60-732, "customized with software release 2022.1 ... a different
  revision", under Vitis 2022.1, 2023.2 AND 2024.2).
  THE ADMIN FIX, and it is the only one: install Vitis/Vivado 2022.2 alongside
  the others, or install a shell for this card built against a release that is
  already here. Nothing in this package can work around a locked IP revision.
  MEANWHILE, the field-viable route is the U250:
      ODIN_CARD=u250 ./run_all.sh          (or ./bootstrap_hacc.sh --card u250)
  which pairs xilinx_u250_gen3x16_xdma_3_1_202020_1 with Vitis 2020.2 — both
  installed on this cluster.
  This refusal cost no queue slot: it was decided from $(odin_platform_root)
  and the installed Vitis list, before anything was submitted.
BODY
}

# 0 = refuse (and the reason is on stdout); 1 = nothing known against this card.
odin_card_refusal() {
    local card="${1:-}" platform available u55c_shells
    [ -n "${card}" ] || card="$(odin_card)"
    platform="$(odin_effective_platform "${card}")"

    case " ${ODIN_DEAD_2022_2_PLATFORMS} " in
        *" ${platform} "*)
            printf 'REFUSING: platform %s is a 2022.2-era shell, and this cluster has no 2022.2.\n' \
                "${platform}"
            _odin_deadlock_body
            return 0
            ;;
    esac

    available="$(odin_platforms_available)" || return 1

    case "${card}" in
        u55c)
            u55c_shells="$(printf '%s\n' "${available}" | grep '^xilinx_u55c' || true)"
            # No U55C shell at all is a different problem; let the build name it.
            [ -n "${u55c_shells}" ] || return 1
            # SELF-HEAL: any U55C shell that is not the 2022.2-era one clears this.
            if printf '%s\n' "${u55c_shells}" | grep -qv '_202210_1$'; then
                return 1
            fi
            # SELF-HEAL: admins installed 2022.2.
            if _odin_has_2022_2; then
                return 1
            fi
            printf 'REFUSING: ODIN_CARD=u55c is DEADLOCKED on this cluster (field-observed 2026-08-25).\n'
            printf '  The only U55C platform installed anywhere (hacc-node0 AND hacc-gpu3)\n'
            printf '  is %s.\n' "$(printf '%s\n' "${u55c_shells}" | tr '\n' ' ' | sed 's/ *$//')"
            _odin_deadlock_body
            return 0
            ;;
        u250)
            if printf '%s\n' "${available}" | grep -qx "${platform}"; then
                return 1
            fi
            if printf '%s\n' "${available}" | grep -qx xilinx_u250_gen3x16_xdma_4_1_202210_1; then
                printf 'REFUSING: the only U250 shell here is xilinx_u250_gen3x16_xdma_4_1_202210_1,\n'
                printf '  which is 2022.2-era and hits exactly the deadlock the 3_1_202020_1 shell\n'
                printf '  avoids.\n'
                _odin_deadlock_body
                return 0
            fi
            return 1
            ;;
    esac
    return 1
}

# Resolve the card once and echo the decision. Callers get CARD/PLATFORM/PART/
# CFG set and ODIN_VITIS_PREFER exported for toolchain.sh; an unknown card or a
# refused one exits 2 right here rather than downstream.
odin_card_resolve() {
    local reason
    CARD="$(odin_card)"
    if ! odin_card_profile "${CARD}" > /dev/null 2>&1; then
        echo "REFUSING: unknown ODIN_CARD='${CARD}'. Known cards: $(odin_card_names)." >&2
        echo "  The per-card facts live in scripts/hacc/cards.sh — add a profile" >&2
        echo "  there rather than special-casing a card anywhere else." >&2
        exit 2
    fi
    PLATFORM="$(odin_effective_platform "${CARD}")"
    PART="${ODIN_FPGA_PART:-$(odin_card_field part "${CARD}")}"
    CFG="${ODIN_VXX_CONFIG:-$(odin_card_field cfg "${CARD}")}"
    if [ -z "${ODIN_VITIS_PREFER:-}" ]; then
        ODIN_VITIS_PREFER="$(odin_card_field vitis_prefer "${CARD}")"
        export ODIN_VITIS_PREFER
    fi
    if reason="$(odin_card_refusal "${CARD}")"; then
        printf '%s\n' "${reason}" >&2
        exit 2
    fi
}
