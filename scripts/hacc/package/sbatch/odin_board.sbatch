#!/bin/bash
#SBATCH --chdir /tmp/
#SBATCH --account=slurm
#SBATCH --job-name=odin_board
#SBATCH --output=slurm-odin-board-%j.out
#SBATCH --error=slurm-odin-board-%j.err
#
# Run the ODIN driver against an allocated card. run_all.sh picks the partition
# from the CARD PROFILE (scripts/hacc/cards.sh) and the cluster's own answers.
#   U55C: -p xilinx_u55c_gen3x16_xdma_3_202210_1  (hacc-gpu1/2/3, ONE HOUR cap)
#         -p mi210_u280_u55c                      (hacc-gpu1, 12 h)
#   U250: -p xilinx_u250_gen3x16_xdma_3_1_202020_1 (1 h) or the 12 h / 2 d pools
#
# ODIN_MODE=probe is B0: introspect the xclbin, load it, open the CU EXCLUSIVE,
# execute a one-token NULL PROGRAM and read back the capture header the fabric
# DMA'd home. It reads NO control register -- pyxrt binds none, which is what
# field-falsified the previous driver on this cluster on 2026-08-25.
#
# ODIN_MODE=deploy is "HACC NUS - ODIN Deployment": the sealed bundle under
# deployment/ executed as one host-mediated PASS per core -- program the core
# once, run each sample, transcode the counts into the next core's axon slots.
# ODIN_BUNDLE names a bundle other than the packaged default; ODIN_CHIP names
# WHICH FABRIC this bitstream holds (the executor refuses a bundle whose own
# chip_config claims name the other one); ODIN_DEPLOY_SAMPLES bounds the
# campaign; ODIN_ARM=resident programs each core once and runs every sample
# with program_words = 1 (the default, as_run, is the archived behaviour).
#
# ODIN_KERNEL_CORES=NC declares the cores the bitstream was built with (the same
# variable build_xclbn.sh packaged it at; unset: 1); the driver and the executor
# read it. Deploy places pass k on fabric core (position in the pass order) mod
# NC; ODIN_PLACEMENTS=rotations runs every fixture at every rotation of the NC
# fabric cores (default identity: core k on core k, the archived run).
#
# ODIN_MODE=power is the card power session: ODIN_POWER_PLAN names the session
# plan (power/plan_hacc_u55c_tcad.json), which carries its own xclbins and
# bundles, so ODIN_XCLBIN is not used. host/odin_power_protocol.py resets the
# card to its shell, starts the sidecar sampler (xrt-smi electrical and
# thermal reports, or pyxrt get_info), and runs the plan: shell-only window,
# telemetry pilot (the go or no-go gate), resident recertification, rotated
# kernel-idle and loop blocks per design, drift reference. ODIN_FAKE_PYXRT runs
# the same session against the fake pyxrt (no card; structure, not silicon).
#
# FIELD-OBSERVED 2026-08-25: the U55C board partition's AllowGroups is
# lab,hgpu,fpga_u55c — reachable through `hgpu` without fpga_u55c membership —
# and mi210_vck_u55c, which the vendor doc names, refuses this account outright.
# The u250 partitions are lab,fpga_u250, and this account IS in fpga_u250.
#
# THE CARD IS ALLOCATED, NOT AMBIENT. hacc-gpu3 answered '0 devices found' under
# xbutil from a non-FPGA partition on 2026-08-25: the cards are passed into the
# job's VM at allocation time (doc/1's board-VM model, now field-confirmed), so
# `xbutil examine` finding nothing here means the wrong partition, not a dead
# card.
#
# The first two #SBATCH lines are the cluster's required preamble (that doc's
# example.sh lines 143-145). --output/--error stay under --chdir (/tmp) because
# slurm opens them before the job body runs.
#
#   ODIN_PKG=... ODIN_XCLBIN=... ODIN_MODE=probe|run \
#       sbatch --wait -p <partition> scripts/hacc/odin_board.sbatch
#   ODIN_PKG=... ODIN_MODE=power ODIN_POWER_PLAN=.../power/plan_hacc_u55c_tcad.json \
#       sbatch --wait -p <partition> scripts/hacc/odin_board.sbatch

set -euo pipefail

# Slurm's own --output/--error live on the node's /tmp (see the preamble above),
# so everything this job prints is ALSO mirrored under ODIN_LOG on /data — that
# mirror, not the slurm file, is what run_all.sh reads and what comes home.
if [[ -n "${ODIN_LOG:-}" ]]; then
    mkdir -p "$(dirname "${ODIN_LOG}")"
    exec > >(tee -a "${ODIN_LOG}") 2>&1
fi

PKG="${ODIN_PKG:?ODIN_PKG (the unpacked package root) must be exported}"
MODE="${ODIN_MODE:-run}"
if [[ "${MODE}" == "power" ]]; then
    # The power plan names every bitstream it loads; there is no single one.
    XCLBIN="${ODIN_XCLBIN:-(named per design by ${ODIN_POWER_PLAN:-the plan})}"
else
    XCLBIN="${ODIN_XCLBIN:?ODIN_XCLBIN (the staged bitstream) must be exported}"
fi
RESULTS="${ODIN_RESULTS:-${PKG}/results/board}"
XRT_SETUP="${XRT_SETUP:-/opt/xilinx/xrt/setup.sh}"
XRT_BIN="${XRT_BIN:-/opt/xilinx/xrt/bin}"

echo "[board] job ${SLURM_JOB_ID:-?} on $(hostname) partition ${SLURM_JOB_PARTITION:-?}"
echo "[board] package ${PKG}"
echo "[board] xclbin  ${XCLBIN}"
echo "[board] mode    ${MODE}"

if [[ ! -f "${XRT_SETUP}" ]]; then
    echo "REFUSING: no XRT at ${XRT_SETUP}. This job runs ONLY on an allocated" >&2
    echo "  HACC@NUS board node; /opt/xilinx/xrt is the cluster's XRT root" >&2
    echo "  (hacc_demo/doc/0-login.md line 55)." >&2
    exit 2
fi

mkdir -p "${RESULTS}"
# shellcheck disable=SC1090  # cluster-side script, absent in this repo
source "${XRT_SETUP}"
"${XRT_BIN}/xbutil" examine | tee "${RESULTS}/xbutil_examine.log"

case "${MODE}" in
    probe)
        python3 "${PKG}/host/odin_board_driver.py" --probe \
            --xclbin "${XCLBIN}" --results "${RESULTS}"
        ;;
    run)
        python3 "${PKG}/host/odin_board_driver.py" \
            --xclbin "${XCLBIN}" --fixtures "${PKG}/fixtures" \
            --placements "${ODIN_PLACEMENTS:-identity}" \
            --results "${RESULTS}"
        ;;
    deploy)
        DEPLOY_ARGS=(--xclbin "${XCLBIN}" --results "${RESULTS}")
        if [[ -n "${ODIN_BUNDLE:-}" ]]; then
            DEPLOY_ARGS+=(--bundle "${ODIN_BUNDLE}")
        fi
        if [[ -n "${ODIN_CHIP:-}" ]]; then
            DEPLOY_ARGS+=(--chip "${ODIN_CHIP}")
        fi
        if [[ -n "${ODIN_ARM:-}" ]]; then
            DEPLOY_ARGS+=(--arm "${ODIN_ARM}")
        fi
        python3 "${PKG}/host/odin_deployment_executor.py" "${DEPLOY_ARGS[@]}"
        ;;
    power)
        PLAN="${ODIN_POWER_PLAN:-${PKG}/power/plan_hacc_u55c_tcad.json}"
        if [[ ! -f "${PLAN}" ]]; then
            echo "REFUSING: no power plan at ${PLAN} (ODIN_POWER_PLAN)." >&2
            exit 2
        fi
        POWER_ARGS=(session --plan "${PLAN}" --results "${RESULTS}")
        if [[ -n "${ODIN_FAKE_PYXRT:-}" ]]; then
            POWER_ARGS+=(--fake-pyxrt "${ODIN_FAKE_PYXRT}")
        fi
        python3 "${PKG}/host/odin_power_protocol.py" "${POWER_ARGS[@]}"
        ;;
    *)
        echo "REFUSING: unknown ODIN_MODE '${MODE}' (probe | run | deploy | power)." >&2
        exit 2
        ;;
esac
