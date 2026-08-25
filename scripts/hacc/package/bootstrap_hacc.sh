#!/usr/bin/env bash
# The ONE command you run on hacchead after uploading odin_hacc_package.zip.
#
#   ./bootstrap_hacc.sh [--card u55c|u250] [/path/to/odin_hacc_package.zip]
#
# THE CARD IS A PARAMETER (v3). --card picks the profile in
# scripts/hacc/cards.sh and travels into the detached run as ODIN_CARD. On
# HACC@NUS today that means `--card u250`: the only U55C shell installed is
# locked to Vitis 2022.2, which this cluster does not have, and the u55c
# default refuses at startup with that reason rather than queueing a job to
# rediscover it.
#
# It verifies the upload against the sha256 that was baked in when the zip was
# built, ARCHIVES whatever install is already there (it never deletes silently),
# extracts a fresh one under ${ODIN_DATA_ROOT:-/data}/${USER}, and launches
# run_all.sh DETACHED — setsid + nohup, output to results/run_all.log — so the
# bring-up survives your ssh session closing.
#
# FIELD LESSON (2026-08-25). v1 ran run_all.sh in the foreground with
# `sbatch --wait`, which meant every hiccup needed a human at the keyboard to
# re-invoke it. Nobody should have to babysit a six-hour place-and-route.
#
# ODIN_DATA_ROOT overrides /data (the only path the head node and the board VMs
# share) so this whole flow can be rehearsed off-cluster.

set -euo pipefail

# Injected by scripts/hacc/make_package.py AFTER the zip is written, into the
# STANDALONE copy that ships beside it. The copy inside the zip necessarily
# keeps the placeholder — no file can contain the hash of an archive that
# contains that same file — and says so instead of pretending to verify.
ZIP_SHA256="__ODIN_ZIP_SHA256__"

HERE="$(cd "$(dirname "$0")" && pwd)"
DATA_ROOT="${ODIN_DATA_ROOT:-/data}"
DEST_PARENT="${DATA_ROOT}/${USER}"
TARGET="${DEST_PARENT}/odin_hacc_package"

say() { printf '%s\n' "$*"; }
refuse() {
    say "REFUSING: $1"
    shift
    for line in "$@"; do say "  ${line}"; done
    exit 2
}

CARD="${ODIN_CARD:-}"
ZIP_ARG=""
while [ $# -gt 0 ]; do
    case "$1" in
        --card) CARD="${2:-}"; shift ;;
        --card=*) CARD="${1#--card=}" ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        -*) refuse "unknown option '$1'." "Usage: ./bootstrap_hacc.sh [--card u55c|u250] [zip]" ;;
        *) ZIP_ARG="$1" ;;
    esac
    shift
done

for tool in unzip tar sha256sum setsid nohup; do
    command -v "${tool}" > /dev/null 2>&1 \
        || refuse "no '${tool}' on PATH." \
            "This script needs unzip, tar, sha256sum, setsid and nohup — all" \
            "of them are on hacchead. Run it there, next to the uploaded zip."
done

# --- 1. find the zip ---------------------------------------------------------
ZIP="${ZIP_ARG:-${ODIN_ZIP:-${HERE}/odin_hacc_package.zip}}"
if [ ! -f "${ZIP}" ]; then
    refuse "no zip at ${ZIP}." \
        "Upload odin_hacc_package.zip next to this script, or pass its path:" \
        "  ./bootstrap_hacc.sh /path/to/odin_hacc_package.zip"
fi
ZIP="$(cd "$(dirname "${ZIP}")" && pwd)/$(basename "${ZIP}")"

# --- 2. verify it is the zip this script was cut for -------------------------
measured="$(sha256sum "${ZIP}" | cut -d' ' -f1)"
if [ "${#ZIP_SHA256}" -eq 64 ] && [ -z "${ZIP_SHA256//[0-9a-f]/}" ]; then
    if [ "${measured}" != "${ZIP_SHA256}" ]; then
        refuse "the upload is not the package this bootstrap was cut for." \
            "expected sha256 ${ZIP_SHA256}" \
            "measured sha256 ${measured}" \
            "A truncated or resumed upload fails here instead of failing six" \
            "hours into a build. Re-upload the zip and this script TOGETHER."
    fi
    say "[bootstrap] sha256 ${measured} — matches the packaged value."
else
    say "[bootstrap] NOTE: this is the copy that travels INSIDE the zip, so it"
    say "            carries no hash to check (it cannot contain its own"
    say "            archive's digest). Measured ${measured}. The standalone"
    say "            bootstrap_hacc.sh shipped beside the zip does verify."
fi

# --- 3. never delete an install silently -------------------------------------
mkdir -p "${DEST_PARENT}" 2>/dev/null \
    || refuse "cannot create ${DEST_PARENT}." \
        "${DATA_ROOT} is the only path the head node and the board VMs share." \
        "Set ODIN_DATA_ROOT if this cluster keeps it elsewhere."
[ -w "${DEST_PARENT}" ] || refuse "${DEST_PARENT} is not writable by ${USER}."

if [ -d "${TARGET}" ]; then
    live=""
    [ -f "${TARGET}/.run_all.lock/pid" ] && live="$(cat "${TARGET}/.run_all.lock/pid")"
    if [ -n "${live}" ] && kill -0 "${live}" 2>/dev/null; then
        refuse "a bring-up is still running in ${TARGET} as pid ${live}." \
            "Archiving it now would pull the floor out from under a live job." \
            "Watch it :  tail -f ${TARGET}/results/run_all.log" \
            "Stop it  :  kill ${live}    — completed work is kept" \
            "Then re-run this bootstrap."
    fi
    archive="${DEST_PARENT}/odin_prev_$(date -u +%Y%m%dT%H%M%SZ).tar.gz"
    say "[bootstrap] ${TARGET} exists — archiving it whole to ${archive}"
    tar -czf "${archive}" -C "${DEST_PARENT}" odin_hacc_package \
        || refuse "the archive of the existing install failed." \
            "Nothing was deleted. Free some space under ${DEST_PARENT} and retry."
    say "[bootstrap] archived $(du -h "${archive}" | cut -f1); now replacing it."
    rm -rf "${TARGET}"
fi

# --- 4. extract and launch ---------------------------------------------------
unzip -q "${ZIP}" -d "${DEST_PARENT}"
[ -x "${TARGET}/run_all.sh" ] || chmod +x "${TARGET}/run_all.sh"
chmod +x "${TARGET}/collect_results.sh" "${TARGET}/scripts/status.sh" \
    "${TARGET}/bootstrap_hacc.sh" 2>/dev/null || true
mkdir -p "${TARGET}/results"
say "[bootstrap] extracted $(find "${TARGET}" -type f | wc -l) files into ${TARGET}"

# The card name is validated against the SSOT that just came out of the zip —
# scripts/hacc/cards.sh — so a typo stops here instead of detaching a run that
# will refuse into a log nobody is watching.
if [ -n "${CARD}" ]; then
    # shellcheck disable=SC1090  # it is inside the package we just unpacked
    . "${TARGET}/scripts/hacc/cards.sh"
    odin_card_profile "${CARD}" > /dev/null 2>&1 \
        || refuse "unknown --card '${CARD}'." \
            "Known cards: $(odin_card_names)." \
            "The per-card facts live in scripts/hacc/cards.sh inside the package."
    say "[bootstrap] card ${CARD} — platform $(odin_card_field platform "${CARD}")"
fi

cd "${TARGET}"
LOG="${TARGET}/results/run_all.log"
{
    printf '\n===== bootstrap %s =====\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'zip     : %s (sha256 %s)\n' "${ZIP}" "${measured}"
    printf 'package : %s\n' "${TARGET}"
    printf 'card    : %s\n' "${CARD:-u55c (default)}"
} >> "${LOG}"

if [ -n "${CARD}" ]; then
    setsid env ODIN_CARD="${CARD}" nohup ./run_all.sh >> "${LOG}" 2>&1 < /dev/null &
else
    setsid nohup ./run_all.sh >> "${LOG}" 2>&1 < /dev/null &
fi
disown 2>/dev/null || true

# Wait only for PROOF that it started: the lock it takes first, or its own
# opening line if it was quick enough to be finished already.
pid=""
for _ in $(seq 1 20); do
    if [ -f "${TARGET}/.run_all.lock/pid" ]; then
        pid="$(cat "${TARGET}/.run_all.lock/pid")"
        break
    fi
    if grep -q 'ODIN bring-up starting' "${LOG}"; then
        break
    fi
    sleep 0.5
done
if [ -n "${pid}" ]; then
    say "[bootstrap] run_all.sh is detached, running as pid ${pid}."
elif grep -q 'ODIN bring-up starting' "${LOG}"; then
    say "[bootstrap] run_all.sh is detached; it has already released its lock,"
    say "            so it either finished or stopped. Read the log."
else
    say "[bootstrap] WARNING: run_all.sh never started within 10s. Read the log:"
    say "            something refused before phase 0."
fi

say "----------------------------------------------------------------------"
say "tail -f ${LOG}"
say "${TARGET}/scripts/status.sh"
say "Safe to log out now — the bring-up no longer needs your shell."
say "----------------------------------------------------------------------"
