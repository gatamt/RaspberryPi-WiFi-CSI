#!/usr/bin/env bash
# Deploy pi_streamer_c to pi via rsync + remote build.
#
# This script is NON-DESTRUCTIVE by default:
#   - it only rsyncs source files (not the build/ directory)
#   - it does not overwrite cmdline.txt (that's bootstrap_pi.sh)
#   - it asks before restarting the systemd service
#
# Usage:
#   scripts/deploy.sh              # sync + build + confirm before restart
#   AUTO_RESTART=1 scripts/deploy.sh  # no confirmation before restart

set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE="${REMOTE:-pi}"
REMOTE_DIR="${REMOTE_DIR:-/home/pi/RaspberryPi-WiFi-CSI/pi/pi_streamer_c}"
SERVICE_NAME="${SERVICE_NAME:-pi-streamer-c.service}"

echo "[deploy] remote=${REMOTE}"
echo "[deploy] local source dir: ${HERE}"
echo "[deploy] remote dir:       ${REMOTE_DIR}"

# ---- rsync ------------------------------------------------------------
# Exclude build artifacts — they'd be wasted bytes and (on Mac→Pi deploys)
# the wrong architecture anyway.
echo "[deploy] rsync source → ${REMOTE}:${REMOTE_DIR}"
rsync -avz --delete \
    --exclude 'build/' \
    --exclude 'build-*/' \
    --exclude '.cache/' \
    --exclude 'compile_commands.json' \
    --exclude '.DS_Store' \
    "${HERE}/" \
    "${REMOTE}:${REMOTE_DIR}/"

# ---- remote build ------------------------------------------------------
echo "[deploy] remote build"
ssh "${REMOTE}" bash -lc "'cd ${REMOTE_DIR} && ./scripts/build.sh'"

# ---- systemd unit install ---------------------------------------------
SERVICE_SRC="${HERE}/../systemd/pi-streamer-c.service"
if [[ -f "${SERVICE_SRC}" ]]; then
    echo "[deploy] install systemd unit"
    ssh "${REMOTE}" "sudo install -m 0644 \
        ${REMOTE_DIR}/../systemd/pi-streamer-c.service \
        /etc/systemd/system/${SERVICE_NAME} && \
        sudo systemctl daemon-reload"
fi

# ---- service restart (gated) ------------------------------------------
if [[ "${AUTO_RESTART:-0}" == "1" ]]; then
    confirm="y"
else
    read -rp "[deploy] restart ${SERVICE_NAME} now? [y/N] " confirm
fi
if [[ "${confirm:-n}" =~ ^[Yy]$ ]]; then
    ssh "${REMOTE}" "sudo systemctl restart ${SERVICE_NAME} && \
                     sudo systemctl status --no-pager ${SERVICE_NAME}"
else
    echo "[deploy] service not restarted (manual: sudo systemctl restart ${SERVICE_NAME})"
fi

echo "[deploy] done."
