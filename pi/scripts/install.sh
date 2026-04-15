#!/usr/bin/env bash
#
# One-shot installer for pi_streamer on a fresh Raspberry Pi 5.
# Assumes hailo-h10-all is already installed and /dev/hailo0 is functional.
#
# Usage:
#   cd ~/RaspberryPi-WiFi-CSI/pi
#   ./scripts/install.sh
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PI_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
AVAHI_SRC="${PI_DIR}/avahi/wificsi.service"
AVAHI_DST="/etc/avahi/services/wificsi.service"

echo "=== pi_streamer install ==="
echo "  pi dir: ${PI_DIR}"

echo "--- 1/5: apt deps ---"
sudo apt update
sudo apt install -y \
    python3-venv \
    python3-picamera2 \
    python3-libcamera \
    libgl1 \
    avahi-daemon \
    iw

echo "--- 2/5: Python venv ---"
VENV="${PI_DIR}/.venv"
if [[ ! -d "${VENV}" ]]; then
    # Use --system-site-packages so picamera2 + libcamera (installed via apt) are reachable
    python3 -m venv --system-site-packages "${VENV}"
fi
# shellcheck disable=SC1090
source "${VENV}/bin/activate"

echo "--- 3/5: pip deps ---"
pip install --upgrade pip
pip install -r "${PI_DIR}/requirements.txt"

echo "--- 4/5: Hailo python bindings sanity check ---"
if python -c 'import hailo_platform; print("hailo_platform", hailo_platform.__version__)'; then
    echo "hailo_platform import OK"
else
    echo "WARNING: hailo_platform not importable in this venv."
    echo "         It is provided by the python3-h10-hailort apt package."
    echo "         The venv was created with --system-site-packages so it"
    echo "         should be visible. Check with:"
    echo "           dpkg -L python3-h10-hailort | grep hailo_platform"
fi

echo "--- 5/5: avahi mDNS service ---"
if [[ ! -f "${AVAHI_SRC}" ]]; then
    echo "ERROR: ${AVAHI_SRC} is missing"
    exit 1
fi
sudo install -m 0644 "${AVAHI_SRC}" "${AVAHI_DST}"
sudo systemctl restart avahi-daemon

echo ""
echo "=== install complete ==="
echo ""
echo "Run the streamer with:"
echo "  source ${VENV}/bin/activate"
echo "  python -m pi_streamer.main --log-level INFO"
echo ""
echo "Or for a camera-only smoke test without Hailo:"
echo "  python -m pi_streamer.main --no-inference"
