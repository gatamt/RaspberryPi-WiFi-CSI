#!/usr/bin/env bash
# Install all pi_streamer_c build dependencies on Raspberry Pi OS Bookworm.
#
# This script is idempotent: safe to re-run. It does NOT touch cmdline.txt,
# systemd services, or any runtime state. For those, see:
#   - scripts/bootstrap_pi.sh   (isolcpus + reboot, interactive)
#   - pi/systemd/pi-streamer-c.service
#
# Assumed platform: Raspberry Pi 5, Raspberry Pi OS Bookworm 64-bit,
# kernel 6.1.0-rpi-2712+.
#
# Pinned upstream versions (taken as-is from Bookworm's apt repos unless noted):
#   libcamera       0.5.x   (libcamera-dev)
#   libx264         0.164+  (libx264-dev)
#   liburing        2.2.x+  (liburing-dev)
#   HailoRT         5.1.1+  (installed out-of-band from Hailo developer zone)

set -euo pipefail

# Re-exec as root if not already.
if [[ ${EUID} -ne 0 ]]; then
    echo "[install_deps] escalating to root via sudo"
    exec sudo -E "$0" "$@"
fi

APT_PKGS=(
    # Build toolchain
    build-essential
    cmake
    ninja-build
    pkg-config
    git

    # Pi-side libraries
    libcamera-dev
    libcamera-apps-lite
    libx264-dev
    liburing-dev

    # Static analysis / dynamic analysis tooling
    valgrind
    clang-tidy
    cppcheck
)

echo "[install_deps] apt-get update"
apt-get update

echo "[install_deps] installing packages: ${APT_PKGS[*]}"
apt-get install -y --no-install-recommends "${APT_PKGS[@]}"

# ---- HailoRT ----------------------------------------------------------------
# HailoRT does not ship in Debian/Raspberry Pi OS apt. It is installed via
# the .deb from https://hailo.ai/developer-zone/ . Detect whether it's
# already present; warn (non-fatal) if absent.
if pkg-config --exists hailort 2>/dev/null; then
    echo "[install_deps] HailoRT detected via pkg-config"
elif [[ -f /usr/lib/aarch64-linux-gnu/libhailort.so ]]; then
    echo "[install_deps] HailoRT detected via libhailort.so"
else
    cat >&2 <<'HAILO_WARN'
[install_deps] WARNING: HailoRT is NOT installed on this system.
  The build will still succeed — real inference support is gated on
  find_package(HailoRT) in CMakeLists.txt, and the stub is a no-op when
  absent. To install HailoRT proper:
    1. Register at https://hailo.ai/developer-zone/
    2. Download the HailoRT .deb for aarch64 Linux (pinned ≥5.1.1)
    3. sudo dpkg -i hailort_*.deb
HAILO_WARN
fi

# ---- Capability grants for SCHED_FIFO ---------------------------------------
# The deployed unit file grants these via AmbientCapabilities. This section is
# documentation only — apt install itself does not need capabilities.
cat <<'CAP_NOTE'
[install_deps] Note: running the built binary under SCHED_FIFO requires
  either CAP_SYS_NICE (granted by the systemd unit) or rtprio in
  /etc/security/limits.conf. See pi/systemd/pi-streamer-c.service.
CAP_NOTE

echo "[install_deps] done."
