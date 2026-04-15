#!/usr/bin/env bash
# bootstrap_pi.sh — one-time setup that modifies /boot/firmware/cmdline.txt
# to enable isolcpus=2,3 + nohz_full + rcu_nocbs for real-time thread
# scheduling on Cortex-A76 cores.
#
# DESTRUCTIVE OPERATIONS — requires explicit confirmation:
#   1. Backs up cmdline.txt as cmdline.txt.bak.YYYYMMDD_HHMMSS
#   2. Appends isolcpus parameters (idempotent — checks if already set)
#   3. Offers to reboot so the new cmdline takes effect
#
# Recovery if the Pi fails to boot after this:
#   - Put the SD card in another machine
#   - Edit /boot/firmware/cmdline.txt directly, restore from the .bak file
#   - Re-insert and boot
#
# References:
#   - `man 7 sched` for SCHED_FIFO + isolcpus semantics
#   - https://docs.kernel.org/admin-guide/kernel-parameters.html
#     (search for isolcpus / nohz_full / rcu_nocbs)

set -euo pipefail

CMDLINE_FILE="/boot/firmware/cmdline.txt"
# Params we want to add. Core 0/1 stay for normal Linux tasks + BLE;
# cores 2/3 get isolated for encoder + Hailo workers.
ISOL_PARAMS="isolcpus=nohz,domain,managed_irq,2,3 nohz_full=2,3 rcu_nocbs=2,3 irqaffinity=0-1"

if [[ ${EUID} -ne 0 ]]; then
    echo "[bootstrap_pi] escalating to root"
    exec sudo -E "$0" "$@"
fi

if [[ ! -f "${CMDLINE_FILE}" ]]; then
    echo "[bootstrap_pi] ERROR: ${CMDLINE_FILE} not found."
    echo "  This script only runs on Raspberry Pi OS Bookworm 64-bit where"
    echo "  cmdline.txt lives in /boot/firmware/."
    exit 1
fi

echo
echo "============================================================"
echo " pi_streamer_c — bootstrap_pi.sh"
echo "============================================================"
echo
echo "This will modify ${CMDLINE_FILE} to isolate CPU cores 2 and 3"
echo "for the pi_streamer_c real-time pipeline:"
echo
echo "   ADD: ${ISOL_PARAMS}"
echo
echo "Current cmdline.txt contents:"
echo "------------------------------------------------------------"
cat "${CMDLINE_FILE}"
echo "------------------------------------------------------------"
echo
read -rp "Continue? This WILL require a reboot. [y/N] " confirm
if [[ ! "${confirm}" =~ ^[Yy]$ ]]; then
    echo "[bootstrap_pi] aborted by user"
    exit 0
fi

# --- Already applied? ---
if grep -q "isolcpus=" "${CMDLINE_FILE}"; then
    echo "[bootstrap_pi] ${CMDLINE_FILE} already contains isolcpus=. Leaving as-is."
    echo "[bootstrap_pi] current line:"
    cat "${CMDLINE_FILE}"
    exit 0
fi

# --- Backup ---
timestamp="$(date +%Y%m%d_%H%M%S)"
backup="${CMDLINE_FILE}.bak.${timestamp}"
echo "[bootstrap_pi] backing up → ${backup}"
cp "${CMDLINE_FILE}" "${backup}"

# --- Append (cmdline.txt is a single line; DO NOT add a newline) ---
# shellcheck disable=SC2002
existing="$(cat "${CMDLINE_FILE}")"
new_line="${existing% } ${ISOL_PARAMS}"
echo "[bootstrap_pi] new cmdline:"
echo "  ${new_line}"
printf '%s\n' "${new_line}" > "${CMDLINE_FILE}"

echo
read -rp "Reboot now to apply? [y/N] " reboot_confirm
if [[ "${reboot_confirm}" =~ ^[Yy]$ ]]; then
    echo "[bootstrap_pi] rebooting in 3 seconds. CTRL-C to abort."
    sleep 3
    /sbin/reboot
else
    echo "[bootstrap_pi] cmdline updated; reboot manually to apply."
    echo "[bootstrap_pi] rollback: sudo cp ${backup} ${CMDLINE_FILE} && sudo reboot"
fi
