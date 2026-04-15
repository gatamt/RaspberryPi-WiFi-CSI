#!/usr/bin/env bash
# Install and enable the pi_ble BLE GATT server as a systemd service.
# Run as root: sudo bash scripts/install_ble.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PI_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== Installing pi_ble BLE server ==="

# 1. Disable BlueZ battery plugin (prevents iOS pairing popup on every connect)
echo "  Disabling BlueZ battery plugin..."
BT_SERVICE="/lib/systemd/system/bluetooth.service"
if ! grep -q "\-P battery" "$BT_SERVICE" 2>/dev/null; then
    cp "$BT_SERVICE" "${BT_SERVICE}.bak.$(date +%s)"
    sed -i 's|ExecStart=.*bluetoothd.*|ExecStart=/usr/libexec/bluetooth/bluetoothd -P battery|' "$BT_SERVICE"
    echo "  Battery plugin disabled in $BT_SERVICE"
else
    echo "  Battery plugin already disabled"
fi

# 2. Apply main.conf with LE connection parameters
echo "  Configuring BlueZ main.conf..."
BLUEZ_CONF="/etc/bluetooth/main.conf"
cp "$BLUEZ_CONF" "${BLUEZ_CONF}.bak.$(date +%s)" 2>/dev/null || true
cp "$PI_DIR/bluetooth/main.conf" "$BLUEZ_CONF"
echo "  main.conf installed"

# 3. Enable and restart bluetooth
echo "  Enabling bluetooth..."
systemctl daemon-reload
systemctl enable bluetooth
systemctl restart bluetooth
sleep 2

# 4. Configure btmgmt (BLE-only, NoInputNoOutput)
echo "  Configuring btmgmt..."
btmgmt bredr off 2>/dev/null || true
btmgmt io-cap 3 2>/dev/null || true
btmgmt bondable on 2>/dev/null || true

# 5. Set supervision timeout (420ms → 2000ms)
echo "  Setting supervision timeout..."
echo 200 > /sys/kernel/debug/bluetooth/hci0/supervision_timeout 2>/dev/null || true

# 6. Clear stale bonds (fresh start)
echo "  Clearing stale BLE bonds..."
ADAPTER_MAC=$(hciconfig hci0 2>/dev/null | grep "BD Address" | awk '{print $3}' | tr ':' '-' | tr 'a-f' 'A-F')
if [ -d "/var/lib/bluetooth/$ADAPTER_MAC" ]; then
    rm -rf /var/lib/bluetooth/$ADAPTER_MAC/cache/
    echo "  Cleared BLE cache for $ADAPTER_MAC"
fi

# 3. Install systemd service
echo "  Installing systemd service..."
cp "$PI_DIR/systemd/pi-ble.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable pi-ble
systemctl start pi-ble

# 4. Verify
echo ""
echo "=== Verification ==="
systemctl status pi-ble --no-pager -l || true
echo ""
echo "=== Done ==="
echo "pi_ble BLE server is installed and running."
echo "View logs: journalctl -u pi-ble -f"
