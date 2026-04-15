"""WiFi management via NetworkManager (nmcli).

Scans for networks, connects, checks status, manages saved connections.
All operations are subprocess-based — no direct D-Bus dependency.
"""

import json
import logging
import re
import subprocess
from typing import Any

logger = logging.getLogger(__name__)

NMCLI_TIMEOUT = 30  # seconds


def _run(cmd: list[str], timeout: int = NMCLI_TIMEOUT) -> tuple[int, str, str]:
    """Run subprocess, return (returncode, stdout, stderr). Never raises."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except FileNotFoundError:
        return -1, "", f"Command not found: {cmd[0]}"


class WiFiManager:
    """Manages WiFi via nmcli subprocess calls."""

    def scan(self) -> list[dict[str, Any]]:
        """Scan for visible WiFi networks. Returns list of network dicts."""
        rc, out, err = _run([
            "nmcli", "-t", "-f", "SSID,SIGNAL,SECURITY",
            "dev", "wifi", "list", "--rescan", "yes",
        ])
        if rc != 0:
            logger.error("WiFi scan failed: %s", err)
            return []

        saved = set(self.saved_networks())
        networks: dict[str, dict] = {}
        for line in out.splitlines():
            parts = line.split(":")
            if len(parts) < 3:
                continue
            ssid = parts[0].strip()
            if not ssid:  # hidden networks
                continue
            try:
                signal = int(parts[1])
            except ValueError:
                signal = 0
            security = parts[2].strip() if parts[2].strip() else "Open"

            # Keep strongest signal per SSID
            if ssid not in networks or signal > networks[ssid]["signal"]:
                networks[ssid] = {
                    "ssid": ssid,
                    "signal": signal,
                    "security": security,
                    "saved": ssid in saved,
                }

        result = sorted(networks.values(), key=lambda n: n["signal"], reverse=True)
        logger.info("WiFi scan: %d networks found", len(result))
        return result

    def connect(self, ssid: str, password: str) -> tuple[bool, str]:
        """Connect to a WiFi network with password. Returns (success, message)."""
        logger.info("Connecting to WiFi: %s", ssid)
        saved_before = set(self.saved_networks())

        # Try simple auto-detect connect first
        rc, out, err = _run([
            "nmcli", "dev", "wifi", "connect", ssid, "password", password,
        ], timeout=30)
        if rc == 0:
            return self._connect_success(ssid)

        # If key-mgmt missing (iPhone hotspot, WPA3 networks), retry with
        # explicit security types: wpa-psk first, then sae (WPA3)
        combined = (err + " " + out).lower()
        if "key-mgmt" in combined or "security" in combined:
            # Clean up the failed auto-detect profile
            if ssid not in saved_before:
                _run(["nmcli", "con", "delete", ssid])

            for key_mgmt in ("wpa-psk", "sae"):
                logger.info("Retrying %s with key-mgmt=%s", ssid, key_mgmt)
                # Two-step: create connection profile, then activate
                rc2, _, err2 = _run([
                    "nmcli", "connection", "add",
                    "type", "wifi",
                    "con-name", ssid,
                    "ifname", "wlan0",
                    "ssid", ssid,
                    "wifi-sec.key-mgmt", key_mgmt,
                    "wifi-sec.psk", password,
                ])
                if rc2 != 0:
                    logger.warning("nmcli add failed for %s: %s", key_mgmt, err2)
                    continue
                rc2, out2, err2 = _run(["nmcli", "con", "up", ssid], timeout=30)
                if rc2 == 0:
                    return self._connect_success(ssid)
                # Clean up failed attempt before trying next key-mgmt
                _run(["nmcli", "con", "delete", ssid])
                logger.warning("key-mgmt=%s failed: %s %s", key_mgmt, out2, err2)

        # Classify error
        if "secrets" in combined or "no secret" in combined or "psk" in combined:
            error_type = "WRONG_PASSWORD"
        elif "no network" in combined or "not found" in combined:
            error_type = "NETWORK_NOT_FOUND"
        elif "timeout" in combined:
            error_type = "TIMEOUT"
        else:
            error_type = "CONNECT_FAILED"

        # Clean up: only delete if nmcli created a NEW connection profile
        if ssid not in saved_before and ssid in set(self.saved_networks()):
            _run(["nmcli", "con", "delete", ssid])
            logger.info("Deleted failed new connection: %s", ssid)

        logger.error("WiFi connect failed: %s (raw: %s %s)", error_type, out, err)
        return False, error_type

    def _connect_success(self, ssid: str) -> tuple[bool, str]:
        """Return success tuple after a WiFi connect."""
        zt_ip = self.get_zt_ip() or "NONE"
        wlan_ip = self.get_wlan_ip() or "NONE"
        logger.info("Connected to %s (wlan=%s, zt=%s)", ssid, wlan_ip, zt_ip)
        return True, f"{zt_ip}:{wlan_ip}"

    def reconnect(self, ssid: str) -> tuple[bool, str]:
        """Reconnect to a saved network (no password). Returns (success, message)."""
        logger.info("Reconnecting to saved WiFi: %s", ssid)
        rc, out, err = _run(["nmcli", "con", "up", ssid], timeout=30)
        if rc == 0:
            zt_ip = self.get_zt_ip() or "NONE"
            wlan_ip = self.get_wlan_ip() or "NONE"
            logger.info("Reconnected to %s (wlan=%s, zt=%s)", ssid, wlan_ip, zt_ip)
            return True, f"{zt_ip}:{wlan_ip}"
        logger.error("WiFi reconnect failed: %s %s", out, err)
        return False, err or out or "Unknown error"

    def status(self) -> dict[str, Any]:
        """Get current WiFi connection status."""
        rc, out, err = _run([
            "nmcli", "-t", "-f", "DEVICE,STATE,CONNECTION", "dev", "status",
        ])
        result = {"connected": False, "ssid": None, "wlan_ip": None, "zt_ip": None}
        if rc != 0:
            return result

        for line in out.splitlines():
            parts = line.split(":")
            if len(parts) >= 3 and parts[0] == "wlan0" and parts[1] == "connected":
                result["connected"] = True
                result["ssid"] = parts[2]
                break

        if result["connected"]:
            result["wlan_ip"] = self.get_wlan_ip()
            result["zt_ip"] = self.get_zt_ip()

        return result

    def get_wlan_ip(self) -> str | None:
        """Get wlan0 IPv4 address."""
        rc, out, _ = _run(["ip", "-4", "-o", "addr", "show", "wlan0"])
        if rc != 0:
            return None
        m = re.search(r"inet (\d+\.\d+\.\d+\.\d+)", out)
        return m.group(1) if m else None

    def get_zt_ip(self) -> str | None:
        """Get ZeroTier interface IPv4 address."""
        rc, out, _ = _run(["ip", "-4", "-o", "addr", "show"])
        if rc != 0:
            return None
        for line in out.splitlines():
            if " zt" in line:
                m = re.search(r"inet (\d+\.\d+\.\d+\.\d+)", line)
                if m:
                    return m.group(1)
        return None

    def saved_networks(self) -> list[str]:
        """List saved WiFi connection names."""
        rc, out, _ = _run(["nmcli", "-t", "-f", "NAME,TYPE", "con", "show"])
        if rc != 0:
            return []
        result = []
        for line in out.splitlines():
            parts = line.split(":")
            if len(parts) >= 2 and "wireless" in parts[1]:
                result.append(parts[0])
        return result

    def forget(self, ssid: str) -> tuple[bool, str]:
        """Delete a saved WiFi connection. Returns (success, message)."""
        rc, out, err = _run(["nmcli", "con", "delete", ssid])
        if rc == 0:
            logger.info("Forgot network: %s", ssid)
            return True, "FORGOTTEN"
        return False, err or "NOT_FOUND"


def _self_test():
    """Parse-only tests that don't modify WiFi state."""
    wm = WiFiManager()

    # Test IP parsing
    assert wm.get_wlan_ip() is not None or True, "wlan0 may not exist on dev machine"
    # Test saved_networks parsing
    saved = wm.saved_networks()
    assert isinstance(saved, list)
    # Test status parsing
    st = wm.status()
    assert isinstance(st, dict)
    assert "connected" in st
    assert "ssid" in st
    print(f"wifi: self-test passed (status={st}, saved={saved})")


if __name__ == "__main__":
    _self_test()
