"""BLE command protocol handler.

Dispatches UTF-8 commands from the BLE Command characteristic to the
appropriate handler (auth, wifi, stream) and formats responses.

Command format: CATEGORY:ACTION[:arg1[:arg2...]]
Response format: OK:<data> or ERR:<message>
"""

import json
import logging
from typing import Any

from .auth import BLESession, PinAuthenticator
from .stream import StreamManager
from .wifi import WiFiManager

logger = logging.getLogger(__name__)

DEFAULT_MTU = 185  # conservative BLE MTU after 3-byte ATT header


def chunk_response(data: str, mtu: int = DEFAULT_MTU) -> list[str]:
    """Split a large response into CHUNK:i/N:<data> notifications."""
    if len(data.encode()) <= mtu:
        return [data]

    # Chunk by bytes, respecting UTF-8 boundaries
    encoded = data.encode()
    # Reserve space for "CHUNK:NN/NN:" prefix (max ~12 bytes)
    payload_size = mtu - 15
    chunks: list[bytes] = []
    pos = 0
    while pos < len(encoded):
        end = min(pos + payload_size, len(encoded))
        # Don't split in the middle of a UTF-8 character
        while end > pos and (encoded[end - 1] & 0xC0) == 0x80:
            end -= 1
        chunks.append(encoded[pos:end])
        pos = end

    total = len(chunks)
    return [f"CHUNK:{i + 1}/{total}:{c.decode()}" for i, c in enumerate(chunks)]


class CommandHandler:
    """Parses and dispatches BLE commands."""

    def __init__(
        self,
        auth: PinAuthenticator,
        wifi: WiFiManager,
        stream: StreamManager,
    ):
        self._auth = auth
        self._wifi = wifi
        self._stream = stream

    def handle(self, command: str, session: BLESession) -> str | list[str]:
        """Handle a command string. Returns response string or list of chunks.

        WiFi Networks scan results are returned via a separate characteristic,
        so WIFI:SCAN returns "OK:SCANNING" on the response characteristic and
        the actual data is sent via the wifi_networks callback.
        """
        command = command.strip()
        if not command:
            return "ERR:EMPTY_COMMAND"

        parts = command.split(":", maxsplit=1)
        category = parts[0].upper()
        rest = parts[1] if len(parts) > 1 else ""

        if category == "AUTH":
            return self._handle_auth(rest, session)
        elif category == "WIFI":
            if not session.require_auth():
                return "ERR:NOT_AUTHENTICATED"
            return self._handle_wifi(rest)
        elif category == "STREAM":
            if not session.require_auth():
                return "ERR:NOT_AUTHENTICATED"
            return self._handle_stream(rest)
        else:
            return f"ERR:UNKNOWN_CATEGORY:{category}"

    # ── Auth ──────────────────────────────────────────────

    def _handle_auth(self, rest: str, session: BLESession) -> str:
        parts = rest.split(":", maxsplit=2)
        action = parts[0].upper() if parts else ""

        if action == "STATUS":
            return "OK:HAS_PIN" if self._auth.has_pin() else "OK:NO_PIN"

        elif action == "SET":
            if len(parts) < 2:
                return "ERR:MISSING_PIN"
            pin = parts[1]
            ok, msg = self._auth.set_initial_pin(pin)
            if ok:
                # Auto-authenticate after setting initial PIN
                session.authenticated = True
                return f"OK:{msg}"
            return f"ERR:{msg}"

        elif action == "VERIFY":
            if len(parts) < 2:
                return "ERR:MISSING_PIN"
            pin = parts[1]
            if session.authenticate(self._auth, pin):
                return "OK:VERIFIED"
            return "ERR:WRONG_PIN"

        elif action == "CHANGE":
            if len(parts) < 3:
                return "ERR:MISSING_PINS"
            old_pin = parts[1]
            new_pin = parts[2]
            ok, msg = self._auth.change_pin(old_pin, new_pin)
            return f"OK:{msg}" if ok else f"ERR:{msg}"

        return f"ERR:UNKNOWN_AUTH_ACTION:{action}"

    # ── WiFi ──────────────────────────────────────────────

    def _handle_wifi(self, rest: str) -> str | list[str]:
        # Split with maxsplit=3 to handle WIFI:CONNECT:SSID:password-with-colons
        parts = rest.split(":", maxsplit=2)
        action = parts[0].upper() if parts else ""

        if action == "SCAN":
            networks = self._wifi.scan()
            data = json.dumps(networks, ensure_ascii=False)
            # Return chunked if needed — caller sends via wifi_networks characteristic
            return chunk_response(data)

        elif action == "CONNECT":
            if len(parts) < 3:
                return "ERR:MISSING_SSID_OR_PASSWORD"
            ssid = parts[1]
            password = parts[2]
            ok, info = self._wifi.connect(ssid, password)
            if ok:
                return f"OK:CONNECTED:{info}"
            return f"ERR:CONNECT_FAILED:{info}"

        elif action == "RECONNECT":
            if len(parts) < 2:
                return "ERR:MISSING_SSID"
            ssid = parts[1]
            ok, info = self._wifi.reconnect(ssid)
            if ok:
                return f"OK:CONNECTED:{info}"
            return f"ERR:CONNECT_FAILED:{info}"

        elif action == "STATUS":
            st = self._wifi.status()
            if st["connected"]:
                zt = st["zt_ip"] or "NONE"
                wlan = st["wlan_ip"] or "NONE"
                return f"CONNECTED:{st['ssid']}:{zt}:{wlan}"
            return "DISCONNECTED"

        elif action == "SAVED":
            saved = self._wifi.saved_networks()
            return "OK:" + ",".join(saved) if saved else "OK:"

        elif action == "FORGET":
            if len(parts) < 2:
                return "ERR:MISSING_SSID"
            ssid = parts[1]
            ok, msg = self._wifi.forget(ssid)
            return f"OK:{msg}" if ok else f"ERR:{msg}"

        return f"ERR:UNKNOWN_WIFI_ACTION:{action}"

    # ── Stream ────────────────────────────────────────────

    def _handle_stream(self, rest: str) -> str:
        action = rest.upper().split(":")[0] if rest else ""

        if action == "START":
            ok, msg = self._stream.start()
            return f"OK:{msg}" if ok else f"ERR:{msg}"

        elif action == "STOP":
            ok, msg = self._stream.stop()
            return f"OK:{msg}" if ok else f"ERR:{msg}"

        elif action == "STATUS":
            return "RUNNING" if self._stream.is_running() else "STOPPED"

        return f"ERR:UNKNOWN_STREAM_ACTION:{action}"


def _self_test():
    import tempfile
    from pathlib import Path
    from unittest.mock import MagicMock

    d = Path(tempfile.mkdtemp())
    try:
        auth = PinAuthenticator(d)
        wifi = MagicMock(spec=WiFiManager)
        stream = MagicMock(spec=StreamManager)
        handler = CommandHandler(auth, wifi, stream)
        session = BLESession()

        # Auth flow
        assert handler.handle("AUTH:STATUS", session) == "OK:NO_PIN"
        assert handler.handle("AUTH:SET:1234", session) == "OK:PIN_SET"
        assert session.authenticated  # auto-auth after set
        assert handler.handle("AUTH:STATUS", session) == "OK:HAS_PIN"

        # Reset session
        session.reset()
        assert not session.authenticated

        # Unauthenticated WiFi should fail
        assert handler.handle("WIFI:SCAN", session) == "ERR:NOT_AUTHENTICATED"
        assert handler.handle("STREAM:STATUS", session) == "ERR:NOT_AUTHENTICATED"

        # Verify
        assert handler.handle("AUTH:VERIFY:0000", session) == "ERR:WRONG_PIN"
        assert handler.handle("AUTH:VERIFY:1234", session) == "OK:VERIFIED"
        assert session.authenticated

        # WiFi commands (mocked)
        wifi.scan.return_value = [{"ssid": "Test", "signal": -50, "security": "WPA2", "saved": False}]
        result = handler.handle("WIFI:SCAN", session)
        assert isinstance(result, list)  # chunked response
        wifi.scan.assert_called_once()

        wifi.connect.return_value = (True, "10.50.118.51:192.168.1.5")
        assert "OK:CONNECTED" in handler.handle("WIFI:CONNECT:TestNet:mypassword", session)

        wifi.reconnect.return_value = (True, "10.50.118.51:192.168.1.5")
        assert "OK:CONNECTED" in handler.handle("WIFI:RECONNECT:TestNet", session)

        wifi.status.return_value = {"connected": True, "ssid": "TestNet", "wlan_ip": "192.168.1.5", "zt_ip": "10.50.118.51"}
        assert "CONNECTED:TestNet" in handler.handle("WIFI:STATUS", session)

        wifi.saved_networks.return_value = ["Net1", "Net2"]
        assert handler.handle("WIFI:SAVED", session) == "OK:Net1,Net2"

        wifi.forget.return_value = (True, "FORGOTTEN")
        assert handler.handle("WIFI:FORGET:Net1", session) == "OK:FORGOTTEN"

        # Stream commands (mocked)
        stream.start.return_value = (True, "STARTED")
        assert handler.handle("STREAM:START", session) == "OK:STARTED"

        stream.stop.return_value = (True, "STOPPED")
        assert handler.handle("STREAM:STOP", session) == "OK:STOPPED"

        stream.is_running.return_value = True
        assert handler.handle("STREAM:STATUS", session) == "RUNNING"

        # Change PIN
        assert handler.handle("AUTH:CHANGE:0000:5678", session) == "ERR:WRONG_PIN"
        assert handler.handle("AUTH:CHANGE:1234:5678", session) == "OK:PIN_CHANGED"

        # Chunking test
        long_data = json.dumps([{"ssid": f"Network{i}", "signal": -50 + i, "security": "WPA2", "saved": False} for i in range(20)])
        chunks = chunk_response(long_data, mtu=100)
        assert len(chunks) > 1
        assert all(c.startswith("CHUNK:") for c in chunks)
        # Reassemble
        reassembled = "".join(c.split(":", maxsplit=2)[2] for c in chunks)
        assert json.loads(reassembled) == json.loads(long_data)

        print("protocol: all tests passed")
    finally:
        import shutil
        shutil.rmtree(d)


if __name__ == "__main__":
    _self_test()
