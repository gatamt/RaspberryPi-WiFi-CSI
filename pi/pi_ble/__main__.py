"""Entry point for pi_ble BLE GATT server.

Usage: python3 -m pi_ble [--config-dir DIR] [--log-level LEVEL]

Wires together: PinAuthenticator, WiFiManager, StreamManager,
CommandHandler, and the BlueZ GATT server.
"""

import argparse
import logging
import signal
import sys
from pathlib import Path

from .auth import BLESession, PinAuthenticator
from .config import DEFAULT_CONFIG_DIR
from .protocol import CommandHandler, chunk_response
from .server import PiSetupApplication, run_server
from .stream import StreamManager
from .wifi import WiFiManager

logger = logging.getLogger("pi_ble")

# Global reference for signal handler
_mainloop_quit = None


def main():
    parser = argparse.ArgumentParser(description="Pi BLE Setup Server")
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=DEFAULT_CONFIG_DIR,
        help=f"Config directory (default: {DEFAULT_CONFIG_DIR})",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )

    logger.info("Starting pi_ble server (config=%s)", args.config_dir)

    # Create managers
    auth = PinAuthenticator(args.config_dir)
    wifi = WiFiManager()
    stream = StreamManager()
    handler = CommandHandler(auth, wifi, stream)
    session = BLESession()

    # Reference to the BLE app (set in on_started callback)
    ble_app: PiSetupApplication | None = None

    def on_command(cmd: str):
        """Handle a BLE command from iOS."""
        nonlocal ble_app
        if ble_app is None:
            logger.error("BLE app not initialized, dropping command: %s", cmd)
            return

        result = handler.handle(cmd, session)

        # WIFI:SCAN returns a list of chunks for the WiFi Networks characteristic
        if cmd.strip().upper().startswith("WIFI:SCAN"):
            if isinstance(result, list):
                for chunk in result:
                    ble_app.send_wifi_networks(chunk)
            else:
                ble_app.send_wifi_networks(result)
            ble_app.send_response("OK:SCANNING")
            return

        # WIFI:STATUS updates the WiFi Status characteristic
        if cmd.strip().upper().startswith("WIFI:STATUS"):
            if isinstance(result, str):
                ble_app.send_wifi_status(result)
                ble_app.send_response(f"OK:{result}")
            else:
                ble_app.send_response("ERR:BAD_WIFI_STATUS")
            return

        # STREAM:STATUS updates the Stream Status characteristic
        if cmd.strip().upper().startswith("STREAM:STATUS"):
            if isinstance(result, str):
                ble_app.send_stream_status(result)
                ble_app.send_response(f"OK:{result}")
            else:
                ble_app.send_response("ERR:BAD_STREAM_STATUS")
            return

        # All other commands: send response on Response characteristic
        if isinstance(result, list):
            for chunk in result:
                ble_app.send_response(chunk)
        else:
            ble_app.send_response(result)

        # Proactive status push after auth success (both SET and VERIFY)
        cmd_upper = cmd.strip().upper()
        is_auth_success = (
            (cmd_upper.startswith("AUTH:VERIFY") and isinstance(result, str) and "VERIFIED" in result)
            or (cmd_upper.startswith("AUTH:SET") and isinstance(result, str) and "PIN_SET" in result)
        )
        if is_auth_success:
            st = "RUNNING" if stream.is_running() else "STOPPED"
            ble_app.send_stream_status(st)
            wifi_st = wifi.status()
            if wifi_st["connected"]:
                zt = wifi_st["zt_ip"] or "NONE"
                wlan = wifi_st["wlan_ip"] or "NONE"
                ble_app.send_wifi_status(f"CONNECTED:{wifi_st['ssid']}:{zt}:{wlan}")
            else:
                ble_app.send_wifi_status("DISCONNECTED")
            logger.info("Pushed status after auth: stream=%s wifi=%s", st, wifi_st.get("ssid", "none"))

        # Proactive stream status push after start/stop
        if cmd_upper.startswith("STREAM:START") or cmd_upper.startswith("STREAM:STOP"):
            st = "RUNNING" if stream.is_running() else "STOPPED"
            ble_app.send_stream_status(st)

    def on_started(app: PiSetupApplication):
        """Called when BLE server is registered and ready."""
        nonlocal ble_app
        ble_app = app
        logger.info("BLE GATT server ready, accepting connections")

    # Run (blocks until SIGINT/SIGTERM)
    run_server(on_command=on_command, on_started=on_started)


if __name__ == "__main__":
    main()
