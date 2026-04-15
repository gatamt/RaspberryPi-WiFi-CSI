"""BlueZ D-Bus BLE GATT server.

Implements the BlueZ GATT Application pattern using dbus-python + GLib.
Registers one Service with 5 Characteristics and an LE Advertisement.

Requires: system Python with dbus-python, PyGObject (pre-installed on Pi OS).
Must run as root for BlueZ GATT registration.
"""

import logging
from typing import Callable

import dbus
import dbus.exceptions
import dbus.mainloop.glib
import dbus.service
from gi.repository import GLib

logger = logging.getLogger(__name__)

# BlueZ D-Bus constants
BLUEZ = "org.bluez"
ADAPTER_IFACE = "org.bluez.Adapter1"
GATT_MANAGER = "org.bluez.GattManager1"
LE_ADV_MANAGER = "org.bluez.LEAdvertisingManager1"
GATT_SERVICE_IFACE = "org.bluez.GattService1"
GATT_CHRC_IFACE = "org.bluez.GattCharacteristic1"
LE_ADV_IFACE = "org.bluez.LEAdvertisement1"
DBUS_OM_IFACE = "org.freedesktop.DBus.ObjectManager"
DBUS_PROP_IFACE = "org.freedesktop.DBus.Properties"

# Our custom UUIDs
SERVICE_UUID = "a0000001-b000-c000-d000-e00000000000"
COMMAND_UUID = "a0000002-b000-c000-d000-e00000000000"
RESPONSE_UUID = "a0000003-b000-c000-d000-e00000000000"
WIFI_NETWORKS_UUID = "a0000004-b000-c000-d000-e00000000000"
WIFI_STATUS_UUID = "a0000005-b000-c000-d000-e00000000000"
STREAM_STATUS_UUID = "a0000006-b000-c000-d000-e00000000000"

LOCAL_NAME = "GataPi5"
BASE_PATH = "/org/bluez/pisetup"


# ─── D-Bus helpers ────────────────────────────────────────

def find_adapter(bus: dbus.SystemBus) -> str:
    """Find the first BlueZ adapter object path (e.g. /org/bluez/hci0)."""
    manager = dbus.Interface(
        bus.get_object(BLUEZ, "/"), DBUS_OM_IFACE
    )
    for path, ifaces in manager.GetManagedObjects().items():
        if GATT_MANAGER in ifaces:
            return path
    raise RuntimeError("No BlueZ adapter with GATT support found")


# ─── GATT Application (ObjectManager) ────────────────────

class Application(dbus.service.Object):
    """Root GATT Application that registers services with BlueZ."""

    def __init__(self, bus: dbus.SystemBus):
        self.path = BASE_PATH
        self.services: list["Service"] = []
        dbus.service.Object.__init__(self, bus, self.path)

    def add_service(self, service: "Service"):
        self.services.append(service)

    @dbus.service.method(DBUS_OM_IFACE, out_signature="a{oa{sa{sv}}}")
    def GetManagedObjects(self):
        response = dbus.Dictionary({}, signature="oa{sa{sv}}")
        for svc in self.services:
            response[dbus.ObjectPath(svc.path)] = svc.get_properties()
            for chrc in svc.characteristics:
                response[dbus.ObjectPath(chrc.path)] = chrc.get_properties()
        return response


# ─── GATT Service ─────────────────────────────────────────

class Service(dbus.service.Object):
    def __init__(self, bus: dbus.SystemBus, index: int, uuid: str, primary: bool = True):
        self.path = f"{BASE_PATH}/service{index}"
        self.bus = bus
        self.uuid = uuid
        self.primary = primary
        self.characteristics: list["Characteristic"] = []
        dbus.service.Object.__init__(self, bus, self.path)

    def add_characteristic(self, chrc: "Characteristic"):
        self.characteristics.append(chrc)

    def get_properties(self) -> dict:
        return dbus.Dictionary({
            GATT_SERVICE_IFACE: dbus.Dictionary({
                "UUID": dbus.String(self.uuid),
                "Primary": dbus.Boolean(self.primary),
                "Characteristics": dbus.Array(
                    [dbus.ObjectPath(c.path) for c in self.characteristics],
                    signature="o",
                ),
            }, signature="sv"),
        }, signature="sa{sv}")

    @dbus.service.method(DBUS_PROP_IFACE, in_signature="s", out_signature="a{sv}")
    def GetAll(self, iface):
        if iface == GATT_SERVICE_IFACE:
            return self.get_properties()[GATT_SERVICE_IFACE]
        raise dbus.exceptions.DBusException(
            "org.freedesktop.DBus.Error.InvalidArgs",
            f"Unknown interface: {iface}",
        )


# ─── GATT Characteristic ─────────────────────────────────

class Characteristic(dbus.service.Object):
    def __init__(
        self,
        bus: dbus.SystemBus,
        index: int,
        uuid: str,
        flags: list[str],
        service: Service,
    ):
        self.path = f"{service.path}/char{index}"
        self.bus = bus
        self.uuid = uuid
        self.flags = flags
        self.service = service
        self.value: bytes = b""
        self._notifying = False
        dbus.service.Object.__init__(self, bus, self.path)
        service.add_characteristic(self)

    def get_properties(self) -> dict:
        return dbus.Dictionary({
            GATT_CHRC_IFACE: dbus.Dictionary({
                "Service": dbus.ObjectPath(self.service.path),
                "UUID": dbus.String(self.uuid),
                "Flags": dbus.Array(self.flags, signature="s"),
                "Descriptors": dbus.Array([], signature="o"),
            }, signature="sv"),
        }, signature="sa{sv}")

    @dbus.service.method(DBUS_PROP_IFACE, in_signature="s", out_signature="a{sv}")
    def GetAll(self, iface):
        if iface == GATT_CHRC_IFACE:
            return self.get_properties()[GATT_CHRC_IFACE]
        raise dbus.exceptions.DBusException(
            "org.freedesktop.DBus.Error.InvalidArgs",
            f"Unknown interface: {iface}",
        )

    @dbus.service.method(GATT_CHRC_IFACE, in_signature="a{sv}", out_signature="ay")
    def ReadValue(self, options):
        return dbus.Array(self.value, signature="y")

    @dbus.service.method(GATT_CHRC_IFACE, in_signature="aya{sv}")
    def WriteValue(self, value, options):
        # Override in subclass
        pass

    @dbus.service.method(GATT_CHRC_IFACE)
    def StartNotify(self):
        if self._notifying:
            return
        self._notifying = True
        logger.debug("StartNotify on %s", self.uuid)

    @dbus.service.method(GATT_CHRC_IFACE)
    def StopNotify(self):
        self._notifying = False
        logger.debug("StopNotify on %s", self.uuid)

    @dbus.service.signal(DBUS_PROP_IFACE, signature="sa{sv}as")
    def PropertiesChanged(self, iface, changed, invalidated):
        pass

    def send_notification(self, data: bytes):
        """Update value and emit PropertiesChanged signal if notifying."""
        self.value = data
        if self._notifying:
            self.PropertiesChanged(
                GATT_CHRC_IFACE,
                {"Value": dbus.Array(data, signature="y")},
                [],
            )


# ─── BlueZ Agent (NoInputNoOutput → Just Works pairing) ──

AGENT_IFACE = "org.bluez.Agent1"
AGENT_MANAGER_IFACE = "org.bluez.AgentManager1"
AGENT_PATH = f"{BASE_PATH}/agent"


class NoInputNoOutputAgent(dbus.service.Object):
    """Auto-accept pairing agent. Prevents iOS pairing popup for BLE."""

    @dbus.service.method(AGENT_IFACE, in_signature="", out_signature="")
    def Release(self):
        logger.info("Agent released")

    @dbus.service.method(AGENT_IFACE, in_signature="os", out_signature="")
    def AuthorizeService(self, device, uuid):
        logger.info("Agent: AuthorizeService %s %s", device, uuid)

    @dbus.service.method(AGENT_IFACE, in_signature="o", out_signature="s")
    def RequestPinCode(self, device):
        logger.info("Agent: RequestPinCode %s", device)
        return "0000"

    @dbus.service.method(AGENT_IFACE, in_signature="o", out_signature="u")
    def RequestPasskey(self, device):
        logger.info("Agent: RequestPasskey %s", device)
        return dbus.UInt32(0)

    @dbus.service.method(AGENT_IFACE, in_signature="ouq", out_signature="")
    def DisplayPasskey(self, device, passkey, entered):
        pass

    @dbus.service.method(AGENT_IFACE, in_signature="ou", out_signature="")
    def RequestConfirmation(self, device, passkey):
        logger.info("Agent: auto-confirming pairing for %s", device)

    @dbus.service.method(AGENT_IFACE, in_signature="o", out_signature="")
    def RequestAuthorization(self, device):
        logger.info("Agent: auto-authorizing %s", device)

    @dbus.service.method(AGENT_IFACE, in_signature="", out_signature="")
    def Cancel(self):
        logger.info("Agent: pairing cancelled")


def _register_agent(bus: dbus.SystemBus):
    """Register a NoInputNoOutput agent as the default BlueZ agent."""
    NoInputNoOutputAgent(bus, AGENT_PATH)
    agent_mgr = dbus.Interface(
        bus.get_object(BLUEZ, "/org/bluez"), AGENT_MANAGER_IFACE
    )
    agent_mgr.RegisterAgent(AGENT_PATH, "NoInputNoOutput")
    agent_mgr.RequestDefaultAgent(AGENT_PATH)
    logger.info("NoInputNoOutput agent registered as default")


# ─── LE Advertisement ────────────────────────────────────

class Advertisement(dbus.service.Object):
    def __init__(self, bus: dbus.SystemBus, index: int):
        self.path = f"{BASE_PATH}/advertisement{index}"
        self.bus = bus
        self.ad_type = "peripheral"
        self.service_uuids = [SERVICE_UUID]
        self.local_name = LOCAL_NAME
        dbus.service.Object.__init__(self, bus, self.path)

    @dbus.service.method(DBUS_PROP_IFACE, in_signature="s", out_signature="a{sv}")
    def GetAll(self, iface):
        if iface != LE_ADV_IFACE:
            raise dbus.exceptions.DBusException(
                "org.freedesktop.DBus.Error.InvalidArgs",
                f"Unknown interface: {iface}",
            )
        return {
            "Type": self.ad_type,
            "ServiceUUIDs": dbus.Array(self.service_uuids, signature="s"),
            "LocalName": dbus.String(self.local_name),
            "Includes": dbus.Array(["tx-power"], signature="s"),
        }

    @dbus.service.method(LE_ADV_IFACE, in_signature="", out_signature="")
    def Release(self):
        logger.info("Advertisement released")


# ─── Pi Setup GATT Application ───────────────────────────

class CommandCharacteristic(Characteristic):
    """Write-only characteristic for receiving commands from iOS."""

    def __init__(self, bus, service, on_command: Callable[[str], None]):
        super().__init__(bus, 0, COMMAND_UUID, ["write", "write-without-response"], service)
        self._on_command = on_command

    @dbus.service.method(GATT_CHRC_IFACE, in_signature="aya{sv}")
    def WriteValue(self, value, options):
        cmd = bytes(value).decode("utf-8", errors="replace")
        logger.info("BLE command received: %s", cmd[:80])
        self._on_command(cmd)


class NotifyCharacteristic(Characteristic):
    """Read+Notify characteristic for sending data to iOS."""

    def __init__(self, bus, index, uuid, service):
        super().__init__(bus, index, uuid, ["read", "notify"], service)


class PiSetupApplication:
    """High-level BLE application wiring GATT server + advertisement."""

    def __init__(
        self,
        bus: dbus.SystemBus,
        on_command: Callable[[str], None],
    ):
        self.bus = bus

        # GATT Application
        self.app = Application(bus)
        self.service = Service(bus, 0, SERVICE_UUID)
        self.app.add_service(self.service)

        # Characteristics
        self.command_chrc = CommandCharacteristic(bus, self.service, on_command)
        self.response_chrc = NotifyCharacteristic(bus, 1, RESPONSE_UUID, self.service)
        self.wifi_networks_chrc = NotifyCharacteristic(bus, 2, WIFI_NETWORKS_UUID, self.service)
        self.wifi_status_chrc = NotifyCharacteristic(bus, 3, WIFI_STATUS_UUID, self.service)
        self.stream_status_chrc = NotifyCharacteristic(bus, 4, STREAM_STATUS_UUID, self.service)

        # Advertisement
        self.adv = Advertisement(bus, 0)

    def send_response(self, data: str):
        self.response_chrc.send_notification(data.encode())

    def send_wifi_networks(self, data: str):
        self.wifi_networks_chrc.send_notification(data.encode())

    def send_wifi_status(self, data: str):
        self.wifi_status_chrc.send_notification(data.encode())

    def send_stream_status(self, data: str):
        self.stream_status_chrc.send_notification(data.encode())

    def register(self, adapter_path: str):
        """Register GATT application and advertisement with BlueZ."""
        # Make adapter discoverable
        adapter_props = dbus.Interface(
            self.bus.get_object(BLUEZ, adapter_path), DBUS_PROP_IFACE
        )
        adapter_props.Set(ADAPTER_IFACE, "Discoverable", dbus.Boolean(True))
        adapter_props.Set(ADAPTER_IFACE, "Alias", dbus.String(LOCAL_NAME))
        logger.info("Adapter set discoverable with name '%s'", LOCAL_NAME)

        # Register GATT application
        gatt_mgr = dbus.Interface(
            self.bus.get_object(BLUEZ, adapter_path), GATT_MANAGER
        )
        gatt_mgr.RegisterApplication(
            self.app.path, {},
            reply_handler=lambda: logger.info("GATT application registered"),
            error_handler=lambda e: logger.error("GATT registration failed: %s", e),
        )

        # Register advertisement
        adv_mgr = dbus.Interface(
            self.bus.get_object(BLUEZ, adapter_path), LE_ADV_MANAGER
        )
        adv_mgr.RegisterAdvertisement(
            self.adv.path, {},
            reply_handler=lambda: logger.info("Advertisement registered"),
            error_handler=lambda e: logger.error("Advertisement registration failed: %s", e),
        )


def run_server(
    on_command: Callable[[str], None],
    on_started: Callable[["PiSetupApplication"], None] | None = None,
):
    """Initialize D-Bus, register GATT application, and run GLib main loop.

    Args:
        on_command: called with the UTF-8 command string when iOS writes to Command characteristic
        on_started: called with the PiSetupApplication after registration, for wiring up response sending
    """
    dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
    bus = dbus.SystemBus()
    adapter_path = find_adapter(bus)
    logger.info("Using adapter: %s", adapter_path)

    # Register NoInputNoOutput agent before GATT app — prevents iOS pairing popup
    try:
        _register_agent(bus)
    except Exception as exc:
        logger.warning("Agent registration failed (non-fatal): %s", exc)

    app = PiSetupApplication(bus, on_command)
    app.register(adapter_path)

    if on_started:
        on_started(app)

    logger.info("BLE GATT server running. Waiting for connections...")
    mainloop = GLib.MainLoop()
    try:
        mainloop.run()
    except KeyboardInterrupt:
        logger.info("BLE server shutting down")
        mainloop.quit()
