"""PIN authentication for BLE sessions.

PinAuthenticator manages persistent PIN storage and verification.
BLESession tracks per-connection authentication state.
"""

import logging
from pathlib import Path

from . import config

logger = logging.getLogger(__name__)

PIN_MIN_LENGTH = 4
PIN_MAX_LENGTH = 8


def _validate_pin_format(pin: str) -> str | None:
    """Return error message if PIN is invalid, else None."""
    if not pin.isdigit():
        return "PIN must contain only digits"
    if len(pin) < PIN_MIN_LENGTH:
        return f"PIN must be at least {PIN_MIN_LENGTH} digits"
    if len(pin) > PIN_MAX_LENGTH:
        return f"PIN must be at most {PIN_MAX_LENGTH} digits"
    return None


class PinAuthenticator:
    """Manages PIN lifecycle: set, verify, change."""

    def __init__(self, config_dir: Path = config.DEFAULT_CONFIG_DIR):
        self._config_dir = config_dir
        self._pin = config.load_pin(config_dir)
        logger.info("PinAuthenticator: PIN %s", "loaded" if self._pin else "not set")

    def has_pin(self) -> bool:
        return self._pin is not None

    def verify(self, pin: str) -> bool:
        if self._pin is None:
            return False
        return self._pin == pin

    def set_initial_pin(self, pin: str) -> tuple[bool, str]:
        """Set PIN for first time. Returns (success, message)."""
        if self._pin is not None:
            return False, "PIN_ALREADY_SET"
        err = _validate_pin_format(pin)
        if err:
            return False, err
        config.save_pin(pin, self._config_dir)
        self._pin = pin
        logger.info("Initial PIN set")
        return True, "PIN_SET"

    def change_pin(self, old_pin: str, new_pin: str) -> tuple[bool, str]:
        """Change PIN. Requires correct old PIN. Returns (success, message)."""
        if self._pin is None:
            return False, "NO_PIN_SET"
        if self._pin != old_pin:
            return False, "WRONG_PIN"
        err = _validate_pin_format(new_pin)
        if err:
            return False, err
        config.save_pin(new_pin, self._config_dir)
        self._pin = new_pin
        logger.info("PIN changed")
        return True, "PIN_CHANGED"


class BLESession:
    """Per-connection authentication state. Resets on BLE disconnect."""

    def __init__(self):
        self.authenticated = False

    def authenticate(self, auth: PinAuthenticator, pin: str) -> bool:
        """Verify PIN and mark session as authenticated."""
        if auth.verify(pin):
            self.authenticated = True
            return True
        return False

    def require_auth(self) -> bool:
        """Check if session is authenticated. Returns False if not."""
        return self.authenticated

    def reset(self):
        """Reset session state (called on BLE disconnect)."""
        self.authenticated = False


def _self_test():
    import tempfile
    from pathlib import Path
    d = Path(tempfile.mkdtemp())
    try:
        auth = PinAuthenticator(d)
        assert not auth.has_pin()

        # Reject too-short PIN
        ok, msg = auth.set_initial_pin("12")
        assert not ok and "at least" in msg

        # Reject non-numeric
        ok, msg = auth.set_initial_pin("abcd")
        assert not ok and "digits" in msg

        # Set valid PIN
        ok, msg = auth.set_initial_pin("1234")
        assert ok and msg == "PIN_SET"
        assert auth.has_pin()

        # Can't set again
        ok, msg = auth.set_initial_pin("5678")
        assert not ok and msg == "PIN_ALREADY_SET"

        # Verify
        assert auth.verify("1234")
        assert not auth.verify("0000")

        # Session
        sess = BLESession()
        assert not sess.require_auth()
        assert not sess.authenticate(auth, "0000")
        assert not sess.require_auth()
        assert sess.authenticate(auth, "1234")
        assert sess.require_auth()
        sess.reset()
        assert not sess.require_auth()

        # Change PIN
        ok, msg = auth.change_pin("0000", "5678")
        assert not ok and msg == "WRONG_PIN"
        ok, msg = auth.change_pin("1234", "5678")
        assert ok and msg == "PIN_CHANGED"
        assert auth.verify("5678")
        assert not auth.verify("1234")

        # Persistence — reload from disk
        auth2 = PinAuthenticator(d)
        assert auth2.has_pin()
        assert auth2.verify("5678")

        print("auth: all tests passed")
    finally:
        import shutil
        shutil.rmtree(d)


if __name__ == "__main__":
    _self_test()
