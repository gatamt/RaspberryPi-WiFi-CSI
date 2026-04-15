"""Persistent configuration storage for pi_ble.

Stores PIN and settings in ~/.pi_ble_config/ with restricted permissions.
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_DIR = Path.home() / ".pi_ble_config"
PIN_FILE = "pin.json"


def _ensure_config_dir(config_dir: Path) -> None:
    if not config_dir.exists():
        config_dir.mkdir(mode=0o700, parents=True)
        logger.info("Created config dir: %s", config_dir)


def _pin_path(config_dir: Path) -> Path:
    return config_dir / PIN_FILE


def load_pin(config_dir: Path = DEFAULT_CONFIG_DIR) -> str | None:
    """Load stored PIN, or None if no PIN has been set."""
    path = _pin_path(config_dir)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return data.get("pin")
    except (json.JSONDecodeError, OSError) as e:
        logger.error("Failed to read PIN file %s: %s", path, e)
        return None


def save_pin(pin: str, config_dir: Path = DEFAULT_CONFIG_DIR) -> None:
    """Save PIN to config dir with restricted permissions."""
    _ensure_config_dir(config_dir)
    path = _pin_path(config_dir)
    data = {"pin": pin, "set_at": datetime.now(timezone.utc).isoformat()}
    # Write atomically via temp file
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    os.chmod(tmp, 0o600)
    tmp.rename(path)
    logger.info("PIN saved to %s", path)


def has_pin(config_dir: Path = DEFAULT_CONFIG_DIR) -> bool:
    """Check if a PIN has been configured."""
    return load_pin(config_dir) is not None


def _self_test():
    import tempfile
    d = Path(tempfile.mkdtemp())
    try:
        assert not has_pin(d), "fresh dir should have no PIN"
        assert load_pin(d) is None
        save_pin("1234", d)
        assert has_pin(d)
        assert load_pin(d) == "1234"
        # Check file permissions
        p = _pin_path(d)
        assert oct(p.stat().st_mode & 0o777) == "0o600", f"bad perms: {oct(p.stat().st_mode)}"
        # Overwrite
        save_pin("5678", d)
        assert load_pin(d) == "5678"
        print("config: all tests passed")
    finally:
        import shutil
        shutil.rmtree(d)


if __name__ == "__main__":
    _self_test()
