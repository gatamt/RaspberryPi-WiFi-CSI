"""pi_streamer process lifecycle management.

Starts, stops, and monitors the pi_streamer video pipeline.
CRITICAL: Stop uses kill -9 (SIGKILL). Regular SIGTERM does NOT work
because the process hangs in Hailo device calls.
"""

import logging
import os
import re
import subprocess
import time
from pathlib import Path

logger = logging.getLogger(__name__)

WORK_DIR = Path(__file__).resolve().parents[1]
# Switched to the C11 low-level streamer (pi_streamer_c) — the Python
# pi_streamer.main is retained only as a reference/fallback. If the C
# binary is missing, we fall back to the venv Python module.
STREAMER_BINARY = WORK_DIR / "pi_streamer_c" / "build" / "pi_streamer_c"
VENV_PYTHON = WORK_DIR / ".venv" / "bin" / "python"
STREAMER_MODULE = "pi_streamer.main"  # legacy / fallback
# pgrep pattern that matches either launcher so stop/status works across
# a transition period.
STREAMER_PGREP_PATTERN = "pi_streamer_c|pi_streamer.main"
LOG_FILE = WORK_DIR / "pi_streamer.log"
POSE_HEF_PATH = Path("/home/pi/models/yolov8m_pose.hef")
OBJECT_HEF_PATH = Path("/home/pi/models/yolo26m.hef")
HAND_HEF_PATH = Path("/home/pi/models/hand_landmark_lite.hef")

# Args for the C streamer (uses key=value style parsed by pi_pipeline_parse_args).
C_DEFAULT_ARGS = [
    "--log-level=INFO",
    "--width=1280",
    "--height=720",
    "--fps=30",
    "--bitrate-bps=2500000",
    "--gop-size=30",
    "--udp-port=3334",
]

# Legacy Python args kept as fallback only.
PYTHON_DEFAULT_ARGS = [
    "--log-level", "INFO",
    "--hand-roi", "wrist",
    "--hand-hef", str(HAND_HEF_PATH),
]

HAILO_RELEASE_WAIT = 1.0  # seconds to wait for /dev/hailo0 release after kill
STARTUP_CHECK_WAIT = 2.0  # seconds to wait before checking if process is alive
KILL_VERIFY_TIMEOUT = 3.0  # max seconds to wait for process to die
LOG_TAIL_LINES = 40


def _required_paths() -> list[Path]:
    return [POSE_HEF_PATH, OBJECT_HEF_PATH, HAND_HEF_PATH]


def _use_c_binary() -> bool:
    """True if the C streamer binary is present and executable."""
    return STREAMER_BINARY.exists() and os.access(STREAMER_BINARY, os.X_OK)


def _startup_preflight() -> str | None:
    if not WORK_DIR.is_dir():
        return f"STARTUP_FAILED: workdir missing ({WORK_DIR})"
    if _use_c_binary():
        # HEF files are still needed for Hailo inference models.
        for path in _required_paths():
            if not path.exists():
                return f"STARTUP_FAILED: missing model ({path})"
        return None
    # Legacy Python path
    if not VENV_PYTHON.exists():
        return f"STARTUP_FAILED: python missing ({VENV_PYTHON})"
    for path in _required_paths():
        if not path.exists():
            return f"STARTUP_FAILED: missing model ({path})"
    return None


def _normalize_log_line(line: str) -> str:
    stripped = line.strip()
    stripped = re.sub(r"^\d{2}:\d{2}:\d{2}\s+\w+\s+\S+\s+", "", stripped)
    stripped = re.sub(r"^\[[^\]]+\]\s+\[[^\]]+\]\s+\w+\s+", "", stripped)
    return stripped


def _startup_failure_reason() -> str:
    if not LOG_FILE.exists():
        return "startup failed (no log output)"

    try:
        lines = LOG_FILE.read_text(errors="replace").splitlines()[-LOG_TAIL_LINES:]
    except OSError as exc:
        return f"startup failed (log unreadable: {exc})"

    for line in reversed(lines):
        normalized = _normalize_log_line(line)
        if "startup failed:" in normalized:
            return normalized.split("startup failed:", maxsplit=1)[1].strip()[:160]
        if normalized.startswith("Failed to start camera:"):
            return normalized[:160]
        if "Failed to acquire camera:" in normalized:
            return normalized[:160]
        if "Failed to open Hailo" in normalized or "failed to open Hailo" in normalized:
            return normalized[:160]
        if normalized:
            fallback = normalized[:160]
    return fallback if "fallback" in locals() else "startup failed"


def _pgrep() -> int | None:
    """Find pi_streamer PID via pgrep. Matches BOTH the C binary and the
    legacy Python module so we can stop/status either one during the
    transition period. Returns PID or None."""
    try:
        r = subprocess.run(
            ["pgrep", "-f", STREAMER_PGREP_PATTERN],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            # May return multiple PIDs (parent + child); take the first
            return int(r.stdout.strip().splitlines()[0])
    except (subprocess.TimeoutExpired, ValueError):
        pass
    return None


class StreamManager:
    """Manages pi_streamer process lifecycle."""

    def is_running(self) -> bool:
        return _pgrep() is not None

    def get_pid(self) -> int | None:
        return _pgrep()

    def start(self) -> tuple[bool, str]:
        """Start pi_streamer. Returns (success, message)."""
        if self.is_running():
            logger.info("Streamer already running (PID %s)", self.get_pid())
            return True, "ALREADY_RUNNING"

        preflight_error = _startup_preflight()
        if preflight_error is not None:
            logger.error(preflight_error)
            return False, preflight_error

        # Build the command based on which streamer is available.
        # The C binary is preferred (isolcpus + SCHED_FIFO + 4 real
        # backends). pi-ble.service runs as root so it inherits
        # CAP_SYS_NICE for the sched_setscheduler calls — no sudo
        # gymnastics needed on the C path.
        if _use_c_binary():
            cmd = [str(STREAMER_BINARY)] + C_DEFAULT_ARGS
        else:
            # Legacy fallback — launch the Python module as pi so
            # picamera2 / /dev/hailo0 perms work.
            cmd = [
                "sudo",
                "-u",
                "pi",
                "env",
                f"PATH={VENV_PYTHON.parent}:/usr/bin:/bin",
                str(VENV_PYTHON),
                "-m",
                STREAMER_MODULE,
            ] + PYTHON_DEFAULT_ARGS
        logger.info("Starting streamer: %s", " ".join(cmd))

        log_fd = None
        try:
            log_fd = open(LOG_FILE, "w", buffering=1)
            proc = subprocess.Popen(
                cmd,
                stdout=log_fd,
                stderr=subprocess.STDOUT,
                cwd=WORK_DIR,
                start_new_session=True,  # detach from our process group
            )
        except (OSError, FileNotFoundError) as e:
            logger.error("Failed to start streamer: %s", e)
            return False, f"STARTUP_FAILED: {e}"
        finally:
            if log_fd is not None:
                log_fd.close()

        # Wait and check if it stayed alive
        time.sleep(STARTUP_CHECK_WAIT)
        if proc.poll() is not None:
            reason = _startup_failure_reason()
            logger.error(
                "Streamer exited immediately with code %s (%s)",
                proc.returncode,
                reason,
            )
            return False, f"STARTUP_FAILED: {reason}"

        logger.info("Streamer started (PID %s)", proc.pid)
        return True, "STARTED"

    def stop(self) -> tuple[bool, str]:
        """Stop pi_streamer with pkill -9. Returns (success, message).

        Uses pkill -9 -f to kill ALL matching processes (bash wrapper + python child).
        SIGKILL is required because SIGTERM hangs in Hailo device calls.
        After killing, waits for /dev/hailo0 to be released.
        """
        if _pgrep() is None:
            logger.info("Streamer not running")
            return True, "NOT_RUNNING"

        logger.info("Killing all pi_streamer processes with pkill -9")
        subprocess.run(
            ["pkill", "-9", "-f", STREAMER_PGREP_PATTERN],
            capture_output=True, timeout=5,
        )

        # Verify all processes are dead
        deadline = time.monotonic() + KILL_VERIFY_TIMEOUT
        while time.monotonic() < deadline:
            if _pgrep() is None:
                break
            time.sleep(0.2)
        else:
            # Still alive — retry
            logger.warning("Processes still alive after %.1fs, retrying pkill -9", KILL_VERIFY_TIMEOUT)
            subprocess.run(
                ["pkill", "-9", "-f", STREAMER_PGREP_PATTERN],
                capture_output=True, timeout=5,
            )
            time.sleep(0.5)
            if _pgrep() is not None:
                logger.error("Failed to kill pi_streamer processes")
                return False, "KILL_FAILED"

        # Wait for /dev/hailo0 release
        logger.info("Waiting %.1fs for /dev/hailo0 release", HAILO_RELEASE_WAIT)
        time.sleep(HAILO_RELEASE_WAIT)

        logger.info("Streamer stopped")
        return True, "STOPPED"

    def restart(self) -> tuple[bool, str]:
        """Stop then start. Returns (success, message)."""
        ok, msg = self.stop()
        if not ok:
            return False, f"RESTART_FAILED: stop={msg}"
        return self.start()


def _self_test():
    sm = StreamManager()
    running = sm.is_running()
    pid = sm.get_pid()
    print(f"stream: self-test passed (running={running}, pid={pid})")


if __name__ == "__main__":
    _self_test()
