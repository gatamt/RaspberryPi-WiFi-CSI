"""Pi Cam 3 Wide capture via picamera2.

The Wide module (IMX708) has three native sensor modes:
  - 4608x2592 @ 14 fps  (full resolution, full FOV)
  - 2304x1296 @ 56 fps  (2x2 binned, FULL WIDE FOV)  <-- we want this
  - 1536x864  @ 120 fps (cropped, loses wide FOV)

We pick the 2304x1296 sensor mode and let libcamera downscale to 1280x720.
``ScalerCrop`` is set explicitly to the full sensor rectangle so there is
no cropping anywhere — the wide 120° field of view is preserved and only
scaled down.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

LOG = logging.getLogger("pi_streamer.camera")

# IMX708 full sensor rectangle (pixels). Used for ScalerCrop to force full FOV.
IMX708_FULL_SENSOR = (0, 0, 4608, 2592)
CAMERA_RETRY_DELAY_S = 0.35
CAMERA_PROFILE_ATTEMPTS = 2


@dataclass(frozen=True)
class CameraConfig:
    width: int = 1280
    height: int = 720
    fps: int = 30
    # Pick the 2304x1296 sensor mode by hinting the raw stream size
    sensor_raw_width: int = 2304
    sensor_raw_height: int = 1296


@dataclass(frozen=True)
class CameraStartupProfile:
    name: str
    include_lores: bool
    raw_size: Optional[Tuple[int, int]]
    scaler_crop: Optional[Tuple[int, int, int, int]]


class PiCamera:
    """picamera2 wrapper that always emits RGB888 frames at fixed size."""

    def __init__(
        self,
        config: CameraConfig = CameraConfig(),
        lores_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        self.config = config
        self._lores_size = lores_size
        self._has_lores = False
        self._picam2 = None
        self._frame_id = 0
        self._profile_name = "unconfigured"

    def _startup_profiles(self) -> list[CameraStartupProfile]:
        profiles = [
            CameraStartupProfile(
                name="preferred",
                include_lores=self._lores_size is not None,
                raw_size=(self.config.sensor_raw_width, self.config.sensor_raw_height),
                scaler_crop=IMX708_FULL_SENSOR,
            ),
            CameraStartupProfile(
                name="no-scalercrop",
                include_lores=self._lores_size is not None,
                raw_size=(self.config.sensor_raw_width, self.config.sensor_raw_height),
                scaler_crop=None,
            ),
            CameraStartupProfile(
                name="auto-raw",
                include_lores=self._lores_size is not None,
                raw_size=None,
                scaler_crop=None,
            ),
            CameraStartupProfile(
                name="main-only",
                include_lores=False,
                raw_size=None,
                scaler_crop=None,
            ),
        ]

        deduped: list[CameraStartupProfile] = []
        seen = set()
        for profile in profiles:
            key = (profile.include_lores, profile.raw_size, profile.scaler_crop)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(profile)
        return deduped

    def _config_kwargs(self, profile: CameraStartupProfile) -> dict:
        controls_dict = {
            "FrameDurationLimits": (
                int(1_000_000 / self.config.fps),
                int(1_000_000 / self.config.fps),
            ),
            "AeEnable": True,
            "AwbEnable": True,
        }
        if profile.scaler_crop is not None:
            controls_dict["ScalerCrop"] = profile.scaler_crop

        cfg_kwargs = {
            "main": {"size": (self.config.width, self.config.height), "format": "RGB888"},
            "controls": controls_dict,
            "buffer_count": 4,
        }
        if profile.raw_size is not None:
            cfg_kwargs["raw"] = {"size": profile.raw_size}
        if profile.include_lores and self._lores_size is not None:
            cfg_kwargs["lores"] = {"size": self._lores_size, "format": "YUV420"}
        return cfg_kwargs

    def start(self) -> None:
        # Imported lazily so the module is importable on dev machines without picamera2.
        from picamera2 import Picamera2  # type: ignore
        from libcamera import controls  # type: ignore

        last_error: Exception | None = None

        for profile in self._startup_profiles():
            for attempt in range(1, CAMERA_PROFILE_ATTEMPTS + 1):
                try:
                    self._picam2 = Picamera2()
                    video_config = self._picam2.create_video_configuration(
                        **self._config_kwargs(profile)
                    )
                    self._picam2.configure(video_config)
                    self._picam2.start()

                    # Force one request through the pipeline so broken configs fail early.
                    request = self._picam2.capture_request()
                    request.release()

                    actual_cfg = self._picam2.camera_configuration()
                    lores_info = actual_cfg.get("lores", {}).get("size")
                    self._has_lores = lores_info is not None
                    self._profile_name = profile.name
                    LOG.info(
                        "camera started: profile=%s main=%s lores=%s raw=%s format=%s",
                        profile.name,
                        actual_cfg["main"]["size"],
                        lores_info,
                        actual_cfg.get("raw", {}).get("size"),
                        actual_cfg["main"]["format"],
                    )
                    if self._lores_size is not None and not self._has_lores:
                        LOG.warning(
                            "lores stream requested at %s but picamera2 did not configure it; "
                            "inference will use full-resolution frames",
                            self._lores_size,
                        )

                    # Try to lock continuous AF if the sensor supports it (no-op if not)
                    try:
                        self._picam2.set_controls(
                            {"AfMode": controls.AfModeEnum.Continuous}  # type: ignore[attr-defined]
                        )
                    except Exception as exc:  # noqa: BLE001
                        LOG.debug("continuous AF not available: %s", exc)
                    return
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    LOG.warning(
                        "camera startup profile=%s attempt=%d failed: %s",
                        profile.name,
                        attempt,
                        exc,
                    )
                    self.stop()
                    time.sleep(CAMERA_RETRY_DELAY_S)

        raise RuntimeError(
            f"Failed to start camera after trying all startup profiles: {last_error}"
        ) from last_error

    def stop(self) -> None:
        if self._picam2 is not None:
            try:
                self._picam2.stop()
                self._picam2.close()
            except Exception as exc:  # noqa: BLE001
                LOG.warning("camera stop error: %s", exc)
            self._picam2 = None

    def capture(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Capture one frame, returning ``(main, lores)```.

        ``main`` is the full-resolution RGB888 (actually BGR memory layout)
        array of shape ``(H, W, 3)``.  ``lores`` is the ISP-downscaled BGR
        array when a lores stream was configured, or ``None`` otherwise.
        """
        if self._picam2 is None:
            raise RuntimeError("camera not started")
        if self._has_lores:
            import cv2 as cv2_mod  # noqa: F811
            request = self._picam2.capture_request()
            try:
                main = request.make_array("main")
                lores_yuv = request.make_array("lores")
                lores = cv2_mod.cvtColor(lores_yuv, cv2_mod.COLOR_YUV420p2BGR)
            finally:
                request.release()
        else:
            main = self._picam2.capture_array("main")
            lores = None
        self._frame_id += 1
        return main, lores

    @property
    def next_frame_id(self) -> int:
        return self._frame_id + 1

    def __enter__(self) -> "PiCamera":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self.stop()
