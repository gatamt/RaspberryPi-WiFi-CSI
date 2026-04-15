"""Entry point for the Raspberry Pi video streamer.

Wires the camera → Hailo inference → overlay → encoder → UDP pipeline
and runs the main loop. Designed to be launched via::

    python -m pi_streamer.main

See ``protocol.md`` for the wire format, and ``PLAN.md`` for the full plan.
"""

from __future__ import annotations

import argparse
import logging
import signal
import subprocess
import sys
import time
from pathlib import Path

from .camera import CameraConfig, PiCamera
from .encoder import H264Encoder
from .hand_roi import WristAnchoredROISource
from .inference import (
    HailoHandInference,
    HailoObjectInference,
    HailoPoseInference,
    create_shared_vdevice,
)
from .overlay import OverlayDrawer
from .tracker import HandTracker
from .udp_protocol import UDPStreamer

LOG = logging.getLogger("pi_streamer.main")

DEFAULT_HEF_PATH = "/home/pi/models/yolov8m_pose.hef"
DEFAULT_OBJECT_HEF_PATH = "/home/pi/models/yolo26m.hef"
DEFAULT_HAND_HEF_PATH = "/home/pi/models/hand_landmark_lite.hef"
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FPS = 30
DEFAULT_BITRATE = 2_500_000
DEFAULT_PORT = 3334


def setup_logging(level: str) -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s %(levelname)-5s %(name)-26s %(message)s",
        datefmt="%H:%M:%S",
    )


def log_wifi_status() -> None:
    """Best-effort report of the Wi-Fi association frequency.

    Warns loudly if the hotspot is on 2.4 GHz — the user should disable
    iOS "Maximize Compatibility" to get 5 GHz performance.
    """
    # Non-interactive SSH sessions don't have /usr/sbin in PATH, so we look
    # in the common locations explicitly.
    iw_candidates = ("/usr/sbin/iw", "/sbin/iw", "/usr/bin/iw", "iw")
    iw_bin = next(
        (p for p in iw_candidates if Path(p).exists() or p == "iw"),
        None,
    )
    if iw_bin is None:
        LOG.info("Wi-Fi status unavailable: iw not found")
        return

    try:
        out = subprocess.run(
            [iw_bin, "dev", "wlan0", "link"],
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        LOG.info("Wi-Fi status unavailable: %s", exc)
        return

    text = out.stdout or ""
    if "Not connected" in text or not text.strip():
        LOG.warning("Wi-Fi: wlan0 not connected")
        return

    freq_mhz: int | None = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("freq:"):
            try:
                # iw reports "freq: 2437.0" (float) so float → int conversion is needed
                freq_mhz = int(float(line.split()[1]))
            except (IndexError, ValueError):
                freq_mhz = None
            break

    if freq_mhz is None:
        LOG.info("Wi-Fi: connected (frequency unknown)")
        return

    band = "5 GHz" if freq_mhz >= 4900 else "2.4 GHz"
    LOG.info("Wi-Fi: connected on %d MHz (%s)", freq_mhz, band)
    if band == "2.4 GHz":
        LOG.warning(
            "Wi-Fi band is 2.4 GHz — for best streaming quality disable "
            "'Maximize Compatibility' in iOS Settings > Personal Hotspot"
        )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="RPi pose+object video streamer")
    ap.add_argument("--hef", default=DEFAULT_HEF_PATH, help="Path to yolov8m_pose.hef")
    ap.add_argument(
        "--object-hef",
        default=DEFAULT_OBJECT_HEF_PATH,
        help="Path to yolo26m.hef",
    )
    ap.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    ap.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    ap.add_argument("--fps", type=int, default=DEFAULT_FPS)
    ap.add_argument("--bitrate", type=int, default=DEFAULT_BITRATE)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    ap.add_argument(
        "--no-inference",
        action="store_true",
        help="Skip both Hailo workers — useful for first-light camera/encode test",
    )
    ap.add_argument(
        "--no-object",
        action="store_true",
        help="Skip yolo26m object detection, keep pose only",
    )
    ap.add_argument(
        "--no-smoothing",
        action="store_true",
        help="Disable temporal smoothing on both pose and object detections (raw decoder output every frame)",
    )
    ap.add_argument(
        "--hand-hef",
        default=DEFAULT_HAND_HEF_PATH,
        help="Path to hand_landmark_lite.hef (MediaPipe 21-point hand landmarks, Hailo Model Zoo v5.1.0/hailo10h)",
    )
    ap.add_argument(
        "--no-hands",
        action="store_true",
        help="Skip the hand-landmark pipeline; runs pose + object only",
    )
    ap.add_argument(
        "--hand-roi",
        choices=["wrist"],
        default="wrist",
        help="Hand ROI source: 'wrist' uses pose wrist+elbow to crop 224x224 regions (Plan A). Plan B 'palm' is deferred to a future session.",
    )
    ap.add_argument(
        "--no-hand-rotate",
        action="store_true",
        help="Disable forearm rotation in WristAnchoredROISource. "
             "Default is rotation ON: the crop's vertical axis aligns "
             "with the forearm so the hand stands up inside the 224x224 "
             "patch, matching the hand_landmark_lite training "
             "distribution. Set this flag to fall back to axis-aligned "
             "crops for A/B comparison.",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    LOG.info("pi_streamer starting")
    LOG.info(
        "config: %dx%d @ %d fps, %d bit/s, port %d",
        args.width,
        args.height,
        args.fps,
        args.bitrate,
        args.port,
    )
    LOG.info("pose hef: %s", args.hef)
    if not args.no_inference and not args.no_object:
        LOG.info("object hef: %s", args.object_hef)
    hands_enabled = not args.no_inference and not args.no_hands
    if hands_enabled:
        LOG.info("hand hef: %s  (roi source: %s)", args.hand_hef, args.hand_roi)

    log_wifi_status()

    if not args.no_inference:
        hef_path = Path(args.hef)
        if not hef_path.exists():
            LOG.error("Pose HEF not found: %s", hef_path)
            return 2
        if not args.no_object:
            obj_hef_path = Path(args.object_hef)
            if not obj_hef_path.exists():
                LOG.error("Object HEF not found: %s", obj_hef_path)
                return 2
        if hands_enabled:
            hand_hef_path = Path(args.hand_hef)
            if not hand_hef_path.exists():
                LOG.error("Hand HEF not found: %s", hand_hef_path)
                return 2

    use_lores = not args.no_inference
    camera = PiCamera(
        CameraConfig(width=args.width, height=args.height, fps=args.fps),
        lores_size=(640, 360) if use_lores else None,
    )
    encoder = H264Encoder(
        width=args.width,
        height=args.height,
        fps=args.fps,
        bitrate=args.bitrate,
    )
    streamer = UDPStreamer(port=args.port, width=args.width, height=args.height)
    overlay = OverlayDrawer(frame_width=args.width, frame_height=args.height)
    inference: HailoPoseInference | None = None
    object_inference: HailoObjectInference | None = None
    hand_inference: HailoHandInference | None = None
    hand_roi_source: WristAnchoredROISource | None = None
    shared_vdevice = None
    if not args.no_inference:
        # Single shared VDevice — the HailoRT scheduler inside it time-slices
        # between pose, object, and hand ConfiguredInferModels. Opening two
        # separate VDevice() instances raises HAILO_OUT_OF_PHYSICAL_DEVICES.
        try:
            shared_vdevice = create_shared_vdevice()
            LOG.info("shared Hailo VDevice opened")
        except Exception as exc:  # noqa: BLE001
            LOG.exception("failed to open Hailo VDevice: %s", exc)
            return 1
        inference = HailoPoseInference(
            hef_path=args.hef,
            frame_width=args.width,
            frame_height=args.height,
            vdevice=shared_vdevice,
            enable_tracking=not args.no_smoothing,
        )
        if not args.no_object:
            object_inference = HailoObjectInference(
                hef_path=args.object_hef,
                frame_width=args.width,
                frame_height=args.height,
                vdevice=shared_vdevice,
                enable_tracking=not args.no_smoothing,
            )
        if hands_enabled:
            hand_tracker = None if args.no_smoothing else HandTracker()
            hand_roi_source = WristAnchoredROISource(
                rotate=not args.no_hand_rotate,
            )
            hand_inference = HailoHandInference(
                hef_path=args.hand_hef,
                frame_width=args.width,
                frame_height=args.height,
                vdevice=shared_vdevice,
                tracker=hand_tracker,
                roi_source=hand_roi_source,
            )
            LOG.info(
                "hand ROI: wrist-anchored  rotate=%s",
                not args.no_hand_rotate,
            )

    shutdown = {"flag": False}

    def handle_signal(signum: int, _frame) -> None:  # noqa: ANN001
        LOG.info("signal %d received — shutting down", signum)
        shutdown["flag"] = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        camera.start()
        encoder.start()
        streamer.start()
        if inference is not None:
            inference.start()
        if object_inference is not None:
            object_inference.start()
        if hand_inference is not None:
            hand_inference.start()
    except Exception as exc:  # noqa: BLE001
        LOG.exception("startup failed: %s", exc)
        return 1

    start_time = time.monotonic()
    sent_frame_count = 0
    last_status_log = start_time

    try:
        while not shutdown["flag"]:
            frame, lores_frame = camera.capture()
            frame_id = camera.next_frame_id
            timestamp_ms = int((time.monotonic() - start_time) * 1000)
            infer_frame = lores_frame if lores_frame is not None else frame

            if inference is not None:
                inference.submit(frame_id=frame_id, frame=infer_frame)
                latest = inference.latest()
            else:
                latest = None

            if object_inference is not None:
                object_inference.submit(frame_id=frame_id, frame=infer_frame)
                latest_objects = object_inference.latest()
            else:
                latest_objects = None

            latest_hands = None
            if hand_inference is not None:
                if hand_inference.worker_dead:
                    LOG.error(
                        "hand inference worker died; exiting main loop"
                    )
                    shutdown["flag"] = True
                    break
                persons = latest.persons if latest is not None else ()
                hand_inference.submit(
                    frame_id=frame_id, frame=frame, persons=persons,
                )
                latest_hands = hand_inference.latest()

            stream_active = streamer.is_active()
            overlay.draw(
                frame,
                latest,
                stream_active=stream_active,
                objects=latest_objects,
                hands=latest_hands,
            )

            if stream_active:
                force_idr = streamer.consume_force_idr()
                for packet in encoder.encode_rgb(
                    frame, pts_ms=timestamp_ms, force_keyframe=force_idr
                ):
                    streamer.send_frame(
                        h264_data=packet.data,
                        frame_id=frame_id,
                        timestamp_ms=packet.pts_ms,
                        is_keyframe=packet.is_keyframe,
                    )
                sent_frame_count += 1

            now = time.monotonic()
            if now - last_status_log >= 5.0:
                LOG.info(
                    "status: state=%s camera_fps=%.1f sent=%d",
                    streamer.state().value,
                    overlay.current_fps,
                    sent_frame_count,
                )
                last_status_log = now
    except Exception as exc:  # noqa: BLE001
        LOG.exception("main loop crashed: %s", exc)
    finally:
        LOG.info("shutting down pipeline")
        if hand_inference is not None:
            hand_inference.stop()
        if object_inference is not None:
            object_inference.stop()
        if inference is not None:
            inference.stop()
        streamer.stop()
        encoder.stop()
        camera.stop()
        if shared_vdevice is not None:
            try:
                shared_vdevice.release()
            except Exception as exc:  # noqa: BLE001
                LOG.debug("VDevice release error: %s", exc)

    return 0


if __name__ == "__main__":
    sys.exit(main())
