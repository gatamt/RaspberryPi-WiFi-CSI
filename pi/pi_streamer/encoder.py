"""PyAV libx264 software encoder wrapper.

Pi 5 has no hardware H.264 encoder (Broadcom removed it from BCM2712) so we
use software x264 via PyAV. For 1280x720 @ 30 FPS with ``ultrafast`` +
``zerolatency``, the Cortex-A76 quad-core handles it at roughly 30-50% CPU
total, leaving plenty of headroom for camera capture, Hailo, and overlays.

Output format is Annex-B H.264 (start code prefixes), chunked by the UDP
streamer into 1400-byte datagrams. Each encoded packet is marked as keyframe
or P-frame via PyAV's packet flags, and those bits are forwarded into the
28-byte chunk header's ``frame_type`` field.
"""

from __future__ import annotations

import fractions
import logging
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np

LOG = logging.getLogger("pi_streamer.encoder")


@dataclass
class EncodedPacket:
    data: bytes
    pts_ms: int
    is_keyframe: bool


class H264Encoder:
    """Software H.264 encoder via PyAV / libx264."""

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        bitrate: int = 2_500_000,
        gop_size: int = 30,
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.bitrate = bitrate
        self.gop_size = gop_size

        self._codec_ctx = None
        self._frame_index = 0

    def start(self) -> None:
        import av  # lazy import

        codec = av.CodecContext.create("libx264", "w")
        codec.width = self.width
        codec.height = self.height
        codec.pix_fmt = "yuv420p"
        codec.bit_rate = self.bitrate
        codec.time_base = fractions.Fraction(1, self.fps)
        codec.framerate = fractions.Fraction(self.fps, 1)
        codec.gop_size = self.gop_size
        codec.max_b_frames = 0  # zerolatency: no B-frames
        codec.options = {
            "preset": "ultrafast",
            "tune": "zerolatency",
            "profile": "baseline",
            "x264-params": (
                f"keyint={self.gop_size}:min-keyint={self.gop_size}:"
                "no-scenecut=1:repeat-headers=1:annexb=1"
            ),
        }
        codec.open()
        self._codec_ctx = codec
        LOG.info(
            "H.264 encoder: %dx%d @ %d fps, %d bit/s, GOP=%d, ultrafast+zerolatency",
            self.width,
            self.height,
            self.fps,
            self.bitrate,
            self.gop_size,
        )

    def stop(self) -> None:
        if self._codec_ctx is not None:
            try:
                # Flush any queued packets
                for _ in self._codec_ctx.encode(None):
                    pass
            except Exception as exc:  # noqa: BLE001
                LOG.debug("flush error: %s", exc)
            try:
                self._codec_ctx.close()
            except Exception as exc:  # noqa: BLE001
                LOG.debug("close error: %s", exc)
            self._codec_ctx = None

    def encode_rgb(
        self,
        rgb_frame: np.ndarray,
        pts_ms: int,
        force_keyframe: bool = False,
    ) -> Iterator[EncodedPacket]:
        """Encode one camera frame and yield any resulting H.264 packets.

        picamera2 configured with ``format="RGB888"`` actually stores pixels in
        BGR order in memory (a well-known libcamera/picamera2 quirk), so we
        pass ``format="bgr24"`` to PyAV. This also matches OpenCV's native
        byte order, so overlay.py can draw directly in-place without a
        cvtColor call.
        """
        import av  # lazy import

        if self._codec_ctx is None:
            raise RuntimeError("encoder not started")

        av_frame = av.VideoFrame.from_ndarray(rgb_frame, format="bgr24")
        av_frame = av_frame.reformat(format="yuv420p")
        av_frame.pts = self._frame_index
        av_frame.time_base = fractions.Fraction(1, self.fps)

        if force_keyframe:
            av_frame.pict_type = av.video.frame.PictureType.I
        else:
            av_frame.pict_type = av.video.frame.PictureType.NONE

        self._frame_index += 1

        for packet in self._codec_ctx.encode(av_frame):
            data = bytes(packet)
            if not data:
                continue
            # PyAV exposes ``is_keyframe`` on Packet as the canonical API.
            # Older versions also had ``packet.flags & Packet.Flags.keyframe``
            # but that attribute was removed in PyAV 14.
            is_keyframe = bool(packet.is_keyframe)
            yield EncodedPacket(
                data=data,
                pts_ms=pts_ms,
                is_keyframe=is_keyframe,
            )
