"""P4-idf6 compatible UDP control + video streaming protocol.

This is a direct Python port of ``udp_stream.c`` from the ESP32-P4 project.
The state machine, control messages, chunking, and pacing all match exactly
so the existing iOS client can connect without modification.

See ``protocol.md`` at the project root for the full wire specification.
"""

from __future__ import annotations

import logging
import socket
import struct
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .wire_format import MAX_DATAGRAM_SIZE, build_chunks

LOG = logging.getLogger("pi_streamer.udp_protocol")

VIDEO_UDP_PORT = 3334
HEARTBEAT_TIMEOUT_S = 10.0
CHUNK_PACING_S = 0.0002  # 200 µs, matches usleep(200) in firmware

# 4-byte ASCII control messages
MSG_VID0 = b"VID0"
MSG_BEAT = b"BEAT"
MSG_PAWS = b"PAWS"
MSG_GONE = b"GONE"
MSG_VACK = b"VACK"
MSG_LEN = 4


class StreamState(Enum):
    IDLE = "IDLE"
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"


@dataclass
class _ClientInfo:
    addr: tuple[str, int]
    last_heartbeat_s: float


class UDPStreamer:
    """UDP video streamer with P4-idf6 compatible control protocol.

    Spawns a background listener thread that handles VID0/BEAT/PAWS/GONE.
    Frames are sent from the main thread via :meth:`send_frame`, which chunks
    them according to the wire format and throttles between chunks.
    """

    def __init__(self, port: int = VIDEO_UDP_PORT, width: int = 1280, height: int = 720) -> None:
        self.port = port
        self.width = width
        self.height = height

        self._sock: Optional[socket.socket] = None
        self._listener: Optional[threading.Thread] = None
        self._running = threading.Event()

        self._state = StreamState.IDLE
        self._state_lock = threading.RLock()
        self._client: Optional[_ClientInfo] = None
        self._force_idr = False

        # stats
        self._total_frames_sent = 0
        self._total_chunks_sent = 0
        self._total_bytes_sent = 0
        self._total_dropped_chunks = 0

    # ---- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self._listener is not None:
            return

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 32768)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 524288)
        # 50 ms send timeout: bounds main-loop stalls under transient Wi-Fi/BT
        # contention while preserving backpressure semantics from the working
        # main-branch code. SO_SNDTIMEO is set independently of
        # socket.settimeout() below so the listener thread's 2.0s recv
        # timeout is unaffected. struct.pack("ll", sec, usec) matches the
        # Linux struct timeval layout on glibc aarch64 (Raspberry Pi OS).
        self._sock.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_SNDTIMEO,
            struct.pack("ll", 0, 50_000),
        )
        self._sock.bind(("0.0.0.0", self.port))
        self._sock.settimeout(2.0)

        self._running.set()
        self._listener = threading.Thread(
            target=self._listener_loop, name="udp-ctrl-listener", daemon=True
        )
        self._listener.start()
        LOG.info("UDP streamer started on port %d (waiting for VID0)", self.port)

    def stop(self) -> None:
        self._running.clear()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        if self._listener is not None:
            self._listener.join(timeout=1.0)
            self._listener = None
        LOG.info("UDP streamer stopped")

    # ---- public state ------------------------------------------------------

    def state(self) -> StreamState:
        with self._state_lock:
            return self._state

    def consume_force_idr(self) -> bool:
        """Return True once if an IDR should be forced on the next encode."""
        with self._state_lock:
            if self._force_idr:
                self._force_idr = False
                return True
            return False

    def is_active(self) -> bool:
        return self.state() == StreamState.ACTIVE

    # ---- frame sending -----------------------------------------------------

    def send_frame(
        self,
        h264_data: bytes,
        frame_id: int,
        timestamp_ms: int,
        is_keyframe: bool,
    ) -> bool:
        """Chunk and send one encoded H.264 frame.

        Returns False if there is no active client and the frame was dropped.
        """
        with self._state_lock:
            if self._state != StreamState.ACTIVE or self._client is None or self._sock is None:
                return False
            client_addr = self._client.addr
            sock = self._sock

        datagrams = build_chunks(
            frame_id=frame_id,
            width=self.width,
            height=self.height,
            timestamp_ms=timestamp_ms,
            h264_data=h264_data,
            is_keyframe=is_keyframe,
        )

        if not datagrams:
            return False

        bytes_this_frame = 0
        dropped_chunks = 0
        for i, dgram in enumerate(datagrams):
            try:
                sock.sendto(dgram, client_addr)
            except (BlockingIOError, socket.timeout):
                # SO_SNDTIMEO (50ms) elapsed — the Wi-Fi TX queue is severely
                # congested. Drop the chunk; iOS will re-sync via waitingForIDR
                # on the next complete keyframe (or trigger its packet-silence
                # watchdog if drops persist across multiple IDRs).
                dropped_chunks += 1
                continue
            except OSError as exc:
                LOG.warning("sendto failed: %s", exc)
                return False
            bytes_this_frame += len(dgram)
            if i + 1 < len(datagrams):
                time.sleep(CHUNK_PACING_S)

        if dropped_chunks:
            if self._total_dropped_chunks == 0:
                LOG.warning(
                    "first chunk drop on frame %d (%d/%d chunks lost) — "
                    "Wi-Fi link may be congested",
                    frame_id,
                    dropped_chunks,
                    len(datagrams),
                )
            self._total_dropped_chunks += dropped_chunks

        self._total_frames_sent += 1
        self._total_chunks_sent += len(datagrams)
        self._total_bytes_sent += bytes_this_frame

        if self._total_frames_sent % 300 == 0:
            drop_info = f", {self._total_dropped_chunks} dropped" if self._total_dropped_chunks else ""
            LOG.info(
                "sent %d frames, %d chunks, %.1f MB total%s",
                self._total_frames_sent,
                self._total_chunks_sent,
                self._total_bytes_sent / 1_048_576,
                drop_info,
            )
        return True

    # ---- listener ----------------------------------------------------------

    def _listener_loop(self) -> None:
        assert self._sock is not None
        sock = self._sock
        LOG.info("control listener running on port %d", self.port)

        while self._running.is_set():
            try:
                data, addr = sock.recvfrom(64)
            except socket.timeout:
                self._check_heartbeat_timeout()
                continue
            except OSError as exc:
                if self._running.is_set():
                    LOG.warning("recvfrom error: %s", exc)
                break

            self._check_heartbeat_timeout()

            if len(data) < MSG_LEN:
                LOG.warning("short packet %d bytes from %s", len(data), addr)
                continue

            msg = data[:MSG_LEN]
            if msg == MSG_VID0:
                self._handle_vid0(addr)
            elif msg == MSG_BEAT:
                self._handle_beat(addr)
            elif msg == MSG_PAWS:
                self._handle_paws()
            elif msg == MSG_GONE:
                self._handle_gone()
            else:
                LOG.warning(
                    "unknown control msg %s from %s",
                    msg.hex(),
                    addr,
                )

        LOG.info("control listener exit")

    def _handle_vid0(self, addr: tuple[str, int]) -> None:
        with self._state_lock:
            self._client = _ClientInfo(addr=addr, last_heartbeat_s=time.monotonic())
            self._state = StreamState.ACTIVE
            self._force_idr = True
        self._send_control(MSG_VACK, addr)
        LOG.info("-> ACTIVE (VID0 from %s)", addr)

    def _handle_beat(self, addr: tuple[str, int]) -> None:
        with self._state_lock:
            now = time.monotonic()
            if self._state == StreamState.IDLE:
                # Re-register on BEAT in IDLE (app may have resumed after timeout)
                self._client = _ClientInfo(addr=addr, last_heartbeat_s=now)
                self._state = StreamState.ACTIVE
                self._force_idr = True
                self._send_control(MSG_VACK, addr)
                LOG.info("-> ACTIVE (BEAT re-registered %s)", addr)
                return
            if self._client is not None:
                self._client.last_heartbeat_s = now
            if self._state == StreamState.PAUSED:
                self._state = StreamState.ACTIVE
                self._force_idr = True
                LOG.info("-> ACTIVE (resumed via BEAT)")

    def _handle_paws(self) -> None:
        with self._state_lock:
            if self._state == StreamState.ACTIVE:
                self._state = StreamState.PAUSED
                LOG.info("-> PAUSED (client PAWS)")

    def _handle_gone(self) -> None:
        self._transition_idle("client sent GONE")

    def _check_heartbeat_timeout(self) -> None:
        with self._state_lock:
            if self._state == StreamState.IDLE or self._client is None:
                return
            elapsed = time.monotonic() - self._client.last_heartbeat_s
            if elapsed > HEARTBEAT_TIMEOUT_S:
                self._transition_idle(f"heartbeat timeout ({elapsed:.1f}s)")

    def _transition_idle(self, reason: str) -> None:
        with self._state_lock:
            if self._state == StreamState.IDLE:
                return
            self._state = StreamState.IDLE
            self._client = None
            self._force_idr = False
        LOG.warning("-> IDLE (%s)", reason)

    def _send_control(self, msg: bytes, addr: tuple[str, int]) -> None:
        if self._sock is None:
            return
        try:
            self._sock.sendto(msg, addr)
        except OSError as exc:
            LOG.warning("control send %s failed: %s", msg.hex(), exc)
