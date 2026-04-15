"""Wire format for the P4-idf6 compatible video stream.

The chunk header is a 28-byte little-endian struct, bit-compatible with
``h264_chunk_header_t`` in the ESP32-P4 firmware. Any drift here means the
iOS client will fail to parse the stream — so this module is the single
source of truth on the Python side and is self-tested below.

See ``protocol.md`` at the project root for the full wire specification.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

HEADER_FORMAT = "<IHHIIHHII"
"""struct format string — see ``protocol.md`` §"Video chunk header"."""

HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
assert HEADER_SIZE == 28, f"header must be 28 bytes, got {HEADER_SIZE}"

MAX_PAYLOAD_SIZE = 1400
"""Max H.264 payload bytes per UDP datagram. Total datagram is 1428 bytes."""

MAX_DATAGRAM_SIZE = HEADER_SIZE + MAX_PAYLOAD_SIZE


@dataclass(frozen=True)
class ChunkHeader:
    """In-memory representation of one 28-byte chunk header."""

    frame_id: int
    width: int
    height: int
    timestamp_ms: int
    total_len: int
    chunk_idx: int
    chunk_count: int
    frame_type: int  # 1 = keyframe (IDR), 0 = P-frame
    reserved: int = 0

    def pack(self) -> bytes:
        """Serialize to the 28-byte little-endian wire format."""
        return struct.pack(
            HEADER_FORMAT,
            self.frame_id & 0xFFFFFFFF,
            self.width & 0xFFFF,
            self.height & 0xFFFF,
            self.timestamp_ms & 0xFFFFFFFF,
            self.total_len & 0xFFFFFFFF,
            self.chunk_idx & 0xFFFF,
            self.chunk_count & 0xFFFF,
            self.frame_type & 0xFFFFFFFF,
            self.reserved & 0xFFFFFFFF,
        )

    @classmethod
    def unpack(cls, data: bytes) -> "ChunkHeader":
        """Parse the 28-byte little-endian wire format."""
        if len(data) < HEADER_SIZE:
            raise ValueError(f"need {HEADER_SIZE} bytes, got {len(data)}")
        values = struct.unpack(HEADER_FORMAT, data[:HEADER_SIZE])
        return cls(*values)


def build_chunks(
    frame_id: int,
    width: int,
    height: int,
    timestamp_ms: int,
    h264_data: bytes,
    is_keyframe: bool,
) -> list[bytes]:
    """Split one encoded H.264 frame into UDP-ready datagrams.

    Returns a list of full datagram bytes (header + payload) ready to send.
    Matches the chunking loop in ``udp_stream.c::udp_stream_send_frame``.
    """
    total_len = len(h264_data)
    if total_len == 0:
        return []

    chunk_count = (total_len + MAX_PAYLOAD_SIZE - 1) // MAX_PAYLOAD_SIZE
    frame_type = 1 if is_keyframe else 0

    datagrams: list[bytes] = []
    offset = 0
    for idx in range(chunk_count):
        payload_len = min(MAX_PAYLOAD_SIZE, total_len - offset)
        header = ChunkHeader(
            frame_id=frame_id,
            width=width,
            height=height,
            timestamp_ms=timestamp_ms,
            total_len=total_len,
            chunk_idx=idx,
            chunk_count=chunk_count,
            frame_type=frame_type,
        )
        datagrams.append(header.pack() + h264_data[offset : offset + payload_len])
        offset += payload_len

    return datagrams


def _self_test() -> None:
    """Sanity check that matches a hand-computed known-good payload."""
    hdr = ChunkHeader(
        frame_id=0x11223344,
        width=1280,
        height=720,
        timestamp_ms=0xAABBCCDD,
        total_len=0x00010000,
        chunk_idx=2,
        chunk_count=73,
        frame_type=1,
        reserved=0,
    )
    packed = hdr.pack()
    assert len(packed) == 28
    expected = bytes.fromhex(
        "44332211"  # frame_id
        "0005"  # width (1280 = 0x0500)
        "d002"  # height (720 = 0x02d0)
        "ddccbbaa"  # timestamp
        "00000100"  # total_len
        "0200"  # chunk_idx
        "4900"  # chunk_count (73 = 0x49)
        "01000000"  # frame_type
        "00000000"  # reserved
    )
    assert packed == expected, f"expected {expected.hex()}, got {packed.hex()}"

    roundtrip = ChunkHeader.unpack(packed)
    assert roundtrip == hdr

    # chunking test: 3000-byte frame -> 3 chunks of 1400/1400/200
    body = bytes(range(256)) * 12  # 3072 bytes
    body = body[:3000]
    chunks = build_chunks(
        frame_id=1,
        width=1280,
        height=720,
        timestamp_ms=0,
        h264_data=body,
        is_keyframe=True,
    )
    assert len(chunks) == 3
    assert len(chunks[0]) == HEADER_SIZE + 1400
    assert len(chunks[1]) == HEADER_SIZE + 1400
    assert len(chunks[2]) == HEADER_SIZE + 200

    reassembled = b""
    for chunk in chunks:
        reassembled += chunk[HEADER_SIZE:]
    assert reassembled == body

    print("wire_format self-test PASS")


if __name__ == "__main__":
    _self_test()
