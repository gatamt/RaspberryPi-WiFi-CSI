#ifndef PI_STREAMER_WIRE_FORMAT_H
#define PI_STREAMER_WIRE_FORMAT_H

// 28-byte little-endian UDP chunk header — MUST be byte-identical to the
// Python reference in `pi/pi_streamer/wire_format.py` (struct format
// "<IHHIIHHII"). Any drift breaks the existing iOS client.
//
// Each H.264 frame is chunked into datagrams of up to 1428 bytes:
//   28-byte header + up to 1400 bytes of H.264 payload.
//
// Primary references: `protocol.md` at the project root and the Python
// reference at `pi/pi_streamer/wire_format.py::_self_test`. No vault
// atomic-note citations (see docs/Phase1-Atomic-Note-Verification.md).

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define PI_CHUNK_HEADER_SIZE   28u
#define PI_MAX_PAYLOAD_SIZE    1400u
#define PI_MAX_DATAGRAM_SIZE   (PI_CHUNK_HEADER_SIZE + PI_MAX_PAYLOAD_SIZE)

// In-memory representation of a chunk header. This struct is NEVER written
// straight to the wire — always pass through pi_chunk_header_pack to build
// the little-endian byte stream. That keeps natural-alignment padding out
// of the on-wire bytes.
typedef struct {
    uint32_t frame_id;
    uint16_t width;
    uint16_t height;
    uint32_t timestamp_ms;
    uint32_t total_len;
    uint16_t chunk_idx;
    uint16_t chunk_count;
    uint32_t frame_type;   // 1 = keyframe (IDR), 0 = P-frame
    uint32_t reserved;
} pi_chunk_header_t;

// Serialize `hdr` into the wire format (little-endian, 28 bytes).
// Caller provides a 28-byte output buffer.
void pi_chunk_header_pack(const pi_chunk_header_t *hdr,
                          uint8_t out[PI_CHUNK_HEADER_SIZE]);

// Parse a 28-byte wire header into `hdr_out`. Returns 0 on success,
// -1 on NULL arguments.
int pi_chunk_header_unpack(const uint8_t in[PI_CHUNK_HEADER_SIZE],
                           pi_chunk_header_t *hdr_out);

// Number of chunks needed to fit a frame of `frame_size` bytes.
// (frame_size + 1399) / 1400 with a zero-size special case.
static inline size_t pi_chunk_count_for(size_t frame_size) {
    if (frame_size == 0u) return 0u;
    return (frame_size + PI_MAX_PAYLOAD_SIZE - 1u) / PI_MAX_PAYLOAD_SIZE;
}

// Build ONE datagram for chunk index `chunk_idx`. Writes the 28-byte
// header followed by up to 1400 payload bytes into `out_buf`.
//
// Returns the total bytes written (≤ PI_MAX_DATAGRAM_SIZE) on success, or
// 0 on invalid argument (NULL frame/buf, chunk_idx out of range, out_cap
// too small, or frame_size == 0).
size_t pi_build_one_chunk(uint32_t       frame_id,
                          uint16_t       width,
                          uint16_t       height,
                          uint32_t       timestamp_ms,
                          const uint8_t *frame_data,
                          size_t         frame_size,
                          size_t         chunk_idx,
                          bool           is_keyframe,
                          uint8_t       *out_buf,
                          size_t         out_cap);

#endif // PI_STREAMER_WIRE_FORMAT_H
