#include "pi_streamer/wire_format.h"

#include <string.h>

// Little-endian scalar I/O. Kept as static inlines so the compiler can
// collapse them to single MOV/STR instructions on ARMv8 (which is
// little-endian natively).

static inline void write_u16_le(uint8_t *dst, uint16_t v) {
    dst[0] = (uint8_t)(v & 0xFFu);
    dst[1] = (uint8_t)((v >> 8) & 0xFFu);
}

static inline void write_u32_le(uint8_t *dst, uint32_t v) {
    dst[0] = (uint8_t)(v & 0xFFu);
    dst[1] = (uint8_t)((v >> 8) & 0xFFu);
    dst[2] = (uint8_t)((v >> 16) & 0xFFu);
    dst[3] = (uint8_t)((v >> 24) & 0xFFu);
}

static inline uint16_t read_u16_le(const uint8_t *src) {
    return (uint16_t)((uint16_t)src[0] | ((uint16_t)src[1] << 8));
}

static inline uint32_t read_u32_le(const uint8_t *src) {
    return ((uint32_t)src[0])        |
           ((uint32_t)src[1] << 8)   |
           ((uint32_t)src[2] << 16)  |
           ((uint32_t)src[3] << 24);
}

void pi_chunk_header_pack(const pi_chunk_header_t *hdr,
                          uint8_t out[PI_CHUNK_HEADER_SIZE]) {
    if (!hdr || !out) return;
    // Byte offsets MUST match Python struct format "<IHHIIHHII":
    //   0  frame_id      uint32_le
    //   4  width         uint16_le
    //   6  height        uint16_le
    //   8  timestamp_ms  uint32_le
    //  12  total_len     uint32_le
    //  16  chunk_idx     uint16_le
    //  18  chunk_count   uint16_le
    //  20  frame_type    uint32_le
    //  24  reserved      uint32_le
    write_u32_le(&out[0],  hdr->frame_id);
    write_u16_le(&out[4],  hdr->width);
    write_u16_le(&out[6],  hdr->height);
    write_u32_le(&out[8],  hdr->timestamp_ms);
    write_u32_le(&out[12], hdr->total_len);
    write_u16_le(&out[16], hdr->chunk_idx);
    write_u16_le(&out[18], hdr->chunk_count);
    write_u32_le(&out[20], hdr->frame_type);
    write_u32_le(&out[24], hdr->reserved);
}

int pi_chunk_header_unpack(const uint8_t in[PI_CHUNK_HEADER_SIZE],
                           pi_chunk_header_t *hdr_out) {
    if (!in || !hdr_out) return -1;
    hdr_out->frame_id     = read_u32_le(&in[0]);
    hdr_out->width        = read_u16_le(&in[4]);
    hdr_out->height       = read_u16_le(&in[6]);
    hdr_out->timestamp_ms = read_u32_le(&in[8]);
    hdr_out->total_len    = read_u32_le(&in[12]);
    hdr_out->chunk_idx    = read_u16_le(&in[16]);
    hdr_out->chunk_count  = read_u16_le(&in[18]);
    hdr_out->frame_type   = read_u32_le(&in[20]);
    hdr_out->reserved     = read_u32_le(&in[24]);
    return 0;
}

size_t pi_build_one_chunk(uint32_t       frame_id,
                          uint16_t       width,
                          uint16_t       height,
                          uint32_t       timestamp_ms,
                          const uint8_t *frame_data,
                          size_t         frame_size,
                          size_t         chunk_idx,
                          bool           is_keyframe,
                          uint8_t       *out_buf,
                          size_t         out_cap) {
    if (!frame_data || !out_buf) return 0u;
    if (frame_size == 0u) return 0u;
    if (out_cap < PI_CHUNK_HEADER_SIZE) return 0u;

    const size_t chunk_count = pi_chunk_count_for(frame_size);
    if (chunk_idx >= chunk_count) return 0u;

    const size_t offset      = chunk_idx * PI_MAX_PAYLOAD_SIZE;
    const size_t remaining   = frame_size - offset;
    const size_t payload_len = remaining < PI_MAX_PAYLOAD_SIZE
                                 ? remaining
                                 : PI_MAX_PAYLOAD_SIZE;
    if (out_cap < PI_CHUNK_HEADER_SIZE + payload_len) return 0u;

    const pi_chunk_header_t hdr = {
        .frame_id     = frame_id,
        .width        = width,
        .height       = height,
        .timestamp_ms = timestamp_ms,
        .total_len    = (uint32_t)frame_size,
        .chunk_idx    = (uint16_t)chunk_idx,
        .chunk_count  = (uint16_t)chunk_count,
        .frame_type   = is_keyframe ? 1u : 0u,
        .reserved     = 0u,
    };

    pi_chunk_header_pack(&hdr, out_buf);
    memcpy(out_buf + PI_CHUNK_HEADER_SIZE,
           frame_data + offset,
           payload_len);
    return PI_CHUNK_HEADER_SIZE + payload_len;
}
