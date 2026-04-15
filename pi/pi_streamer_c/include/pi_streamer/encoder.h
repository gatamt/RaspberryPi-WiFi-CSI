#ifndef PI_STREAMER_ENCODER_H
#define PI_STREAMER_ENCODER_H

// Software H.264 encoder C ABI. Pi 5 has no hardware encoder — libx264 is
// the only realistic path (verified via Raspberry Pi Foundation docs and
// the existing Python `encoder.py` which already uses PyAV/libx264).
//
// Two backends share this header:
//   Host / TDD:   src/encoder_mock.c   (always buildable, synthetic NALs)
//   Pi production: src/encoder_x264.c  (compiled only when find_package
//                   (LIBX264) succeeds; wraps libx264 C API directly).
//
// Configuration mirrors the Python reference (`pi_streamer/encoder.py`):
//   preset=ultrafast, tune=zerolatency, profile=baseline, b_frames=0,
//   x264-params: keyint=N:min-keyint=N:no-scenecut=1:repeat-headers=1:annexb=1
//
// Primary reference: x264.h header + FFmpeg wiki on x264 presets.

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct pi_encoder pi_encoder_t;

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t fps;
    uint32_t bitrate_bps;
    uint32_t gop_size;
    bool     zero_latency;  // preset=ultrafast + tune=zerolatency + baseline
} pi_encoder_config_t;

typedef struct {
    const uint8_t *nal;        // Annex-B H.264 payload
    size_t         nal_size;
    uint64_t       pts_ms;
    bool           is_keyframe;
} pi_encoded_packet_t;

pi_encoder_t *pi_encoder_create(const pi_encoder_config_t *cfg);
void          pi_encoder_destroy(pi_encoder_t *enc);

// Submit a planar YUV420 frame. The encoder may return 0+ output packets;
// iterate with pi_encoder_next_packet until it returns -1.
int pi_encoder_submit_yuv420(pi_encoder_t  *enc,
                             const uint8_t *y, size_t y_stride,
                             const uint8_t *u, size_t u_stride,
                             const uint8_t *v, size_t v_stride,
                             uint64_t       pts_ms,
                             bool           force_keyframe);

// Retrieve the next encoded packet. Returns 0 if one is available, -1 if
// not. Packet memory is owned by the encoder and remains valid until the
// next call to any encoder function.
int pi_encoder_next_packet(pi_encoder_t *enc, pi_encoded_packet_t *out);

// Flush any queued packets (call before destroy).
void pi_encoder_flush(pi_encoder_t *enc);

#endif // PI_STREAMER_ENCODER_H
