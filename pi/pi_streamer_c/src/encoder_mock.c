#include "pi_streamer/encoder.h"

#include <stdlib.h>
#include <string.h>

// Mock H.264 encoder. Produces one fixed-size NAL unit per submitted
// frame, with an Annex-B start code and a frame-counter-dependent body
// so tests can tell consecutive packets apart. Emits a keyframe every
// `gop_size` frames so the wire-format chunking test can see both I and
// P frame types.

#define MOCK_NAL_BYTES 512u

struct pi_encoder {
    pi_encoder_config_t cfg;
    uint8_t             nal[MOCK_NAL_BYTES];
    size_t              nal_size;
    uint64_t            pts_ms;
    bool                is_keyframe;
    bool                has_pending;
    uint64_t            frame_counter;
};

pi_encoder_t *pi_encoder_create(const pi_encoder_config_t *cfg) {
    if (!cfg || cfg->width == 0 || cfg->height == 0 || cfg->gop_size == 0) {
        return NULL;
    }
    pi_encoder_t *enc = calloc(1, sizeof *enc);
    if (!enc) return NULL;
    enc->cfg = *cfg;
    return enc;
}

void pi_encoder_destroy(pi_encoder_t *enc) {
    free(enc);
}

int pi_encoder_submit_yuv420(pi_encoder_t  *enc,
                             const uint8_t *y, size_t y_stride,
                             const uint8_t *u, size_t u_stride,
                             const uint8_t *v, size_t v_stride,
                             uint64_t       pts_ms,
                             bool           force_keyframe) {
    (void)y; (void)y_stride;
    (void)u; (void)u_stride;
    (void)v; (void)v_stride;
    if (!enc) return -1;

    // Annex-B start code + fill byte for the rest.
    enc->nal[0] = 0x00;
    enc->nal[1] = 0x00;
    enc->nal[2] = 0x00;
    enc->nal[3] = 0x01;
    const uint8_t fill = (uint8_t)(enc->frame_counter & 0xFFu);
    memset(enc->nal + 4, fill, MOCK_NAL_BYTES - 4u);

    enc->nal_size    = MOCK_NAL_BYTES;
    enc->pts_ms      = pts_ms;
    enc->is_keyframe =
        force_keyframe ||
        (enc->frame_counter % enc->cfg.gop_size) == 0u;
    enc->has_pending = true;
    enc->frame_counter++;
    return 0;
}

int pi_encoder_next_packet(pi_encoder_t *enc, pi_encoded_packet_t *out) {
    if (!enc || !out) return -1;
    if (!enc->has_pending) return -1;
    out->nal         = enc->nal;
    out->nal_size    = enc->nal_size;
    out->pts_ms      = enc->pts_ms;
    out->is_keyframe = enc->is_keyframe;
    enc->has_pending = false;
    return 0;
}

void pi_encoder_flush(pi_encoder_t *enc) {
    if (enc) enc->has_pending = false;
}
