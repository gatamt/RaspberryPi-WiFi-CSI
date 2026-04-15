#include "pi_streamer/encoder.h"
#include "pi_streamer/logger.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <x264.h>

// Real software H.264 encoder backed by libx264 (Pi 5 has no hardware
// encoder — Broadcom removed the BCM2712 H.264 block, so x264 on the
// Cortex-A76 quad-core is the only realistic path). All parameters are
// kept 1:1 with the Python reference at pi_streamer/encoder.py so wire
// behavior is identical between the legacy Python streamer and this
// low-latency C streamer.
//
// Output is Annex-B H.264, repeat-headers on so every IDR carries fresh
// SPS/PPS (the iOS UDP receiver may join mid-stream and needs them).
//
// Threading model: i_threads=2 + b_sliced_threads=1 gives us intra-frame
// slice parallelism so a single encode call uses two cores at once on the
// camera/encoder pinned core pair, with no encoder-internal lookahead
// queue (i_sync_lookahead=0) — that's required for true zerolatency.
//
// Zero-copy: pi_encoder_submit_yuv420 overwrites pic_in.img.plane[0..2]
// with the caller-provided buffer pointers. x264_picture_alloc's internal
// buffer is left allocated only so x264_picture_clean has something to
// free; we never write into it.

#define PI_LOG_MOD "encoder_x264"

struct pi_encoder {
    pi_encoder_config_t cfg;
    x264_param_t        param;
    x264_t             *h;
    x264_picture_t      pic_in;
    x264_picture_t      pic_out;

    // Pending-packet staging (single slot — the streamer drains
    // next_packet immediately after each submit, so one frame at a time
    // is sufficient).
    x264_nal_t *pending_nals;
    int         pending_nal_count;
    int         pending_frame_size;
    uint64_t    pending_pts_ms;
    bool        pending_is_keyframe;
    bool        has_pending;

    // x264_encoder_encode returns an array of x264_nal_t pointing into
    // x264's own internal buffer (valid only until the next encode call).
    // We concatenate them into our own buffer so the streamer can hand a
    // single contiguous Annex-B blob to the chunker.
    uint8_t *concat_buf;
    size_t   concat_cap;
    size_t   concat_size;
};

pi_encoder_t *pi_encoder_create(const pi_encoder_config_t *cfg) {
    if (!cfg || cfg->width == 0 || cfg->height == 0 ||
        cfg->fps == 0 || cfg->gop_size == 0 || cfg->bitrate_bps == 0) {
        PI_WARN(PI_LOG_MOD, "invalid config: w=%u h=%u fps=%u gop=%u bps=%u",
                cfg ? cfg->width : 0u,
                cfg ? cfg->height : 0u,
                cfg ? cfg->fps : 0u,
                cfg ? cfg->gop_size : 0u,
                cfg ? cfg->bitrate_bps : 0u);
        return NULL;
    }

    pi_encoder_t *enc = calloc(1, sizeof *enc);
    if (!enc) return NULL;
    enc->cfg = *cfg;

    // Mirror Python: preset=ultrafast + tune=zerolatency + profile=baseline.
    if (x264_param_default_preset(&enc->param, "ultrafast", "zerolatency") < 0) {
        PI_WARN(PI_LOG_MOD, "x264_param_default_preset failed");
        free(enc);
        return NULL;
    }

    enc->param.i_width              = (int)cfg->width;
    enc->param.i_height             = (int)cfg->height;
    enc->param.i_csp                = X264_CSP_I420;
    enc->param.i_bitdepth           = 8;
    enc->param.i_fps_num            = (int)cfg->fps;
    enc->param.i_fps_den            = 1;

    // x264-params from Python: keyint=N:min-keyint=N:no-scenecut=1:
    // repeat-headers=1:annexb=1.
    enc->param.i_keyint_max         = (int)cfg->gop_size;
    enc->param.i_keyint_min         = (int)cfg->gop_size;
    enc->param.i_scenecut_threshold = 0;
    enc->param.b_repeat_headers     = 1;  // SPS/PPS on every IDR
    enc->param.b_annexb             = 1;

    // Zerolatency essentials.
    enc->param.i_bframe             = 0;
    enc->param.b_intra_refresh      = 0;
    enc->param.i_sync_lookahead     = 0;
    enc->param.b_vfr_input          = 0;

    // Threading: 2 slice threads on the camera/encoder core pair.
    enc->param.i_threads            = 2;
    enc->param.b_sliced_threads     = 1;

    // ABR rate control. libx264 uses kbps for i_bitrate.
    enc->param.rc.i_rc_method       = X264_RC_ABR;
    enc->param.rc.i_bitrate         = (int)(cfg->bitrate_bps / 1000u);
    enc->param.rc.i_vbv_max_bitrate = enc->param.rc.i_bitrate;
    // 1-second VBV buffer matches libx264's default and the Python
    // reference (PyAV sets no explicit vbv_buffer_size). The previous
    // bitrate/fps "single-frame VBV" forced the rate controller to crush
    // every IDR down to ~1/fps of the bitrate, producing a visible 1 Hz
    // blockiness pulse on the iOS receiver. Giving the rate controller a
    // full second of bits lets IDRs be 4-10x larger than P-frames without
    // QP-spiking — which is how H.264 is supposed to look.
    enc->param.rc.i_vbv_buffer_size = enc->param.rc.i_bitrate;

    if (x264_param_apply_profile(&enc->param, "baseline") < 0) {
        PI_WARN(PI_LOG_MOD, "x264_param_apply_profile(baseline) failed");
        free(enc);
        return NULL;
    }

    enc->h = x264_encoder_open(&enc->param);
    if (!enc->h) {
        PI_WARN(PI_LOG_MOD, "x264_encoder_open failed");
        free(enc);
        return NULL;
    }

    if (x264_picture_alloc(&enc->pic_in, X264_CSP_I420,
                           (int)cfg->width, (int)cfg->height) < 0) {
        PI_WARN(PI_LOG_MOD, "x264_picture_alloc failed");
        x264_encoder_close(enc->h);
        free(enc);
        return NULL;
    }

    enc->concat_cap = 256u * 1024u;  // grows on demand
    enc->concat_buf = malloc(enc->concat_cap);
    if (!enc->concat_buf) {
        PI_WARN(PI_LOG_MOD, "concat buffer alloc failed");
        x264_picture_clean(&enc->pic_in);
        x264_encoder_close(enc->h);
        free(enc);
        return NULL;
    }
    enc->concat_size = 0;
    enc->has_pending = false;

    PI_INFO(PI_LOG_MOD,
            "x264 encoder: %ux%u @ %u fps, %u bit/s, GOP=%u, ultrafast+zerolatency+baseline",
            cfg->width, cfg->height, cfg->fps, cfg->bitrate_bps, cfg->gop_size);

    return enc;
}

void pi_encoder_destroy(pi_encoder_t *enc) {
    if (!enc) return;
    if (enc->h) {
        x264_encoder_close(enc->h);
        enc->h = NULL;
    }
    x264_picture_clean(&enc->pic_in);
    free(enc->concat_buf);
    free(enc);
}

int pi_encoder_submit_yuv420(pi_encoder_t  *enc,
                             const uint8_t *y, size_t y_stride,
                             const uint8_t *u, size_t u_stride,
                             const uint8_t *v, size_t v_stride,
                             uint64_t       pts_ms,
                             bool           force_keyframe) {
    if (!enc || !enc->h || !y || !u || !v) return -1;

    // Zero-copy: hand x264 the caller's plane pointers directly. Safe
    // because x264_encoder_encode reads the planes synchronously before
    // returning (no async ownership transfer in the pure C API). The
    // double cast (uintptr_t first) strips const without tripping
    // -Wcast-qual; x264's pic_in.img.plane[] field is non-const by API.
    enc->pic_in.img.i_csp       = X264_CSP_I420;
    enc->pic_in.img.i_plane     = 3;
    enc->pic_in.img.plane[0]    = (uint8_t *)(uintptr_t)y;
    enc->pic_in.img.plane[1]    = (uint8_t *)(uintptr_t)u;
    enc->pic_in.img.plane[2]    = (uint8_t *)(uintptr_t)v;
    enc->pic_in.img.i_stride[0] = (int)y_stride;
    enc->pic_in.img.i_stride[1] = (int)u_stride;
    enc->pic_in.img.i_stride[2] = (int)v_stride;

    enc->pic_in.i_pts  = (int64_t)pts_ms;
    enc->pic_in.i_type = force_keyframe ? X264_TYPE_IDR : X264_TYPE_AUTO;

    int frame_size = x264_encoder_encode(enc->h,
                                         &enc->pending_nals,
                                         &enc->pending_nal_count,
                                         &enc->pic_in,
                                         &enc->pic_out);
    if (frame_size < 0) {
        PI_WARN(PI_LOG_MOD,
                "x264_encoder_encode failed (rc=%d), strides Y=%zu U=%zu V=%zu",
                frame_size, y_stride, u_stride, v_stride);
        enc->has_pending = false;
        return -1;
    }
    if (frame_size == 0) {
        // Encoder consumed the frame but is buffering — no output yet.
        enc->has_pending = false;
        return 0;
    }

    // Concatenate all NALs into our own buffer. x264 guarantees the NAL
    // payloads are laid out back-to-back inside its internal buffer so a
    // single memcpy of `frame_size` bytes from nals[0].p_payload would
    // also work — but we copy NAL-by-NAL to stay defensive against any
    // future libx264 changes that might add padding.
    if (enc->concat_cap < (size_t)frame_size) {
        size_t new_cap = (size_t)frame_size * 2u;
        uint8_t *new_buf = realloc(enc->concat_buf, new_cap);
        if (!new_buf) {
            PI_WARN(PI_LOG_MOD, "concat buffer realloc failed (need %d bytes)",
                    frame_size);
            enc->has_pending = false;
            return -1;
        }
        enc->concat_buf = new_buf;
        enc->concat_cap = new_cap;
    }

    size_t off = 0;
    for (int i = 0; i < enc->pending_nal_count; ++i) {
        const x264_nal_t *nal = &enc->pending_nals[i];
        if (nal->i_payload <= 0) continue;
        memcpy(enc->concat_buf + off, nal->p_payload, (size_t)nal->i_payload);
        off += (size_t)nal->i_payload;
    }
    enc->concat_size         = off;
    enc->pending_frame_size  = frame_size;
    enc->pending_pts_ms      = (uint64_t)enc->pic_out.i_pts;
    enc->pending_is_keyframe = (enc->pic_out.b_keyframe != 0);
    enc->has_pending         = true;
    return 0;
}

int pi_encoder_next_packet(pi_encoder_t *enc, pi_encoded_packet_t *out) {
    if (!enc || !out) return -1;
    if (!enc->has_pending) return -1;

    out->nal         = enc->concat_buf;
    out->nal_size    = enc->concat_size;
    out->pts_ms      = enc->pending_pts_ms;
    out->is_keyframe = enc->pending_is_keyframe;
    enc->has_pending = false;
    return 0;
}

void pi_encoder_flush(pi_encoder_t *enc) {
    if (!enc || !enc->h) return;

    // Drain any frames still in x264's internal queue. We're shutting
    // down so the bytes are discarded — the iOS receiver will see the
    // stream end either way.
    while (x264_encoder_delayed_frames(enc->h) > 0) {
        x264_nal_t *nals = NULL;
        int         nal_count = 0;
        int rc = x264_encoder_encode(enc->h, &nals, &nal_count, NULL, &enc->pic_out);
        if (rc < 0) {
            PI_WARN(PI_LOG_MOD, "flush encode rc=%d", rc);
            break;
        }
        if (rc == 0) {
            // Nothing more came out — bail out so we don't spin forever
            // if delayed_frames keeps insisting there are queued frames
            // (shouldn't happen, but be defensive).
            break;
        }
    }
    enc->has_pending = false;
}
