#include "pi_streamer/inference.h"

#include <stdlib.h>
#include <string.h>

// Mock HailoRT inference backend. One VDevice can host multiple models;
// each model has a one-slot "pending result" FIFO so submit followed by
// poll produces a deterministic test fixture.
//
// Real Pi backend will live in src/inference_hailort.c and use the
// HailoRT C API (VDevice + InferModel + AsyncInferJob). The mock matches
// the ABI so the rest of the pipeline is testable without HailoRT.

#define MOCK_RAW_OUTPUT_BYTES 64u

struct pi_vdevice {
    int dummy;
};

struct pi_infer_model {
    pi_vdevice_t           *vd;
    pi_infer_model_config_t cfg;
    // Single-slot pending result.
    bool      has_result;
    uint64_t  frame_id;
    uint8_t   raw_output[MOCK_RAW_OUTPUT_BYTES];
};

pi_vdevice_t *pi_vdevice_create(void) {
    return calloc(1, sizeof(pi_vdevice_t));
}

void pi_vdevice_destroy(pi_vdevice_t *vd) {
    free(vd);
}

pi_infer_model_t *pi_infer_model_create(pi_vdevice_t                   *vd,
                                        const pi_infer_model_config_t *cfg) {
    if (!vd || !cfg) return NULL;
    if ((unsigned)cfg->kind >= PI_INFER__COUNT) return NULL;
    if (cfg->input_width == 0 || cfg->input_height == 0) return NULL;

    pi_infer_model_t *m = calloc(1, sizeof *m);
    if (!m) return NULL;
    m->vd  = vd;
    m->cfg = *cfg;
    return m;
}

void pi_infer_model_destroy(pi_infer_model_t *model) {
    free(model);
}

int pi_infer_submit(pi_infer_model_t *model,
                    const uint8_t    *input,
                    size_t            input_size,
                    uint64_t          frame_id) {
    if (!model || !input) return -1;
    if (input_size == 0) return -1;
    if (model->has_result) {
        // Back-pressure: mock queue depth is 1. Caller should drain via
        // pi_infer_poll before submitting again.
        return -1;
    }

    // Fill the raw output with a deterministic, kind-dependent pattern.
    const uint8_t fill = (uint8_t)((uint32_t)model->cfg.kind * 0x40u +
                                   (uint8_t)(frame_id & 0x0Fu));
    memset(model->raw_output, fill, MOCK_RAW_OUTPUT_BYTES);

    model->frame_id   = frame_id;
    model->has_result = true;
    return 0;
}

int pi_infer_poll(pi_infer_model_t *model, pi_infer_result_t *out) {
    if (!model || !out) return -1;
    if (!model->has_result) return -1;

    // Zero the whole result so the caller never observes uninitialised
    // tail fields (pi_detection_t array + num_detections). This matters
    // because pi_inference_state_publish memcpys the entire struct value
    // into a seqlock slot; leaking stack garbage there would cause the
    // overlay renderer to draw phantom boxes.
    memset(out, 0, sizeof *out);

    out->frame_id        = model->frame_id;
    out->kind            = model->cfg.kind;
    out->raw_output      = model->raw_output;
    out->raw_output_size = MOCK_RAW_OUTPUT_BYTES;
    out->latency_ns      = 0u;  // mock is instantaneous

    // Mock detection: produce ONE kind-specific bbox that visibly drifts
    // with the frame id so a host smoke-test / visual check can tell
    // whether the overlay pipeline is wired end-to-end without needing
    // real Hailo hardware. Coordinates target the default 1280x720 frame;
    // the overlay renderer clamps if the camera is configured smaller.
    //
    // Layout per kind (visually distinct corners of the frame):
    //   POSE   — 320x360 box in the upper-left quadrant
    //   OBJECT — 300x300 box in the upper-right quadrant
    //   HAND   — 200x200 box lower-centre
    // All three shift left/right by up to ±32 px based on frame id so
    // the viewer can tell the boxes are refreshing, not stale pixels.
    const int32_t drift = (int32_t)((model->frame_id % 64u) - 32);
    pi_detection_t *d = &out->detections[0];
    switch (model->cfg.kind) {
        case PI_INFER_POSE:
            d->x1 = 80  + drift;
            d->y1 = 80;
            d->x2 = 400 + drift;
            d->y2 = 440;
            d->score    = 0.92f;
            d->class_id = 0;   // person
            break;
        case PI_INFER_OBJECT:
            d->x1 = 880 + drift;
            d->y1 = 80;
            d->x2 = 1180 + drift;
            d->y2 = 380;
            d->score    = 0.85f;
            d->class_id = 56;  // "chair" in COCO
            break;
        case PI_INFER_HAND:
            d->x1 = 540 + drift;
            d->y1 = 480;
            d->x2 = 740 + drift;
            d->y2 = 680;
            d->score    = 0.78f;
            d->class_id = 0;
            break;
        default:
            out->num_detections = 0;
            model->has_result    = false;
            return 0;
    }
    out->num_detections = 1;

    model->has_result = false;
    return 0;
}

const char *pi_infer_kind_name(pi_infer_kind_t k) {
    switch (k) {
        case PI_INFER_POSE:   return "pose";
        case PI_INFER_OBJECT: return "object";
        case PI_INFER_HAND:   return "hand";
        case PI_INFER__COUNT: return "<invalid>";
    }
    return "<unknown>";
}
