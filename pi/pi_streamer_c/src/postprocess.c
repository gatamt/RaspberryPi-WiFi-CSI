#include "pi_streamer/postprocess.h"

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

// -----------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------
//
// Upper bound on candidate count collected across all 3 scales after the
// score threshold is applied. 256 is comfortable for a single-class
// person detector — a realistic indoor scene produces <30 candidates
// post-threshold and well below 10 post-NMS.
#define PI_PP_MAX_CANDIDATES 256

// Maximum DFL regression bins the decoder knows how to softmax. YOLOv8
// ships with reg_max = 16; we hard-cap at 32 so the stack-local softmax
// buffer stays bounded even if a future model bumps the value.
#define PI_PP_MAX_REG_MAX    32

// -----------------------------------------------------------------------
// Internal types
// -----------------------------------------------------------------------

typedef struct {
    float  x1, y1, x2, y2;   // letterbox coordinates
    float  score;
} pp_candidate_t;

// Grouping of the 3 tensors that belong to one feature scale.
// `kpts` is looked up so the caller can validate the HEF produced a
// complete pose head, but the decoder itself doesn't read it.
typedef struct {
    int32_t                 grid_h;
    int32_t                 grid_w;
    int32_t                 stride;
    const pi_tensor_view_t *bbox;
    const pi_tensor_view_t *obj;
    const pi_tensor_view_t *kpts;  // present but unused
} pp_scale_t;

// -----------------------------------------------------------------------
// Dequantization
// -----------------------------------------------------------------------

// Read one scalar from a quantized tensor and dequantize it.
// flat_idx = spatial_idx * channels + channel_idx (channels-last layout,
// which is what the Hailo compiler emits by default for yolo heads).
static inline float dequant_at(const pi_tensor_view_t *t, size_t flat_idx) {
    float raw;
    if (t->dtype == PI_TENSOR_DTYPE_U16) {
        raw = (float)((const uint16_t *)t->data)[flat_idx];
    } else {
        raw = (float)((const uint8_t  *)t->data)[flat_idx];
    }
    return (raw - t->zero_point) * t->scale;
}

// -----------------------------------------------------------------------
// Softmax over reg_max bins (small, numerically stable)
// -----------------------------------------------------------------------

static void softmax_reg_max(const float *in, float *out, int reg_max) {
    float vmax = in[0];
    for (int i = 1; i < reg_max; i++) {
        if (in[i] > vmax) vmax = in[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < reg_max; i++) {
        out[i] = expf(in[i] - vmax);
        sum   += out[i];
    }
    const float inv = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
    for (int i = 0; i < reg_max; i++) {
        out[i] *= inv;
    }
}

// -----------------------------------------------------------------------
// Scale grouping — port of Python _group_outputs
// -----------------------------------------------------------------------

// Find the single tensor in cfg->tensors matching the spatial grid and
// the target channel count. Returns NULL if none found. This replaces
// the dict-by-name lookup Python uses — channel count is a stable,
// compiler-independent fingerprint so we key on it.
static const pi_tensor_view_t *find_tensor(const pi_postprocess_pose_cfg_t *cfg,
                                           int32_t grid_h,
                                           int32_t grid_w,
                                           int32_t channels) {
    for (size_t i = 0; i < cfg->num_tensors; i++) {
        const pi_tensor_view_t *t = &cfg->tensors[i];
        if (t->grid_h == grid_h && t->grid_w == grid_w &&
            t->channels == channels) {
            return t;
        }
    }
    return NULL;
}

// Populate `out` with one pp_scale_t per distinct (grid_h, grid_w) in
// cfg->tensors. Returns the number of scales found, up to 3 — we never
// expect more than 3 scales from yolov8's p3/p4/p5 head.
static size_t group_scales(const pi_postprocess_pose_cfg_t *cfg,
                           pp_scale_t out[3]) {
    const int32_t bbox_ch = cfg->reg_max * 4;
    const int32_t obj_ch  = 1;
    const int32_t kpts_ch = 17 * 3;

    size_t n_scales = 0;
    // Track unique grids we've already processed so we don't emit the
    // same scale twice from the 3 bbox/obj/kpts tensors pointing at it.
    int32_t seen_h[3] = {0};
    int32_t seen_w[3] = {0};
    size_t  seen_n    = 0;

    for (size_t i = 0; i < cfg->num_tensors && n_scales < 3; i++) {
        const int32_t gh = cfg->tensors[i].grid_h;
        const int32_t gw = cfg->tensors[i].grid_w;

        bool already = false;
        for (size_t j = 0; j < seen_n; j++) {
            if (seen_h[j] == gh && seen_w[j] == gw) { already = true; break; }
        }
        if (already) continue;

        const pi_tensor_view_t *tb = find_tensor(cfg, gh, gw, bbox_ch);
        const pi_tensor_view_t *to = find_tensor(cfg, gh, gw, obj_ch);
        const pi_tensor_view_t *tk = find_tensor(cfg, gh, gw, kpts_ch);
        if (!tb || !to) continue;  // incomplete scale — skip

        // Compute stride. Assume square strides (Python does too).
        if (gw <= 0 || gh <= 0) continue;
        if (cfg->input_size % gw != 0) continue;
        const int32_t stride = cfg->input_size / gw;
        if (cfg->input_size / gh != stride) continue;

        out[n_scales].grid_h = gh;
        out[n_scales].grid_w = gw;
        out[n_scales].stride = stride;
        out[n_scales].bbox   = tb;
        out[n_scales].obj    = to;
        out[n_scales].kpts   = tk;
        n_scales++;

        if (seen_n < 3) {
            seen_h[seen_n] = gh;
            seen_w[seen_n] = gw;
            seen_n++;
        }
    }
    return n_scales;
}

// -----------------------------------------------------------------------
// Per-scale anchor decode
// -----------------------------------------------------------------------

// Process one (grid_h x grid_w) feature scale. For each anchor above the
// score threshold, push a pp_candidate_t into `candidates`. Candidate
// coordinates are in letterbox pixel space — the main decode function
// reverses the letterbox transform after NMS.
//
// Returns the number of new candidates appended. Stops early if the
// candidate array would overflow.
static size_t decode_scale(const pp_scale_t *scale,
                           float             score_threshold,
                           int               reg_max,
                           pp_candidate_t   *candidates,
                           size_t            cand_size,
                           size_t            cand_cap) {
    const int32_t bbox_ch = reg_max * 4;
    const float   stride_f = (float)scale->stride;

    float dfl_logits[PI_PP_MAX_REG_MAX];
    float dfl_prob[PI_PP_MAX_REG_MAX];

    size_t added = 0;
    for (int32_t row = 0; row < scale->grid_h; row++) {
        for (int32_t col = 0; col < scale->grid_w; col++) {
            const size_t spatial_idx =
                (size_t)row * (size_t)scale->grid_w + (size_t)col;

            // Objectness first — if below threshold we skip the bbox
            // decode entirely. This is the hot early-out that keeps the
            // per-frame cost bounded no matter how many anchors there are.
            const float obj = dequant_at(scale->obj, spatial_idx);
            if (obj < score_threshold) continue;

            // Decode 4 DFL distances from the 64 bbox channels. Layout
            // inside the channel dimension is [l_0..l_15, t_0..t_15,
            // r_0..r_15, b_0..b_15] per Ultralytics DFL convention.
            float dist[4];
            for (int side = 0; side < 4; side++) {
                // Gather the 16 logits for this side into a tiny buffer
                // so softmax can run on contiguous memory.
                for (int b = 0; b < reg_max; b++) {
                    const size_t ch = (size_t)(side * reg_max + b);
                    const size_t flat =
                        spatial_idx * (size_t)bbox_ch + ch;
                    dfl_logits[b] = dequant_at(scale->bbox, flat);
                }
                softmax_reg_max(dfl_logits, dfl_prob, reg_max);
                float d = 0.0f;
                for (int b = 0; b < reg_max; b++) {
                    d += dfl_prob[b] * (float)b;
                }
                dist[side] = d;
            }

            // Anchor centre + stride scale.
            const float anchor_x = (float)col + 0.5f;
            const float anchor_y = (float)row + 0.5f;
            const float x1 = (anchor_x - dist[0]) * stride_f;
            const float y1 = (anchor_y - dist[1]) * stride_f;
            const float x2 = (anchor_x + dist[2]) * stride_f;
            const float y2 = (anchor_y + dist[3]) * stride_f;

            if (x2 <= x1 || y2 <= y1) continue;

            if (cand_size + added >= cand_cap) {
                // Candidate budget exhausted — this shouldn't happen in
                // practice (PI_PP_MAX_CANDIDATES = 256 vs typical <30)
                // but stay safe and drop further hits rather than
                // overflow the caller's buffer.
                return added;
            }

            candidates[cand_size + added] = (pp_candidate_t){
                .x1 = x1, .y1 = y1, .x2 = x2, .y2 = y2, .score = obj,
            };
            added++;
        }
    }
    return added;
}

// -----------------------------------------------------------------------
// NMS (single class)
// -----------------------------------------------------------------------

static inline float max_f(float a, float b) { return a > b ? a : b; }
static inline float min_f(float a, float b) { return a < b ? a : b; }

// Sort descending by score (small N — selection sort is fine here).
// We sort indices into `cands` so the original array can stay immutable,
// which makes the NMS loop easier to reason about.
static void sort_indices_by_score_desc(const pp_candidate_t *cands,
                                       int32_t              *idx,
                                       int                   n) {
    for (int i = 0; i < n; i++) idx[i] = i;
    for (int i = 0; i < n; i++) {
        int best = i;
        for (int j = i + 1; j < n; j++) {
            if (cands[idx[j]].score > cands[idx[best]].score) best = j;
        }
        if (best != i) {
            int32_t t = idx[i]; idx[i] = idx[best]; idx[best] = t;
        }
    }
}

static float iou(const pp_candidate_t *a, const pp_candidate_t *b) {
    const float ix1 = max_f(a->x1, b->x1);
    const float iy1 = max_f(a->y1, b->y1);
    const float ix2 = min_f(a->x2, b->x2);
    const float iy2 = min_f(a->y2, b->y2);
    const float iw  = max_f(0.0f, ix2 - ix1);
    const float ih  = max_f(0.0f, iy2 - iy1);
    const float inter = iw * ih;
    const float area_a = max_f(0.0f, a->x2 - a->x1) *
                          max_f(0.0f, a->y2 - a->y1);
    const float area_b = max_f(0.0f, b->x2 - b->x1) *
                          max_f(0.0f, b->y2 - b->y1);
    const float denom = area_a + area_b - inter;
    if (denom <= 0.0f) return 0.0f;
    return inter / denom;
}

// In-place NMS. Writes kept indices to `keep[]`, returns count.
static int nms_candidates(const pp_candidate_t *cands,
                          int                   n,
                          float                 iou_threshold,
                          int32_t              *keep) {
    if (n <= 0) return 0;
    int32_t idx[PI_PP_MAX_CANDIDATES];
    if (n > PI_PP_MAX_CANDIDATES) n = PI_PP_MAX_CANDIDATES;
    sort_indices_by_score_desc(cands, idx, n);

    bool suppressed[PI_PP_MAX_CANDIDATES] = { false };
    int  n_keep = 0;
    for (int i = 0; i < n; i++) {
        if (suppressed[i]) continue;
        keep[n_keep++] = idx[i];
        for (int j = i + 1; j < n; j++) {
            if (suppressed[j]) continue;
            if (iou(&cands[idx[i]], &cands[idx[j]]) > iou_threshold) {
                suppressed[j] = true;
            }
        }
    }
    return n_keep;
}

// -----------------------------------------------------------------------
// Letterbox reverse
// -----------------------------------------------------------------------
//
// The model consumes a square input_size x input_size letterbox image
// produced from a (frame_width x frame_height) camera frame by:
//   1. scale = min(input / frame_w, input / frame_h)
//   2. resized_w = frame_w * scale;   resized_h = frame_h * scale
//   3. pad_x = (input - resized_w) / 2
//   4. pad_y = (input - resized_h) / 2
// So a letterbox coordinate maps back to a frame coordinate via:
//   frame_x = (letterbox_x - pad_x) / scale
//   frame_y = (letterbox_y - pad_y) / scale
static void letterbox_reverse_bbox(float  input_size,
                                   float  frame_w,
                                   float  frame_h,
                                   float *x1, float *y1,
                                   float *x2, float *y2) {
    const float s_w   = input_size / frame_w;
    const float s_h   = input_size / frame_h;
    const float scale = (s_w < s_h) ? s_w : s_h;
    const float pad_x = (input_size - frame_w * scale) * 0.5f;
    const float pad_y = (input_size - frame_h * scale) * 0.5f;
    const float inv   = (scale > 0.0f) ? (1.0f / scale) : 0.0f;

    *x1 = (*x1 - pad_x) * inv;
    *y1 = (*y1 - pad_y) * inv;
    *x2 = (*x2 - pad_x) * inv;
    *y2 = (*y2 - pad_y) * inv;
}

// -----------------------------------------------------------------------
// Public decode entry point
// -----------------------------------------------------------------------

size_t pi_postprocess_pose_decode(const pi_postprocess_pose_cfg_t *cfg,
                                  pi_detection_t                  *out_dets,
                                  size_t                           out_cap) {
    if (!cfg || !out_dets || out_cap == 0) return 0;
    if (cfg->num_tensors == 0 || cfg->reg_max <= 0 ||
        cfg->reg_max > PI_PP_MAX_REG_MAX) {
        return 0;
    }
    if (cfg->input_size <= 0 || cfg->frame_width <= 0 || cfg->frame_height <= 0) {
        return 0;
    }

    pp_scale_t scales[3];
    const size_t n_scales = group_scales(cfg, scales);
    if (n_scales == 0) return 0;

    pp_candidate_t candidates[PI_PP_MAX_CANDIDATES];
    size_t         n_cands = 0;
    for (size_t s = 0; s < n_scales; s++) {
        n_cands += decode_scale(&scales[s],
                                cfg->score_threshold,
                                cfg->reg_max,
                                candidates, n_cands,
                                PI_PP_MAX_CANDIDATES);
    }
    if (n_cands == 0) return 0;

    int32_t keep[PI_PP_MAX_CANDIDATES];
    const int n_keep = nms_candidates(candidates, (int)n_cands,
                                      cfg->iou_threshold, keep);
    if (n_keep <= 0) return 0;

    // Letterbox reverse + integer round + cap to out_cap.
    size_t written = 0;
    for (int i = 0; i < n_keep && written < out_cap; i++) {
        pp_candidate_t c = candidates[keep[i]];
        letterbox_reverse_bbox((float)cfg->input_size,
                               (float)cfg->frame_width,
                               (float)cfg->frame_height,
                               &c.x1, &c.y1, &c.x2, &c.y2);

        pi_detection_t *d = &out_dets[written++];
        d->x1 = (int32_t)(c.x1 + 0.5f);
        d->y1 = (int32_t)(c.y1 + 0.5f);
        d->x2 = (int32_t)(c.x2 + 0.5f);
        d->y2 = (int32_t)(c.y2 + 0.5f);
        d->score    = c.score;
        d->class_id = 0;       // yolov8 pose head is single-class "person"
        d->_reserved = 0;
    }
    return written;
}
