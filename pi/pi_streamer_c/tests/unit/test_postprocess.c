#include "pi_streamer/postprocess.h"
#include "pi_streamer/detection.h"

#include "unity.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

// -----------------------------------------------------------------------
// Tiny synthetic YOLOv8 pose head at 320x320 input.
//
// Using 320 keeps the three scale grids small and the test buffers
// stack-friendly (40x40, 20x20, 10x10). The decoder's math is scale
// invariant — whatever works at 320 works at 640 in production.
//
// reg_max = 16, 1 object class (person), 17 keypoints.
// -----------------------------------------------------------------------

#define INPUT_SIZE 320
#define REG_MAX    16
#define BBOX_CH    (REG_MAX * 4)  // 64
#define OBJ_CH     1
#define KPTS_CH    (17 * 3)       // 51

#define P3_H 40
#define P3_W 40
#define P4_H 20
#define P4_W 20
#define P5_H 10
#define P5_W 10

// Allocate one tensor's worth of u8 storage.
static uint8_t p3_bbox[P3_H * P3_W * BBOX_CH];
static uint8_t p3_obj [P3_H * P3_W * OBJ_CH];
static uint8_t p3_kpts[P3_H * P3_W * KPTS_CH];
static uint8_t p4_bbox[P4_H * P4_W * BBOX_CH];
static uint8_t p4_obj [P4_H * P4_W * OBJ_CH];
static uint8_t p4_kpts[P4_H * P4_W * KPTS_CH];
static uint8_t p5_bbox[P5_H * P5_W * BBOX_CH];
static uint8_t p5_obj [P5_H * P5_W * OBJ_CH];
static uint8_t p5_kpts[P5_H * P5_W * KPTS_CH];

void setUp(void) {
    memset(p3_bbox, 0, sizeof p3_bbox);
    memset(p3_obj,  0, sizeof p3_obj);
    memset(p3_kpts, 0, sizeof p3_kpts);
    memset(p4_bbox, 0, sizeof p4_bbox);
    memset(p4_obj,  0, sizeof p4_obj);
    memset(p4_kpts, 0, sizeof p4_kpts);
    memset(p5_bbox, 0, sizeof p5_bbox);
    memset(p5_obj,  0, sizeof p5_obj);
    memset(p5_kpts, 0, sizeof p5_kpts);
}

void tearDown(void) {}

// Build a default cfg backed by the static buffers above. obj tensors
// use the real "post-sigmoid probability via scale = 1/255" convention;
// bbox + kpts use scale = 1 so tests can set raw DFL bin weights
// directly without quantization gymnastics.
static void make_cfg(pi_postprocess_pose_cfg_t *cfg,
                     int32_t frame_w, int32_t frame_h) {
    memset(cfg, 0, sizeof *cfg);
    cfg->num_tensors     = 9;
    cfg->input_size      = INPUT_SIZE;
    cfg->reg_max         = REG_MAX;
    cfg->frame_width     = frame_w;
    cfg->frame_height    = frame_h;
    cfg->score_threshold = 0.25f;
    cfg->iou_threshold   = 0.7f;

    const struct {
        void   *data;
        int32_t gh, gw, ch;
    } plan[9] = {
        { p3_bbox, P3_H, P3_W, BBOX_CH },
        { p3_obj,  P3_H, P3_W, OBJ_CH  },
        { p3_kpts, P3_H, P3_W, KPTS_CH },
        { p4_bbox, P4_H, P4_W, BBOX_CH },
        { p4_obj,  P4_H, P4_W, OBJ_CH  },
        { p4_kpts, P4_H, P4_W, KPTS_CH },
        { p5_bbox, P5_H, P5_W, BBOX_CH },
        { p5_obj,  P5_H, P5_W, OBJ_CH  },
        { p5_kpts, P5_H, P5_W, KPTS_CH },
    };
    for (int i = 0; i < 9; i++) {
        cfg->tensors[i] = (pi_tensor_view_t){
            .data       = plan[i].data,
            .dtype      = PI_TENSOR_DTYPE_U8,
            .scale      = (plan[i].ch == OBJ_CH) ? (1.0f / 255.0f) : 1.0f,
            .zero_point = 0.0f,
            .grid_h     = plan[i].gh,
            .grid_w     = plan[i].gw,
            .channels   = plan[i].ch,
        };
    }
}

// Plant a single "detection" at the given cell in the p3 scale. The
// bbox logits are set so the DFL softmax collapses onto a known distance
// (via a very tall peak at bin `d`). Objectness is set to 255 so post-
// dequantize it is exactly 1.0.
static void plant_p3(int row, int col,
                     int d_left, int d_top, int d_right, int d_bottom) {
    const size_t spatial = (size_t)row * P3_W + (size_t)col;
    uint8_t *bb = p3_bbox + spatial * (size_t)BBOX_CH;

    // Peak at the target bin; every other bin is 0 (neutral baseline
    // for softmax). Peak value 200 is plenty — e^200 dominates e^0.
    const int dists[4] = { d_left, d_top, d_right, d_bottom };
    for (int side = 0; side < 4; side++) {
        const int peak_bin = dists[side];
        for (int b = 0; b < REG_MAX; b++) {
            bb[side * REG_MAX + b] = (b == peak_bin) ? 200u : 0u;
        }
    }

    p3_obj[spatial] = 255;
}

// Plant a detection in the p5 scale with the same convention.
static void plant_p5(int row, int col,
                     int d_left, int d_top, int d_right, int d_bottom) {
    const size_t spatial = (size_t)row * P5_W + (size_t)col;
    uint8_t *bb = p5_bbox + spatial * (size_t)BBOX_CH;
    const int dists[4] = { d_left, d_top, d_right, d_bottom };
    for (int side = 0; side < 4; side++) {
        const int peak_bin = dists[side];
        for (int b = 0; b < REG_MAX; b++) {
            bb[side * REG_MAX + b] = (b == peak_bin) ? 200u : 0u;
        }
    }
    p5_obj[spatial] = 255;
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

void test_null_and_empty(void) {
    pi_detection_t out[4];
    TEST_ASSERT_EQUAL_UINT(0, pi_postprocess_pose_decode(NULL, out, 4));

    pi_postprocess_pose_cfg_t cfg = {0};
    TEST_ASSERT_EQUAL_UINT(0, pi_postprocess_pose_decode(&cfg, out, 4));
}

void test_no_detection_when_all_zero(void) {
    pi_postprocess_pose_cfg_t cfg;
    make_cfg(&cfg, 320, 320);

    pi_detection_t out[4];
    TEST_ASSERT_EQUAL_UINT(
        0, pi_postprocess_pose_decode(&cfg, out, 4));
}

void test_single_detection_p3(void) {
    // frame == input: no letterbox padding, no scale — easy to verify
    // the bbox math directly. p3 stride = 320 / 40 = 8.
    pi_postprocess_pose_cfg_t cfg;
    make_cfg(&cfg, 320, 320);

    // Anchor at (col=5, row=5) → anchor centre = (5.5, 5.5) cells.
    // DFL distances 2, 2, 2, 2 → bbox in cells is [3.5..7.5, 3.5..7.5].
    // Multiplied by stride 8 → pixels [28..60, 28..60]. After letterbox
    // reverse with scale 1, pad 0 → same.
    plant_p3(5, 5, /*l*/2, /*t*/2, /*r*/2, /*b*/2);

    pi_detection_t out[4] = {0};
    const size_t n = pi_postprocess_pose_decode(&cfg, out, 4);
    TEST_ASSERT_EQUAL_UINT(1, n);

    TEST_ASSERT_INT32_WITHIN(1, 28, out[0].x1);
    TEST_ASSERT_INT32_WITHIN(1, 28, out[0].y1);
    TEST_ASSERT_INT32_WITHIN(1, 60, out[0].x2);
    TEST_ASSERT_INT32_WITHIN(1, 60, out[0].y2);
    TEST_ASSERT_EQUAL_INT16(0, out[0].class_id);
    // post-sigmoid probability ~1.0
    TEST_ASSERT_TRUE(out[0].score > 0.9f);
}

void test_nms_suppresses_overlapping(void) {
    pi_postprocess_pose_cfg_t cfg;
    make_cfg(&cfg, 320, 320);

    // Two anchors stamped one cell apart, both with the same box
    // extent. Their bboxes overlap heavily (IoU > 0.7) so NMS must
    // drop one. We pick p3 row=5 and row=6 at col=5, bbox extent ±3
    // cells each direction (24 px wide/tall at stride 8).
    plant_p3(5, 5, 3, 3, 3, 3);
    plant_p3(6, 5, 3, 3, 3, 3);

    // Push the second anchor's score slightly lower so the sort is
    // deterministic (the decoder sorts DESC by score, so p3[5,5]
    // should win NMS and appear in the output).
    p3_obj[5 * P3_W + 5] = 255;
    p3_obj[6 * P3_W + 5] = 200;

    pi_detection_t out[4];
    const size_t n = pi_postprocess_pose_decode(&cfg, out, 4);
    TEST_ASSERT_EQUAL_UINT(1, n);
}

void test_nms_keeps_separated(void) {
    pi_postprocess_pose_cfg_t cfg;
    make_cfg(&cfg, 320, 320);

    // Two detections far apart in different scales — IoU ≈ 0 → both
    // survive NMS.
    plant_p3(5, 5, 2, 2, 2, 2);    // ~[28..60, 28..60]
    plant_p5(8, 8, 2, 2, 2, 2);    // p5 stride 32: [208..240, 208..240]

    pi_detection_t out[4] = {0};
    const size_t n = pi_postprocess_pose_decode(&cfg, out, 4);
    TEST_ASSERT_EQUAL_UINT(2, n);
}

void test_score_threshold_blocks_below(void) {
    pi_postprocess_pose_cfg_t cfg;
    make_cfg(&cfg, 320, 320);
    cfg.score_threshold = 0.9f;

    plant_p3(5, 5, 2, 2, 2, 2);
    // Objectness 100/255 ≈ 0.39 — below 0.9 threshold.
    p3_obj[5 * P3_W + 5] = 100;

    pi_detection_t out[4];
    const size_t n = pi_postprocess_pose_decode(&cfg, out, 4);
    TEST_ASSERT_EQUAL_UINT(0, n);
}

void test_letterbox_reverse_wider_frame(void) {
    // Frame 1280x720, model input 320 → letterbox scale 0.25 (limited
    // by width), resized 320x180, pad_y = (320-180)/2 = 70. A letterbox
    // coordinate at (160, 160) should map to frame (640, (160-70)/0.25)
    // = (640, 360) — centre of the frame.
    pi_postprocess_pose_cfg_t cfg;
    make_cfg(&cfg, 1280, 720);

    // Plant a tight box around (160, 160) in letterbox pixel space.
    // p3 stride = 8, so cell = 20. Anchor centre = 20.5 cells = 164 px.
    // Distances 0.5 each → box [160..168, 160..168]. After letterbox
    // reverse: frame_x = (160 - 0) / 0.25 = 640, frame_y = (160 - 70) /
    // 0.25 = 360.
    //
    // Because the DFL bin weighting gives integer distances only (our
    // planting function picks a single peak), we use 1 instead of 0.5
    // and adjust expectations accordingly.
    plant_p3(20, 20, 1, 1, 1, 1);

    pi_detection_t out[4];
    const size_t n = pi_postprocess_pose_decode(&cfg, out, 4);
    TEST_ASSERT_EQUAL_UINT(1, n);

    // Cell 20, anchor centre (20.5, 20.5) cells → (164, 164) letterbox
    // px. With distance 1 on each side: box [156..172, 156..172]
    // letterbox, reverse → frame [624..688, (156-70)/0.25 .. (172-70)/
    // 0.25] = [624..688, 344..408].
    TEST_ASSERT_INT32_WITHIN(2, 624, out[0].x1);
    TEST_ASSERT_INT32_WITHIN(2, 688, out[0].x2);
    TEST_ASSERT_INT32_WITHIN(2, 344, out[0].y1);
    TEST_ASSERT_INT32_WITHIN(2, 408, out[0].y2);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_null_and_empty);
    RUN_TEST(test_no_detection_when_all_zero);
    RUN_TEST(test_single_detection_p3);
    RUN_TEST(test_nms_suppresses_overlapping);
    RUN_TEST(test_nms_keeps_separated);
    RUN_TEST(test_score_threshold_blocks_below);
    RUN_TEST(test_letterbox_reverse_wider_frame);
    return UNITY_END();
}
