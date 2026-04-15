#include "pi_streamer/overlay.h"
#include "pi_streamer/detection.h"

#include "unity.h"

#include <string.h>

// Small canvas so the tests are easy to read. 32x16 with stride == width.
#define W 32
#define H 16
#define Y_STRIDE W

static uint8_t canvas[H * Y_STRIDE];

void setUp(void) {
    memset(canvas, 0, sizeof canvas);
}

void tearDown(void) {}

// Count pixels with the given value across the whole canvas.
static size_t count_value(uint8_t v) {
    size_t n = 0;
    for (size_t i = 0; i < sizeof canvas; i++) {
        if (canvas[i] == v) n++;
    }
    return n;
}

static size_t count_value_in_row(int row, uint8_t v) {
    size_t n = 0;
    for (int c = 0; c < W; c++) {
        if (canvas[row * Y_STRIDE + c] == v) n++;
    }
    return n;
}

static size_t count_value_in_col(int col, uint8_t v) {
    size_t n = 0;
    for (int r = 0; r < H; r++) {
        if (canvas[r * Y_STRIDE + col] == v) n++;
    }
    return n;
}

// ---- Tests ----

void test_null_ptr_no_crash(void) {
    pi_detection_t d = { .x1 = 2, .y1 = 2, .x2 = 10, .y2 = 8 };
    pi_overlay_draw_detections(NULL, W, H, Y_STRIDE, &d, 1, 1, 255);
    // Also dets=NULL + n>0 is a no-op, not a crash.
    pi_overlay_draw_detections(canvas, W, H, Y_STRIDE, NULL, 3, 1, 255);
    TEST_ASSERT_EQUAL_UINT(0, count_value(255));
}

void test_empty_detections_no_op(void) {
    pi_overlay_draw_detections(canvas, W, H, Y_STRIDE, NULL, 0, 1, 255);
    TEST_ASSERT_EQUAL_UINT(0, count_value(255));
}

void test_single_bbox_thickness_1(void) {
    // 10x8 bbox at (2, 2) .. (11, 9). 10 wide, 8 tall.
    //
    // Top + bottom edge: 10 pixels each = 20
    // Left + right edge: 8 pixels each  = 16
    // Corners are shared; with thickness 1 the top/bottom bars are
    // the full 10 pixels and the left/right bars are the full 8
    // pixels BUT the corners overlap (4 pixels double-counted).
    //
    // Total unique stamped pixels = 10 + 10 + 8 + 8 - 4 = 32.
    pi_detection_t d = { .x1 = 2, .y1 = 2, .x2 = 11, .y2 = 9 };
    pi_overlay_draw_detections(canvas, W, H, Y_STRIDE, &d, 1, 1, 255);

    TEST_ASSERT_EQUAL_UINT(32, count_value(255));

    // Row 2 (top edge) has exactly 10 stamped pixels.
    TEST_ASSERT_EQUAL_UINT(10, count_value_in_row(2, 255));
    // Row 9 (bottom edge) has exactly 10 stamped pixels.
    TEST_ASSERT_EQUAL_UINT(10, count_value_in_row(9, 255));
    // Column 2 (left edge) has exactly 8 stamped pixels.
    TEST_ASSERT_EQUAL_UINT(8, count_value_in_col(2, 255));
    // Column 11 (right edge) has 8 stamped pixels.
    TEST_ASSERT_EQUAL_UINT(8, count_value_in_col(11, 255));

    // Inside the box must be untouched.
    for (int r = 3; r <= 8; r++) {
        for (int c = 3; c <= 10; c++) {
            TEST_ASSERT_EQUAL_UINT8(0, canvas[r * Y_STRIDE + c]);
        }
    }
}

void test_corner_clamp_partial_off_screen(void) {
    // Bbox extends past the right + bottom edges. x2=40 clamps to 31,
    // y2=20 clamps to 15. Must not crash and must not write past the
    // canvas.
    pi_detection_t d = { .x1 = 28, .y1 = 12, .x2 = 40, .y2 = 20 };
    pi_overlay_draw_detections(canvas, W, H, Y_STRIDE, &d, 1, 1, 200);

    // All stamped pixels must live inside the canvas.
    size_t stamped = 0;
    for (int r = 0; r < H; r++) {
        for (int c = 0; c < W; c++) {
            if (canvas[r * Y_STRIDE + c] == 200) stamped++;
        }
    }
    TEST_ASSERT_TRUE(stamped > 0);

    // Row 12 (top edge) must have pixels from col 28..31 (4 pixels).
    TEST_ASSERT_EQUAL_UINT(4, count_value_in_row(12, 200));
    // Col 28 (left edge) must have pixels from row 12..15 (4 pixels).
    TEST_ASSERT_EQUAL_UINT(4, count_value_in_col(28, 200));
}

void test_fully_off_screen_no_op(void) {
    // Bbox entirely outside the canvas — left of x=0.
    pi_detection_t d = { .x1 = -20, .y1 = 4, .x2 = -1, .y2 = 10 };
    pi_overlay_draw_detections(canvas, W, H, Y_STRIDE, &d, 1, 1, 200);
    TEST_ASSERT_EQUAL_UINT(0, count_value(200));

    // Below the canvas.
    pi_detection_t d2 = { .x1 = 4, .y1 = 20, .x2 = 10, .y2 = 30 };
    pi_overlay_draw_detections(canvas, W, H, Y_STRIDE, &d2, 1, 1, 200);
    TEST_ASSERT_EQUAL_UINT(0, count_value(200));
}

void test_degenerate_zero_area_no_op(void) {
    // x1 == x2 (zero width)
    pi_detection_t d = { .x1 = 5, .y1 = 5, .x2 = 5, .y2 = 10 };
    pi_overlay_draw_detections(canvas, W, H, Y_STRIDE, &d, 1, 1, 100);
    TEST_ASSERT_EQUAL_UINT(0, count_value(100));

    // y1 == y2 (zero height)
    pi_detection_t d2 = { .x1 = 5, .y1 = 7, .x2 = 10, .y2 = 7 };
    pi_overlay_draw_detections(canvas, W, H, Y_STRIDE, &d2, 1, 1, 100);
    TEST_ASSERT_EQUAL_UINT(0, count_value(100));
}

void test_multiple_boxes(void) {
    pi_detection_t dets[2] = {
        { .x1 = 1, .y1 = 1, .x2 = 6,  .y2 = 4  },  // 6x4 bbox
        { .x1 = 20, .y1 = 8, .x2 = 27, .y2 = 14 }, // 8x7 bbox
    };
    pi_overlay_draw_detections(canvas, W, H, Y_STRIDE, dets, 2, 1, 128);

    // Verify box 1's top edge: row 1, cols 1..6.
    for (int c = 1; c <= 6; c++) {
        TEST_ASSERT_EQUAL_UINT8(128, canvas[1 * Y_STRIDE + c]);
    }
    // Verify box 2's top edge: row 8, cols 20..27.
    for (int c = 20; c <= 27; c++) {
        TEST_ASSERT_EQUAL_UINT8(128, canvas[8 * Y_STRIDE + c]);
    }
    // Mid-canvas gap between the boxes must be empty.
    TEST_ASSERT_EQUAL_UINT8(0, canvas[5 * Y_STRIDE + 15]);
}

void test_thickness_2(void) {
    // 10x8 bbox. Thickness 2 means top+bottom bars are each 2 rows,
    // left+right bars each 2 columns. Overlapping corners = 4 x 4 = 16.
    //
    // Stamped pixels:
    //   top bar:   rows 2..3, cols 2..11 → 20
    //   bottom:    rows 8..9, cols 2..11 → 20
    //   left bar:  rows 2..9, cols 2..3  → 16
    //   right bar: rows 2..9, cols 10..11→ 16
    // Unique: 20 + 20 + 16 + 16 - (4 corners * 4 pixels each) = 56.
    pi_detection_t d = { .x1 = 2, .y1 = 2, .x2 = 11, .y2 = 9 };
    pi_overlay_draw_detections(canvas, W, H, Y_STRIDE, &d, 1, 2, 255);

    TEST_ASSERT_EQUAL_UINT(56, count_value(255));

    // Inner hollow region (rows 4..7, cols 4..9) must still be zero.
    for (int r = 4; r <= 7; r++) {
        for (int c = 4; c <= 9; c++) {
            TEST_ASSERT_EQUAL_UINT8(0, canvas[r * Y_STRIDE + c]);
        }
    }
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_null_ptr_no_crash);
    RUN_TEST(test_empty_detections_no_op);
    RUN_TEST(test_single_bbox_thickness_1);
    RUN_TEST(test_corner_clamp_partial_off_screen);
    RUN_TEST(test_fully_off_screen_no_op);
    RUN_TEST(test_degenerate_zero_area_no_op);
    RUN_TEST(test_multiple_boxes);
    RUN_TEST(test_thickness_2);
    return UNITY_END();
}
