#include "pi_streamer/camera.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

static pi_camera_config_t default_cfg(void) {
    const pi_camera_config_t cfg = {
        .width      = 1280,
        .height     = 720,
        .fps        = 30,
        .raw_width  = 2304,
        .raw_height = 1296,
    };
    return cfg;
}

static void test_rejects_null_and_zero_dims(void) {
    TEST_ASSERT_NULL(pi_camera_create(NULL));
    pi_camera_config_t bad = default_cfg();
    bad.width = 0;
    TEST_ASSERT_NULL(pi_camera_create(&bad));
    bad = default_cfg();
    bad.height = 0;
    TEST_ASSERT_NULL(pi_camera_create(&bad));
}

static void test_start_stop_lifecycle(void) {
    pi_camera_config_t cfg = default_cfg();
    pi_camera_t *cam = pi_camera_create(&cfg);
    TEST_ASSERT_NOT_NULL(cam);
    TEST_ASSERT_FALSE(pi_camera_is_running(cam));

    TEST_ASSERT_EQUAL_INT(0, pi_camera_start(cam));
    TEST_ASSERT_TRUE(pi_camera_is_running(cam));

    TEST_ASSERT_EQUAL_INT(0, pi_camera_stop(cam));
    TEST_ASSERT_FALSE(pi_camera_is_running(cam));

    pi_camera_destroy(cam);
}

static void test_capture_without_start_fails(void) {
    pi_camera_config_t cfg = default_cfg();
    pi_camera_t *cam = pi_camera_create(&cfg);
    pi_camera_frame_t f = {0};
    TEST_ASSERT_EQUAL_INT(-1, pi_camera_capture(cam, &f, 100));
    pi_camera_destroy(cam);
}

static void test_capture_produces_monotonic_frame_ids(void) {
    pi_camera_config_t cfg = default_cfg();
    pi_camera_t *cam = pi_camera_create(&cfg);
    pi_camera_start(cam);

    pi_camera_frame_t f1 = {0}, f2 = {0}, f3 = {0};
    TEST_ASSERT_EQUAL_INT(0, pi_camera_capture(cam, &f1, 100));
    TEST_ASSERT_EQUAL_INT(0, pi_camera_capture(cam, &f2, 100));
    TEST_ASSERT_EQUAL_INT(0, pi_camera_capture(cam, &f3, 100));

    TEST_ASSERT_EQUAL_UINT64(0, f1.frame_id);
    TEST_ASSERT_EQUAL_UINT64(1, f2.frame_id);
    TEST_ASSERT_EQUAL_UINT64(2, f3.frame_id);
    TEST_ASSERT_EQUAL_UINT32(1280, f1.width);
    TEST_ASSERT_EQUAL_UINT32(720,  f1.height);
    // T0: stride is now the Y plane stride (1 byte per pixel), not BGR24.
    TEST_ASSERT_EQUAL_UINT32(1280, f1.stride);
    TEST_ASSERT_EQUAL_INT(-1, f1.dma_fd);
    TEST_ASSERT_NOT_NULL(f1.pixels);
    TEST_ASSERT_GREATER_THAN_UINT64(0, f1.timestamp_ns);

    // T0: YUV420 planar metadata — I420 layout.
    TEST_ASSERT_EQUAL_UINT32(1280, f1.y_stride);
    TEST_ASSERT_EQUAL_UINT32(640,  f1.uv_stride);
    TEST_ASSERT_EQUAL_UINT64((uint64_t)1280 * 720,                       (uint64_t)f1.u_offset);
    TEST_ASSERT_EQUAL_UINT64((uint64_t)f1.u_offset + (uint64_t)640 * 360,(uint64_t)f1.v_offset);
    TEST_ASSERT_EQUAL_UINT64((uint64_t)1280 * 720 * 3u / 2u,             (uint64_t)f1.pixels_size);

    pi_camera_release(cam, &f1);
    pi_camera_release(cam, &f2);
    pi_camera_release(cam, &f3);
    pi_camera_destroy(cam);
}

// Check the mock's luma pattern per frame independently so we don't race
// on the shared buffer.
static void test_capture_luma_pattern_per_frame(void) {
    pi_camera_config_t cfg = default_cfg();
    pi_camera_t *cam = pi_camera_create(&cfg);
    pi_camera_start(cam);

    pi_camera_frame_t f = {0};
    TEST_ASSERT_EQUAL_INT(0, pi_camera_capture(cam, &f, 100));
    TEST_ASSERT_EQUAL_UINT8(0x00, f.pixels[0]);
    TEST_ASSERT_EQUAL_UINT8(0x80, f.pixels[f.u_offset]);
    TEST_ASSERT_EQUAL_UINT8(0x80, f.pixels[f.v_offset]);
    pi_camera_release(cam, &f);

    TEST_ASSERT_EQUAL_INT(0, pi_camera_capture(cam, &f, 100));
    TEST_ASSERT_EQUAL_UINT8(0x01, f.pixels[0]);
    pi_camera_release(cam, &f);

    pi_camera_destroy(cam);
}

static void test_null_is_safe(void) {
    pi_camera_destroy(NULL);
    pi_camera_release(NULL, NULL);
    TEST_ASSERT_EQUAL_INT(-1, pi_camera_start(NULL));
    TEST_ASSERT_EQUAL_INT(-1, pi_camera_stop(NULL));
    TEST_ASSERT_FALSE(pi_camera_is_running(NULL));
    TEST_PASS();
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_rejects_null_and_zero_dims);
    RUN_TEST(test_start_stop_lifecycle);
    RUN_TEST(test_capture_without_start_fails);
    RUN_TEST(test_capture_produces_monotonic_frame_ids);
    RUN_TEST(test_capture_luma_pattern_per_frame);
    RUN_TEST(test_null_is_safe);
    return UNITY_END();
}
