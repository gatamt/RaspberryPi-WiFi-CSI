#include "pi_streamer/encoder.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

static pi_encoder_config_t default_cfg(void) {
    const pi_encoder_config_t cfg = {
        .width        = 1280,
        .height       = 720,
        .fps          = 30,
        .bitrate_bps  = 2500000,
        .gop_size     = 30,
        .zero_latency = true,
    };
    return cfg;
}

static void test_create_rejects_invalid(void) {
    TEST_ASSERT_NULL(pi_encoder_create(NULL));
    pi_encoder_config_t bad = default_cfg();
    bad.gop_size = 0;
    TEST_ASSERT_NULL(pi_encoder_create(&bad));
    bad = default_cfg();
    bad.width = 0;
    TEST_ASSERT_NULL(pi_encoder_create(&bad));
}

static void test_submit_then_next_packet(void) {
    pi_encoder_config_t cfg = default_cfg();
    pi_encoder_t *enc = pi_encoder_create(&cfg);
    TEST_ASSERT_NOT_NULL(enc);

    // Dummy YUV pointers (mock doesn't read them).
    uint8_t y[16] = {0}, u[4] = {0}, v[4] = {0};
    TEST_ASSERT_EQUAL_INT(0, pi_encoder_submit_yuv420(
        enc, y, 4, u, 2, v, 2, /*pts_ms=*/100, /*force_keyframe=*/true));

    pi_encoded_packet_t pkt;
    TEST_ASSERT_EQUAL_INT(0, pi_encoder_next_packet(enc, &pkt));
    TEST_ASSERT_EQUAL_UINT64(100, pkt.pts_ms);
    TEST_ASSERT_TRUE(pkt.is_keyframe);
    TEST_ASSERT_NOT_NULL(pkt.nal);
    TEST_ASSERT_GREATER_THAN_UINT(4u, pkt.nal_size);
    // Annex-B start code
    TEST_ASSERT_EQUAL_UINT8(0x00, pkt.nal[0]);
    TEST_ASSERT_EQUAL_UINT8(0x00, pkt.nal[1]);
    TEST_ASSERT_EQUAL_UINT8(0x00, pkt.nal[2]);
    TEST_ASSERT_EQUAL_UINT8(0x01, pkt.nal[3]);

    // No second packet queued.
    TEST_ASSERT_EQUAL_INT(-1, pi_encoder_next_packet(enc, &pkt));

    pi_encoder_destroy(enc);
}

static void test_gop_cycle_produces_periodic_keyframes(void) {
    pi_encoder_config_t cfg = default_cfg();
    cfg.gop_size = 4;
    pi_encoder_t *enc = pi_encoder_create(&cfg);

    uint8_t y[16] = {0}, u[4] = {0}, v[4] = {0};
    const bool expected_keyframe[8] = {
        true, false, false, false,   // first GOP: frame 0 is I
        true, false, false, false,   // second GOP
    };
    for (int i = 0; i < 8; i++) {
        TEST_ASSERT_EQUAL_INT(0, pi_encoder_submit_yuv420(
            enc, y, 4, u, 2, v, 2, /*pts_ms=*/(uint64_t)i, false));
        pi_encoded_packet_t pkt;
        TEST_ASSERT_EQUAL_INT(0, pi_encoder_next_packet(enc, &pkt));
        TEST_ASSERT_EQUAL(expected_keyframe[i], pkt.is_keyframe);
    }

    pi_encoder_destroy(enc);
}

static void test_flush_and_destroy_null_safe(void) {
    pi_encoder_flush(NULL);
    pi_encoder_destroy(NULL);
    TEST_PASS();
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_create_rejects_invalid);
    RUN_TEST(test_submit_then_next_packet);
    RUN_TEST(test_gop_cycle_produces_periodic_keyframes);
    RUN_TEST(test_flush_and_destroy_null_safe);
    return UNITY_END();
}
