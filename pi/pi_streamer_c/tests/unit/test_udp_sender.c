#include "pi_streamer/udp_sender.h"
#include "unity.h"

#include <string.h>

void setUp(void) {}
void tearDown(void) {}

static pi_udp_sender_config_t default_cfg(void) {
    const pi_udp_sender_config_t cfg = {
        .bind_ip        = "0.0.0.0",
        .bind_port      = 3334,
        .sq_depth       = 256,
        .send_buf_bytes = 524288,
    };
    return cfg;
}

static void test_create_destroy(void) {
    pi_udp_sender_config_t cfg = default_cfg();
    pi_udp_sender_t *s = pi_udp_sender_create(&cfg);
    TEST_ASSERT_NOT_NULL(s);
    pi_udp_sender_destroy(s);
}

static void test_send_increments_stats(void) {
    pi_udp_sender_config_t cfg = default_cfg();
    pi_udp_sender_t *s = pi_udp_sender_create(&cfg);

    const uint8_t payload[] = {'V', 'I', 'D', '0'};
    TEST_ASSERT_EQUAL_INT(0,
        pi_udp_sender_send(s, "127.0.0.1", 5000, payload, sizeof payload));

    pi_udp_sender_stats_t stats = {0};
    pi_udp_sender_stats(s, &stats);
    TEST_ASSERT_EQUAL_UINT64(1, stats.datagrams_sent);
    TEST_ASSERT_EQUAL_UINT64(sizeof payload, stats.bytes_sent);

    pi_udp_sender_destroy(s);
}

static void test_send_rejects_null_and_empty(void) {
    pi_udp_sender_config_t cfg = default_cfg();
    pi_udp_sender_t *s = pi_udp_sender_create(&cfg);

    const uint8_t payload[] = {0x01};
    TEST_ASSERT_EQUAL_INT(-1,
        pi_udp_sender_send(NULL, "127.0.0.1", 5000, payload, 1));
    TEST_ASSERT_EQUAL_INT(-1,
        pi_udp_sender_send(s, NULL, 5000, payload, 1));
    TEST_ASSERT_EQUAL_INT(-1,
        pi_udp_sender_send(s, "127.0.0.1", 5000, NULL, 1));
    TEST_ASSERT_EQUAL_INT(-1,
        pi_udp_sender_send(s, "127.0.0.1", 5000, payload, 0));

    pi_udp_sender_destroy(s);
}

static void test_poll_and_try_recv_are_safe(void) {
    pi_udp_sender_config_t cfg = default_cfg();
    pi_udp_sender_t *s = pi_udp_sender_create(&cfg);

    TEST_ASSERT_EQUAL_INT(0, pi_udp_sender_poll(s));

    uint8_t buf[64];
    size_t  received = 42;  // sentinel
    char    ip[32]   = {0};
    uint16_t port    = 42;
    TEST_ASSERT_EQUAL_INT(0,
        pi_udp_sender_try_recv(s, buf, sizeof buf, &received, ip, sizeof ip, &port));
    TEST_ASSERT_EQUAL_UINT64(0, received);
    TEST_ASSERT_EQUAL_UINT16(0, port);

    pi_udp_sender_destroy(s);
}

static void test_null_safe(void) {
    pi_udp_sender_destroy(NULL);
    TEST_ASSERT_EQUAL_INT(-1, pi_udp_sender_poll(NULL));
    pi_udp_sender_stats_t stats = { .datagrams_sent = 99 };
    pi_udp_sender_stats(NULL, &stats);
    TEST_ASSERT_EQUAL_UINT64(0, stats.datagrams_sent);
    TEST_PASS();
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_create_destroy);
    RUN_TEST(test_send_increments_stats);
    RUN_TEST(test_send_rejects_null_and_empty);
    RUN_TEST(test_poll_and_try_recv_are_safe);
    RUN_TEST(test_null_safe);
    return UNITY_END();
}
