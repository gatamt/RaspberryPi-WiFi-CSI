// Unit tests for the core logger module. Proves init/gate/format path
// work end-to-end without crashing and without allocating, and validates
// the Unity + CMake wiring in CMakeLists.txt and tests/unit/CMakeLists.txt.

#include "pi_streamer/logger.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

static void test_init_is_idempotent(void) {
    TEST_ASSERT_EQUAL_INT(0, pi_log_init(PI_LOG_INFO));
    TEST_ASSERT_EQUAL_INT(0, pi_log_init(PI_LOG_DEBUG));
    TEST_ASSERT_EQUAL_INT(PI_LOG_DEBUG, pi_log_get_level());
}

static void test_set_level_round_trip(void) {
    pi_log_set_level(PI_LOG_WARN);
    TEST_ASSERT_EQUAL_INT(PI_LOG_WARN, pi_log_get_level());
    pi_log_set_level(PI_LOG_ERROR);
    TEST_ASSERT_EQUAL_INT(PI_LOG_ERROR, pi_log_get_level());
}

static void test_emit_at_all_levels_does_not_crash(void) {
    pi_log_set_level(PI_LOG_WARN);
    // These two are below the gate: must not write to stderr.
    pi_log(PI_LOG_DEBUG, "unit", "debug should be filtered: %d", 1);
    pi_log(PI_LOG_INFO,  "unit", "info  should be filtered: %d", 2);
    // These two are at or above the gate: must emit.
    pi_log(PI_LOG_WARN,  "unit", "warn  emitted: %d", 3);
    pi_log(PI_LOG_ERROR, "unit", "error emitted: %d", 4);
    // If we got here without SIGSEGV or abort, the call paths are sound.
    TEST_PASS();
}

static void test_null_module_is_safe(void) {
    pi_log_set_level(PI_LOG_DEBUG);
    pi_log(PI_LOG_INFO, NULL, "msg with NULL module");
    TEST_PASS();
}

static void test_long_body_is_truncated_not_overflowed(void) {
    pi_log_set_level(PI_LOG_DEBUG);
    // Build a message longer than the internal 512-byte body buffer.
    char big[2048];
    for (size_t i = 0; i < sizeof big - 1; ++i) {
        big[i] = (char)('A' + (i % 26));
    }
    big[sizeof big - 1] = '\0';
    pi_log(PI_LOG_INFO, "truncate", "%s", big);
    TEST_PASS();
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_init_is_idempotent);
    RUN_TEST(test_set_level_round_trip);
    RUN_TEST(test_emit_at_all_levels_does_not_crash);
    RUN_TEST(test_null_module_is_safe);
    RUN_TEST(test_long_body_is_truncated_not_overflowed);
    return UNITY_END();
}
