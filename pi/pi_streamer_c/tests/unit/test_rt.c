// Unit tests for pi_streamer/rt.h. These run on BOTH host and Pi:
//   - On host (PI_TARGET unset), every helper is a no-op and returns 0.
//   - On Pi, the helpers actually hit the kernel; mlockall + name + pin
//     succeed under systemd's capability set, while promote_fifo may or
//     may not depending on CAP_SYS_NICE — we accept either for safety.

#include "pi_streamer/rt.h"
#include "unity.h"

#include <pthread.h>

void setUp(void) {}
void tearDown(void) {}

static void test_lock_memory_returns_zero_or_host_noop(void) {
    int rc = pi_rt_lock_memory();
    TEST_ASSERT_EQUAL_INT(0, rc);  // host no-op OR Pi + systemd caps = OK
}

static void test_name_thread_current(void) {
    int rc = pi_rt_name_thread(pthread_self(), "pi-test");
    TEST_ASSERT_EQUAL_INT(0, rc);
}

static void test_pin_thread_current(void) {
    // On host this is a no-op and returns 0. On Pi, cpu 0 always exists.
    int rc = pi_rt_pin_thread(pthread_self(), 0);
    TEST_ASSERT_EQUAL_INT(0, rc);
}

static void test_promote_fifo_rejected_gracefully(void) {
    // Without CAP_SYS_NICE + root, this will likely fail on Pi, return 0
    // on host. Either is acceptable — we just check the function doesn't
    // segfault on a valid pthread_t.
    (void)pi_rt_promote_fifo(pthread_self(), 1);
    TEST_PASS();
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_lock_memory_returns_zero_or_host_noop);
    RUN_TEST(test_name_thread_current);
    RUN_TEST(test_pin_thread_current);
    RUN_TEST(test_promote_fifo_rejected_gracefully);
    return UNITY_END();
}
