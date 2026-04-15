#include "pi_streamer/ring_buffer.h"
#include "unity.h"

#include <pthread.h>
#include <sched.h>
#include <stdint.h>

void setUp(void) {}
void tearDown(void) {}

// ---------- Basic single-threaded correctness ---------------------------

static void test_reject_non_power_of_two(void) {
    TEST_ASSERT_NULL(pi_ring_create(0));
    TEST_ASSERT_NULL(pi_ring_create(1));   // must be >= 2
    TEST_ASSERT_NULL(pi_ring_create(3));
    TEST_ASSERT_NULL(pi_ring_create(5));
    TEST_ASSERT_NULL(pi_ring_create(100));
}

static void test_happy_push_pop(void) {
    pi_ring_buffer_t *r = pi_ring_create(4);
    TEST_ASSERT_NOT_NULL(r);

    TEST_ASSERT_EQUAL_UINT64(4, pi_ring_capacity(r));
    TEST_ASSERT_TRUE(pi_ring_is_empty(r));
    TEST_ASSERT_FALSE(pi_ring_is_full(r));
    TEST_ASSERT_EQUAL_UINT64(0, pi_ring_size(r));

    int items[4] = {10, 20, 30, 40};
    for (int i = 0; i < 4; i++) {
        TEST_ASSERT_EQUAL_INT(0, pi_ring_push(r, &items[i]));
    }

    TEST_ASSERT_TRUE(pi_ring_is_full(r));
    TEST_ASSERT_EQUAL_UINT64(4, pi_ring_size(r));
    TEST_ASSERT_EQUAL_INT(-1, pi_ring_push(r, &items[0]));

    void *out = NULL;
    for (int i = 0; i < 4; i++) {
        TEST_ASSERT_EQUAL_INT(0, pi_ring_pop(r, &out));
        TEST_ASSERT_EQUAL_PTR(&items[i], out);
    }

    TEST_ASSERT_TRUE(pi_ring_is_empty(r));
    TEST_ASSERT_EQUAL_INT(-1, pi_ring_pop(r, &out));
    TEST_ASSERT_NULL(out);

    pi_ring_destroy(r);
}

static void test_wrap_around_preserves_order(void) {
    pi_ring_buffer_t *r = pi_ring_create(2);
    TEST_ASSERT_NOT_NULL(r);

    int a = 1, b = 2, c = 3, d = 4;
    void *out = NULL;

    TEST_ASSERT_EQUAL_INT(0, pi_ring_push(r, &a));
    TEST_ASSERT_EQUAL_INT(0, pi_ring_push(r, &b));
    TEST_ASSERT_EQUAL_INT(-1, pi_ring_push(r, &c));  // full

    TEST_ASSERT_EQUAL_INT(0, pi_ring_pop(r, &out));
    TEST_ASSERT_EQUAL_PTR(&a, out);

    TEST_ASSERT_EQUAL_INT(0, pi_ring_push(r, &c));
    TEST_ASSERT_EQUAL_INT(0, pi_ring_pop(r, &out));
    TEST_ASSERT_EQUAL_PTR(&b, out);

    TEST_ASSERT_EQUAL_INT(0, pi_ring_push(r, &d));
    TEST_ASSERT_EQUAL_INT(0, pi_ring_pop(r, &out));
    TEST_ASSERT_EQUAL_PTR(&c, out);
    TEST_ASSERT_EQUAL_INT(0, pi_ring_pop(r, &out));
    TEST_ASSERT_EQUAL_PTR(&d, out);

    TEST_ASSERT_TRUE(pi_ring_is_empty(r));
    pi_ring_destroy(r);
}

static void test_null_handling(void) {
    TEST_ASSERT_EQUAL_INT(-1, pi_ring_push(NULL, (void *)0xdead));
    void *out = (void *)0x1;
    TEST_ASSERT_EQUAL_INT(-1, pi_ring_pop(NULL, &out));
    TEST_ASSERT_NULL(out);

    pi_ring_destroy(NULL);     // NULL-safe
    TEST_PASS();
}

// ---------- Multi-threaded SPSC stress test -----------------------------

#define STRESS_CAPACITY 1024u
#define STRESS_ITEMS    200000u

typedef struct {
    pi_ring_buffer_t *ring;
    uintptr_t         produced;
    uintptr_t         consumed;
    int               consumer_mismatch;
} stress_ctx_t;

static void *producer_thread(void *arg) {
    stress_ctx_t *ctx = (stress_ctx_t *)arg;
    for (uintptr_t i = 1; i <= STRESS_ITEMS; i++) {
        void *payload = (void *)i;
        while (pi_ring_push(ctx->ring, payload) != 0) {
            sched_yield();
        }
        ctx->produced = i;
    }
    return NULL;
}

static void *consumer_thread(void *arg) {
    stress_ctx_t *ctx = (stress_ctx_t *)arg;
    uintptr_t expected = 1;
    while (expected <= STRESS_ITEMS) {
        void *got = NULL;
        if (pi_ring_pop(ctx->ring, &got) == 0) {
            const uintptr_t val = (uintptr_t)got;
            if (val != expected) {
                ctx->consumer_mismatch = 1;
                return NULL;
            }
            expected++;
            ctx->consumed = expected - 1;
        } else {
            sched_yield();
        }
    }
    return NULL;
}

static void test_stress_spsc_order_preserved(void) {
    pi_ring_buffer_t *r = pi_ring_create(STRESS_CAPACITY);
    TEST_ASSERT_NOT_NULL(r);

    stress_ctx_t ctx = { .ring = r };
    pthread_t prod, cons;
    TEST_ASSERT_EQUAL_INT(0,
        pthread_create(&cons, NULL, consumer_thread, &ctx));
    TEST_ASSERT_EQUAL_INT(0,
        pthread_create(&prod, NULL, producer_thread, &ctx));

    pthread_join(prod, NULL);
    pthread_join(cons, NULL);

    TEST_ASSERT_FALSE(ctx.consumer_mismatch);
    TEST_ASSERT_EQUAL_UINT64(STRESS_ITEMS, ctx.produced);
    TEST_ASSERT_EQUAL_UINT64(STRESS_ITEMS, ctx.consumed);
    TEST_ASSERT_TRUE(pi_ring_is_empty(r));

    pi_ring_destroy(r);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_reject_non_power_of_two);
    RUN_TEST(test_happy_push_pop);
    RUN_TEST(test_wrap_around_preserves_order);
    RUN_TEST(test_null_handling);
    RUN_TEST(test_stress_spsc_order_preserved);
    return UNITY_END();
}
