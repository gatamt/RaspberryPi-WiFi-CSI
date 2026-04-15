#include "pi_streamer/inference_state.h"
#include "unity.h"

#include <stdatomic.h>
#include <stdint.h>
#include <string.h>

void setUp(void) {}
void tearDown(void) {}

// Sentinel tensor pointer — the mock inference backend never dereferences
// raw_output in the publish path, so any stable address is fine here.
static const uint8_t kFakeTensor[32] = {0};

static pi_infer_result_t make_result(pi_infer_kind_t kind,
                                     uint64_t        frame_id,
                                     uint64_t        latency_ns) {
    pi_infer_result_t r = {
        .frame_id        = frame_id,
        .kind            = kind,
        .raw_output      = kFakeTensor,
        .raw_output_size = sizeof kFakeTensor,
        .latency_ns      = latency_ns,
    };
    return r;
}

// ---- 1. pi_inference_state_init zeroes every slot ------------------------
static void test_init_is_zero(void) {
    pi_inference_state_t state;
    // Fill with garbage first so we can detect that init really clears it.
    memset(&state, 0xA5, sizeof state);

    pi_inference_state_init(&state);

    for (int k = 0; k < PI_INFER__COUNT; k++) {
        TEST_ASSERT_EQUAL_UINT(0u,
            atomic_load(&state.slots[k].seq));
        TEST_ASSERT_EQUAL_UINT64(0u, state.slots[k].value.frame_id);
        TEST_ASSERT_EQUAL_INT(0,    (int)state.slots[k].value.kind);
        TEST_ASSERT_NULL(state.slots[k].value.raw_output);
        TEST_ASSERT_EQUAL_size_t(0u, state.slots[k].value.raw_output_size);
        TEST_ASSERT_EQUAL_UINT64(0u, state.slots[k].value.latency_ns);
    }
}

static void test_init_null_safe(void) {
    pi_inference_state_init(NULL);  // must not crash
    TEST_PASS();
}

// ---- 2. Publish then snapshot round-trips the value ---------------------
static void test_publish_then_snapshot(void) {
    pi_inference_state_t state;
    pi_inference_state_init(&state);

    const pi_infer_result_t r = make_result(PI_INFER_POSE, 4242u, 123456u);
    pi_inference_state_publish(&state, PI_INFER_POSE, &r);

    pi_inference_state_t snap;
    const bool ok = pi_inference_state_snapshot(&state, &snap);
    TEST_ASSERT_TRUE(ok);

    // Pose slot should contain the published result.
    const pi_infer_result_t *got = &snap.slots[PI_INFER_POSE].value;
    TEST_ASSERT_EQUAL_UINT64(4242u,   got->frame_id);
    TEST_ASSERT_EQUAL_INT(PI_INFER_POSE, got->kind);
    TEST_ASSERT_EQUAL_PTR(kFakeTensor, got->raw_output);
    TEST_ASSERT_EQUAL_size_t(sizeof kFakeTensor, got->raw_output_size);
    TEST_ASSERT_EQUAL_UINT64(123456u, got->latency_ns);

    // Seq should be even (publication completed) and non-zero (one publish
    // happened, bumping it to 2).
    TEST_ASSERT_EQUAL_UINT(2u, atomic_load(&state.slots[PI_INFER_POSE].seq));
}

static void test_publish_null_safe(void) {
    pi_inference_state_t state;
    pi_inference_state_init(&state);
    pi_inference_state_publish(NULL, PI_INFER_POSE, NULL);
    const pi_infer_result_t r = make_result(PI_INFER_POSE, 1u, 1u);
    pi_inference_state_publish(&state, (pi_infer_kind_t)(-1), &r);
    pi_inference_state_publish(&state, (pi_infer_kind_t)PI_INFER__COUNT, &r);
    pi_inference_state_publish(&state, PI_INFER_POSE, NULL);
    TEST_ASSERT_EQUAL_UINT(0u, atomic_load(&state.slots[PI_INFER_POSE].seq));
}

static void test_snapshot_null_safe(void) {
    pi_inference_state_t state;
    pi_inference_state_init(&state);
    pi_inference_state_t out;
    TEST_ASSERT_FALSE(pi_inference_state_snapshot(NULL, &out));
    TEST_ASSERT_FALSE(pi_inference_state_snapshot(&state, NULL));
}

// ---- 3. Publish/snapshot loop — no races since we are single-threaded ---
static void test_snapshot_consistency_loop(void) {
    pi_inference_state_t state;
    pi_inference_state_init(&state);

    for (uint64_t i = 0; i < 10000u; i++) {
        // Alternate which kind we publish to so every slot gets exercised.
        const pi_infer_kind_t k = (pi_infer_kind_t)(i % PI_INFER__COUNT);
        const pi_infer_result_t r = make_result(k, i, i * 100u);

        pi_inference_state_publish(&state, k, &r);

        pi_inference_state_t snap;
        const bool ok = pi_inference_state_snapshot(&state, &snap);
        TEST_ASSERT_TRUE(ok);

        // The slot we just wrote should contain i.
        TEST_ASSERT_EQUAL_UINT64(i, snap.slots[k].value.frame_id);
        TEST_ASSERT_EQUAL_UINT64(i * 100u, snap.slots[k].value.latency_ns);
    }
}

// ---- 4. Publishing to one slot does not disturb the others ----------------
static void test_three_kinds_independent(void) {
    pi_inference_state_t state;
    pi_inference_state_init(&state);

    const pi_infer_result_t r = make_result(PI_INFER_POSE, 7u, 5u);
    pi_inference_state_publish(&state, PI_INFER_POSE, &r);

    pi_inference_state_t snap;
    TEST_ASSERT_TRUE(pi_inference_state_snapshot(&state, &snap));

    // Pose got the data.
    TEST_ASSERT_EQUAL_UINT64(7u, snap.slots[PI_INFER_POSE].value.frame_id);

    // Object and hand are still pristine.
    TEST_ASSERT_EQUAL_UINT64(0u, snap.slots[PI_INFER_OBJECT].value.frame_id);
    TEST_ASSERT_EQUAL_UINT64(0u, snap.slots[PI_INFER_HAND].value.frame_id);
    TEST_ASSERT_EQUAL_UINT64(0u, snap.slots[PI_INFER_OBJECT].value.latency_ns);
    TEST_ASSERT_EQUAL_UINT64(0u, snap.slots[PI_INFER_HAND].value.latency_ns);
}

// ---- 5. Publishing to all three kinds in sequence -----------------------
static void test_publish_all_three(void) {
    pi_inference_state_t state;
    pi_inference_state_init(&state);

    const pi_infer_result_t rp = make_result(PI_INFER_POSE,   100u, 1u);
    const pi_infer_result_t ro = make_result(PI_INFER_OBJECT, 200u, 2u);
    const pi_infer_result_t rh = make_result(PI_INFER_HAND,   300u, 3u);

    pi_inference_state_publish(&state, PI_INFER_POSE,   &rp);
    pi_inference_state_publish(&state, PI_INFER_OBJECT, &ro);
    pi_inference_state_publish(&state, PI_INFER_HAND,   &rh);

    pi_inference_state_t snap;
    TEST_ASSERT_TRUE(pi_inference_state_snapshot(&state, &snap));

    TEST_ASSERT_EQUAL_UINT64(100u, snap.slots[PI_INFER_POSE].value.frame_id);
    TEST_ASSERT_EQUAL_UINT64(200u, snap.slots[PI_INFER_OBJECT].value.frame_id);
    TEST_ASSERT_EQUAL_UINT64(300u, snap.slots[PI_INFER_HAND].value.frame_id);
    TEST_ASSERT_EQUAL_UINT64(1u, snap.slots[PI_INFER_POSE].value.latency_ns);
    TEST_ASSERT_EQUAL_UINT64(2u, snap.slots[PI_INFER_OBJECT].value.latency_ns);
    TEST_ASSERT_EQUAL_UINT64(3u, snap.slots[PI_INFER_HAND].value.latency_ns);
}

// ---- 6. Repeated publish to the same slot — last write wins -------------
static void test_repeated_publish_last_wins(void) {
    pi_inference_state_t state;
    pi_inference_state_init(&state);

    for (uint64_t i = 1u; i <= 500u; i++) {
        const pi_infer_result_t r = make_result(PI_INFER_OBJECT, i, i);
        pi_inference_state_publish(&state, PI_INFER_OBJECT, &r);
    }

    pi_inference_state_t snap;
    TEST_ASSERT_TRUE(pi_inference_state_snapshot(&state, &snap));
    TEST_ASSERT_EQUAL_UINT64(500u, snap.slots[PI_INFER_OBJECT].value.frame_id);
    TEST_ASSERT_EQUAL_UINT64(500u, snap.slots[PI_INFER_OBJECT].value.latency_ns);

    // Every publish bumps seq by 2 (odd -> even), so 500 publishes = 1000.
    TEST_ASSERT_EQUAL_UINT(1000u,
        atomic_load(&state.slots[PI_INFER_OBJECT].seq));
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_init_is_zero);
    RUN_TEST(test_init_null_safe);
    RUN_TEST(test_publish_then_snapshot);
    RUN_TEST(test_publish_null_safe);
    RUN_TEST(test_snapshot_null_safe);
    RUN_TEST(test_snapshot_consistency_loop);
    RUN_TEST(test_three_kinds_independent);
    RUN_TEST(test_publish_all_three);
    RUN_TEST(test_repeated_publish_last_wins);
    return UNITY_END();
}
