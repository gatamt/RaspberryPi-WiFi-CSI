#include "pi_streamer/state_machine.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

typedef struct {
    int         count;
    pi_state_t  last_from;
    pi_state_t  last_to;
    pi_event_t  last_event;
} obs_t;

static void obs_cb(pi_state_t from, pi_state_t to,
                   pi_event_t event, void *ctx) {
    obs_t *o = (obs_t *)ctx;
    o->count++;
    o->last_from  = from;
    o->last_to    = to;
    o->last_event = event;
}

static void test_initial_state_is_idle(void) {
    pi_hsm_t *hsm = pi_hsm_create(NULL, NULL);
    TEST_ASSERT_NOT_NULL(hsm);
    TEST_ASSERT_EQUAL_INT(PI_STATE_IDLE, pi_hsm_current_state(hsm));
    TEST_ASSERT_FALSE(pi_hsm_is_terminal(hsm));
    pi_hsm_destroy(hsm);
}

static void test_idle_to_streaming_flow(void) {
    obs_t obs = {0};
    pi_hsm_t *hsm = pi_hsm_create(obs_cb, &obs);

    TEST_ASSERT_EQUAL_INT(1, pi_hsm_dispatch(hsm, PI_EVENT_START));
    TEST_ASSERT_EQUAL_INT(PI_STATE_STARTING, pi_hsm_current_state(hsm));

    TEST_ASSERT_EQUAL_INT(1, pi_hsm_dispatch(hsm, PI_EVENT_START_OK));
    TEST_ASSERT_EQUAL_INT(PI_STATE_STREAMING, pi_hsm_current_state(hsm));

    TEST_ASSERT_EQUAL_INT(2, obs.count);
    TEST_ASSERT_EQUAL_INT(PI_STATE_STARTING, obs.last_from);
    TEST_ASSERT_EQUAL_INT(PI_STATE_STREAMING, obs.last_to);
    TEST_ASSERT_EQUAL_INT(PI_EVENT_START_OK, obs.last_event);

    pi_hsm_destroy(hsm);
}

static void test_starting_to_error_on_start_err(void) {
    pi_hsm_t *hsm = pi_hsm_create(NULL, NULL);
    pi_hsm_dispatch(hsm, PI_EVENT_START);
    TEST_ASSERT_EQUAL_INT(1, pi_hsm_dispatch(hsm, PI_EVENT_START_ERR));
    TEST_ASSERT_EQUAL_INT(PI_STATE_ERROR, pi_hsm_current_state(hsm));
    pi_hsm_destroy(hsm);
}

static void test_pause_resume_cycle(void) {
    pi_hsm_t *hsm = pi_hsm_create(NULL, NULL);
    pi_hsm_dispatch(hsm, PI_EVENT_START);
    pi_hsm_dispatch(hsm, PI_EVENT_START_OK);

    TEST_ASSERT_EQUAL_INT(1, pi_hsm_dispatch(hsm, PI_EVENT_PAUSE));
    TEST_ASSERT_EQUAL_INT(PI_STATE_PAUSED, pi_hsm_current_state(hsm));

    TEST_ASSERT_EQUAL_INT(1, pi_hsm_dispatch(hsm, PI_EVENT_RESUME));
    TEST_ASSERT_EQUAL_INT(PI_STATE_STREAMING, pi_hsm_current_state(hsm));

    pi_hsm_destroy(hsm);
}

static void test_streaming_self_transition_on_frame_ready(void) {
    obs_t obs = {0};
    pi_hsm_t *hsm = pi_hsm_create(obs_cb, &obs);
    pi_hsm_dispatch(hsm, PI_EVENT_START);
    pi_hsm_dispatch(hsm, PI_EVENT_START_OK);

    const int before = obs.count;
    TEST_ASSERT_EQUAL_INT(1, pi_hsm_dispatch(hsm, PI_EVENT_FRAME_READY));
    TEST_ASSERT_EQUAL_INT(PI_STATE_STREAMING, pi_hsm_current_state(hsm));
    TEST_ASSERT_EQUAL_INT(before + 1, obs.count);
    TEST_ASSERT_EQUAL_INT(PI_STATE_STREAMING, obs.last_from);
    TEST_ASSERT_EQUAL_INT(PI_STATE_STREAMING, obs.last_to);

    pi_hsm_destroy(hsm);
}

static void test_error_reset_recovers_to_idle(void) {
    pi_hsm_t *hsm = pi_hsm_create(NULL, NULL);
    pi_hsm_dispatch(hsm, PI_EVENT_ERROR);
    TEST_ASSERT_EQUAL_INT(PI_STATE_ERROR, pi_hsm_current_state(hsm));

    TEST_ASSERT_EQUAL_INT(1, pi_hsm_dispatch(hsm, PI_EVENT_RESET));
    TEST_ASSERT_EQUAL_INT(PI_STATE_IDLE, pi_hsm_current_state(hsm));
    pi_hsm_destroy(hsm);
}

static void test_stopped_is_terminal(void) {
    pi_hsm_t *hsm = pi_hsm_create(NULL, NULL);
    pi_hsm_dispatch(hsm, PI_EVENT_STOP);
    TEST_ASSERT_EQUAL_INT(PI_STATE_STOPPED, pi_hsm_current_state(hsm));
    TEST_ASSERT_TRUE(pi_hsm_is_terminal(hsm));

    // Every event in STOPPED must be ignored (returns 0, state unchanged).
    for (int e = 0; e < PI_EVENT__COUNT; ++e) {
        const int rc = pi_hsm_dispatch(hsm, (pi_event_t)e);
        TEST_ASSERT_EQUAL_INT(0, rc);
        TEST_ASSERT_EQUAL_INT(PI_STATE_STOPPED, pi_hsm_current_state(hsm));
    }
    pi_hsm_destroy(hsm);
}

static void test_ignored_event_returns_zero(void) {
    pi_hsm_t *hsm = pi_hsm_create(NULL, NULL);
    // FRAME_READY in IDLE is meaningless but must not be an error.
    TEST_ASSERT_EQUAL_INT(0, pi_hsm_dispatch(hsm, PI_EVENT_FRAME_READY));
    TEST_ASSERT_EQUAL_INT(PI_STATE_IDLE, pi_hsm_current_state(hsm));
    pi_hsm_destroy(hsm);
}

static void test_out_of_range_event_is_error(void) {
    pi_hsm_t *hsm = pi_hsm_create(NULL, NULL);
    TEST_ASSERT_EQUAL_INT(-1,
        pi_hsm_dispatch(hsm, (pi_event_t)PI_EVENT__COUNT));
    TEST_ASSERT_EQUAL_INT(-1,
        pi_hsm_dispatch(hsm, (pi_event_t)99));
    pi_hsm_destroy(hsm);
}

static void test_null_hsm_is_safe(void) {
    TEST_ASSERT_EQUAL_INT(-1, pi_hsm_dispatch(NULL, PI_EVENT_START));
    TEST_ASSERT_EQUAL_INT(PI_STATE__COUNT, pi_hsm_current_state(NULL));
    TEST_ASSERT_FALSE(pi_hsm_is_terminal(NULL));
    pi_hsm_destroy(NULL);
    TEST_PASS();
}

static void test_name_helpers_never_return_null(void) {
    for (int s = 0; s <= PI_STATE__COUNT; s++) {
        TEST_ASSERT_NOT_NULL(pi_state_name((pi_state_t)s));
    }
    for (int e = 0; e <= PI_EVENT__COUNT; e++) {
        TEST_ASSERT_NOT_NULL(pi_event_name((pi_event_t)e));
    }
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_initial_state_is_idle);
    RUN_TEST(test_idle_to_streaming_flow);
    RUN_TEST(test_starting_to_error_on_start_err);
    RUN_TEST(test_pause_resume_cycle);
    RUN_TEST(test_streaming_self_transition_on_frame_ready);
    RUN_TEST(test_error_reset_recovers_to_idle);
    RUN_TEST(test_stopped_is_terminal);
    RUN_TEST(test_ignored_event_returns_zero);
    RUN_TEST(test_out_of_range_event_is_error);
    RUN_TEST(test_null_hsm_is_safe);
    RUN_TEST(test_name_helpers_never_return_null);
    return UNITY_END();
}
