#include "pi_streamer/inference.h"
#include "unity.h"

#include <string.h>

void setUp(void) {}
void tearDown(void) {}

static pi_infer_model_config_t pose_cfg(void) {
    const pi_infer_model_config_t cfg = {
        .kind               = PI_INFER_POSE,
        .hef_path           = "/home/pi/models/yolov8m_pose.hef",
        .input_width        = 640,
        .input_height       = 640,
        .scheduler_priority = 18,
    };
    return cfg;
}

static pi_infer_model_config_t hand_cfg(void) {
    const pi_infer_model_config_t cfg = {
        .kind               = PI_INFER_HAND,
        .hef_path           = "/home/pi/models/hand_landmark_lite.hef",
        .input_width        = 224,
        .input_height       = 224,
        .scheduler_priority = 15,
    };
    return cfg;
}

static void test_vdevice_create_destroy(void) {
    pi_vdevice_t *vd = pi_vdevice_create();
    TEST_ASSERT_NOT_NULL(vd);
    pi_vdevice_destroy(vd);
    pi_vdevice_destroy(NULL);
}

static void test_model_create_rejects_bad(void) {
    pi_vdevice_t *vd = pi_vdevice_create();
    TEST_ASSERT_NULL(pi_infer_model_create(NULL, NULL));

    pi_infer_model_config_t bad = pose_cfg();
    bad.input_width = 0;
    TEST_ASSERT_NULL(pi_infer_model_create(vd, &bad));

    bad = pose_cfg();
    bad.kind = (pi_infer_kind_t)PI_INFER__COUNT;
    TEST_ASSERT_NULL(pi_infer_model_create(vd, &bad));

    pi_vdevice_destroy(vd);
}

static void test_three_models_on_shared_vdevice(void) {
    pi_vdevice_t *vd = pi_vdevice_create();

    pi_infer_model_config_t pose = pose_cfg();
    pi_infer_model_config_t hand = hand_cfg();
    pi_infer_model_config_t obj  = pose_cfg();
    obj.kind = PI_INFER_OBJECT;
    obj.hef_path = "/home/pi/models/yolo26m.hef";
    obj.scheduler_priority = 17;

    pi_infer_model_t *m_pose = pi_infer_model_create(vd, &pose);
    pi_infer_model_t *m_obj  = pi_infer_model_create(vd, &obj);
    pi_infer_model_t *m_hand = pi_infer_model_create(vd, &hand);
    TEST_ASSERT_NOT_NULL(m_pose);
    TEST_ASSERT_NOT_NULL(m_obj);
    TEST_ASSERT_NOT_NULL(m_hand);

    pi_infer_model_destroy(m_pose);
    pi_infer_model_destroy(m_obj);
    pi_infer_model_destroy(m_hand);
    pi_vdevice_destroy(vd);
}

static void test_submit_then_poll(void) {
    pi_vdevice_t *vd = pi_vdevice_create();
    pi_infer_model_config_t cfg = pose_cfg();
    pi_infer_model_t *m = pi_infer_model_create(vd, &cfg);

    const uint8_t input[32] = {0xAA};
    TEST_ASSERT_EQUAL_INT(0, pi_infer_submit(m, input, sizeof input, 42));

    pi_infer_result_t res = {0};
    TEST_ASSERT_EQUAL_INT(0, pi_infer_poll(m, &res));
    TEST_ASSERT_EQUAL_UINT64(42, res.frame_id);
    TEST_ASSERT_EQUAL_INT(PI_INFER_POSE, res.kind);
    TEST_ASSERT_NOT_NULL(res.raw_output);
    TEST_ASSERT_GREATER_THAN_UINT(0u, res.raw_output_size);

    // Nothing more to poll.
    TEST_ASSERT_EQUAL_INT(-1, pi_infer_poll(m, &res));

    pi_infer_model_destroy(m);
    pi_vdevice_destroy(vd);
}

static void test_submit_when_full_fails(void) {
    pi_vdevice_t *vd = pi_vdevice_create();
    pi_infer_model_config_t cfg = pose_cfg();
    pi_infer_model_t *m = pi_infer_model_create(vd, &cfg);

    const uint8_t input[8] = {0};
    TEST_ASSERT_EQUAL_INT(0, pi_infer_submit(m, input, sizeof input, 1));
    // Mock queue depth is 1 — submitting without polling should fail.
    TEST_ASSERT_EQUAL_INT(-1, pi_infer_submit(m, input, sizeof input, 2));

    pi_infer_result_t res;
    pi_infer_poll(m, &res);
    // After poll, submitting is allowed again.
    TEST_ASSERT_EQUAL_INT(0, pi_infer_submit(m, input, sizeof input, 3));

    pi_infer_model_destroy(m);
    pi_vdevice_destroy(vd);
}

static void test_kind_name_never_null(void) {
    for (int k = 0; k <= PI_INFER__COUNT; k++) {
        TEST_ASSERT_NOT_NULL(pi_infer_kind_name((pi_infer_kind_t)k));
    }
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_vdevice_create_destroy);
    RUN_TEST(test_model_create_rejects_bad);
    RUN_TEST(test_three_models_on_shared_vdevice);
    RUN_TEST(test_submit_then_poll);
    RUN_TEST(test_submit_when_full_fails);
    RUN_TEST(test_kind_name_never_null);
    return UNITY_END();
}
