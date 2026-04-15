#ifndef PI_STREAMER_INFERENCE_H
#define PI_STREAMER_INFERENCE_H

// Hailo inference C ABI. Wraps the HailoRT C API (thread-per-stream async
// model) to run 3 HEFs on a shared VDevice: pose (yolov8m_pose), object
// (yolo26m), hand (hand_landmark_lite). On host, a mock backend satisfies
// the same interface with synthesized zero-latency results.
//
// HailoRT specifics (applied in the Pi backend, not the mock):
//   - One `hailo_vdevice` shared by all 3 models.
//   - Scheduler priorities per HEF: POSE=18, OBJECT=17, HAND=15 — pose
//     must run every frame to feed the hand ROI source, hand fills the
//     scheduler gaps.
//   - One worker thread per model — `hailo_vstream_write_raw_buffer` +
//     `hailo_vstream_read_raw_buffer` in async mode.
//
// Primary reference: `hailort.h` and the `libhailort/examples/c/` samples
// at github.com/hailo-ai/hailort.

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "pi_streamer/detection.h"

typedef struct pi_vdevice     pi_vdevice_t;
typedef struct pi_infer_model pi_infer_model_t;

typedef enum {
    PI_INFER_POSE = 0,
    PI_INFER_OBJECT,
    PI_INFER_HAND,
    PI_INFER__COUNT,
} pi_infer_kind_t;

typedef struct {
    pi_infer_kind_t kind;
    const char     *hef_path;           // /home/pi/models/<model>.hef
    uint32_t        input_width;
    uint32_t        input_height;
    int             scheduler_priority; // HailoRT scheduler priority
    // Camera frame dimensions — needed by the real backend's postprocess
    // step to reverse the letterbox transform so detections land back in
    // frame pixel coordinates. Zero means "default to input_width/height"
    // (i.e. no letterbox). Mock backend ignores these fields.
    uint32_t        frame_width;
    uint32_t        frame_height;
} pi_infer_model_config_t;

typedef struct {
    uint64_t         frame_id;
    pi_infer_kind_t  kind;
    const void      *raw_output;      // tensor output, model-specific layout
    size_t           raw_output_size;
    uint64_t         latency_ns;      // from submit to complete
    // Decoded detections. Populated inside pi_infer_poll (real and mock
    // backends) after running the model-specific postprocess. These fields
    // are what the overlay renderer consumes via inference_state_snapshot
    // — raw_output cannot be used downstream because its pointer becomes
    // stale on the next publish (see inference_state.h for the race model).
    uint8_t          num_detections;
    pi_detection_t   detections[PI_MAX_DETECTIONS_PER_KIND];
} pi_infer_result_t;

pi_vdevice_t *pi_vdevice_create(void);
void          pi_vdevice_destroy(pi_vdevice_t *vd);

pi_infer_model_t *pi_infer_model_create(pi_vdevice_t                   *vd,
                                        const pi_infer_model_config_t *cfg);
void              pi_infer_model_destroy(pi_infer_model_t *model);

// Submit an input buffer for inference. Non-blocking. Result (if any)
// is returned via pi_infer_poll. Returns 0 on enqueue, -1 on error/full.
int pi_infer_submit(pi_infer_model_t *model,
                    const uint8_t    *input,
                    size_t            input_size,
                    uint64_t          frame_id);

// Retrieve the next completed inference result for this model. Returns 0
// on success, -1 if nothing is ready. Result is valid until the next
// pi_infer_* call on this model.
int pi_infer_poll(pi_infer_model_t *model, pi_infer_result_t *out);

// Friendly name for logging / diagnostics.
const char *pi_infer_kind_name(pi_infer_kind_t k);

#endif // PI_STREAMER_INFERENCE_H
