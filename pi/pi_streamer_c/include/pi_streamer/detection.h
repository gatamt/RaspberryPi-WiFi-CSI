#ifndef PI_STREAMER_DETECTION_H
#define PI_STREAMER_DETECTION_H

// Minimal detection geometry — what the overlay renderer needs to draw a
// bounding box on a video frame.
//
// Coordinates are in the encoder's pixel space (1280x720 by default), NOT
// the model's letterbox space. The postprocess step is responsible for
// reversing the letterbox transform before populating a pi_detection_t.
//
// The struct is deliberately POD so it can live inside pi_infer_result_t
// and be copied through the seqlock-based inference_state publish path
// without pointer-lifetime concerns (see inference_state.h for the race
// model that motivated this choice).

#include <stdint.h>

// Upper bound on detections we carry per model. 16 is generous for the
// typical robot-facing scene (1-3 people, maybe 5 objects, 2 hands) and
// keeps sizeof(pi_infer_result_t) below 1 KB so the seqlock value copy
// stays cache-friendly.
#define PI_MAX_DETECTIONS_PER_KIND 16

typedef struct {
    // Top-left + bottom-right in frame pixel coordinates. int32 so a
    // postprocess that returns a negative (off-screen) coordinate can
    // signal it cheaply; the overlay renderer clamps before drawing.
    int32_t x1;
    int32_t y1;
    int32_t x2;
    int32_t y2;
    // 0..1 confidence after the model's own sigmoid/softmax.
    float   score;
    // Class id, model-dependent. yolov8_pose is single-class (person=0)
    // so this is always 0 for PI_INFER_POSE. yolo26 uses COCO classes.
    int16_t class_id;
    // Small reserved field so the struct is 4-byte aligned on 32-bit
    // targets without trailing padding getting copied by the seqlock.
    int16_t _reserved;
} pi_detection_t;

#endif  // PI_STREAMER_DETECTION_H
