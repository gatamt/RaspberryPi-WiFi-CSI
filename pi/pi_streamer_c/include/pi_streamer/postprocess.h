#ifndef PI_STREAMER_POSTPROCESS_H
#define PI_STREAMER_POSTPROCESS_H

// Hailo tensor postprocessing — decodes raw model outputs into
// pi_detection_t bounding boxes in frame pixel coordinates.
//
// This is a C port of pi/pi_streamer/postprocess.py (YOLOv8 pose head).
// It implements the minimal path needed for bbox overlay rendering:
//
//   1. Dequantize the 9 output tensors via (raw - zero_point) * scale
//   2. DFL softmax + weighted sum for 4 distances per anchor
//   3. Anchor-relative bbox decode with stride-adjusted anchor centres
//   4. Confidence threshold on the post-sigmoid objectness
//   5. NMS (single class)
//   6. Letterbox → frame coordinate reverse
//
// Keypoints are explicitly NOT decoded here — they would double the
// per-anchor state budget and the overlay renderer only needs bbox
// rectangles for v1. Keypoint support can be layered on later without
// touching the bbox path by adding an extra output field.
//
// References:
//   - Python reference: pi/pi_streamer/postprocess.py
//   - Ultralytics YOLOv8 head: ultralytics/nn/modules/head.py (Pose class)
//   - DFL: "Generalized Focal Loss" (Li et al., 2020)

#include "pi_streamer/detection.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// YOLOv8 pose tensor output spec (what the HEF exposes to us).
//
// Hailo outputs are laid out one per "output stream" with names assigned
// by the compiler. We key by (grid_h, grid_w, channel_count) instead so
// we don't have to hardcode compiler-assigned names.
typedef enum {
    PI_TENSOR_DTYPE_U8 = 0,
    PI_TENSOR_DTYPE_U16,
} pi_tensor_dtype_t;

typedef struct {
    // Pointer to the raw tensor bytes inside Hailo's output buffer.
    // Interpretation depends on `dtype`.
    const void       *data;
    pi_tensor_dtype_t dtype;

    // Quantization parameters. Dequantized value is:
    //   (raw_as_float - zero_point) * scale
    float             scale;
    float             zero_point;

    // Spatial grid dimensions. For yolov8m_pose at 640x640 input:
    //   p3: 80x80  p4: 40x40  p5: 20x20
    int32_t           grid_h;
    int32_t           grid_w;

    // Channel count. Typical yolov8 pose:
    //   bbox:  64 (= reg_max * 4)
    //   obj:    1
    //   kpts:  51 (= 17 * 3) — IGNORED by this decoder in v1
    int32_t           channels;
} pi_tensor_view_t;

// Per-model configuration. Pass the same pose-head tensors for every
// frame (they live in the Hailo worker thread's scratch buffer).
typedef struct {
    // The 9 tensors that make up a yolov8 pose head. Ordering does not
    // matter — they are grouped into (bbox, obj, kpts) triples by
    // spatial dimension internally.
    pi_tensor_view_t tensors[9];
    size_t           num_tensors;

    // Model input resolution (square). yolov8m_pose on Hailo 10H is 640.
    int32_t          input_size;

    // DFL regression bins. Ultralytics YOLOv8 default is 16.
    int32_t          reg_max;

    // Frame resolution that the postprocess should return bbox coords
    // in. The decoder applies a letterbox reverse (scale + pad_y) so the
    // caller can hand the detection straight to the overlay renderer.
    int32_t          frame_width;
    int32_t          frame_height;

    // Score + NMS thresholds. Match the Python reference defaults.
    float            score_threshold;
    float            iou_threshold;
} pi_postprocess_pose_cfg_t;

// Decode one frame's worth of yolov8_pose output into `out_dets`.
// Returns the number of detections written (<= out_cap).
//
// The decode is stateless so the same cfg can be reused without any
// reset; all per-call scratch is stack-allocated.
size_t pi_postprocess_pose_decode(const pi_postprocess_pose_cfg_t *cfg,
                                  pi_detection_t                  *out_dets,
                                  size_t                           out_cap);

#ifdef __cplusplus
}
#endif

#endif  // PI_STREAMER_POSTPROCESS_H
