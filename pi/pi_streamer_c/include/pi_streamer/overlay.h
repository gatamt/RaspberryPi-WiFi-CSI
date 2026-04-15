#ifndef PI_STREAMER_OVERLAY_H
#define PI_STREAMER_OVERLAY_H

// YUV420 luminance-plane bbox renderer.
//
// Draws flat-color rectangles on the Y (luminance) plane of a YUV420p frame
// in place. Chroma planes are left untouched — this gives the boxes a
// bright, color-neutral outline against the underlying video, which is
// exactly what a debug overlay needs without a full color-space pass.
//
// Why Y-plane only:
//   - H.264 encodes from YUV420, so editing the Y plane directly skips
//     the BGR→YUV conversion the Python reference's overlay.py was paying.
//   - Chroma is subsampled 2x2 — drawing on it is 4x more book-keeping for
//     no perceptual gain on thin rectangle outlines.
//   - The encoder's i420 input already points at the Y plane, so we can
//     mutate it with zero copies.
//
// Coordinate convention: pixel (0, 0) is top-left. Coordinates outside
// the frame are clipped (not wrapped). Zero-area boxes and empty detection
// lists are no-ops.

#include "pi_streamer/detection.h"

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Draw `n_det` bbox outlines onto the Y-plane of a YUV420 frame.
//
//   y_plane   — pointer to the top-left luminance pixel
//   width     — frame width in pixels
//   height    — frame height in pixels
//   y_stride  — Y-plane stride in bytes (>= width)
//   dets      — detections array (may be NULL iff n_det == 0)
//   n_det     — number of valid entries in dets
//   thickness — outline thickness in pixels (clamped to [1, 16])
//   value     — Y pixel value to stamp (typically 235 for bright "white"
//               under limited-range BT.709 / BT.601; 255 under full range)
//
// The function is pure in the sense that it only writes to y_plane[...]
// and only reads dets[...]. Safe to call from any thread that owns the
// frame buffer.
void pi_overlay_draw_detections(uint8_t              *y_plane,
                                int32_t               width,
                                int32_t               height,
                                size_t                y_stride,
                                const pi_detection_t *dets,
                                size_t                n_det,
                                int                   thickness,
                                uint8_t               value);

#ifdef __cplusplus
}
#endif

#endif  // PI_STREAMER_OVERLAY_H
