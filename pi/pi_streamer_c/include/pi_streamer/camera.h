#ifndef PI_STREAMER_CAMERA_H
#define PI_STREAMER_CAMERA_H

// Camera C ABI. Two backends share this header:
//
//   Host / TDD:   src/camera_mock.c     (always buildable, synthetic frames)
//   Pi production: src/camera_libcamera.cpp  (compiled only when
//                   find_package(libcamera) succeeds; wraps the libcamera
//                   C++ API and exposes this C interface via DMA-BUF
//                   zero-copy)
//
// Primary reference: libcamera API https://libcamera.org/api-html/,
// rpicam-apps source at github.com/raspberrypi/rpicam-apps. No vault atomic
// note citations — see docs/Phase1-Atomic-Note-Verification.md.

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct pi_camera pi_camera_t;

typedef struct {
    uint32_t width;       // main stream output width, e.g. 1280
    uint32_t height;      // main stream output height, e.g. 720
    uint32_t fps;         // target frame rate, e.g. 30
    uint32_t raw_width;   // IMX708 wide native = 2304
    uint32_t raw_height;  // IMX708 wide native = 1296
} pi_camera_config_t;

// A captured frame. `pixels` may point into a DMA-BUF mapping on the Pi,
// or a mock buffer on host. Caller MUST pi_camera_release when done so the
// backend can recycle the buffer.
//
// Pixel format is planar YUV420 (I420): the Y plane starts at `pixels`,
// the U plane at `pixels + u_offset`, and the V plane at `pixels + v_offset`.
// For a 1280x720 frame: y_stride == 1280, uv_stride == 640,
// u_offset == 1280*720, v_offset == u_offset + 640*360. Total buffer size
// is `y_stride*height + 2*uv_stride*(height/2) == pixels_size`.
//
// `stride` is kept as a back-compat alias for `y_stride` so existing
// single-plane consumers keep compiling.
typedef struct {
    const uint8_t *pixels;
    size_t         pixels_size;
    uint32_t       width;
    uint32_t       height;
    uint32_t       stride;        // bytes per row of Y plane (== y_stride)
    uint64_t       timestamp_ns;  // CLOCK_MONOTONIC
    uint64_t       frame_id;      // monotonic counter from backend
    int            dma_fd;        // -1 for mock, real fd on Pi
    void          *opaque;        // backend-private handle for release
    // --- YUV420 planar metadata (T0, append-only) ---
    uint32_t       y_stride;      // bytes per row of Y plane (usually == width)
    uint32_t       uv_stride;     // bytes per row of U and V planes (usually == width/2)
    size_t         u_offset;      // byte offset from `pixels` to start of U plane
    size_t         v_offset;      // byte offset from `pixels` to start of V plane
} pi_camera_frame_t;

pi_camera_t *pi_camera_create(const pi_camera_config_t *cfg);
void         pi_camera_destroy(pi_camera_t *cam);

int  pi_camera_start(pi_camera_t *cam);
int  pi_camera_stop (pi_camera_t *cam);
bool pi_camera_is_running(const pi_camera_t *cam);

// Capture the next frame. On Pi this blocks until libcamera signals a
// completed request (up to `timeout_ms`). On host/mock it returns
// immediately with a synthetic frame.
//
// Returns 0 on success, -1 on timeout or error.
int pi_camera_capture(pi_camera_t       *cam,
                      pi_camera_frame_t *out_frame,
                      uint32_t           timeout_ms);

// Return a frame to the backend for buffer reuse. NULL-safe. Must be
// called exactly once per successful capture or DMA-BUF slots leak.
void pi_camera_release(pi_camera_t *cam, pi_camera_frame_t *frame);

#endif // PI_STREAMER_CAMERA_H
