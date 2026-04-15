#include "pi_streamer/camera.h"

#include <stdlib.h>
#include <string.h>
#include <time.h>

// Mock camera backend. Produces synthetic planar YUV420 (I420) frames with
// a frame-id-dependent luma pattern and a flat grey chroma. No threads,
// no hardware. The buffer layout matches what the real libcamera backend
// delivers so encoder_x264 can be tested against the same plane pointers.
//
// Real libcamera backend lives in src/camera_libcamera.cpp and is added
// to the build only when find_package(libcamera) succeeds and the build
// target is Pi.

struct pi_camera {
    pi_camera_config_t cfg;
    bool               running;
    uint64_t           frame_id;
    uint8_t           *buffer;
    size_t             buffer_size;
    // YUV420 plane geometry, computed once in pi_camera_create.
    uint32_t           y_stride;
    uint32_t           uv_stride;
    size_t             u_offset;
    size_t             v_offset;
};

pi_camera_t *pi_camera_create(const pi_camera_config_t *cfg) {
    if (!cfg || cfg->width == 0 || cfg->height == 0) {
        return NULL;
    }
    // YUV420 requires even width/height so the chroma planes line up.
    if ((cfg->width & 1u) || (cfg->height & 1u)) {
        return NULL;
    }
    pi_camera_t *cam = calloc(1, sizeof *cam);
    if (!cam) return NULL;
    cam->cfg = *cfg;
    cam->y_stride  = cfg->width;
    cam->uv_stride = cfg->width / 2u;
    const size_t y_size  = (size_t)cam->y_stride  * (size_t)cfg->height;
    const size_t uv_size = (size_t)cam->uv_stride * (size_t)(cfg->height / 2u);
    cam->u_offset = y_size;
    cam->v_offset = y_size + uv_size;
    cam->buffer_size = y_size + 2u * uv_size; // == width*height*3/2
    cam->buffer = calloc(1, cam->buffer_size);
    if (!cam->buffer) {
        free(cam);
        return NULL;
    }
    return cam;
}

void pi_camera_destroy(pi_camera_t *cam) {
    if (!cam) return;
    free(cam->buffer);
    free(cam);
}

int pi_camera_start(pi_camera_t *cam) {
    if (!cam) return -1;
    cam->running = true;
    return 0;
}

int pi_camera_stop(pi_camera_t *cam) {
    if (!cam) return -1;
    cam->running = false;
    return 0;
}

bool pi_camera_is_running(const pi_camera_t *cam) {
    return cam && cam->running;
}

int pi_camera_capture(pi_camera_t       *cam,
                      pi_camera_frame_t *out_frame,
                      uint32_t           timeout_ms) {
    (void)timeout_ms;
    if (!cam || !out_frame) return -1;
    if (!cam->running) return -1;

    // Paint the Y plane with a frame-id-dependent byte so consecutive
    // frames can be told apart, and flood the chroma planes with neutral
    // grey (0x80). This yields a valid YUV420 frame that encoder_x264
    // (and the mock encoder) can consume without fixing up strides.
    const uint8_t fid = (uint8_t)(cam->frame_id & 0xFFu);
    memset(cam->buffer,                  fid,  cam->u_offset);
    memset(cam->buffer + cam->u_offset,  0x80, cam->buffer_size - cam->u_offset);

    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);

    out_frame->pixels       = cam->buffer;
    out_frame->pixels_size  = cam->buffer_size;
    out_frame->width        = cam->cfg.width;
    out_frame->height       = cam->cfg.height;
    out_frame->stride       = cam->y_stride;          // == y_stride
    out_frame->timestamp_ns = (uint64_t)ts.tv_sec * 1000000000ull +
                              (uint64_t)ts.tv_nsec;
    out_frame->frame_id     = cam->frame_id;
    out_frame->dma_fd       = -1;
    out_frame->opaque       = NULL;
    out_frame->y_stride     = cam->y_stride;
    out_frame->uv_stride    = cam->uv_stride;
    out_frame->u_offset     = cam->u_offset;
    out_frame->v_offset     = cam->v_offset;

    cam->frame_id++;
    return 0;
}

void pi_camera_release(pi_camera_t *cam, pi_camera_frame_t *frame) {
    (void)cam;
    (void)frame;
    // Mock owns a single shared buffer — nothing to reclaim per frame.
}
