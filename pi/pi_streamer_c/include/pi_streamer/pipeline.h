#ifndef PI_STREAMER_PIPELINE_H
#define PI_STREAMER_PIPELINE_H

// Top-level pipeline runner. Wires camera → encoder → wire_format → UDP
// sender and (optionally) camera → Hailo inference. Exposes a single
// function `pi_pipeline_run` so that both main.c (production) and
// test_pipeline.c (host TDD) can drive it.
//
// Thread model:
//   The ground-truth path is a single-threaded run-to-completion loop in
//   the calling thread. Tests run against it. The threaded variant (gated
//   behind a flag) splits the work across:
//     - camera capture thread (SCHED_FIFO prio 40, pinned to core 2)
//     - x264 encoder thread (SCHED_FIFO prio 50, pinned to core 2)
//     - 3 Hailo worker threads (SCHED_FIFO prio 30/30/20, pinned to core 3)
//     - UDP sender thread (SCHED_FIFO prio 45, pinned to core 1)
//     - main thread (event loop, dispatch, drain output rings — core 1)

#include "pi_streamer/logger.h"

#include <stdbool.h>
#include <stdint.h>

typedef struct {
    pi_log_level_t log_level;
    uint32_t       width;
    uint32_t       height;
    uint32_t       fps;
    uint32_t       bitrate_bps;
    uint32_t       gop_size;
    uint16_t       udp_port;        // bind port, usually 3334
    uint64_t       max_frames;      // 0 = run until SIGINT/SIGTERM
    bool           enable_inference;
    // Test hook: when the UDP sender is the mock, we have no way to
    // receive a VID0 from a real client, so the caller supplies the
    // destination directly. On Pi with the real backend this becomes the
    // address captured from the latest VID0.
    const char    *dest_ip;
    uint16_t       dest_port;
} pi_pipeline_args_t;

// Fill `out` with defaults that match the Python reference.
void pi_pipeline_args_defaults(pi_pipeline_args_t *out);

// Parse argv into `out` on top of defaults. Returns 0 on success,
// -1 on unknown flag or bad value. Call defaults first.
int pi_pipeline_parse_args(int argc, char **argv, pi_pipeline_args_t *out);

// Run the pipeline. Returns 0 on clean exit. Installs SIGINT/SIGTERM
// handlers for graceful shutdown; they are removed on return.
int pi_pipeline_run(const pi_pipeline_args_t *args);

// Request the currently-running pipeline to stop. Safe from signal
// handlers. Takes effect at the next loop iteration.
void pi_pipeline_request_stop(void);

#endif // PI_STREAMER_PIPELINE_H
