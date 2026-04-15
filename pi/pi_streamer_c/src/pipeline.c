#include "pi_streamer/pipeline.h"

#include "compat/arch.h"
#include "pi_streamer/_test_hooks.h"
#include "pi_streamer/camera.h"
#include "pi_streamer/detection.h"
#include "pi_streamer/encoder.h"
#include "pi_streamer/inference.h"
#include "pi_streamer/inference_state.h"
#include "pi_streamer/logger.h"
#include "pi_streamer/overlay.h"
#include "pi_streamer/ring_buffer.h"
#include "pi_streamer/rt.h"
#include "pi_streamer/state_machine.h"
#include "pi_streamer/udp_sender.h"
#include "pi_streamer/wire_format.h"

#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// ---------------------------------------------------------------------------
// Client registration / control-plane state
// ---------------------------------------------------------------------------
// Tracks the single iOS client that has registered via VID0. The pipeline
// only emits encoded chunks while a client is active and not paused, which
// matches the Python reference (pi/pi_streamer/udp_protocol.py:141-213).
//
// The struct lives on the stack inside pi_pipeline_run so each call gets a
// fresh state — there is no global singleton, which keeps test repeatability
// trivial.
typedef struct {
    bool      active;
    bool      paused;
    char      ip[64];
    uint16_t  port;
    uint64_t  last_heartbeat_ms;
    bool      force_idr;
} pi_client_state_t;

// ---------------------------------------------------------------------------
// Encoded-packet slot (Pi only)
// ---------------------------------------------------------------------------
// The sender_thread owns UDP send, not the encoder thread. x264's NAL
// buffer is reused per-encode, so the encoder thread must copy the bytes
// out before handing ownership to the sender. To keep malloc out of the
// hot path, pipeline_spawn_workers pre-allocates 4 slots with a single
// 4 x PI_ENCODED_SLOT_CAP backing buffer. Each slot carries a snapshot
// of the destination (ip, port) as well as the frame metadata so the
// sender can chunk + send without touching ctx->client.
#ifdef PI_TARGET
#define PI_ENCODED_SLOT_CAP   (256u * 1024u)  // 256 KB per slot
#define PI_ENCODED_SLOT_COUNT 4u

typedef struct {
    uint8_t  *nal;            // points into encoded_slot_nals, cap = PI_ENCODED_SLOT_CAP
    size_t    nal_size;       // valid bytes in `nal`
    uint64_t  pts_ms;
    bool      is_keyframe;
    uint64_t  frame_id;
    char      dest_ip[64];    // snapshot of client.ip at encode time
    uint16_t  dest_port;
} encoded_slot_t;
#endif

// ---------------------------------------------------------------------------
// Pipeline run context
// ---------------------------------------------------------------------------
// Holds every long-lived object owned by a single pi_pipeline_run() call,
// so the 4 phase helpers (setup / spawn_workers / main_loop / teardown)
// can share state through one pointer. Each call gets a fresh stack-local
// instance, which keeps test isolation identical to the pre-refactor shape.
//
// All pointer fields start NULL and teardown relies on that to do partial
// cleanup if setup() fails mid-way — do not introduce non-NULL defaults
// without also auditing pipeline_teardown.
typedef struct {
    const pi_pipeline_args_t *args;
    pi_camera_t              *cam;
    pi_encoder_t             *enc;
    pi_udp_sender_t          *udp;
    pi_hsm_t                 *hsm;
    pi_vdevice_t             *vdev;
    pi_infer_model_t         *models[PI_INFER__COUNT];
    pi_client_state_t         client;
    uint64_t                  frames_done;
    bool                      camera_started;   // only stop() if start() ran
    struct sigaction          old_sigint;       // saved to restore in teardown
    struct sigaction          old_sigterm;
    bool                      sighandlers_installed;
    // Shared inference state — readable from any thread via the
    // seqlock-guarded snapshot API. Writes come from hailo_worker_loop on
    // the Pi threaded build and from pipeline_main_loop on the host build.
    // Declared outside PI_TARGET so the host build path can publish mock
    // results into it and read them back when drawing the overlay.
    pi_inference_state_t      inference_state;
#ifdef PI_TARGET
    // ----- Camera + encoder worker threads (Pi only) -----
    // On Pi the main thread becomes a thin control-plane loop; capture and
    // encode run on their own pthreads pinned to core 2. Host builds keep
    // the single-threaded main loop and never touch any of these fields.
    pthread_t              camera_thread;
    pthread_t              encoder_thread;
    bool                   camera_thread_created;
    bool                   encoder_thread_created;
    pi_ring_buffer_t      *camera_ring;      // 8-slot SPSC, camera -> encoder
    pi_ring_buffer_t      *encoder_ring;     // 8-slot SPSC, encoder -> sender
    // Pre-allocated frame slot pool — recycled FIFO so capture never malloc's
    // in the hot path. The encoder thread returns each slot to frame_pool_ring
    // after releasing the camera frame and pushing to the sender ring.
    //
    // pi_ring_buffer_t is SPSC, but both camera_thread and encoder_thread
    // push to frame_pool_ring (camera pushes on capture errors; encoder
    // pushes on normal recycle). frame_pool_mtx serialises pushes so the
    // ring is effectively MPSC-via-mutex. The critical section is a
    // pointer store + seq_cst RMW — sub-microsecond.
    pi_camera_frame_t      frame_pool[8];
    pi_ring_buffer_t      *frame_pool_ring;
    pthread_mutex_t        frame_pool_mtx;
    bool                   mutexes_initialized;  // guards teardown destroys
    atomic_bool            threads_running;
    atomic_bool            camera_thread_ok;
    atomic_bool            encoder_thread_ok;

    // ----- Sender worker thread (Pi only) -----
    // sender_thread pops encoded_slot_t* off encoder_ring, chunks + sends
    // via pi_udp_sender_send, then recycles the slot back into
    // encoded_slot_pool. It also owns the control-plane RX drain — both
    // TX and RX sit on core 1 with prio 45, leaving core 2 entirely for
    // capture + encode.
    //
    // encoder_thread pushes back onto encoded_slot_pool when encoder_ring
    // is full (fallback path), AND sender_thread pushes on normal recycle.
    // Two producers into an SPSC ring are a race → encoded_slot_pool_mtx
    // serialises them (MPSC-via-mutex).
    pthread_t              sender_thread;
    bool                   sender_thread_created;
    atomic_bool            sender_thread_ok;
    pi_ring_buffer_t      *encoded_slot_pool;                 // free-list of encoded_slot_t*
    pthread_mutex_t        encoded_slot_pool_mtx;
    encoded_slot_t         encoded_slots[PI_ENCODED_SLOT_COUNT];
    uint8_t               *encoded_slot_nals;                 // backing store, 4 x 256 KB

    // ----- Hailo worker threads (Pi only) -----
    // One pthread per inference kind (pose / object / hand). Each worker:
    //   1. Pops a camera-frame pointer from its own input ring.
    //   2. Calls pi_infer_submit + pi_infer_poll on its assigned model.
    //   3. Publishes the result into `inference_state` via seqlock.
    //
    // The input rings are fed by encoder_thread_main AFTER it finishes
    // submitting a frame to x264 but BEFORE it releases the camera slot.
    // That buys a short window where the frame pointer is still valid —
    // long enough for the Hailo workers to dispatch the submit call.
    //
    // All 3 workers are pinned to core 3 (leaving core 0 for kernel and
    // housekeeping, core 1 for sender, core 2 for camera+encoder). The
    // SCHED_FIFO priorities MATCH the HailoRT scheduler priorities baked
    // into the HEF model configs in pipeline_setup (pose=30, object=30,
    // hand=20) so the Linux scheduler never re-orders what the Hailo
    // scheduler already sequenced.
    pthread_t              hailo_threads[PI_INFER__COUNT];
    bool                   hailo_thread_created[PI_INFER__COUNT];
    atomic_bool            hailo_thread_ok[PI_INFER__COUNT];
    pi_ring_buffer_t      *hailo_input_rings[PI_INFER__COUNT];  // 4-slot each

    // Mutex that serialises writes to ctx->client (from sender_thread's
    // control-plane drain) against reads from encoder_thread. Critical
    // sections are sub-microsecond (<200 ns), so there is no priority-
    // inversion concern on an RT pipeline and the mutex is the simplest
    // correct primitive. The field is still declared on host (inside
    // PI_TARGET) since the host main loop is single-threaded and never
    // takes it.
    pthread_mutex_t        client_mtx;
#endif
} pipeline_ctx_t;

// Heartbeat timeout matches HEARTBEAT_TIMEOUT_S = 10.0 in
// pi/pi_streamer/udp_protocol.py:26. If no BEAT arrives within this window
// we drop the client and return to IDLE; a re-registering BEAT or fresh VID0
// brings us back to ACTIVE with force_idr set.
static const uint64_t HEARTBEAT_TIMEOUT_MS = 10000ULL;

// ---------------------------------------------------------------------------
// Test-only time offset / iteration counter
// ---------------------------------------------------------------------------
// Tests cannot wall-clock-sleep for 11 seconds, so they shift the pipeline's
// notion of "now" forward via this offset. They also use the iteration
// counter to schedule control-message deliveries through the mock UDP
// backend's pending queue. Production code never touches either symbol —
// the setters and the counter sit in pipeline.c which is compiled into
// both libraries, but the production library has no caller and the
// linker's --gc-sections pass strips them from the final executable. See
// include/pi_streamer/_test_hooks.h for the full rationale.
static int64_t  g_pipeline_test_time_offset_ms = 0;
static int64_t  g_pipeline_test_time_step_ms   = 0;
static uint64_t g_pipeline_iteration_count     = 0;

void pi_pipeline_test_set_time_offset_ms(int64_t offset_ms) {
    g_pipeline_test_time_offset_ms = offset_ms;
}

void pi_pipeline_test_set_time_step_ms(int64_t step_ms) {
    g_pipeline_test_time_step_ms = step_ms;
}

uint64_t pi_pipeline_test_get_iteration_count(void) {
    return g_pipeline_iteration_count;
}

void pi_pipeline_test_reset_iteration_count(void) {
    g_pipeline_iteration_count = 0;
}

// Wrap pi_monotonic_ms() with the test offset baked in. uint64_t + int64_t
// is well-defined arithmetic in C (the int64_t is converted to uint64_t),
// and a negative offset wraps modulo 2^64 — fine for elapsed-time math
// because elapsed = now_after - last is correct under modular subtraction.
static inline uint64_t pipeline_now_ms(void) {
    return pi_monotonic_ms() + (uint64_t)g_pipeline_test_time_offset_ms;
}

// Cross-module shutdown flag.
//
// `volatile sig_atomic_t` is the one POSIX-guaranteed type a signal
// handler may write to. stdatomic operations are NOT listed in
// async-signal-safe functions (man 7 signal-safety), so the original
// `atomic_store_explicit` call from handle_signal was technically UB.
// sig_atomic_t stores give us the same portable guarantee without the
// stdatomic dependency in signal context. The main loop still needs an
// acquire barrier to observe the write — see the loop in pi_pipeline_run.
static volatile sig_atomic_t g_shutdown_requested = 0;

static void handle_signal(int sig) {
    (void)sig;
    g_shutdown_requested = 1;
}

void pi_pipeline_request_stop(void) {
    // Same flag, written from the test harness (main-thread context).
    // Using plain assignment is safe: sig_atomic_t writes are atomic per
    // POSIX, and the main loop does a barrier-fenced read before acting.
    g_shutdown_requested = 1;
}

void pi_pipeline_args_defaults(pi_pipeline_args_t *out) {
    if (!out) return;
    out->log_level        = PI_LOG_INFO;
    out->width            = 1280;
    out->height           = 720;
    out->fps              = 30;
    out->bitrate_bps      = 2500000;
    out->gop_size         = 30;
    out->udp_port         = 3334;
    out->max_frames       = 0;   // unlimited
    out->enable_inference = true;
    out->dest_ip          = "127.0.0.1";
    out->dest_port        = 5000;
}

static int parse_u32(const char *s, uint32_t *out) {
    if (!s || !out) return -1;
    char *end = NULL;
    unsigned long v = strtoul(s, &end, 10);
    if (!end || *end != '\0') return -1;
    *out = (uint32_t)v;
    return 0;
}

static int parse_u64(const char *s, uint64_t *out) {
    if (!s || !out) return -1;
    char *end = NULL;
    unsigned long long v = strtoull(s, &end, 10);
    if (!end || *end != '\0') return -1;
    *out = (uint64_t)v;
    return 0;
}

int pi_pipeline_parse_args(int argc, char **argv, pi_pipeline_args_t *out) {
    if (!out) return -1;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        const char *eq = strchr(a, '=');
        if (!eq) {
            fprintf(stderr, "pi_streamer: expected --flag=value, got '%s'\n", a);
            return -1;
        }
        const size_t flag_len = (size_t)(eq - a);
        const char  *val     = eq + 1;

        #define FLAG_IS(name) (flag_len == sizeof(name) - 1 && \
                               memcmp(a, (name), sizeof(name) - 1) == 0)

        if (FLAG_IS("--log-level")) {
            if (strcmp(val, "DEBUG") == 0)      out->log_level = PI_LOG_DEBUG;
            else if (strcmp(val, "INFO")  == 0) out->log_level = PI_LOG_INFO;
            else if (strcmp(val, "WARN")  == 0) out->log_level = PI_LOG_WARN;
            else if (strcmp(val, "ERROR") == 0) out->log_level = PI_LOG_ERROR;
            else return -1;
        } else if (FLAG_IS("--width"))       { if (parse_u32(val, &out->width))       return -1; }
        else if (FLAG_IS("--height"))        { if (parse_u32(val, &out->height))      return -1; }
        else if (FLAG_IS("--fps"))           { if (parse_u32(val, &out->fps))         return -1; }
        else if (FLAG_IS("--bitrate-bps"))   { if (parse_u32(val, &out->bitrate_bps)) return -1; }
        else if (FLAG_IS("--gop-size"))      { if (parse_u32(val, &out->gop_size))    return -1; }
        else if (FLAG_IS("--udp-port"))      { uint32_t v; if (parse_u32(val, &v)) return -1; out->udp_port = (uint16_t)v; }
        else if (FLAG_IS("--max-frames"))    { if (parse_u64(val, &out->max_frames))  return -1; }
        else if (FLAG_IS("--enable-inference")) { out->enable_inference = strcmp(val, "true") == 0 || strcmp(val, "1") == 0; }
        else if (FLAG_IS("--dest-ip"))       { out->dest_ip = val; }
        else if (FLAG_IS("--dest-port"))     { uint32_t v; if (parse_u32(val, &v)) return -1; out->dest_port = (uint16_t)v; }
        else {
            fprintf(stderr, "pi_streamer: unknown flag '%.*s'\n",
                    (int)flag_len, a);
            return -1;
        }
        #undef FLAG_IS
    }
    return 0;
}

// Observer for state machine transitions — logs every state change.
static void hsm_log_observer(pi_state_t from, pi_state_t to,
                             pi_event_t event, void *ctx) {
    (void)ctx;
    PI_INFO("hsm", "%s -> %s on %s",
            pi_state_name(from),
            pi_state_name(to),
            pi_event_name(event));
}

// ---------------------------------------------------------------------------
// Overlay rendering (bbox draw on Y-plane)
// ---------------------------------------------------------------------------
// Snapshot the published inference state, flatten detections from all
// three kinds into one array (up to PI_MAX_DETECTIONS_PER_KIND entries per
// kind), then stamp bbox outlines directly on the Y plane of the camera
// frame. Chroma is left untouched on purpose — the outline comes out as a
// colour-neutral bright line against the underlying video, which is
// enough for a debug overlay and avoids a full YUV420 paint loop on the
// encoder hot path.
//
// Why the Y plane and not an offscreen composite:
//   libcamera hands us an mmap'd FrameBuffer that is also x264's input
//   buffer. Drawing in place means zero extra copies and keeps the
//   encoder's zero-copy contract intact. libcamera does not re-read the
//   buffer after pi_camera_capture returns until pi_camera_release, so
//   mutating it between capture and encode is safe — this is the same
//   in-place pattern the Python reference's overlay.py used on BGR
//   frames (pi_streamer/main.py:335, pi_streamer/overlay.py:148-204).
//
// Thickness + brightness:
//   `PI_OVERLAY_THICKNESS = 3` is fat enough to be visible at 1280x720
//   after H.264 compression without dominating the frame.
//   `PI_OVERLAY_Y_VALUE = 235` matches limited-range BT.601/BT.709 peak
//   white, which is what x264's baseline profile produces by default.
#define PI_OVERLAY_THICKNESS 3
#define PI_OVERLAY_Y_VALUE   235

// Flatten detections from every inference kind into `out`, up to `out_cap`.
// Returns the number of detections actually written. Older entries win if
// `out_cap` is exceeded — this is fine because all kinds have the same
// MAX so overflow only happens if three models each emit their maximum.
static size_t collect_detections(const pi_inference_state_t *snap,
                                 pi_detection_t             *out,
                                 size_t                      out_cap) {
    if (!snap || !out || out_cap == 0) return 0;
    size_t written = 0;
    for (int k = 0; k < PI_INFER__COUNT; k++) {
        const pi_infer_result_t *r = &snap->slots[k].value;
        const size_t n = r->num_detections;
        for (size_t i = 0; i < n && written < out_cap; i++) {
            out[written++] = r->detections[i];
        }
    }
    return written;
}

// Render all currently-published detections onto the Y plane of a
// captured camera frame. Safe no-op when inference is disabled, when the
// snapshot read fails, or when no detections are live.
static void draw_overlay_on_frame(const pi_inference_state_t *state,
                                  const pi_camera_frame_t    *frame,
                                  uint32_t                    frame_width,
                                  uint32_t                    frame_height) {
    if (!state || !frame || !frame->pixels) return;
    if (frame_width == 0 || frame_height == 0) return;

    pi_inference_state_t snap;
    if (!pi_inference_state_snapshot(state, &snap)) {
        // Seqlock reader retry budget exhausted — writers are thrashing
        // the slots. Skip this frame's overlay rather than draw stale
        // data. Over time this manifests as the bbox "stuttering" but
        // never as corruption, which is the safest failure mode.
        return;
    }

    pi_detection_t flat[PI_MAX_DETECTIONS_PER_KIND * PI_INFER__COUNT];
    const size_t n = collect_detections(&snap, flat,
                                        sizeof flat / sizeof flat[0]);
    if (n == 0) return;

    // pi_camera_frame_t::pixels is declared `const uint8_t *` to protect
    // the camera buffer from accidental mutation in non-overlay code. The
    // overlay path is the one exception: we intentionally draw directly
    // into libcamera's DMA buffer between capture and release, same as
    // Python overlay.py did on BGR. Cast away const at this single
    // controlled site rather than weakening the camera ABI globally.
    uint8_t *y_plane = (uint8_t *)(uintptr_t)frame->pixels;

    pi_overlay_draw_detections(y_plane,
                               (int32_t)frame_width,
                               (int32_t)frame_height,
                               frame->y_stride,
                               flat, n,
                               PI_OVERLAY_THICKNESS,
                               PI_OVERLAY_Y_VALUE);
}

// Chunk an encoded packet and push every chunk into the UDP sender.
// Returns 0 on full success, -1 if any send failed.
//
// The destination is passed explicitly so callers on different threads
// can snapshot the client state at their own cadence. On the host path
// the main loop passes `client->ip`/`client->port` directly; on the Pi
// path the sender_thread passes values it copied out of an encoded_slot_t
// at the moment the encoder thread produced the packet. Either way the
// CLI's --dest-ip/--dest-port are NOT consulted here — they only ever
// prime the client state.
static int send_encoded_packet(pi_udp_sender_t            *udp,
                               const char                 *dst_ip,
                               uint16_t                    dst_port,
                               uint32_t                    width,
                               uint32_t                    height,
                               uint64_t                    frame_id,
                               const pi_encoded_packet_t  *pkt) {
    const size_t n_chunks = pi_chunk_count_for(pkt->nal_size);
    uint8_t buf[PI_MAX_DATAGRAM_SIZE];
    // 200 µs inter-chunk pacing matches Python's CHUNK_PACING_S in
    // pi_streamer/udp_protocol.py:27. Without it a 22-chunk IDR fires as
    // a sub-millisecond burst into the brcmfmac/mt76 TX queue, which
    // overflows under WiFi contention and silently drops chunks — the
    // visible symptom is partial-IDR corruption on the iOS receiver.
    static const struct timespec chunk_pace = { 0, 200000 };  // 200 µs
    for (size_t i = 0; i < n_chunks; i++) {
        const size_t n = pi_build_one_chunk(
            (uint32_t)frame_id,
            (uint16_t)width,
            (uint16_t)height,
            (uint32_t)pkt->pts_ms,
            pkt->nal, pkt->nal_size,
            i, pkt->is_keyframe,
            buf, sizeof buf);
        if (n == 0) {
            PI_WARN("pipeline", "pi_build_one_chunk returned 0 for chunk %zu",
                    i);
            return -1;
        }
        if (pi_udp_sender_send(udp, dst_ip, dst_port, buf, n) != 0) {
            PI_WARN("pipeline", "udp send failed for chunk %zu", i);
            return -1;
        }
        if (i + 1 < n_chunks) {
            nanosleep(&chunk_pace, NULL);
        }
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Phase helpers for pi_pipeline_run
// ---------------------------------------------------------------------------
// pi_pipeline_run is split into four sequential phases:
//   1. pipeline_setup        — log init, sighandlers, construct backends,
//                              pi_camera_start, HSM START dispatch, reset
//                              client state and test-iteration counter.
//   2. pipeline_spawn_workers— spawns the Pi-only worker threads.
//   3. pipeline_main_loop    — the drain/capture/encode/send super-loop.
//   4. pipeline_teardown     — send polite GONE, HSM STOP, destroy all
//                              backends in reverse order, restore sighandlers.
// All state that the original monolithic function held in locals lives on
// pipeline_ctx_t so each phase can share it through one pointer.

static int pipeline_setup(pipeline_ctx_t *ctx) {
    const pi_pipeline_args_t *args = ctx->args;

    pi_log_init(args->log_level);
    PI_INFO("main",
            "starting pipeline %ux%u@%u, bitrate=%u, gop=%u, port=%u, "
            "max_frames=%llu, inference=%s, dest=%s:%u",
            args->width, args->height, args->fps,
            args->bitrate_bps, args->gop_size, (unsigned)args->udp_port,
            (unsigned long long)args->max_frames,
            args->enable_inference ? "on" : "off",
            args->dest_ip ? args->dest_ip : "<null>",
            (unsigned)args->dest_port);

    // Lock the process's memory so SCHED_FIFO workers never page-fault
    // during the hot path. mlockall + 128 KiB stack prefault. Best-
    // effort: if RLIMIT_MEMLOCK is too small or we lack permissions,
    // pi_rt_lock_memory logs a warning and returns -1, and we continue
    // in degraded mode. Host builds are a no-op.
    (void)pi_rt_lock_memory();

    // Zero-initialise the shared inference state. On the Pi threaded build
    // pipeline_spawn_workers calls this again just before the worker
    // threads start; calling it twice is safe (init is idempotent). On
    // the host build this is the ONLY call site, and it has to happen
    // before the host main loop calls pi_inference_state_publish.
    pi_inference_state_init(&ctx->inference_state);

    // Install signal handlers for graceful shutdown. Originals are saved
    // into ctx so teardown can restore them exactly once.
    struct sigaction sa = {0};
    sa.sa_handler = handle_signal;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT,  &sa, &ctx->old_sigint);
    sigaction(SIGTERM, &sa, &ctx->old_sigterm);
    ctx->sighandlers_installed = true;
    g_shutdown_requested = 0;

    // ---------------- Construct everything ----------------
    const pi_camera_config_t cam_cfg = {
        .width      = args->width,
        .height     = args->height,
        .fps        = args->fps,
        .raw_width  = 2304,
        .raw_height = 1296,
    };
    ctx->cam = pi_camera_create(&cam_cfg);

    const pi_encoder_config_t enc_cfg = {
        .width        = args->width,
        .height       = args->height,
        .fps          = args->fps,
        .bitrate_bps  = args->bitrate_bps,
        .gop_size     = args->gop_size,
        .zero_latency = true,
    };
    ctx->enc = pi_encoder_create(&enc_cfg);

    const pi_udp_sender_config_t udp_cfg = {
        .bind_ip        = "0.0.0.0",
        .bind_port      = args->udp_port,
        .sq_depth       = 256,
        .send_buf_bytes = 524288,
    };
    ctx->udp = pi_udp_sender_create(&udp_cfg);

    if (args->enable_inference) {
        ctx->vdev = pi_vdevice_create();
        const pi_infer_model_config_t mcfg[PI_INFER__COUNT] = {
            { PI_INFER_POSE,
              "/home/pi/models/yolov8m_pose.hef",   640, 640, 18,
              args->width, args->height },
            { PI_INFER_OBJECT,
              "/home/pi/models/yolo26m.hef",        640, 640, 17,
              args->width, args->height },
            { PI_INFER_HAND,
              "/home/pi/models/hand_landmark_lite.hef", 224, 224, 15,
              args->width, args->height },
        };
        for (int k = 0; k < PI_INFER__COUNT; k++) {
            ctx->models[k] = pi_infer_model_create(ctx->vdev, &mcfg[k]);
        }
    }

    ctx->hsm = pi_hsm_create(hsm_log_observer, NULL);

    if (!ctx->cam || !ctx->enc || !ctx->udp || !ctx->hsm) {
        PI_ERROR("main", "failed to allocate core pipeline objects");
        return -1;
    }

    // ---------------- Start ----------------
    if (pi_camera_start(ctx->cam) != 0) {
        PI_ERROR("main", "camera_start failed");
        return -1;
    }
    ctx->camera_started = true;

    pi_hsm_dispatch(ctx->hsm, PI_EVENT_START);
    pi_hsm_dispatch(ctx->hsm, PI_EVENT_START_OK);

    // Reset per-run state. Client starts IDLE and the test iteration
    // counter starts at zero so each pi_pipeline_run() call is isolated.
    // ctx is already zero-initialised by the caller, but we set these
    // explicitly to document the invariant.
    memset(&ctx->client, 0, sizeof ctx->client);
    ctx->frames_done = 0;
    g_pipeline_iteration_count = 0;

    return 0;
}

// ---------------------------------------------------------------------------
// Shared control-plane service routine
// ---------------------------------------------------------------------------
// Drains pending VID0/BEAT/PAWS/GONE control datagrams and checks the
// heartbeat timeout. Called once per main-loop iteration by BOTH the
// single-threaded host loop and the Pi threaded loop — extracting it avoids
// duplicating the ~70 lines of control-plane logic between the two
// `#ifdef PI_TARGET` branches of pipeline_main_loop.
//
// Reads: ctx->udp (mutating — try_recv / send / poll is called internally),
//        ctx->hsm (mutating — dispatches events on state changes),
//        ctx->client (read + written).
//
// On the Pi path the encoder thread reads ctx->client concurrently; this
// main-thread writer plus one reader is a known TOCTOU hazard, serialised
// by client_mtx below.
static void pipeline_service_control_plane(pipeline_ctx_t *ctx) {
    pi_udp_sender_t   *udp    = ctx->udp;
    pi_hsm_t          *hsm    = ctx->hsm;
    pi_client_state_t *client = &ctx->client;

    // Serialise all writes to ctx->client under client_mtx on the Pi
    // build. Readers (encoder_thread) take the same mutex for a
    // consistent snapshot. Host builds skip the mutex — the single-
    // threaded main loop has no concurrent reader.
#ifdef PI_TARGET
    pthread_mutex_lock(&ctx->client_mtx);
#endif

    // Check the heartbeat timeout BEFORE draining control messages. If
    // we drained BEAT/VID0 first (which refreshes last_heartbeat_ms), a
    // BEAT arriving 10.1s after the previous one would reset the timer
    // and the pipeline would never transition IDLE → forced-IDR-on-
    // register, unlike the Python reference (udp_protocol.py:292-298)
    // which checks before dispatch. Checking here AND again after the
    // drain (below) covers both "timer expired before we noticed" and
    // "timer expired in-loop" cases.
    if (client->active && !client->paused) {
        const uint64_t elapsed = pipeline_now_ms() - client->last_heartbeat_ms;
        if (elapsed > HEARTBEAT_TIMEOUT_MS) {
            PI_WARN("pipeline",
                    "-> IDLE (heartbeat timeout %llu ms, pre-drain)",
                    (unsigned long long)elapsed);
            client->active = false;
            client->ip[0]  = '\0';
            client->port   = 0;
            (void)pi_hsm_dispatch(hsm, PI_EVENT_RESET);
        }
    }

    // ---- Drain pending control messages (VID0/BEAT/PAWS/GONE) ----
    // This block is the C port of pi/pi_streamer/udp_protocol.py
    // _handle_vid0/_handle_beat/_handle_paws/_handle_gone (lines
    // 257-307). Drain semantics:
    //   - VID0: register/replace client, send VACK, force IDR, RESUME.
    //   - BEAT: refresh heartbeat; if IDLE -> register like VID0;
    //           if PAUSED -> unpause, force IDR, RESUME.
    //   - PAWS: ACTIVE -> PAUSED (PAUSE event into HSM).
    //   - GONE: drop client, RESET HSM.
    {
        uint8_t  cbuf[64];
        size_t   cn = 0;
        char     sip[64] = {0};
        uint16_t sp = 0;
        int rc_recv;
        while ((rc_recv = pi_udp_sender_try_recv(
                   udp, cbuf, sizeof cbuf, &cn, sip, sizeof sip, &sp)) == 1) {
            if (cn < 4u) continue;  // short / malformed
            const uint64_t now_ms = pipeline_now_ms();
            if (memcmp(cbuf, "VID0", 4) == 0) {
                client->active           = true;
                client->paused           = false;
                client->force_idr        = true;
                {
                    const size_t n = strnlen(sip, sizeof client->ip - 1u);
                    memcpy(client->ip, sip, n);
                    client->ip[n] = '\0';
                }
                client->port             = sp;
                client->last_heartbeat_ms = now_ms;
                (void)pi_udp_sender_send(udp, sip, sp,
                                         (const uint8_t *)"VACK", 4u);
                (void)pi_hsm_dispatch(hsm, PI_EVENT_RESUME);
                PI_INFO("pipeline", "-> ACTIVE (VID0 from %s:%u)",
                        client->ip, (unsigned)client->port);
            } else if (memcmp(cbuf, "BEAT", 4) == 0) {
                if (!client->active) {
                    // Re-register on BEAT in IDLE — Python parity
                    // (udp_protocol.py:268-275).
                    client->active    = true;
                    client->paused    = false;
                    client->force_idr = true;
                    {
                        const size_t n = strnlen(sip, sizeof client->ip - 1u);
                        memcpy(client->ip, sip, n);
                        client->ip[n] = '\0';
                    }
                    client->port = sp;
                    (void)pi_udp_sender_send(udp, sip, sp,
                                             (const uint8_t *)"VACK", 4u);
                    (void)pi_hsm_dispatch(hsm, PI_EVENT_RESUME);
                    PI_INFO("pipeline",
                            "-> ACTIVE (BEAT re-registered %s:%u)",
                            client->ip, (unsigned)client->port);
                } else if (client->paused) {
                    client->paused    = false;
                    client->force_idr = true;
                    (void)pi_hsm_dispatch(hsm, PI_EVENT_RESUME);
                    PI_INFO("pipeline", "-> ACTIVE (resumed via BEAT)");
                }
                client->last_heartbeat_ms = now_ms;
            } else if (memcmp(cbuf, "PAWS", 4) == 0) {
                if (client->active && !client->paused) {
                    client->paused = true;
                    (void)pi_hsm_dispatch(hsm, PI_EVENT_PAUSE);
                    PI_INFO("pipeline", "-> PAUSED (client PAWS)");
                }
            } else if (memcmp(cbuf, "GONE", 4) == 0) {
                if (client->active) {
                    client->active = false;
                    client->paused = false;
                    client->ip[0]  = '\0';
                    client->port   = 0;
                    (void)pi_hsm_dispatch(hsm, PI_EVENT_RESET);
                    PI_INFO("pipeline", "-> IDLE (client GONE)");
                }
            }
        }
    }

    // ---- Post-drain heartbeat timeout recheck ----
    // Second check covers the case where the timer expired DURING the
    // drain above (e.g. no BEAT present but we spent time on PAWS/GONE).
    if (client->active && !client->paused) {
        const uint64_t elapsed = pipeline_now_ms() - client->last_heartbeat_ms;
        if (elapsed > HEARTBEAT_TIMEOUT_MS) {
            PI_WARN("pipeline",
                    "-> IDLE (heartbeat timeout %llu ms)",
                    (unsigned long long)elapsed);
            client->active = false;
            client->ip[0]  = '\0';
            client->port   = 0;
            (void)pi_hsm_dispatch(hsm, PI_EVENT_RESET);
        }
    }

#ifdef PI_TARGET
    pthread_mutex_unlock(&ctx->client_mtx);
#endif
}

// ---------------------------------------------------------------------------
// Pi-only worker threads
// ---------------------------------------------------------------------------
// On the Pi build the main thread hands off frame capture and encoding to
// two dedicated pthreads pinned to core 2, connected by an 8-slot SPSC
// camera_ring carrying pi_camera_frame_t* slots drawn from frame_pool_ring.
// Host builds keep pipeline_main_loop single-threaded and never see these
// functions — the compiler drops them as dead code under --gc-sections.
#ifdef PI_TARGET
static void *camera_thread_main(void *arg) {
    pipeline_ctx_t *ctx = (pipeline_ctx_t *)arg;
    pi_rt_name_thread(pthread_self(), "pi-camera");
    pi_rt_pin_thread(pthread_self(), 2);     // core 2
    pi_rt_promote_fifo(pthread_self(), 40);  // prio 40 — below encoder

    atomic_store(&ctx->camera_thread_ok, true);

    while (atomic_load(&ctx->threads_running)) {
        pi_barrier_acquire();
        if (g_shutdown_requested) break;

        // Borrow a free slot from the pool. Pool-exhausted means the
        // encoder thread is behind; back off briefly and retry rather
        // than dropping frames on the floor.
        void *slot_ptr = NULL;
        if (pi_ring_pop(ctx->frame_pool_ring, &slot_ptr) != 0) {
            struct timespec ts = { 0, 100000 };  // 100 us
            nanosleep(&ts, NULL);
            continue;
        }
        pi_camera_frame_t *slot = (pi_camera_frame_t *)slot_ptr;

        // 50 ms timeout keeps the loop responsive to shutdown requests.
        // Shutdown-latency contract: libcamera's completed-frame condvar
        // is waited on internally inside pi_camera_capture with exactly
        // this timeout, so even when libcamera has no frames this call
        // returns every 50ms. Worst-case exit latency after
        // `threads_running = false + pi_barrier_release()` is ~50ms,
        // which is the dominant bound for the whole shutdown path —
        // every other thread uses sub-millisecond back-offs.
        if (pi_camera_capture(ctx->cam, slot, 50) != 0) {
            // Capture timeout / error: return the unused slot to the
            // pool. Serialise with encoder_thread's normal recycle.
            pthread_mutex_lock(&ctx->frame_pool_mtx);
            (void)pi_ring_push(ctx->frame_pool_ring, slot);
            pthread_mutex_unlock(&ctx->frame_pool_mtx);
            continue;
        }

        // Hand off to the encoder. With matching 8-slot ring + 8-slot pool
        // the push can only fail if the encoder has starved for a long
        // time, in which case drop the frame and return both resources.
        if (pi_ring_push(ctx->camera_ring, slot) != 0) {
            pi_camera_release(ctx->cam, slot);
            pthread_mutex_lock(&ctx->frame_pool_mtx);
            (void)pi_ring_push(ctx->frame_pool_ring, slot);
            pthread_mutex_unlock(&ctx->frame_pool_mtx);
        }
    }
    atomic_store(&ctx->camera_thread_ok, false);
    return NULL;
}

static void *encoder_thread_main(void *arg) {
    pipeline_ctx_t *ctx = (pipeline_ctx_t *)arg;
    pi_rt_name_thread(pthread_self(), "pi-encoder");
    pi_rt_pin_thread(pthread_self(), 2);     // same core as camera
    pi_rt_promote_fifo(pthread_self(), 50);  // prio 50 — above camera

    atomic_store(&ctx->encoder_thread_ok, true);

    while (atomic_load(&ctx->threads_running)) {
        pi_barrier_acquire();
        if (g_shutdown_requested) break;

        void *slot_ptr = NULL;
        if (pi_ring_pop(ctx->camera_ring, &slot_ptr) != 0) {
            struct timespec ts = { 0, 200000 };  // 200 us
            nanosleep(&ts, NULL);
            continue;
        }
        pi_camera_frame_t *f = (pi_camera_frame_t *)slot_ptr;

        // Snapshot the client state under client_mtx so we get a
        // consistent view of (active, paused, ip, port) even if
        // sender_thread's control-plane drain races with us. Also
        // consume + clear force_idr atomically so only one frame wins
        // the keyframe flag after a VID0/BEAT arrival.
        bool     c_active   = false;
        bool     c_paused   = false;
        bool     c_force    = false;
        char     c_ip[64]   = {0};
        uint16_t c_port     = 0;
        pthread_mutex_lock(&ctx->client_mtx);
        c_active = ctx->client.active;
        c_paused = ctx->client.paused;
        c_force  = ctx->client.force_idr;
        ctx->client.force_idr = false;
        {
            const size_t n = strnlen(ctx->client.ip, sizeof c_ip - 1u);
            memcpy(c_ip, ctx->client.ip, n);
            c_ip[n] = '\0';
        }
        c_port = ctx->client.port;
        pthread_mutex_unlock(&ctx->client_mtx);

        if (c_active && !c_paused) {
            // Render bbox outlines onto the Y plane in place before the
            // encoder reads it. See draw_overlay_on_frame for why
            // mutating the libcamera frame in place is safe here.
            if (ctx->args->enable_inference) {
                draw_overlay_on_frame(&ctx->inference_state, f,
                                      ctx->args->width, ctx->args->height);
            }

            const bool force = (ctx->frames_done == 0u) || c_force;
            (void)pi_encoder_submit_yuv420(
                ctx->enc,
                f->pixels,                f->y_stride,
                f->pixels + f->u_offset,  f->uv_stride,
                f->pixels + f->v_offset,  f->uv_stride,
                f->timestamp_ns / 1000000u,
                force);

            pi_encoded_packet_t pkt;
            while (pi_encoder_next_packet(ctx->enc, &pkt) == 0) {
                // Borrow a pre-allocated slot from the pool. If the pool
                // is exhausted the sender is lagging — drop this packet
                // rather than stall the encoder. The x264 NAL buffer is
                // reused on the next encode so we must copy before
                // publishing to the sender thread.
                void *pool_p = NULL;
                if (pi_ring_pop(ctx->encoded_slot_pool, &pool_p) != 0) {
                    // pool exhausted: drop packet
                    continue;
                }
                encoded_slot_t *slot = (encoded_slot_t *)pool_p;

                // A NAL larger than PI_ENCODED_SLOT_CAP used to be
                // silently truncated, producing a malformed keyframe
                // that iOS rejected forever. Drop the packet and return
                // the slot instead — the next GOP boundary recovers us.
                const size_t cap = PI_ENCODED_SLOT_CAP;
                if (pkt.nal_size > cap) {
                    PI_WARN("pipeline",
                            "encoded NAL %zu > slot cap %zu, dropping "
                            "frame (key=%d)",
                            pkt.nal_size, cap, (int)pkt.is_keyframe);
                    pthread_mutex_lock(&ctx->encoded_slot_pool_mtx);
                    (void)pi_ring_push(ctx->encoded_slot_pool, slot);
                    pthread_mutex_unlock(&ctx->encoded_slot_pool_mtx);
                    continue;
                }

                slot->nal_size = pkt.nal_size;
                if (slot->nal_size > 0u) {
                    memcpy(slot->nal, pkt.nal, slot->nal_size);
                }
                slot->pts_ms      = pkt.pts_ms;
                slot->is_keyframe = pkt.is_keyframe;
                slot->frame_id    = f->frame_id;
                // Destination snapshot was taken under client_mtx above.
                {
                    const size_t n = strnlen(c_ip, sizeof slot->dest_ip - 1u);
                    memcpy(slot->dest_ip, c_ip, n);
                    slot->dest_ip[n] = '\0';
                }
                slot->dest_port = c_port;

                if (pi_ring_push(ctx->encoder_ring, slot) != 0) {
                    // ring full: return slot to pool and drop packet.
                    // Serialise with sender_thread's normal recycle.
                    pthread_mutex_lock(&ctx->encoded_slot_pool_mtx);
                    (void)pi_ring_push(ctx->encoded_slot_pool, slot);
                    pthread_mutex_unlock(&ctx->encoded_slot_pool_mtx);
                }
            }
        }

        // Submit the raw pixel buffer to all Hailo models SYNCHRONOUSLY
        // before releasing the camera frame. The old fan-out design
        // pushed `f` into per-kind rings and let worker threads call
        // pi_infer_submit() later — but by then the encoder had already
        // run pi_camera_release(), which zeros the frame struct (see
        // camera_libcamera.cpp pi_camera_release), so workers received
        // `f->pixels == NULL` and every submit failed.
        //
        // Doing the submit here is safe: pi_infer_submit is a memcpy into
        // an internal pinned buffer (see inference_hailort.cpp), so it
        // does not keep `f->pixels` alive past its return. The Hailo
        // worker threads still exist — they now just poll for the result
        // and publish into inference_state. This decouples the actual
        // TPU latency (~15 ms) from the encoder hot path while guaranteeing
        // the pointer we submit is live.
        if (ctx->args->enable_inference) {
            for (int k = 0; k < PI_INFER__COUNT; k++) {
                if (!ctx->models[k]) continue;
                (void)pi_infer_submit(ctx->models[k],
                                      f->pixels, f->pixels_size,
                                      f->frame_id);
            }
        }

        pi_camera_release(ctx->cam, f);
        // Serialise with camera_thread's error-path push.
        pthread_mutex_lock(&ctx->frame_pool_mtx);
        (void)pi_ring_push(ctx->frame_pool_ring, f);
        pthread_mutex_unlock(&ctx->frame_pool_mtx);
        pi_hsm_dispatch(ctx->hsm, PI_EVENT_FRAME_READY);
        ctx->frames_done++;

        // max_frames: request shutdown so the control-plane loop exits.
        if (ctx->args->max_frames > 0u &&
            ctx->frames_done >= ctx->args->max_frames) {
            g_shutdown_requested = 1;
            break;
        }
    }
    atomic_store(&ctx->encoder_thread_ok, false);
    return NULL;
}

// ---------------------------------------------------------------------------
// sender_thread
// ---------------------------------------------------------------------------
// Owns the UDP egress path: pops encoded_slot_t* off encoder_ring, chunks
// the NAL via pi_build_one_chunk, pushes chunks into pi_udp_sender_send,
// drives io_uring completions via pi_udp_sender_poll, and recycles slots
// back into encoded_slot_pool. Also drains the control-plane RX queue
// (VID0/BEAT/PAWS/GONE) so both TX and RX live on core 1 with prio 45,
// leaving core 2 entirely for capture + encode.
//
// Pinned to core 1, SCHED_FIFO prio 45 — sits above camera (40) but below
// encoder (50) so a blocked encoder never starves sender; the blocking
// direction is encoder -> sender via encoder_ring back-pressure, not the
// other way around.
static void *sender_thread_main(void *arg) {
    pipeline_ctx_t *ctx = (pipeline_ctx_t *)arg;
    pi_rt_name_thread(pthread_self(), "pi-sender");
    pi_rt_pin_thread(pthread_self(), 1);     // core 1
    pi_rt_promote_fifo(pthread_self(), 45);

    atomic_store(&ctx->sender_thread_ok, true);

    while (atomic_load(&ctx->threads_running)) {
        pi_barrier_acquire();
        if (g_shutdown_requested) break;

        // Control-plane RX drain + heartbeat check. ctx->client is
        // written here and read by encoder_thread; one-frame staleness
        // is acceptable and is handled by the client_mtx guard.
        pipeline_service_control_plane(ctx);

        // Drive io_uring send completions every iteration so SQ does not
        // back up even when the encoder is temporarily idle.
        (void)pi_udp_sender_poll(ctx->udp);

        void *slot_ptr = NULL;
        if (pi_ring_pop(ctx->encoder_ring, &slot_ptr) != 0) {
            // Idle: back off briefly. 100 us keeps us responsive to
            // shutdown + control-plane traffic without burning the core.
            struct timespec ts = { 0, 100000 };
            nanosleep(&ts, NULL);
            continue;
        }
        encoded_slot_t *slot = (encoded_slot_t *)slot_ptr;

        // The slot carries its own destination snapshot — we do NOT look
        // at ctx->client here, which keeps sender_thread's TX path free
        // of any encoder/sender races.
        const pi_encoded_packet_t pkt = {
            .nal         = slot->nal,
            .nal_size    = slot->nal_size,
            .pts_ms      = slot->pts_ms,
            .is_keyframe = slot->is_keyframe,
        };
        (void)send_encoded_packet(ctx->udp,
                                  slot->dest_ip, slot->dest_port,
                                  ctx->args->width, ctx->args->height,
                                  slot->frame_id, &pkt);

        // Return slot to pool for re-use by the encoder thread.
        // Serialise with encoder_thread's ring-full fallback push so
        // two concurrent producers cannot corrupt the SPSC ring.
        pthread_mutex_lock(&ctx->encoded_slot_pool_mtx);
        const int pool_rc = pi_ring_push(ctx->encoded_slot_pool, slot);
        pthread_mutex_unlock(&ctx->encoded_slot_pool_mtx);
        if (pool_rc != 0) {
            // Pool is bounded by slot count so this should never trigger;
            // log + leak one iteration rather than crash if it somehow
            // does.
            PI_WARN("pipeline", "encoded_slot_pool push failed (leak one slot)");
        }
    }
    atomic_store(&ctx->sender_thread_ok, false);
    return NULL;
}

// ---------------------------------------------------------------------------
// Hailo worker threads — poll-only variant
// ---------------------------------------------------------------------------
// Three pthreads, one per inference kind (pose / object / hand). Each
// worker is a pure POLL loop: encoder_thread_main calls pi_infer_submit
// directly BEFORE releasing the camera frame, so by the time the worker
// wakes up there is already a job in flight. The worker polls
// pi_infer_poll until either the result is ready (publish into
// inference_state) or 20ms elapse (drop).
//
// This avoids the lifetime race the old fan-out design had, where
// workers dereferenced `pi_camera_frame_t` AFTER pi_camera_release had
// zeroed the frame struct. With synchronous submit in the encoder
// thread, the pointer is guaranteed valid for the duration of the
// memcpy into pi_infer_model's pinned input buffer.
//
// All 3 workers are pinned to core 3, SCHED_FIFO, priorities 30 / 30 / 20.
static void *hailo_worker_loop(void *arg, pi_infer_kind_t kind,
                               const char *name, int prio) {
    pipeline_ctx_t *ctx = (pipeline_ctx_t *)arg;
    pi_rt_name_thread(pthread_self(), name);
    pi_rt_pin_thread(pthread_self(), 3);      // core 3 — dedicated to Hailo
    pi_rt_promote_fifo(pthread_self(), prio);

    atomic_store(&ctx->hailo_thread_ok[kind], true);

    pi_infer_model_t *model = ctx->models[kind];

    while (atomic_load(&ctx->threads_running)) {
        pi_barrier_acquire();
        if (g_shutdown_requested) break;

        // If there's no model (enable_inference=false or HEF missing) the
        // thread still runs so teardown's pthread_join has a valid target.
        if (!model) {
            struct timespec ts = { 0, 1000000 };  // 1 ms
            nanosleep(&ts, NULL);
            continue;
        }

        // Poll for a completed inference result. pi_infer_poll is
        // non-blocking — it returns -1 immediately when nothing is ready.
        // Back off 1 ms between attempts so we don't spin a whole core
        // when the encoder is idle (no active client => no submits).
        pi_infer_result_t res = {0};
        if (pi_infer_poll(model, &res) == 0) {
            pi_inference_state_publish(&ctx->inference_state, kind, &res);
        } else {
            struct timespec ts = { 0, 1000000 };  // 1 ms
            nanosleep(&ts, NULL);
        }
    }
    atomic_store(&ctx->hailo_thread_ok[kind], false);
    return NULL;
}

static void *hailo_worker_pose(void *arg) {
    return hailo_worker_loop(arg, PI_INFER_POSE,   "pi-hailo-pose",   30);
}
static void *hailo_worker_object(void *arg) {
    return hailo_worker_loop(arg, PI_INFER_OBJECT, "pi-hailo-object", 30);
}
static void *hailo_worker_hand(void *arg) {
    return hailo_worker_loop(arg, PI_INFER_HAND,   "pi-hailo-hand",   20);
}
#endif  // PI_TARGET

static int pipeline_spawn_workers(pipeline_ctx_t *ctx) {
#ifdef PI_TARGET
    // Init the three MPSC-via-mutex guards BEFORE any thread is spawned
    // so workers can take them as soon as they start running.
    // `mutexes_initialized` guards teardown so a setup-phase failure
    // doesn't pthread_mutex_destroy an uninitialised handle (UB).
    pthread_mutex_init(&ctx->frame_pool_mtx,        NULL);
    pthread_mutex_init(&ctx->encoded_slot_pool_mtx, NULL);
    pthread_mutex_init(&ctx->client_mtx,            NULL);
    ctx->mutexes_initialized = true;

    // Create the three 8-slot SPSC rings that wire the Pi pipeline:
    //   camera_ring     — camera_thread -> encoder_thread
    //   encoder_ring    — encoder_thread -> sender_thread
    //   frame_pool_ring — encoder_thread -> camera_thread (slot recycling)
    ctx->camera_ring     = pi_ring_create(8);
    ctx->encoder_ring    = pi_ring_create(8);
    ctx->frame_pool_ring = pi_ring_create(8);
    if (!ctx->camera_ring || !ctx->encoder_ring || !ctx->frame_pool_ring) {
        PI_ERROR("pipeline", "failed to allocate thread rings");
        return -1;
    }

    // Populate the frame pool ring with every slot from the pre-allocated
    // frame_pool[]. Capture dequeues a slot before each pi_camera_capture
    // and the encoder thread puts it back after releasing the frame.
    for (int i = 0; i < 8; i++) {
        memset(&ctx->frame_pool[i], 0, sizeof ctx->frame_pool[i]);
        (void)pi_ring_push(ctx->frame_pool_ring, &ctx->frame_pool[i]);
    }

    // Encoded-slot pool. One contiguous 4 x 256 KB backing buffer
    // plus a 4-slot SPSC free-list. calloc() is called exactly once
    // here — never in the hot path — and both live for the duration of
    // the pipeline run.
    ctx->encoded_slot_pool = pi_ring_create(PI_ENCODED_SLOT_COUNT);
    if (!ctx->encoded_slot_pool) {
        PI_ERROR("pipeline", "failed to allocate encoded_slot_pool");
        return -1;
    }
    ctx->encoded_slot_nals =
        (uint8_t *)calloc(PI_ENCODED_SLOT_COUNT, PI_ENCODED_SLOT_CAP);
    if (!ctx->encoded_slot_nals) {
        PI_ERROR("pipeline", "failed to allocate encoded_slot_nals");
        return -1;
    }
    for (unsigned i = 0u; i < PI_ENCODED_SLOT_COUNT; i++) {
        memset(&ctx->encoded_slots[i], 0, sizeof ctx->encoded_slots[i]);
        ctx->encoded_slots[i].nal =
            ctx->encoded_slot_nals + (size_t)i * PI_ENCODED_SLOT_CAP;
        (void)pi_ring_push(ctx->encoded_slot_pool, &ctx->encoded_slots[i]);
    }

    // Start threads. threads_running is the shutdown flag every worker
    // polls on; flipping it to false in teardown is what breaks the
    // loops. Track creation success explicitly so teardown knows which
    // pthread_t handles are valid.
    atomic_store(&ctx->threads_running, true);
    atomic_store(&ctx->camera_thread_ok, false);
    atomic_store(&ctx->encoder_thread_ok, false);
    atomic_store(&ctx->sender_thread_ok, false);

    if (pthread_create(&ctx->camera_thread, NULL,
                       camera_thread_main, ctx) != 0) {
        PI_ERROR("pipeline", "pthread_create camera failed");
        atomic_store(&ctx->threads_running, false);
        return -1;
    }
    ctx->camera_thread_created = true;

    if (pthread_create(&ctx->encoder_thread, NULL,
                       encoder_thread_main, ctx) != 0) {
        PI_ERROR("pipeline", "pthread_create encoder failed");
        atomic_store(&ctx->threads_running, false);
        pthread_join(ctx->camera_thread, NULL);
        ctx->camera_thread_created = false;
        return -1;
    }
    ctx->encoder_thread_created = true;

    if (pthread_create(&ctx->sender_thread, NULL,
                       sender_thread_main, ctx) != 0) {
        PI_ERROR("pipeline", "pthread_create sender failed");
        atomic_store(&ctx->threads_running, false);
        pthread_join(ctx->encoder_thread, NULL);
        pthread_join(ctx->camera_thread, NULL);
        ctx->encoder_thread_created = false;
        ctx->camera_thread_created  = false;
        return -1;
    }
    ctx->sender_thread_created = true;

    // ----- Hailo worker threads + shared state -----
    // No per-kind input rings: the encoder thread calls pi_infer_submit
    // synchronously before releasing the camera frame, and the workers
    // only call pi_infer_poll + publish. hailo_input_rings[] is kept as
    // NULL for backward-compat with the teardown loop but never used.
    pi_inference_state_init(&ctx->inference_state);
    for (int k = 0; k < PI_INFER__COUNT; k++) {
        ctx->hailo_input_rings[k] = NULL;
        atomic_store(&ctx->hailo_thread_ok[k], false);
        ctx->hailo_thread_created[k] = false;
    }

    // Spawn workers: pose / object / hand. Each one is idempotent with
    // respect to the others — if one fails to create we tear the others
    // down via the shared threads_running flag and let the teardown path
    // join whichever handles were already valid.
    typedef void *(*hailo_worker_fn_t)(void *);
    const hailo_worker_fn_t workers[PI_INFER__COUNT] = {
        hailo_worker_pose,
        hailo_worker_object,
        hailo_worker_hand,
    };
    for (int k = 0; k < PI_INFER__COUNT; k++) {
        if (pthread_create(&ctx->hailo_threads[k], NULL,
                           workers[k], ctx) != 0) {
            PI_ERROR("pipeline", "pthread_create hailo_worker[%d] failed", k);
            atomic_store(&ctx->threads_running, false);
            for (int j = 0; j < k; j++) {
                if (ctx->hailo_thread_created[j]) {
                    pthread_join(ctx->hailo_threads[j], NULL);
                    ctx->hailo_thread_created[j] = false;
                }
            }
            pthread_join(ctx->sender_thread, NULL);
            pthread_join(ctx->encoder_thread, NULL);
            pthread_join(ctx->camera_thread, NULL);
            ctx->sender_thread_created  = false;
            ctx->encoder_thread_created = false;
            ctx->camera_thread_created  = false;
            return -1;
        }
        ctx->hailo_thread_created[k] = true;
    }
#else
    (void)ctx;
#endif
    return 0;
}

static int pipeline_main_loop(pipeline_ctx_t *ctx) {
#ifdef PI_TARGET
    // ---------------- Pi threaded main loop ----------------
    // Capture runs on camera_thread (core 2, prio 40), encode on
    // encoder_thread (core 2, prio 50), TX+control-plane on sender_thread
    // (core 1, prio 45). The main thread is now a minimal supervisor:
    // it polls the shutdown flag + max_frames and sleeps otherwise. All
    // control-plane and UDP completion work lives in sender_thread.
    const pi_pipeline_args_t *args = ctx->args;

    while (1) {
        pi_barrier_acquire();
        if (g_shutdown_requested) break;
        if (args->max_frames > 0u &&
            ctx->frames_done >= args->max_frames) {
            PI_INFO("main", "max_frames=%llu reached, stopping",
                    (unsigned long long)args->max_frames);
            g_shutdown_requested = 1;
            break;
        }

        // 5 ms supervisor wake-up is plenty — we just need to notice
        // shutdown + max_frames in bounded time. The worker threads own
        // the hot path entirely.
        struct timespec ts = { 0, 5000000 };
        nanosleep(&ts, NULL);
    }

    PI_INFO("main", "pipeline stopping after %llu frames",
            (unsigned long long)ctx->frames_done);
    return 0;
#else
    // ---------------- Host single-threaded main loop ----------------
    // Unchanged behaviour: capture, fan-out to Hailo workers, encode,
    // chunk, send — all on one thread. Kept behind #ifndef PI_TARGET so
    // the existing host test suite exercises identical semantics.
    const pi_pipeline_args_t *args   = ctx->args;
    pi_camera_t              *cam    = ctx->cam;
    pi_encoder_t             *enc    = ctx->enc;
    pi_udp_sender_t          *udp    = ctx->udp;
    pi_hsm_t                 *hsm    = ctx->hsm;
    pi_infer_model_t        **models = ctx->models;
    pi_client_state_t        *client = &ctx->client;

    // A barrier here guarantees the signal-handler's write to
    // g_shutdown_requested is visible on the load path. volatile alone is
    // sufficient under POSIX for sig_atomic_t, but an explicit acquire
    // fence documents intent and costs nothing on Cortex-A76 (DMB ISHLD).
    while (1) {
        pi_barrier_acquire();
        if (g_shutdown_requested) break;
        if (args->max_frames > 0 && ctx->frames_done >= args->max_frames) {
            PI_INFO("main", "max_frames=%llu reached, stopping",
                    (unsigned long long)args->max_frames);
            break;
        }

        // Control-plane drain + heartbeat timeout (shared helper).
        pipeline_service_control_plane(ctx);

        pi_camera_frame_t frame = {0};
        if (pi_camera_capture(cam, &frame, 50) != 0) {
            continue;  // timeout or transient error
        }

        // Fan-out: submit the same pixel buffer to all 3 Hailo workers
        // and publish results into ctx->inference_state so the overlay
        // renderer below can read them. Mocks have a one-slot FIFO per
        // model, so submit-then-poll round-trips synchronously inside
        // this loop — on the Pi threaded build the poll+publish happens
        // in hailo_worker_loop instead.
        if (args->enable_inference) {
            for (int k = 0; k < PI_INFER__COUNT; k++) {
                if (!models[k]) continue;
                (void)pi_infer_submit(models[k], frame.pixels,
                                      frame.pixels_size, frame.frame_id);
                pi_infer_result_t res;
                if (pi_infer_poll(models[k], &res) == 0) {
                    pi_inference_state_publish(
                        &ctx->inference_state,
                        (pi_infer_kind_t)k, &res);
                }
            }
        }

        // Encoder: only submit / chunk / send when there is an active,
        // non-paused client. This mirrors Python send_frame's early-out
        // (udp_protocol.py:152-156) and prevents us from burning CPU on
        // x264 encodes that would just be dropped.
        if (client->active && !client->paused) {
            // T3: draw currently-published bbox outlines onto the Y plane
            // in place BEFORE the encoder reads the frame. Everything the
            // encoder sees is what the receiver will decode.
            if (args->enable_inference) {
                draw_overlay_on_frame(&ctx->inference_state, &frame,
                                      args->width, args->height);
            }

            const bool force = (ctx->frames_done == 0u) || client->force_idr;
            client->force_idr = false;
            (void)pi_encoder_submit_yuv420(
                enc,
                frame.pixels,                  frame.y_stride,
                frame.pixels + frame.u_offset, frame.uv_stride,
                frame.pixels + frame.v_offset, frame.uv_stride,
                frame.timestamp_ns / 1000000u,
                force);

            pi_encoded_packet_t pkt;
            while (pi_encoder_next_packet(enc, &pkt) == 0) {
                (void)send_encoded_packet(udp, client->ip, client->port,
                                          args->width, args->height,
                                          frame.frame_id, &pkt);
            }
        }

        pi_udp_sender_poll(udp);
        pi_hsm_dispatch(hsm, PI_EVENT_FRAME_READY);
        pi_camera_release(cam, &frame);
        ctx->frames_done++;

        // Test hook: advance the iteration counter and (optionally) the
        // synthetic time offset. Production builds set both step and
        // counter to zero and skip this branch entirely after the
        // compiler folds the constants. See _test_hooks.h.
        g_pipeline_iteration_count++;
        if (g_pipeline_test_time_step_ms != 0) {
            g_pipeline_test_time_offset_ms += g_pipeline_test_time_step_ms;
        }
    }

    PI_INFO("main", "pipeline stopping after %llu frames",
            (unsigned long long)ctx->frames_done);

    return 0;
#endif
}

// ---------------------------------------------------------------------------
// pipeline_teardown: join-order topology
// ---------------------------------------------------------------------------
// On SIGINT / SIGTERM the signal handler flips `g_shutdown_requested` to 1
// (plain volatile sig_atomic_t store — the only async-signal-safe option per
// POSIX signal-safety rules) and returns. pipeline_teardown then coordinates
// a clean stop of all 6 worker threads + rings + backend objects + sigactions.
//
// Dependency topology between the workers:
//
//        camera_thread ──camera_ring──▶ encoder_thread ──encoder_ring──▶ sender_thread
//                ▲                           │
//                │                           ├──hailo_input_rings[pose]──▶ hailo_pose
//                │                           ├──hailo_input_rings[obj]───▶ hailo_object
//                │                           └──hailo_input_rings[hand]──▶ hailo_hand
//                │                           │
//                └────frame_pool_ring─────────┘    (recycling)
//
// None of the workers block on a mutex / cv / sem held by the signal path,
// so the joins only have to respect *resource* ordering (rings and backend
// objects), not thread-liveness ordering — every worker exits on its own
// polled check of `threads_running` + `g_shutdown_requested`.
//
// Join order (Pi):
//   1. Clear threads_running flag + pi_barrier_release()
//   2. Join hailo workers (pose, object, hand) — drain hailo_input_rings,
//      may take up to ~20 ms per worker for the pi_infer_poll retry loop
//      to bail (short-circuited on g_shutdown_requested).
//   3. Join sender thread — drains encoder_ring + does control plane RX.
//      Back-off is 100 us so worst case is one iteration (~a few ms).
//   4. Join encoder thread — produces encoded_slot_pool back-pressure.
//      Back-off is 200 us; x264 zero-latency submit+pull is sub-millisecond.
//   5. Join camera thread — last because it supplies frames. libcamera
//      capture has a 50 ms internal timeout, so this is the dominant
//      worst-case exit latency (~50 ms).
//
// After all threads joined:
//   6. Drain residual ring contents so allocations are returned to pools
//   7. Destroy rings
//   8. Destroy backend objects in REVERSE creation order
//      (hsm, hailo models, vdevice, camera, encoder, udp)
//   9. Restore sighandlers, free ctx
//
// Total worst-case: ~50 ms (camera) + ~20 ms (Hailo poll) overlapped with
// each other = ~50 ms wall-clock on cold libcamera. Well under the 500 ms
// shutdown budget. The timing is instrumented via a log line below.
static void pipeline_teardown(pipeline_ctx_t *ctx) {
#ifdef PI_TARGET
    // Stop worker threads FIRST so they can no longer touch backend
    // objects while we destroy them. All worker threads poll on
    // threads_running + g_shutdown_requested, so join order is just a
    // cleanup preference — no thread blocks waiting on another. We join
    // in reverse spawn order (hailo -> sender -> encoder -> camera) so
    // producer rings are drained before their ring memory disappears.
    //
    // Timing instrumentation: measure the total join duration so tests
    // can assert the 500 ms shutdown budget is met.
    const uint64_t t0 = pi_monotonic_ms();
    PI_INFO("pipeline", "shutdown: joining workers");
    atomic_store(&ctx->threads_running, false);
    // Publish the flag clear to every worker's next pi_barrier_acquire(). On
    // AArch64 this lowers to DMB ISHST and ensures workers that were already
    // past their loop-top check will observe false on their next iteration.
    pi_barrier_release();
    // Hailo workers first — they're the most recent spawn and must stop
    // before we destroy the Hailo input rings or the inference models
    // they reference via ctx->models[].
    for (int k = 0; k < PI_INFER__COUNT; k++) {
        if (ctx->hailo_thread_created[k]) {
            pthread_join(ctx->hailo_threads[k], NULL);
            ctx->hailo_thread_created[k] = false;
        }
    }
    if (ctx->sender_thread_created) {
        pthread_join(ctx->sender_thread, NULL);
        ctx->sender_thread_created = false;
    }
    if (ctx->encoder_thread_created) {
        pthread_join(ctx->encoder_thread, NULL);
        ctx->encoder_thread_created = false;
    }
    if (ctx->camera_thread_created) {
        pthread_join(ctx->camera_thread, NULL);
        ctx->camera_thread_created = false;
    }
    PI_INFO("pipeline", "shutdown: joins done in %llu ms",
            (unsigned long long)(pi_monotonic_ms() - t0));
    // Drain any in-flight frames BEFORE destroying the rings so their
    // backend-allocated opaque (CompletedSlot* on libcamera — heap-allocated
    // in camera_libcamera.cpp:549) is handed back to pi_camera_release and
    // freed cleanly. Without this, a race between camera_thread capturing
    // and shutdown can leave a 32-byte CompletedSlot leaked on teardown.
    // Safe because all worker threads have already been joined above —
    // no concurrent producer/consumer can touch these rings now.
    // pi_camera_release tolerates a NULL opaque and also zeros the frame
    // struct (camera_libcamera.cpp:688-710), so a partially-populated slot
    // in the frame pool is still safe to pass in.
    if (ctx->camera_ring && ctx->cam) {
        void *p = NULL;
        while (pi_ring_pop(ctx->camera_ring, &p) == 0) {
            pi_camera_frame_t *f = (pi_camera_frame_t *)p;
            pi_camera_release(ctx->cam, f);
        }
    }
    if (ctx->frame_pool_ring && ctx->cam) {
        void *p = NULL;
        while (pi_ring_pop(ctx->frame_pool_ring, &p) == 0) {
            pi_camera_frame_t *f = (pi_camera_frame_t *)p;
            if (f && f->opaque) {
                pi_camera_release(ctx->cam, f);
            }
        }
    }
    if (ctx->camera_ring) {
        pi_ring_destroy(ctx->camera_ring);
        ctx->camera_ring = NULL;
    }
    if (ctx->encoder_ring) {
        pi_ring_destroy(ctx->encoder_ring);
        ctx->encoder_ring = NULL;
    }
    if (ctx->frame_pool_ring) {
        pi_ring_destroy(ctx->frame_pool_ring);
        ctx->frame_pool_ring = NULL;
    }
    if (ctx->encoded_slot_pool) {
        pi_ring_destroy(ctx->encoded_slot_pool);
        ctx->encoded_slot_pool = NULL;
    }
    if (ctx->encoded_slot_nals) {
        free(ctx->encoded_slot_nals);
        ctx->encoded_slot_nals = NULL;
    }
    for (int k = 0; k < PI_INFER__COUNT; k++) {
        if (ctx->hailo_input_rings[k]) {
            pi_ring_destroy(ctx->hailo_input_rings[k]);
            ctx->hailo_input_rings[k] = NULL;
        }
    }
    // Destroy the producer-serialisation mutexes after every worker has
    // joined so no concurrent holder exists. Guarded by
    // mutexes_initialized so a teardown after a pipeline_setup() failure
    // (spawn_workers never ran) doesn't pthread_mutex_destroy an
    // uninitialised handle.
    if (ctx->mutexes_initialized) {
        pthread_mutex_destroy(&ctx->frame_pool_mtx);
        pthread_mutex_destroy(&ctx->encoded_slot_pool_mtx);
        pthread_mutex_destroy(&ctx->client_mtx);
        ctx->mutexes_initialized = false;
    }
#endif

    // Polite-disconnect: tell any still-registered client that the server
    // is going away. The iOS app uses this to clear its waitingForIDR
    // banner immediately rather than waiting for its own packet-silence
    // watchdog. Matches Python UDPStreamer.stop() behaviour for the
    // server-initiated shutdown case.
    if (ctx->udp && ctx->client.active) {
        (void)pi_udp_sender_send(ctx->udp, ctx->client.ip, ctx->client.port,
                                 (const uint8_t *)"GONE", 4u);
    }

    if (ctx->hsm) {
        pi_hsm_dispatch(ctx->hsm, PI_EVENT_STOP);
    }

    // Reverse order of construction.
    if (ctx->hsm) {
        pi_hsm_destroy(ctx->hsm);
        ctx->hsm = NULL;
    }
    if (ctx->args && ctx->args->enable_inference) {
        for (int k = 0; k < PI_INFER__COUNT; k++) {
            if (ctx->models[k]) {
                pi_infer_model_destroy(ctx->models[k]);
                ctx->models[k] = NULL;
            }
        }
        if (ctx->vdev) {
            pi_vdevice_destroy(ctx->vdev);
            ctx->vdev = NULL;
        }
    }
    if (ctx->cam) {
        // Only call stop() if start() succeeded. On the mock backend
        // this is a no-op, but on libcamera stop-without-start is
        // undefined.
        if (ctx->camera_started) {
            pi_camera_stop(ctx->cam);
            ctx->camera_started = false;
        }
        pi_camera_destroy(ctx->cam);
        ctx->cam = NULL;
    }
    if (ctx->enc) {
        pi_encoder_flush(ctx->enc);
        pi_encoder_destroy(ctx->enc);
        ctx->enc = NULL;
    }
    if (ctx->udp) {
        pi_udp_sender_destroy(ctx->udp);
        ctx->udp = NULL;
    }

    // Restore previous signal handlers if setup installed them.
    if (ctx->sighandlers_installed) {
        sigaction(SIGINT,  &ctx->old_sigint,  NULL);
        sigaction(SIGTERM, &ctx->old_sigterm, NULL);
        ctx->sighandlers_installed = false;
    }
}

int pi_pipeline_run(const pi_pipeline_args_t *args) {
    if (!args) return -1;

    pipeline_ctx_t ctx = {0};
    ctx.args = args;

    int rc = pipeline_setup(&ctx);
    if (rc != 0) {
        pipeline_teardown(&ctx);
        return rc;
    }

    rc = pipeline_spawn_workers(&ctx);
    if (rc != 0) {
        pipeline_teardown(&ctx);
        return rc;
    }

    rc = pipeline_main_loop(&ctx);

    pipeline_teardown(&ctx);
    return rc;
}
