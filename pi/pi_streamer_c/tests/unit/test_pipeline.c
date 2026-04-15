#include "pi_streamer/pipeline.h"
#include "pi_streamer/wire_format.h"
#include "unity.h"

#ifdef PI_ENABLE_TEST_HOOKS
#  include "pi_streamer/_test_hooks.h"
#endif

#include <string.h>

// Reset every process-global piece of test state between sub-tests.
// Without this, a stray pending injection or a stale time offset from
// one test would leak into the next and the failure mode would look
// unrelated to the actual cause.
void setUp(void) {
#ifdef PI_ENABLE_TEST_HOOKS
    pi_udp_mock_pending_clear();
    pi_udp_mock_last_reset();
    pi_pipeline_test_set_time_offset_ms(0);
    pi_pipeline_test_set_time_step_ms(0);
    pi_pipeline_test_reset_iteration_count();
#endif
}

void tearDown(void) {
#ifdef PI_ENABLE_TEST_HOOKS
    pi_udp_mock_pending_clear();
    pi_pipeline_test_set_time_offset_ms(0);
    pi_pipeline_test_set_time_step_ms(0);
#endif
}

static void test_defaults_are_sane(void) {
    pi_pipeline_args_t args = {0};
    pi_pipeline_args_defaults(&args);
    TEST_ASSERT_EQUAL_UINT32(1280,    args.width);
    TEST_ASSERT_EQUAL_UINT32(720,     args.height);
    TEST_ASSERT_EQUAL_UINT32(30,      args.fps);
    TEST_ASSERT_EQUAL_UINT32(2500000, args.bitrate_bps);
    TEST_ASSERT_EQUAL_UINT32(30,      args.gop_size);
    TEST_ASSERT_EQUAL_UINT16(3334,    args.udp_port);
    TEST_ASSERT_EQUAL_UINT64(0,       args.max_frames);
    TEST_ASSERT_TRUE(args.enable_inference);
    TEST_ASSERT_EQUAL_STRING("127.0.0.1", args.dest_ip);
}

static void test_parse_known_flags(void) {
    pi_pipeline_args_t args;
    pi_pipeline_args_defaults(&args);

    char *argv[] = {
        (char *)"pi_streamer_c",
        (char *)"--width=640",
        (char *)"--height=480",
        (char *)"--fps=15",
        (char *)"--bitrate-bps=1000000",
        (char *)"--gop-size=15",
        (char *)"--udp-port=3335",
        (char *)"--max-frames=100",
        (char *)"--enable-inference=false",
        (char *)"--dest-ip=192.168.1.5",
        (char *)"--dest-port=5500",
        (char *)"--log-level=DEBUG",
    };
    const int argc = (int)(sizeof argv / sizeof argv[0]);

    TEST_ASSERT_EQUAL_INT(0, pi_pipeline_parse_args(argc, argv, &args));
    TEST_ASSERT_EQUAL_UINT32(640,       args.width);
    TEST_ASSERT_EQUAL_UINT32(480,       args.height);
    TEST_ASSERT_EQUAL_UINT32(15,        args.fps);
    TEST_ASSERT_EQUAL_UINT32(1000000,   args.bitrate_bps);
    TEST_ASSERT_EQUAL_UINT32(15,        args.gop_size);
    TEST_ASSERT_EQUAL_UINT16(3335,      args.udp_port);
    TEST_ASSERT_EQUAL_UINT64(100,       args.max_frames);
    TEST_ASSERT_FALSE(args.enable_inference);
    TEST_ASSERT_EQUAL_STRING("192.168.1.5", args.dest_ip);
    TEST_ASSERT_EQUAL_UINT16(5500,      args.dest_port);
}

static void test_parse_rejects_unknown_flag(void) {
    pi_pipeline_args_t args;
    pi_pipeline_args_defaults(&args);
    char *argv[] = {
        (char *)"pi_streamer_c",
        (char *)"--no-such-flag=1",
    };
    TEST_ASSERT_EQUAL_INT(-1, pi_pipeline_parse_args(2, argv, &args));
}

static void test_parse_rejects_missing_equals(void) {
    pi_pipeline_args_t args;
    pi_pipeline_args_defaults(&args);
    char *argv[] = {
        (char *)"pi_streamer_c",
        (char *)"--width",
    };
    TEST_ASSERT_EQUAL_INT(-1, pi_pipeline_parse_args(2, argv, &args));
}

static void test_run_bounded_by_max_frames(void) {
    pi_pipeline_args_t args;
    pi_pipeline_args_defaults(&args);
    args.max_frames       = 5;
    args.enable_inference = true;
    args.log_level        = PI_LOG_WARN;   // quieter test output

    const int rc = pi_pipeline_run(&args);
    TEST_ASSERT_EQUAL_INT(0, rc);
}

static void test_run_without_inference(void) {
    pi_pipeline_args_t args;
    pi_pipeline_args_defaults(&args);
    args.max_frames       = 3;
    args.enable_inference = false;
    args.log_level        = PI_LOG_WARN;

    const int rc = pi_pipeline_run(&args);
    TEST_ASSERT_EQUAL_INT(0, rc);
}

static void test_run_rejects_null_args(void) {
    TEST_ASSERT_EQUAL_INT(-1, pi_pipeline_run(NULL));
}

#ifdef PI_ENABLE_TEST_HOOKS

// Helper: build a minimal pipeline_args_t for control-plane tests.
//
// We disable inference (the mocks do nothing useful at this layer and only
// make the test slower), set the log level to ERROR so the test output
// stays readable, and pick a width/height divisible by 16 so the encoder
// mock and YUV plane offsets line up with whatever real camera_libcamera
// would deliver. dest_ip/dest_port are deliberately bogus — the pipeline
// only uses them as defaults until VID0 arrives.
static void control_plane_args(pi_pipeline_args_t *args, uint64_t max_frames) {
    pi_pipeline_args_defaults(args);
    args->max_frames       = max_frames;
    args->enable_inference = false;
    args->log_level        = PI_LOG_ERROR;
    args->width            = 320;
    args->height           = 240;
    args->fps              = 30;
    args->bitrate_bps      = 1000000;
    args->gop_size         = 30;
    args->dest_ip          = "0.0.0.0";   // never used once VID0 lands
    args->dest_port        = 0;
}

// Search the recorded send ring for the index of the first datagram that
// looks like a video chunk (28-byte wire-format header + payload >= 30B
// total). Control responses such as VACK are 4-byte payloads, so the size
// filter is sufficient. Returns -1 if none found.
static int find_first_video_send(uint64_t total) {
    for (size_t i = 0; i < total; i++) {
        uint8_t buf[PI_MAX_DATAGRAM_SIZE];
        const size_t n = pi_udp_mock_last_send_at(i, buf, sizeof buf);
        if (n >= PI_CHUNK_HEADER_SIZE + 1u) return (int)i;
    }
    return -1;
}

// Search the recorded send ring for the first 4-byte payload matching
// `tag` (e.g. "VACK"). Returns -1 if not found.
static int find_first_control_send(uint64_t total, const char tag[4]) {
    for (size_t i = 0; i < total; i++) {
        uint8_t buf[16];
        const size_t n = pi_udp_mock_last_send_at(i, buf, sizeof buf);
        if (n == 4u && memcmp(buf, tag, 4) == 0) return (int)i;
    }
    return -1;
}

// Test 1: VID0 registration + first chunk is a keyframe.
//
// Drops a synthetic VID0 datagram from 127.0.0.1:5000 onto the pending
// queue with deliver_at_iteration=0, runs the pipeline for 5 frames, then
// inspects the send ring shadow:
//   - at least one video chunk was transmitted to the registered client
//   - a VACK was sent back to the same address
//   - the first video chunk's wire-format frame_type field is 1 (IDR)
static void test_client_registration_vid0(void) {
    const uint8_t vid0[] = {'V', 'I', 'D', '0'};
    pi_udp_mock_pending_inject(vid0, sizeof vid0, "127.0.0.1", 5000, 0u);

    pi_pipeline_args_t args;
    control_plane_args(&args, 5);
    TEST_ASSERT_EQUAL_INT(0, pi_pipeline_run(&args));

    const uint64_t total = pi_udp_mock_last_send_count();
    TEST_ASSERT_TRUE_MESSAGE(total >= 2u,
        "expected at least one VACK + one video chunk recorded");

    // VACK should be present and addressed to 127.0.0.1:5000.
    const int vack_idx = find_first_control_send(total, "VACK");
    TEST_ASSERT_TRUE_MESSAGE(vack_idx >= 0, "no VACK found in send ring");
    char     vack_ip[64] = {0};
    uint16_t vack_port   = 0;
    TEST_ASSERT_EQUAL_INT(0,
        pi_udp_mock_last_send_dst((size_t)vack_idx, vack_ip, sizeof vack_ip,
                                  &vack_port));
    TEST_ASSERT_EQUAL_STRING("127.0.0.1", vack_ip);
    TEST_ASSERT_EQUAL_UINT16(5000, vack_port);

    // First video chunk should be a keyframe — pipeline forces an IDR on
    // the first frame after VID0 registration. We parse the wire-format
    // header to confirm, since the encoder mock fills the payload with a
    // counter byte and not real H.264.
    const int video_idx = find_first_video_send(total);
    TEST_ASSERT_TRUE_MESSAGE(video_idx >= 0, "no video chunk in send ring");
    char     vid_ip[64] = {0};
    uint16_t vid_port   = 0;
    TEST_ASSERT_EQUAL_INT(0,
        pi_udp_mock_last_send_dst((size_t)video_idx, vid_ip, sizeof vid_ip,
                                  &vid_port));
    TEST_ASSERT_EQUAL_STRING("127.0.0.1", vid_ip);
    TEST_ASSERT_EQUAL_UINT16(5000, vid_port);

    uint8_t hdr_bytes[PI_CHUNK_HEADER_SIZE];
    const size_t n = pi_udp_mock_last_send_at((size_t)video_idx, hdr_bytes,
                                              sizeof hdr_bytes);
    TEST_ASSERT_EQUAL_size_t(PI_CHUNK_HEADER_SIZE, n);
    pi_chunk_header_t hdr = {0};
    TEST_ASSERT_EQUAL_INT(0, pi_chunk_header_unpack(hdr_bytes, &hdr));
    TEST_ASSERT_EQUAL_UINT32_MESSAGE(1u, hdr.frame_type,
        "first sent video chunk should be a keyframe (frame_type=1)");
}

// Test 2: heartbeat timeout drops the client.
//
// Registers via VID0 on iteration 0, then advances the synthetic time by
// 11 seconds per iteration. After iteration 0 sends some video chunks,
// iteration 1's heartbeat check sees elapsed > 10s and drops the client,
// so no further video chunks are emitted.
static void test_heartbeat_timeout_drops_client(void) {
    const uint8_t vid0[] = {'V', 'I', 'D', '0'};
    pi_udp_mock_pending_inject(vid0, sizeof vid0, "127.0.0.1", 5000, 0u);

    // Each iteration advances synthetic time by 11s. Iter 0 is processed
    // BEFORE the bump, so it sees elapsed=0 and emits chunks. Iter 1 sees
    // elapsed=11s and triggers the 10s timeout.
    pi_pipeline_test_set_time_step_ms(11000);

    pi_pipeline_args_t args;
    control_plane_args(&args, 2);
    TEST_ASSERT_EQUAL_INT(0, pi_pipeline_run(&args));

    const uint64_t total = pi_udp_mock_last_send_count();
    const int first_video = find_first_video_send(total);
    TEST_ASSERT_TRUE_MESSAGE(first_video >= 0,
        "iter 0 should have emitted at least one video chunk before the timeout");

    // Count total video chunks. With 1 chunk per frame (mock encoder
    // emits 512-byte NALs which fit in one 1400-byte payload) and the
    // pipeline only encoding while the client is active, iter 0 produces
    // exactly 1 video chunk and iter 1 produces 0. Allow a little slack
    // in case the mock changes shape later.
    size_t video_chunks = 0;
    for (size_t i = 0; i < total; i++) {
        uint8_t buf[PI_MAX_DATAGRAM_SIZE];
        const size_t n = pi_udp_mock_last_send_at(i, buf, sizeof buf);
        if (n >= PI_CHUNK_HEADER_SIZE + 1u) video_chunks++;
    }
    TEST_ASSERT_EQUAL_size_t_MESSAGE(1u, video_chunks,
        "exactly one video chunk should have been sent before the heartbeat timeout");
}

// Test 3: GONE mid-run stops video sends.
//
// Registers via VID0 on iteration 0 and pre-schedules a GONE on
// iteration 3. Runs for 6 frames. Iters 0..2 should each emit a video
// chunk; iters 3..5 should emit nothing (the GONE drained at the start
// of iter 3 deactivates the client before encoder submit).
static void test_gone_stops_video_sends(void) {
    const uint8_t vid0[] = {'V', 'I', 'D', '0'};
    const uint8_t gone[] = {'G', 'O', 'N', 'E'};
    pi_udp_mock_pending_inject(vid0, sizeof vid0, "127.0.0.1", 5000, 0u);
    pi_udp_mock_pending_inject(gone, sizeof gone, "127.0.0.1", 5000, 3u);

    pi_pipeline_args_t args;
    control_plane_args(&args, 6);
    TEST_ASSERT_EQUAL_INT(0, pi_pipeline_run(&args));

    const uint64_t total = pi_udp_mock_last_send_count();
    size_t video_chunks = 0;
    for (size_t i = 0; i < total; i++) {
        uint8_t buf[PI_MAX_DATAGRAM_SIZE];
        const size_t n = pi_udp_mock_last_send_at(i, buf, sizeof buf);
        if (n >= PI_CHUNK_HEADER_SIZE + 1u) video_chunks++;
    }
    TEST_ASSERT_EQUAL_size_t_MESSAGE(3u, video_chunks,
        "iters 0..2 should each emit one video chunk; GONE on iter 3 stops the rest");
}

#endif // PI_ENABLE_TEST_HOOKS

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_defaults_are_sane);
    RUN_TEST(test_parse_known_flags);
    RUN_TEST(test_parse_rejects_unknown_flag);
    RUN_TEST(test_parse_rejects_missing_equals);
    RUN_TEST(test_run_bounded_by_max_frames);
    RUN_TEST(test_run_without_inference);
    RUN_TEST(test_run_rejects_null_args);
#ifdef PI_ENABLE_TEST_HOOKS
    RUN_TEST(test_client_registration_vid0);
    RUN_TEST(test_heartbeat_timeout_drops_client);
    RUN_TEST(test_gone_stops_video_sends);
#endif
    return UNITY_END();
}
