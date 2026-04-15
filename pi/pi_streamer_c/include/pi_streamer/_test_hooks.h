#ifndef PI_STREAMER__TEST_HOOKS_H
#define PI_STREAMER__TEST_HOOKS_H

// Internal test-only hooks for the host TDD build.
//
// This header is part of the pi_streamer_core_mock library's INTERNAL
// surface — production code (pi_streamer_core with the io_uring/x264/etc
// real backends) MUST NOT include it from end-user-facing call sites.
// pipeline.c does include it because pipeline.c is compiled into BOTH
// libraries: in production builds the test-hook bodies become unreferenced
// dead code that the linker drops via --gc-sections; in mock builds the
// test executable calls into them through this header.
//
// The hook bodies are always compiled (no `#ifdef PI_ENABLE_TEST_HOOKS`
// around them) so that the mock static library exports stable symbols
// regardless of how the test executable was built. A small amount of dead
// code leaks into pi_streamer_core_mock in non-test contexts, which is a
// deliberate trade-off to keep the per-test compile flags off the static
// library's command line.
//
// Why pipeline.c's iteration counter and the mock's pending-injection
// queue are coupled:
//   The pipeline owns its UDP sender (created and destroyed inside
//   pi_pipeline_run), so tests cannot inject into a per-instance FIFO from
//   outside. Instead the mock keeps a process-global queue keyed on a
//   delivery-iteration index, and pipeline.c bumps a global iteration
//   counter at the end of every loop pass. The mock's try_recv consults
//   the counter via pi_pipeline_test_get_iteration_count() before deciding
//   whether to deliver each pending entry. This lets a single
//   pi_pipeline_run() call observe a sequence of control messages at
//   precise iteration boundaries — needed for the GONE-mid-stream and
//   heartbeat-timeout tests.
//
// The iteration-counter + pending-injection pattern is what lets a
// single pi_pipeline_run() call reproduce GONE-mid-stream and
// heartbeat-timeout scenarios deterministically without any sleeps.

#include <stddef.h>
#include <stdint.h>

#include "pi_streamer/udp_sender.h"

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// UDP mock test hooks
// ---------------------------------------------------------------------------

// Enqueue a pending control datagram on the process-global queue. The
// mock UDP sender's pi_udp_sender_try_recv() drains entries whose
// `deliver_at_iteration` is <= the pipeline's current iteration counter.
//
// `data`/`n` are the payload bytes (n must be > 0 and <= 64).
// `src_ip`/`src_port` are reported back from try_recv as if a real client
// at that address sent the datagram.
// `deliver_at_iteration` is the earliest pipeline iteration at which the
//   entry becomes eligible. Use 0 to deliver on the very first iteration.
//
// Capacity is fixed at 16 pending entries; further pushes are dropped.
void pi_udp_mock_pending_inject(const uint8_t *data,
                                size_t         n,
                                const char    *src_ip,
                                uint16_t       src_port,
                                uint64_t       deliver_at_iteration);

// Drop every entry in the process-global pending queue. Tests should call
// this in setUp/tearDown so a leaked entry from one test cannot bleed
// into the next.
void pi_udp_mock_pending_clear(void);

// Snapshot of the most recently destroyed mock sender's stats and ring.
// pi_udp_sender_destroy() copies datagrams_sent + the first 32 send
// records into a process-global shadow before freeing, so tests can read
// what the pipeline sent AFTER pi_pipeline_run() returns and the live
// sender is gone.
//
// The snapshot is overwritten on every destroy, so a test that wants to
// compare two pipeline runs must read the snapshot between calls.

// Number of datagrams the most recently destroyed mock sender accepted.
uint64_t pi_udp_mock_last_send_count(void);

// Copy the i-th recorded payload into `out`. Returns bytes copied
// (clamped to `out_cap`), or 0 if `idx` is out of range.
size_t pi_udp_mock_last_send_at(size_t   idx,
                                uint8_t *out,
                                size_t   out_cap);

// Read the destination IP and port of the i-th recorded send. NULL-safe;
// out parameters that are NULL are skipped. Returns 0 on success, -1 if
// `idx` is out of range.
int pi_udp_mock_last_send_dst(size_t    idx,
                              char     *ip_out,
                              size_t    ip_cap,
                              uint16_t *port_out);

// Reset the snapshot ring. Tests call this in setUp so the previous
// test's destroyed-sender state doesn't pollute the assertions.
void pi_udp_mock_last_reset(void);

// ---------------------------------------------------------------------------
// Pipeline time + iteration hooks
// ---------------------------------------------------------------------------

// Override the millisecond reference used by the pipeline's heartbeat
// tracker. The pipeline reads `pi_monotonic_ms() + offset_ms` everywhere
// it needs to know "now". Pass 0 to clear the override.
void pi_pipeline_test_set_time_offset_ms(int64_t offset_ms);

// Per-iteration time step. The pipeline adds `step_ms` to the time
// offset at the END of each main-loop iteration. Setting step_ms = 11000
// makes time appear to advance by 11 seconds between iterations, which
// lets a single pi_pipeline_run() call trigger the heartbeat timeout
// without sleeping. Pass 0 to disable.
void pi_pipeline_test_set_time_step_ms(int64_t step_ms);

// Read the current pipeline iteration counter. The pipeline resets this
// to 0 at the start of pi_pipeline_run() and increments it at the end of
// every main-loop iteration. The mock UDP sender uses this to decide
// when to deliver pending injections.
uint64_t pi_pipeline_test_get_iteration_count(void);

// Force the iteration counter to a specific value. Tests do not normally
// need this — pipeline_run resets it automatically — but it is exposed for
// completeness and for the (unlikely) case where two tests share state.
void pi_pipeline_test_reset_iteration_count(void);

#ifdef __cplusplus
}
#endif

#endif // PI_STREAMER__TEST_HOOKS_H
