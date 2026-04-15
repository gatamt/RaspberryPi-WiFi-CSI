#ifndef PI_STREAMER_INFERENCE_STATE_H
#define PI_STREAMER_INFERENCE_STATE_H

// Cross-thread-published inference state using a Linux-style seqlock.
//
// Writer flow (each Hailo worker thread, one per pi_infer_kind_t):
//   1. Compute pi_infer_result_t on the local stack.
//   2. pi_inference_state_publish(state, kind, &result)
// Publish takes ~20 ns on Cortex-A76 — two atomic increments and a memcpy.
//
// Reader flow (encoder thread / overlay renderer):
//   1. pi_inference_state_snapshot(state, &snapshot_out)
//   2. Use snapshot_out — it is a self-contained copy.
//
// Protocol:
//   - Each slot holds an atomic uint seq counter.
//   - seq EVEN = stable, seq ODD = writer in progress.
//   - Writer: seq++ (odd) -> memcpy value -> release fence -> seq++ (even).
//   - Reader: load seq1; if odd retry; memcpy value; acquire fence; load
//     seq2; if seq1 != seq2 retry. Up to 8 retries, then return false.
//
// There is ONE seqlock per inference kind (pose / object / hand) so each
// worker can publish independently. Readers snapshot all three in one call
// and are guaranteed a per-slot consistent view (not a global cross-slot
// atomic view — that is not required by the overlay).
//
// The raw_output pointer inside pi_infer_result_t points into the owning
// worker thread's scratch buffer. Readers that need the tensor bytes MUST
// memcpy them out before the next publish cycle. The overlay renderer
// only consumes decoded scalar summaries (frame_id, kind, latency), so
// the pointer staleness window is not an issue in practice.
//
// Reference for the memory-ordering pattern:
//   Linux kernel include/linux/seqlock.h
//   https://www.kernel.org/doc/html/latest/locking/seqlock.html

#include "pi_streamer/inference.h"

#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>

// One published slot per inference kind. The atomic seq counter is the
// synchronisation primitive; `value` is the payload and is ONLY safe to
// touch via pi_inference_state_publish / _snapshot.
typedef struct {
    atomic_uint       seq;
    pi_infer_result_t value;
} pi_inference_slot_t;

typedef struct {
    pi_inference_slot_t slots[PI_INFER__COUNT];
} pi_inference_state_t;

// Zero-initialise every slot (seq = 0, value all zeros). Safe to call more
// than once. NULL-safe.
void pi_inference_state_init(pi_inference_state_t *state);

// Publish a result for one inference kind. Called on the owning Hailo
// worker thread; only one writer per kind. NULL-safe and kind-bounds-safe.
void pi_inference_state_publish(pi_inference_state_t    *state,
                                pi_infer_kind_t          kind,
                                const pi_infer_result_t *result);

// Snapshot every slot into `out`. Returns true on success, false if any
// slot could not be read consistently after PI_INFERENCE_STATE_MAX_RETRIES.
// `out` is overwritten even on failure (partial snapshot should not be
// consumed; caller should check the return value).
bool pi_inference_state_snapshot(const pi_inference_state_t *state,
                                 pi_inference_state_t       *out);

#endif  // PI_STREAMER_INFERENCE_STATE_H
