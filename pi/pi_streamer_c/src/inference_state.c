#include "pi_streamer/inference_state.h"

#include <stddef.h>
#include <string.h>

// Maximum retry count for a single-slot seqlock read. Eight is plenty: a
// reader only re-spins when a writer intersected its copy, and there is
// exactly one writer per kind. At 30 fps the contention window is a
// ~20 ns memcpy per publish, so re-reading more than once is already
// pathological.
#define PI_INFERENCE_STATE_MAX_RETRIES 8

void pi_inference_state_init(pi_inference_state_t *state) {
    if (!state) return;
    for (int k = 0; k < PI_INFER__COUNT; k++) {
        atomic_store_explicit(&state->slots[k].seq, 0u, memory_order_relaxed);
        memset(&state->slots[k].value, 0, sizeof state->slots[k].value);
    }
    // Make the zero-init visible to any thread that subsequently acquires
    // a reference to this struct. Callers should still synchronise ctx
    // publication through their own mechanism, but this ensures that the
    // very first snapshot after init cannot observe partial garbage.
    atomic_thread_fence(memory_order_release);
}

void pi_inference_state_publish(pi_inference_state_t    *state,
                                pi_infer_kind_t          kind,
                                const pi_infer_result_t *result) {
    if (!state || !result) return;
    if ((int)kind < 0 || (int)kind >= PI_INFER__COUNT) return;

    pi_inference_slot_t *slot = &state->slots[kind];

    // Step 1: bump seq to odd. acq_rel is required: the increment must not
    // move past the following memcpy (release side), and any previous
    // publish by this thread on a different slot must be visible before
    // we start writing (acquire side — matters if a worker were to
    // occasionally read its own neighbour's slot, which is not the case
    // today but is cheap to leave correct).
    atomic_fetch_add_explicit(&slot->seq, 1u, memory_order_acq_rel);

    // Step 2: copy the payload. Plain memcpy is fine — readers will notice
    // the torn write via the seq check and retry.
    slot->value = *result;

    // Step 3: release fence so the value store is visible to readers
    // before the second increment flips seq back to even.
    atomic_thread_fence(memory_order_release);

    // Step 4: bump seq to even. Publication is now complete.
    atomic_fetch_add_explicit(&slot->seq, 1u, memory_order_acq_rel);
}

bool pi_inference_state_snapshot(const pi_inference_state_t *state,
                                 pi_inference_state_t       *out) {
    if (!state || !out) return false;

    for (int k = 0; k < PI_INFER__COUNT; k++) {
        unsigned int s1 = 0u;
        unsigned int s2 = 0u;
        int retries = 0;
        bool ok = false;

        while (retries < PI_INFERENCE_STATE_MAX_RETRIES) {
            // Acquire-load seq. If odd, a writer is mid-publish; retry.
            s1 = atomic_load_explicit(&state->slots[k].seq,
                                      memory_order_acquire);
            if ((s1 & 1u) != 0u) {
                retries++;
                continue;
            }

            // Copy the value while seq is (hopefully) stable.
            out->slots[k].value = state->slots[k].value;

            // Acquire fence so the second load is ordered after the copy.
            atomic_thread_fence(memory_order_acquire);

            s2 = atomic_load_explicit(&state->slots[k].seq,
                                      memory_order_relaxed);

            if (s1 == s2) {
                ok = true;
                break;
            }
            retries++;
        }

        if (!ok) {
            // Stamp the out-slot seq so a caller inspecting partial
            // state sees exactly what we gave up on, then bail.
            atomic_store_explicit(&out->slots[k].seq, s1,
                                  memory_order_relaxed);
            return false;
        }
        atomic_store_explicit(&out->slots[k].seq, s1,
                              memory_order_relaxed);
    }
    return true;
}
