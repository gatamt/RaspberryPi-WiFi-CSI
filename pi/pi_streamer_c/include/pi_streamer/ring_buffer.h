#ifndef PI_STREAMER_RING_BUFFER_H
#define PI_STREAMER_RING_BUFFER_H

// Single-producer single-consumer (SPSC) lock-free ring buffer for pointer
// slots. Used between pipeline stages: camera → encoder, encoder → UDP,
// camera → Hailo workers, Hailo → main.
//
// Design notes:
//   - Capacity is always a power of two so modulo becomes `idx & mask`.
//   - `head` and `tail` live on separate cache lines to avoid false sharing
//     between producer and consumer threads (Cortex-A76 has 64-byte lines).
//   - ACQUIRE / RELEASE ordering via C11 atomics. On AArch64 this lowers to
//     DMB ISHLD / ISHST (inner-shareable load / store barriers), which is
//     the minimum required for the producer's slot write to be visible to
//     the consumer once it reads `head`.
//   - The ring holds `void *` slots. It does not own what they point to;
//     callers are responsible for buffer lifetime. This keeps the data path
//     zero-copy — each slot is a pointer into a pre-allocated frame pool.
//
// Primary references:
//   - ARM Architecture Reference Manual for ARMv8-A (DDI 0487), chapter
//     B2 "AArch64 Application Level Memory Model".
//   - GCC __atomic Builtins:
//     https://gcc.gnu.org/onlinedocs/gcc/_005f_005fatomic-Builtins.html
//   - Leslie Lamport, "Specifying Concurrent Program Modules", 1983.
//
// No DTU atomic-note citations — see
// docs/Phase1-Atomic-Note-Verification.md for rationale.

#include <stdbool.h>
#include <stddef.h>

typedef struct pi_ring_buffer pi_ring_buffer_t;

// Create a ring holding up to `capacity` pointer slots. `capacity` MUST be
// a power of two and at least 2. Returns NULL on invalid argument or
// allocation failure.
pi_ring_buffer_t *pi_ring_create(size_t capacity);

// Free all memory owned by the ring. The pointer slots themselves are not
// touched — the ring never owns what they point to. NULL-safe.
void pi_ring_destroy(pi_ring_buffer_t *ring);

// Producer-side push. Non-blocking. Returns 0 on success, -1 if the ring
// is full. Must only be called from one producer thread.
int pi_ring_push(pi_ring_buffer_t *ring, void *slot);

// Consumer-side pop. Non-blocking. On success returns 0 and writes the
// popped slot into `*out`. On empty ring returns -1 and writes NULL to
// `*out` (if out is non-NULL). Must only be called from one consumer
// thread.
int pi_ring_pop(pi_ring_buffer_t *ring, void **out);

// Observability. These read both head and tail with ACQUIRE ordering,
// so they are best-effort snapshots — a producer or consumer can advance
// between the two atomic loads. Safe to call from any thread.
size_t pi_ring_capacity(const pi_ring_buffer_t *ring);
size_t pi_ring_size(const pi_ring_buffer_t *ring);
bool   pi_ring_is_empty(const pi_ring_buffer_t *ring);
bool   pi_ring_is_full(const pi_ring_buffer_t *ring);

#endif // PI_STREAMER_RING_BUFFER_H
