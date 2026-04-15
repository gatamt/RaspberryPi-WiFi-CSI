#include "pi_streamer/ring_buffer.h"
#include "compat/arch.h"

#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

// Internal representation. See ring_buffer.h for rationale.
//
// The "counter-based" head/tail scheme (each side is a monotonically
// increasing 64-bit counter, modulo = counter & mask) lets us distinguish
// empty (head == tail) from full (head - tail == capacity) without wasting
// a slot. Wrap of the counter itself happens at 2^SIZE_MAX which is not a
// real concern: at 1000 frames/sec this would take ~585 million years on
// 64-bit.
struct pi_ring_buffer {
    PI_CACHELINE_ALIGN _Atomic size_t head;  // written by producer
    PI_CACHELINE_ALIGN _Atomic size_t tail;  // written by consumer
    size_t   capacity;
    size_t   mask;
    void   **slots;
};

// The struct must be a multiple of the cache-line size so aligned_alloc
// is happy. The alignas(64) on head + tail forces this because the struct
// inherits its largest-member alignment.
PI_STATIC_ASSERT(sizeof(struct pi_ring_buffer) % PI_CACHE_LINE_BYTES == 0,
                 "ring buffer struct size must be cache-line multiple");

static bool is_power_of_two(size_t v) {
    return v >= 2u && (v & (v - 1u)) == 0u;
}

pi_ring_buffer_t *pi_ring_create(size_t capacity) {
    if (!is_power_of_two(capacity)) {
        return NULL;
    }

    pi_ring_buffer_t *ring = aligned_alloc(PI_CACHE_LINE_BYTES,
                                           sizeof *ring);
    if (!ring) {
        return NULL;
    }
    // aligned_alloc leaves memory uninitialized; zero it so the atomics
    // and pointers start at well-defined values. Zero is the valid
    // "unstarted" counter for head/tail.
    memset(ring, 0, sizeof *ring);

    ring->slots = calloc(capacity, sizeof *ring->slots);
    if (!ring->slots) {
        free(ring);
        return NULL;
    }
    ring->capacity = capacity;
    ring->mask     = capacity - 1u;
    return ring;
}

void pi_ring_destroy(pi_ring_buffer_t *ring) {
    if (!ring) return;
    free(ring->slots);
    free(ring);
}

int pi_ring_push(pi_ring_buffer_t *ring, void *slot) {
    if (PI_UNLIKELY(!ring)) return -1;

    // Producer owns `head` — a relaxed load is sufficient.
    const size_t head =
        atomic_load_explicit(&ring->head, memory_order_relaxed);
    // Tail is published by the consumer with RELEASE ordering. An ACQUIRE
    // load synchronizes with that, giving us an up-to-date free-slot count.
    const size_t tail =
        atomic_load_explicit(&ring->tail, memory_order_acquire);

    if (head - tail >= ring->capacity) {
        return -1;  // full
    }

    ring->slots[head & ring->mask] = slot;
    // Publish the new slot. The consumer's ACQUIRE load of `head` will
    // see the store above due to the release/acquire pair.
    atomic_store_explicit(&ring->head, head + 1u, memory_order_release);
    return 0;
}

int pi_ring_pop(pi_ring_buffer_t *ring, void **out) {
    if (PI_UNLIKELY(!ring)) {
        if (out) *out = NULL;
        return -1;
    }

    const size_t tail =
        atomic_load_explicit(&ring->tail, memory_order_relaxed);
    const size_t head =
        atomic_load_explicit(&ring->head, memory_order_acquire);

    if (head == tail) {
        if (out) *out = NULL;
        return -1;  // empty
    }

    void *slot = ring->slots[tail & ring->mask];
    if (out) *out = slot;
    // Free the slot for the producer.
    atomic_store_explicit(&ring->tail, tail + 1u, memory_order_release);
    return 0;
}

size_t pi_ring_capacity(const pi_ring_buffer_t *ring) {
    return ring ? ring->capacity : 0u;
}

size_t pi_ring_size(const pi_ring_buffer_t *ring) {
    if (!ring) return 0u;
    const size_t h =
        atomic_load_explicit(&ring->head, memory_order_acquire);
    const size_t t =
        atomic_load_explicit(&ring->tail, memory_order_acquire);
    return h - t;
}

bool pi_ring_is_empty(const pi_ring_buffer_t *ring) {
    return pi_ring_size(ring) == 0u;
}

bool pi_ring_is_full(const pi_ring_buffer_t *ring) {
    if (!ring) return false;
    return pi_ring_size(ring) >= ring->capacity;
}
