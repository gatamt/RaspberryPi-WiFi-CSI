#ifndef PI_STREAMER_COMPAT_ARCH_H
#define PI_STREAMER_COMPAT_ARCH_H

// Architecture compatibility shim: cache-line size, memory barriers, NEON
// availability. Targets GCC/Clang on Linux (host arm64 / x86_64 for TDD,
// Cortex-A76 on the Pi for production).
//
// Primary sources:
//   - ARM Architecture Reference Manual for ARMv8-A, DDI 0487
//     (AArch64 memory model: Chapter B2 "The AArch64 Application Level
//     Memory Model", barriers DMB ISH / LDAR/STLR).
//   - GCC __atomic built-ins:
//     https://gcc.gnu.org/onlinedocs/gcc/_005f_005fatomic-Builtins.html
//
// Citations go directly to the primary sources above.

#include <stdalign.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <time.h>

// Cortex-A76 has 64-byte L1 cache lines. Matches ARMv8-A
// generally-implemented CWG/ERG of 4 words (64 bytes).
#ifndef PI_CACHE_LINE_BYTES
#  define PI_CACHE_LINE_BYTES 64
#endif

// Monotonic millisecond clock — wraps clock_gettime(CLOCK_MONOTONIC) and
// converts to a uint64_t millisecond count. This is the canonical time
// source for heartbeat tracking and any other coarse-grained timeout in the
// pipeline. CLOCK_MONOTONIC is unaffected by wall-clock jumps (NTP, manual
// `date` adjustment), which is what we want for elapsed-time checks.
//
// Returning a uint64_t millisecond count gives ~584 million years of range,
// so wrap-around is not a concern in practice.
//
// Reference: POSIX clock_gettime(2). _GNU_SOURCE / _POSIX_C_SOURCE >= 199309
// is already pulled in by the build system (target_compile_definitions
// _GNU_SOURCE on both core libraries).
static inline uint64_t pi_monotonic_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000ULL +
           (uint64_t)(ts.tv_nsec / 1000000L);
}

#define PI_CACHELINE_ALIGN alignas(PI_CACHE_LINE_BYTES)

// Compiler branch hints (cold-path folding for error paths).
#if defined(__GNUC__) || defined(__clang__)
#  define PI_LIKELY(x)   (__builtin_expect(!!(x), 1))
#  define PI_UNLIKELY(x) (__builtin_expect(!!(x), 0))
#else
#  define PI_LIKELY(x)   (x)
#  define PI_UNLIKELY(x) (x)
#endif

// ARMv8-A-mapped C11 atomic fences.
//
// On AArch64 these lower to:
//   ACQUIRE -> DMB ISHLD  (load-acquire barrier, inner shareable)
//   RELEASE -> DMB ISHST  (store-release barrier, inner shareable)
//   SEQ_CST -> DMB ISH    (full barrier)
// See ARMv8 ARM B2.9.3 and GCC ARM back-end atomic lowering.
static inline void pi_barrier_acquire(void) {
    __atomic_thread_fence(__ATOMIC_ACQUIRE);
}

static inline void pi_barrier_release(void) {
    __atomic_thread_fence(__ATOMIC_RELEASE);
}

static inline void pi_barrier_full(void) {
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
}

// Compile-time NEON advertisement. Cortex-A76 (ARMv8.2-A) always has NEON
// (SIMD is mandatory in the A-profile base since ARMv8.0), so runtime
// detection is unnecessary on the Pi. On host builds (x86_64) this simply
// reports 0 and no-ops any NEON codepath we gate on it.
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#  define PI_HAVE_NEON 1
#else
#  define PI_HAVE_NEON 0
#endif

// Static assertion helper that works at file scope in C11.
#define PI_STATIC_ASSERT(cond, msg) _Static_assert((cond), msg)

// Force a variable to be treated as used (for intentionally-unused parameters
// that must stay named for readability).
#define PI_UNUSED(x) ((void)(x))

#endif // PI_STREAMER_COMPAT_ARCH_H
