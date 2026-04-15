#ifndef PI_STREAMER_LOGGER_H
#define PI_STREAMER_LOGGER_H

// Structured, level-gated, RT-safe logger for pi_streamer_c.
//
// Design constraints:
//   1. Safe to call from a SCHED_FIFO thread without inducing priority
//      inversion — must not take glibc stdio locks (printf) and must not
//      malloc. Output goes through writev(STDERR_FILENO, ...) which is a
//      single atomic syscall for payloads < PIPE_BUF (4096 bytes on Linux).
//   2. Per-call stack buffer only — no heap, no static mutex.
//   3. Level gate is an atomic int set once at init, updated rarely.
//   4. Monotonic wall-clock timestamps so log ordering is preserved under
//      NTP-induced REALTIME skews.
//
// Primary references:
//   - Linux kernel printk / printk_safe documentation
//     (docs.kernel.org/core-api/printk-safe.html) for RT-safe pattern.
//   - `man 2 writev` for atomicity guarantees on Linux.
//
// No DTU atomic notes are cited — see docs/Phase1-Atomic-Note-Verification.md.

#include <stdarg.h>
#include <stddef.h>

typedef enum {
    PI_LOG_DEBUG = 0,
    PI_LOG_INFO  = 1,
    PI_LOG_WARN  = 2,
    PI_LOG_ERROR = 3,
} pi_log_level_t;

// Initialize logger with given minimum emission level. Safe to call more
// than once. Returns 0 on success.
int pi_log_init(pi_log_level_t min_level);

// Update the minimum level at runtime (e.g. --log-level flag in main).
// Thread-safe; other threads may see the change at the next pi_log() call.
void pi_log_set_level(pi_log_level_t level);

// Read the current minimum level (mostly for tests).
pi_log_level_t pi_log_get_level(void);

// Emit a formatted log line. `module` is a short tag (e.g. "camera") and
// may be NULL. Format is printf-style; compiler validates it.
void pi_log(pi_log_level_t level,
            const char *module,
            const char *fmt, ...)
    __attribute__((format(printf, 3, 4)));

// Variadic variant used by forwarders.
void pi_log_v(pi_log_level_t level,
              const char *module,
              const char *fmt,
              va_list ap)
    __attribute__((format(printf, 3, 0)));

// Convenience macros so callers don't repeat the level name.
#define PI_DEBUG(mod, ...) pi_log(PI_LOG_DEBUG, (mod), __VA_ARGS__)
#define PI_INFO(mod, ...)  pi_log(PI_LOG_INFO,  (mod), __VA_ARGS__)
#define PI_WARN(mod, ...)  pi_log(PI_LOG_WARN,  (mod), __VA_ARGS__)
#define PI_ERROR(mod, ...) pi_log(PI_LOG_ERROR, (mod), __VA_ARGS__)

#endif // PI_STREAMER_LOGGER_H
