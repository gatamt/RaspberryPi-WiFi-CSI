// _GNU_SOURCE is defined via target_compile_definitions in CMakeLists.txt.
#include "pi_streamer/logger.h"

#include <stdatomic.h>
#include <stdio.h>
#include <string.h>
#include <sys/uio.h>
#include <time.h>
#include <unistd.h>

// See logger.h for design rationale and primary references.

// Internal level gate. Reads in hot paths use acquire, writes use release.
static _Atomic int g_min_level = PI_LOG_INFO;

int pi_log_init(pi_log_level_t min_level) {
    atomic_store_explicit(&g_min_level, (int)min_level, memory_order_release);
    return 0;
}

void pi_log_set_level(pi_log_level_t level) {
    atomic_store_explicit(&g_min_level, (int)level, memory_order_release);
}

pi_log_level_t pi_log_get_level(void) {
    return (pi_log_level_t)atomic_load_explicit(&g_min_level,
                                                memory_order_acquire);
}

static const char *level_label(pi_log_level_t level) {
    switch (level) {
        case PI_LOG_DEBUG: return "DEBUG";
        case PI_LOG_INFO:  return "INFO ";
        case PI_LOG_WARN:  return "WARN ";
        case PI_LOG_ERROR: return "ERROR";
    }
    return "?????";
}

void pi_log_v(pi_log_level_t level,
              const char *module,
              const char *fmt,
              va_list ap) {
    const int min =
        atomic_load_explicit(&g_min_level, memory_order_acquire);
    if ((int)level < min) {
        return;
    }

    // Monotonic timestamp (ms). CLOCK_MONOTONIC is immune to NTP adjustments.
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    const long ms = (long)ts.tv_sec * 1000L + (long)(ts.tv_nsec / 1000000L);

    char prefix[96];
    int plen = snprintf(prefix, sizeof prefix,
                        "[%010ld %s %-8s] ",
                        ms,
                        level_label(level),
                        module ? module : "-");
    if (plen < 0) {
        plen = 0;
    } else if ((size_t)plen > sizeof prefix) {
        plen = (int)sizeof prefix;
    }

    char body[512];
    int blen = vsnprintf(body, sizeof body, fmt, ap);
    if (blen < 0) {
        blen = 0;
    } else if ((size_t)blen > sizeof body - 1) {
        blen = (int)(sizeof body - 1);
    }

    char nl = '\n';
    struct iovec iov[3];
    iov[0].iov_base = prefix;
    iov[0].iov_len  = (size_t)plen;
    iov[1].iov_base = body;
    iov[1].iov_len  = (size_t)blen;
    iov[2].iov_base = &nl;
    iov[2].iov_len  = 1;

    // writev is atomic on Linux for total size <= PIPE_BUF (4096). Our
    // worst case is 96+512+1 = 609 bytes — well within the limit, so no
    // interleaving between threads.
    ssize_t w = writev(STDERR_FILENO, iov, 3);
    (void)w;  // best-effort: we don't retry partial writes in the hot path.
}

void pi_log(pi_log_level_t level,
            const char *module,
            const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    pi_log_v(level, module, fmt, ap);
    va_end(ap);
}
