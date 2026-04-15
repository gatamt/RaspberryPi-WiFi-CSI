// _GNU_SOURCE is defined via target_compile_definitions in CMakeLists.txt.
// Needed for pthread_setaffinity_np, pthread_setname_np, cpu_set_t, and
// sched_setscheduler on glibc.
#include "pi_streamer/rt.h"
#include "pi_streamer/logger.h"

// Host (non-Pi) build: all four helpers are no-ops returning 0 so the
// TDD loop on Mac/Linux x86_64 stays identical. Only the Pi build pulls
// in the Linux-specific RT headers (sched.h cpu_set_t, sys/mman.h).
#ifdef PI_TARGET

#include <errno.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <string.h>
#include <sys/mman.h>

// Prefault size — keep small enough that mlock can't fail on a fresh
// stack but large enough to cover the deepest recursive call in any
// of the Pi RT threads (camera/encoder/sender/workers).
#define PI_RT_PREFAULT_BYTES (128u * 1024u)

int pi_rt_lock_memory(void) {
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
        PI_WARN("rt", "mlockall failed: %s", strerror(errno));
        return -1;
    }

    // Touch each 4 KiB page of a sacrificial stack buffer so the kernel
    // commits + wires the pages now rather than during the first RT loop
    // iteration, which would otherwise add ms-scale jitter. Write AND
    // read through an asm clobber so GCC cannot DCE the loop under
    // -O3 / -Werror=unused-but-set-variable, and cannot strip the
    // volatile via -Wcast-qual.
    volatile uint8_t stack[PI_RT_PREFAULT_BYTES];
    for (size_t i = 0; i < PI_RT_PREFAULT_BYTES; i += 4096u) {
        stack[i] = 0u;
        __asm__ volatile("" : : "r"(stack[i]) : "memory");
    }
    return 0;
}

int pi_rt_pin_thread(pthread_t tid, int cpu) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    int rc = pthread_setaffinity_np(tid, sizeof set, &set);
    if (rc != 0) {
        PI_WARN("rt", "pthread_setaffinity_np(cpu=%d) failed: %s",
                cpu, strerror(rc));
        return -1;
    }
    return 0;
}

int pi_rt_promote_fifo(pthread_t tid, int priority) {
    struct sched_param p;
    memset(&p, 0, sizeof p);
    p.sched_priority = priority;
    int rc = pthread_setschedparam(tid, SCHED_FIFO, &p);
    if (rc != 0) {
        PI_WARN("rt", "pthread_setschedparam(SCHED_FIFO, prio=%d) failed: %s",
                priority, strerror(rc));
        return -1;
    }
    return 0;
}

int pi_rt_name_thread(pthread_t tid, const char *name) {
    if (name == NULL) {
        return -1;
    }
    // Linux caps TASK_COMM_LEN at 16 including NUL. Truncate proactively
    // so pthread_setname_np doesn't reject the call with ERANGE.
    char buf[16];
    size_t n = strlen(name);
    if (n > 15) {
        n = 15;
    }
    memcpy(buf, name, n);
    buf[n] = '\0';
    int rc = pthread_setname_np(tid, buf);
    if (rc != 0) {
        PI_WARN("rt", "pthread_setname_np(\"%s\") failed: %s",
                buf, strerror(rc));
        return -1;
    }
    return 0;
}

#else // !PI_TARGET — host no-op implementation.

int pi_rt_lock_memory(void) {
    return 0;
}

int pi_rt_pin_thread(pthread_t tid, int cpu) {
    (void)tid;
    (void)cpu;
    return 0;
}

int pi_rt_promote_fifo(pthread_t tid, int priority) {
    (void)tid;
    (void)priority;
    return 0;
}

int pi_rt_name_thread(pthread_t tid, const char *name) {
    (void)tid;
    (void)name;
    return 0;
}

#endif // PI_TARGET
