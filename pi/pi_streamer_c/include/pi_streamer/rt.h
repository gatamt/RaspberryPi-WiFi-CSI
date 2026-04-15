#ifndef PI_STREAMER_RT_H
#define PI_STREAMER_RT_H

// Real-time thread helpers. On Pi (PI_TARGET=1) they actually enforce
// mlockall, CPU affinity, SCHED_FIFO, and thread naming. On host builds
// they all degrade to no-ops returning 0 so TDD and mock builds keep
// compiling unchanged.
//
// Reference: Linux kernel Documentation/scheduler/sched-rt-group.rst,
// POSIX pthread_setaffinity_np(3), mlockall(2), sched_setscheduler(2).
// No vault atomic note citations — per docs/Phase1-Atomic-Note-Verification.md.

#include <pthread.h>
#include <stdbool.h>

// mlockall(MCL_CURRENT | MCL_FUTURE) and prefault the first 128 KiB of
// the calling thread's stack so subsequent RT threads never page-fault.
// Returns 0 on success. On host, returns 0 without doing anything.
int pi_rt_lock_memory(void);

// Pin a pthread to a single CPU via pthread_setaffinity_np.
// cpu is a 0-based core index. Returns 0 on success.
int pi_rt_pin_thread(pthread_t tid, int cpu);

// Promote a thread to SCHED_FIFO with the given priority (1-99).
// Requires CAP_SYS_NICE or running as root (systemd grants it).
int pi_rt_promote_fifo(pthread_t tid, int priority);

// Set the thread name visible via pthread_getname_np / /proc/<pid>/task.
// Linux limits names to 15 characters; longer names are truncated.
int pi_rt_name_thread(pthread_t tid, const char *name);

#endif // PI_STREAMER_RT_H
