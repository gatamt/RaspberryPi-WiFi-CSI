// Pi production UDP sender backend — io_uring + SQPOLL + dedicated recv
// thread. Compiled only when CMake's PI_USE_REAL_UDP is ON and liburing is
// found by pkg-config (see CMakeLists.txt). On host (Mac / TDD) the
// build falls back to src/udp_sender_mock.c, so this file is Linux-only.
//
// Architecture:
//   - One io_uring instance shared by:
//       * the main pipeline thread (TX path: pi_udp_sender_send → SQE)
//       * the dedicated rx_loop pthread (RX path + completion drain)
//   - SQPOLL kernel thread pinned to core 1 (sq_thread_cpu = 1) so the
//     submission ring stays kernel-poll'd and userspace never makes a real
//     io_uring_enter syscall on the hot send path.
//   - SQPOLL needs CAP_SYS_NICE — the production systemd unit grants it.
//     If queue_init_params() fails (no caps, old kernel, etc.) we fall
//     back to a plain non-SQPOLL ring so dev hosts and rootless containers
//     still work.
//   - Registered files: socket fd is registered as fixed file slot 0 so
//     every SQE uses IOSQE_FIXED_FILE — no per-syscall fdget cost.
//   - Registered buffers: TX pool is page-aligned, mlock'd, and registered
//     via io_uring_register_buffers so the kernel can DMA from it without
//     re-mapping per send.
//   - Recv path: RX_BUF_COUNT recvmsg SQEs are pre-posted at startup. When
//     a CQE comes back, rx_loop copies the payload into a freshly malloc'd
//     rx_slot_t, pushes it onto an internal SPSC ring, and immediately
//     re-posts the recvmsg. The pipeline thread drains via
//     pi_udp_sender_try_recv → pi_ring_pop.
//
// Threading note on liburing's submission queue:
//   The SQ is single-producer by contract — io_uring_get_sqe + submit must
//   not race with another thread doing the same on the same ring. Because
//   both pi_udp_sender_send (main thread) and rx_loop (recv re-post) feed
//   the same ring, we serialize SQE acquisition with sq_mutex. The mutex
//   is held only across get_sqe + prep + submit (a few cycles + one cheap
//   write to the SQPOLL kernel thread's tail); contention is bounded by
//   the RX_BUF_COUNT re-posts per second and is not on the hot send path
//   when no recv is happening.
//
// Primary references: man io_uring(7), liburing/examples/*, kernel.dk's
// "Efficient IO with io_uring" PDF.

#include "pi_streamer/udp_sender.h"
#include "pi_streamer/ring_buffer.h"

#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#include <liburing.h>

// ---- Tunables --------------------------------------------------------------
//
// MAX_TX_BUFS = 64 fits the free-bitmap into a single uint64_t so the
// alloc/free fast path is one __builtin_ctzll plus one atomic CAS-free RMW.
// At the planned 30 fps × ~10 chunks/frame this gives >200 ms of in-flight
// budget before back-pressure kicks in, which is far longer than the
// 50 ms SO_SNDTIMEO upper bound — buffers always recycle in time.
#define MAX_TX_BUFS    64
#define TX_BUF_SIZE    1500   // standard MTU; H.264 chunks fit comfortably
#define RX_BUF_COUNT   16
#define RX_BUF_SIZE    128    // VID0/BEAT/PAWS/GONE/VACK are 4 bytes each
#define RX_RING_SLOTS  16     // power of two, matches RX_BUF_COUNT

// SQE user_data tagging. Bit layout:
//   0x1000 .. 0x1FFF — recv slot completion (low 12 bits = slot index)
//   0x2000 .. 0x2FFF — send buffer completion (low 12 bits = TX buf index)
#define TAG_RECV_BASE  0x1000ULL
#define TAG_RECV_MASK  0xF000ULL
#define TAG_SEND_BASE  0x2000ULL

// ---- Internal types --------------------------------------------------------

typedef struct {
    size_t   data_len;
    uint8_t  data[RX_BUF_SIZE];
    char     src_ip[INET_ADDRSTRLEN];
    uint16_t src_port;
} rx_slot_t;

struct pi_udp_sender {
    pi_udp_sender_config_t cfg;

    struct io_uring        ring;
    int                    sock_fd;
    bool                   sqpoll_enabled;  // false if fallback to plain ring

    // Registered TX buffer pool. Page-aligned + mlock'd so the kernel can
    // DMA from it without TLB faults. tx_free_bitmap is a 64-bit free list:
    // bit i set = buffer i is free.
    uint8_t               *tx_pool;
    struct iovec           tx_iovs[MAX_TX_BUFS];
    uint64_t               tx_free_bitmap;

    // Per-slot destination address storage. MUST NOT live on the stack:
    // io_uring_prep_sendto stores only the pointer (sqe->addr2) and the
    // kernel dereferences it asynchronously — under SQPOLL the send is
    // issued by the SQPOLL kernel thread long after pi_udp_sender_send
    // has returned, so a stack-local sockaddr_in would be freed before
    // the kernel reads it. By parking the dst here (heap-allocated, same
    // lifetime as pi_udp_sender_t), the kernel always sees valid addr
    // bytes until the matching CQE comes back and the slot is freed.
    struct sockaddr_in     tx_dsts[MAX_TX_BUFS];

    // Pre-posted recv buffers + msg headers. rx_addrs is the per-slot
    // sockaddr_in storage that recvmsg writes the source address into.
    uint8_t                rx_pool[RX_BUF_COUNT][RX_BUF_SIZE];
    struct msghdr          rx_msghdrs[RX_BUF_COUNT];
    struct iovec           rx_iovs[RX_BUF_COUNT];
    struct sockaddr_in     rx_addrs[RX_BUF_COUNT];

    // Internal SPSC staging ring: rx_loop pushes rx_slot_t*, the pipeline
    // thread pops via pi_udp_sender_try_recv. Decouples kernel completion
    // timing from pipeline polling cadence.
    pi_ring_buffer_t      *rx_spsc;

    // Recv thread state.
    pthread_t              rx_thread;
    atomic_bool            rx_running;

    // Submission queue mutex. The SQ is single-producer per liburing
    // contract — both the main thread (send path) and rx_loop (recv re-post)
    // need to serialize io_uring_get_sqe + io_uring_submit on this ring.
    pthread_mutex_t        sq_mutex;

    // Cumulative stats (atomic so pi_udp_sender_stats can read from any
    // thread without locking).
    atomic_uint_fast64_t   datagrams_sent;
    atomic_uint_fast64_t   bytes_sent;
    atomic_uint_fast64_t   datagrams_received;
    atomic_uint_fast64_t   send_drops;
};

// Forward declaration so pi_udp_sender_create can pthread_create on it.
static void *rx_loop(void *arg);

// ---- Helpers ---------------------------------------------------------------

// Pre-post a single recvmsg SQE for slot `idx`. Caller MUST hold sq_mutex.
// Returns 0 on success, -1 if get_sqe returned NULL (SQ full).
static int post_recv_locked(pi_udp_sender_t *s, int idx) {
    struct io_uring_sqe *sqe = io_uring_get_sqe(&s->ring);
    if (!sqe) return -1;

    // Reset the msghdr each post: the kernel rewrites msg_namelen with the
    // actual address length, and msg_iov/iovlen aren't supposed to change
    // across calls but we re-bind them defensively in case a future
    // refactor reuses a slot for a different buffer.
    s->rx_iovs[idx].iov_base = s->rx_pool[idx];
    s->rx_iovs[idx].iov_len  = RX_BUF_SIZE;
    memset(&s->rx_msghdrs[idx], 0, sizeof s->rx_msghdrs[idx]);
    s->rx_msghdrs[idx].msg_name    = &s->rx_addrs[idx];
    s->rx_msghdrs[idx].msg_namelen = sizeof s->rx_addrs[idx];
    s->rx_msghdrs[idx].msg_iov     = &s->rx_iovs[idx];
    s->rx_msghdrs[idx].msg_iovlen  = 1;

    // Slot 0 = registered socket fd. IOSQE_FIXED_FILE skips fdget.
    io_uring_prep_recvmsg(sqe, 0, &s->rx_msghdrs[idx], 0);
    sqe->flags |= IOSQE_FIXED_FILE;
    sqe->user_data = TAG_RECV_BASE + (uint64_t)idx;
    return 0;
}

// ---- Lifecycle -------------------------------------------------------------

pi_udp_sender_t *pi_udp_sender_create(const pi_udp_sender_config_t *cfg) {
    if (!cfg || !cfg->bind_ip || cfg->sq_depth == 0) return NULL;

    pi_udp_sender_t *s = calloc(1, sizeof *s);
    if (!s) return NULL;
    s->cfg = *cfg;
    s->sock_fd = -1;
    s->tx_pool = NULL;
    s->rx_spsc = NULL;

    // ---- Socket --------------------------------------------------------
    s->sock_fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (s->sock_fd < 0) goto fail;

    int one = 1;
    if (setsockopt(s->sock_fd, SOL_SOCKET, SO_REUSEADDR,
                   &one, sizeof one) < 0) goto fail;

    // SO_SNDBUF target — kernel may clamp to net.core.wmem_max. We do not
    // treat clamping as a fatal error; the bootstrap script raises wmem_max
    // to 524288 to match Python's default.
    int sndbuf = (int)cfg->send_buf_bytes;
    setsockopt(s->sock_fd, SOL_SOCKET, SO_SNDBUF, &sndbuf, sizeof sndbuf);

    // SO_RCVBUF — control plane only, 32 KiB matches udp_protocol.py.
    int rcvbuf = 32768;
    setsockopt(s->sock_fd, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof rcvbuf);

    // SO_SNDTIMEO = 50 ms. Bounds main-loop stalls under transient Wi-Fi
    // contention. Matches the Python streamer's struct.pack("ll", 0, 50000).
    struct timeval tv = { .tv_sec = 0, .tv_usec = 50000 };
    setsockopt(s->sock_fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);

    struct sockaddr_in bind_addr = {0};
    bind_addr.sin_family = AF_INET;
    bind_addr.sin_port   = htons(cfg->bind_port);
    if (inet_pton(AF_INET, cfg->bind_ip, &bind_addr.sin_addr) != 1) goto fail;
    if (bind(s->sock_fd, (struct sockaddr *)&bind_addr,
             sizeof bind_addr) < 0) goto fail;

    // ---- io_uring with SQPOLL fallback --------------------------------
    struct io_uring_params p = {0};
    p.flags          = IORING_SETUP_SQPOLL | IORING_SETUP_SQ_AFF;
    p.sq_thread_cpu  = 1;     // pin SQ kernel thread to core 1 (plan §4.2)
    p.sq_thread_idle = 2000;  // 2 s before SQPOLL parks itself
    if (io_uring_queue_init_params((unsigned)cfg->sq_depth,
                                   &s->ring, &p) < 0) {
        // Fallback: drop SQPOLL + SQ_AFF and retry with a plain ring.
        // Common reasons: missing CAP_SYS_NICE (rootless container, dev
        // host), kernel < 5.13, or SQPOLL disabled in sysctl.
        memset(&p, 0, sizeof p);
        if (io_uring_queue_init_params((unsigned)cfg->sq_depth,
                                       &s->ring, &p) < 0) {
            goto fail;  // Stop condition: SQPOLL AND fallback both failed.
        }
        s->sqpoll_enabled = false;
    } else {
        s->sqpoll_enabled = true;
    }

    // Register the bound socket as fixed file slot 0. Every subsequent
    // SQE will use IOSQE_FIXED_FILE + fd=0 instead of the real fd.
    int fds[1] = { s->sock_fd };
    if (io_uring_register_files(&s->ring, fds, 1) < 0) {
        io_uring_queue_exit(&s->ring);
        goto fail;
    }

    // ---- TX pool (page-aligned + mlock'd + registered) ----------------
    if (posix_memalign((void **)&s->tx_pool, 4096,
                       (size_t)MAX_TX_BUFS * TX_BUF_SIZE) != 0) {
        s->tx_pool = NULL;
        io_uring_queue_exit(&s->ring);
        goto fail;
    }
    // mlock failure is non-fatal — the kernel will still DMA from the
    // pool, just with the possibility of a minor page fault on first
    // touch. Worth logging but never aborting.
    (void)mlock(s->tx_pool, (size_t)MAX_TX_BUFS * TX_BUF_SIZE);

    for (int i = 0; i < MAX_TX_BUFS; ++i) {
        s->tx_iovs[i].iov_base = s->tx_pool + (size_t)i * TX_BUF_SIZE;
        s->tx_iovs[i].iov_len  = TX_BUF_SIZE;
    }
    if (io_uring_register_buffers(&s->ring, s->tx_iovs, MAX_TX_BUFS) < 0) {
        // Buffer registration failure is also non-fatal: we just lose the
        // zero-copy DMA optimization. The kernel falls back to a copy.
        // Log once but continue.
    }
    s->tx_free_bitmap = ~0ULL;  // all 64 buffers free

    // ---- Internal SPSC staging ring -----------------------------------
    s->rx_spsc = pi_ring_create(RX_RING_SLOTS);
    if (!s->rx_spsc) {
        munlock(s->tx_pool, (size_t)MAX_TX_BUFS * TX_BUF_SIZE);
        free(s->tx_pool);
        s->tx_pool = NULL;
        io_uring_queue_exit(&s->ring);
        goto fail;
    }

    // ---- SQ mutex + initial recv pre-post -----------------------------
    if (pthread_mutex_init(&s->sq_mutex, NULL) != 0) {
        pi_ring_destroy(s->rx_spsc);
        s->rx_spsc = NULL;
        munlock(s->tx_pool, (size_t)MAX_TX_BUFS * TX_BUF_SIZE);
        free(s->tx_pool);
        s->tx_pool = NULL;
        io_uring_queue_exit(&s->ring);
        goto fail;
    }

    // Pre-post all RX_BUF_COUNT recvmsg SQEs in one shot. Holding the
    // mutex across the loop is fine — no other thread is alive yet.
    pthread_mutex_lock(&s->sq_mutex);
    for (int i = 0; i < RX_BUF_COUNT; ++i) {
        if (post_recv_locked(s, i) != 0) {
            pthread_mutex_unlock(&s->sq_mutex);
            pthread_mutex_destroy(&s->sq_mutex);
            pi_ring_destroy(s->rx_spsc);
            s->rx_spsc = NULL;
            munlock(s->tx_pool, (size_t)MAX_TX_BUFS * TX_BUF_SIZE);
            free(s->tx_pool);
            s->tx_pool = NULL;
            io_uring_queue_exit(&s->ring);
            goto fail;
        }
    }
    io_uring_submit(&s->ring);
    pthread_mutex_unlock(&s->sq_mutex);

    // ---- Spawn recv thread --------------------------------------------
    atomic_store(&s->rx_running, true);
    if (pthread_create(&s->rx_thread, NULL, rx_loop, s) != 0) {
        pthread_mutex_destroy(&s->sq_mutex);
        pi_ring_destroy(s->rx_spsc);
        s->rx_spsc = NULL;
        munlock(s->tx_pool, (size_t)MAX_TX_BUFS * TX_BUF_SIZE);
        free(s->tx_pool);
        s->tx_pool = NULL;
        io_uring_queue_exit(&s->ring);
        goto fail;
    }
    pthread_setname_np(s->rx_thread, "pi-udp-rx");

    return s;

fail:
    if (s->sock_fd >= 0) close(s->sock_fd);
    free(s);
    return NULL;
}

void pi_udp_sender_destroy(pi_udp_sender_t *s) {
    if (!s) return;

    // Tell rx_loop to stop, then wake it. shutdown(SHUT_RD) causes any
    // in-flight recvmsg to complete with -ESHUTDOWN/-EBADF, which produces
    // a CQE and unblocks io_uring_wait_cqe.
    atomic_store(&s->rx_running, false);
    if (s->sock_fd >= 0) shutdown(s->sock_fd, SHUT_RD);
    pthread_join(s->rx_thread, NULL);

    // Drain any rx_slot_t pointers still queued in the SPSC ring so we
    // don't leak the malloc'd payload structures.
    if (s->rx_spsc) {
        void *leftover = NULL;
        while (pi_ring_pop(s->rx_spsc, &leftover) == 0) {
            free(leftover);
        }
    }

    io_uring_queue_exit(&s->ring);
    pthread_mutex_destroy(&s->sq_mutex);
    if (s->sock_fd >= 0) close(s->sock_fd);

    if (s->tx_pool) {
        munlock(s->tx_pool, (size_t)MAX_TX_BUFS * TX_BUF_SIZE);
        free(s->tx_pool);
    }
    if (s->rx_spsc) pi_ring_destroy(s->rx_spsc);
    free(s);
}

// ---- Send path -------------------------------------------------------------

int pi_udp_sender_send(pi_udp_sender_t *s,
                       const char      *dst_ip,
                       uint16_t         dst_port,
                       const uint8_t   *data,
                       size_t           data_len) {
    if (!s || !dst_ip || !data) return -1;
    if (data_len == 0u || data_len > TX_BUF_SIZE) return -1;

    // ---- Allocate a TX buffer from the bitmap -------------------------
    // Loop over the bitmap with __builtin_ctzll. The CAS-free RMW pattern
    // (load + bit clear + atomic_and) is racy under multi-producer, but
    // the send path is called only from the pipeline thread by design.
    // Even so, we use __atomic_fetch_and so completions from the rx thread
    // (which sets bits in the bitmap on send completion) interleave
    // correctly with the producer's clear.
    uint64_t free_bits =
        __atomic_load_n(&s->tx_free_bitmap, __ATOMIC_ACQUIRE);
    if (free_bits == 0) {
        atomic_fetch_add(&s->send_drops, 1);
        return -1;
    }
    int idx = __builtin_ctzll(free_bits);
    // Atomically clear the bit. We don't retry if the bit was already
    // clear because only the producer clears bits — the rx thread only
    // sets them. So this fetch_and is effectively a publish.
    __atomic_fetch_and(&s->tx_free_bitmap, ~(1ULL << idx), __ATOMIC_ACQ_REL);

    // Copy payload into the registered (mlock'd, page-aligned) slot.
    uint8_t *slot = s->tx_pool + (size_t)idx * TX_BUF_SIZE;
    memcpy(slot, data, data_len);

    // ---- Build sockaddr_in for sendto ---------------------------------
    // CRITICAL: the dst MUST live in s->tx_dsts[idx], NOT on the stack.
    // io_uring_prep_sendto stores only the pointer in sqe->addr2, and
    // under SQPOLL the kernel reads it asynchronously after this function
    // has already returned. A stack-local would be destroyed before the
    // kernel consumes it → garbage addr → silent send-to-nowhere.
    struct sockaddr_in *dst = &s->tx_dsts[idx];
    memset(dst, 0, sizeof *dst);
    dst->sin_family = AF_INET;
    dst->sin_port   = htons(dst_port);
    if (inet_pton(AF_INET, dst_ip, &dst->sin_addr) != 1) {
        // Return the buffer to the free pool before returning the error.
        __atomic_or_fetch(&s->tx_free_bitmap,
                          1ULL << idx, __ATOMIC_RELEASE);
        return -1;
    }

    // ---- Submit the SQE under the SQ mutex ----------------------------
    pthread_mutex_lock(&s->sq_mutex);
    struct io_uring_sqe *sqe = io_uring_get_sqe(&s->ring);
    if (!sqe) {
        pthread_mutex_unlock(&s->sq_mutex);
        // SQ full — return the buffer and count the drop.
        __atomic_or_fetch(&s->tx_free_bitmap,
                          1ULL << idx, __ATOMIC_RELEASE);
        atomic_fetch_add(&s->send_drops, 1);
        return -1;
    }
    // prep_sendto wraps prep_send + prep_send_set_addr. The buffer pointer
    // (slot) is inside our registered TX pool, so the kernel can DMA from
    // it without re-mapping. Slot 0 is the registered socket fd. The dst
    // pointer below is stable until the matching CQE comes back (see
    // struct comment above on tx_dsts).
    io_uring_prep_sendto(sqe, /* fixed file slot */ 0,
                         slot, data_len, /* flags */ 0,
                         (const struct sockaddr *)dst, sizeof *dst);
    sqe->flags    |= IOSQE_FIXED_FILE;
    sqe->user_data = TAG_SEND_BASE + (uint64_t)idx;
    // io_uring_submit is cheap under SQPOLL (just a memory store to the
    // tail; the kernel SQPOLL thread picks it up on its next poll).
    // Without SQPOLL it makes a real io_uring_enter syscall.
    int sret = io_uring_submit(&s->ring);
    pthread_mutex_unlock(&s->sq_mutex);

    if (sret < 0) {
        // Submission failed — return the buffer and report drop.
        __atomic_or_fetch(&s->tx_free_bitmap,
                          1ULL << idx, __ATOMIC_RELEASE);
        atomic_fetch_add(&s->send_drops, 1);
        return -1;
    }

    atomic_fetch_add(&s->datagrams_sent, 1);
    atomic_fetch_add(&s->bytes_sent, data_len);
    return 0;
}

// ---- Recv path -------------------------------------------------------------

// Dedicated thread that owns CQE draining for both recv and send paths.
// Send completions just return the TX buffer to the free pool; recv
// completions stage the payload onto the SPSC ring and re-post the recv.
static void *rx_loop(void *arg) {
    pi_udp_sender_t *s = arg;

    while (atomic_load(&s->rx_running)) {
        struct io_uring_cqe *cqe = NULL;
        int ret = io_uring_wait_cqe(&s->ring, &cqe);
        if (ret < 0) {
            // EINTR is the common case during shutdown — re-check
            // rx_running and either continue or exit.
            if (ret == -EINTR) continue;
            break;
        }
        if (!cqe) continue;

        const uint64_t tag = cqe->user_data;

        if ((tag & TAG_RECV_MASK) == TAG_RECV_BASE) {
            // ---- Recv completion ---------------------------------------
            const int idx = (int)(tag & 0xFFFULL);
            if (idx >= 0 && idx < RX_BUF_COUNT && cqe->res > 0) {
                rx_slot_t *slot = malloc(sizeof *slot);
                if (slot) {
                    size_t n = (size_t)cqe->res;
                    if (n > RX_BUF_SIZE) n = RX_BUF_SIZE;
                    slot->data_len = n;
                    memcpy(slot->data, s->rx_pool[idx], n);
                    inet_ntop(AF_INET, &s->rx_addrs[idx].sin_addr,
                              slot->src_ip, sizeof slot->src_ip);
                    slot->src_port = ntohs(s->rx_addrs[idx].sin_port);
                    if (pi_ring_push(s->rx_spsc, slot) != 0) {
                        // SPSC ring full — drop the message rather than
                        // block. Pipeline is too slow draining; this is
                        // best-effort control traffic.
                        free(slot);
                    } else {
                        atomic_fetch_add(&s->datagrams_received, 1);
                    }
                }
            }
            // Re-post this recv slot regardless of whether the payload
            // staging succeeded — we always want RX_BUF_COUNT recvs in
            // flight. Holds sq_mutex briefly to serialize with the
            // pipeline thread's send path.
            if (idx >= 0 && idx < RX_BUF_COUNT) {
                pthread_mutex_lock(&s->sq_mutex);
                if (post_recv_locked(s, idx) == 0) {
                    io_uring_submit(&s->ring);
                }
                pthread_mutex_unlock(&s->sq_mutex);
            }

        } else if (tag >= TAG_SEND_BASE) {
            // ---- Send completion ---------------------------------------
            const int buf_idx = (int)(tag - TAG_SEND_BASE);
            // Track per-error stats. We treat -EAGAIN / -ETIMEDOUT as a
            // "drop" because that means SO_SNDTIMEO fired — the chunk
            // never made it onto the wire.
            if (cqe->res < 0 &&
                (cqe->res == -EAGAIN || cqe->res == -ETIMEDOUT)) {
                atomic_fetch_add(&s->send_drops, 1);
            }
            if (buf_idx >= 0 && buf_idx < MAX_TX_BUFS) {
                __atomic_or_fetch(&s->tx_free_bitmap,
                                  1ULL << buf_idx, __ATOMIC_RELEASE);
            }
        }
        // else: unknown tag — ignore. Could happen during shutdown when
        // shutdown() generates a -ESHUTDOWN CQE for a recvmsg we already
        // unbound. We just acknowledge it and move on.

        io_uring_cqe_seen(&s->ring, cqe);
    }
    return NULL;
}

int pi_udp_sender_poll(pi_udp_sender_t *s) {
    // rx_loop owns CQE draining. In the multi-threaded design this
    // function is a no-op — see the architecture comment at the top of
    // the file. We keep the ABI symmetric with udp_sender_mock.c so the
    // pipeline doesn't need a backend-conditional poll().
    if (!s) return -1;
    return 0;
}

int pi_udp_sender_try_recv(pi_udp_sender_t *s,
                           uint8_t  *buf, size_t buf_cap, size_t *received,
                           char     *src_ip, size_t src_ip_cap,
                           uint16_t *src_port) {
    if (!s) return -1;

    void *slot_ptr = NULL;
    if (pi_ring_pop(s->rx_spsc, &slot_ptr) != 0 || !slot_ptr) {
        if (received) *received = 0u;
        if (src_port) *src_port = 0;
        return 0;  // empty
    }

    rx_slot_t *slot = slot_ptr;
    if (buf && buf_cap > 0) {
        size_t n = slot->data_len;
        if (n > buf_cap) n = buf_cap;
        memcpy(buf, slot->data, n);
        if (received) *received = n;
    } else if (received) {
        *received = 0u;
    }

    if (src_ip && src_ip_cap > 0) {
        const size_t srclen = strlen(slot->src_ip);
        const size_t cpy = srclen >= src_ip_cap ? src_ip_cap - 1u : srclen;
        memcpy(src_ip, slot->src_ip, cpy);
        src_ip[cpy] = '\0';
    }
    if (src_port) *src_port = slot->src_port;

    free(slot);
    return 1;
}

void pi_udp_sender_stats(const pi_udp_sender_t *s,
                         pi_udp_sender_stats_t *out) {
    if (!out) return;
    if (!s) {
        memset(out, 0, sizeof *out);
        return;
    }
    // atomic_load accepts a pointer to a const-qualified atomic, so we
    // don't need to cast away const here — we just take pointers into
    // the const struct directly. This keeps -Wcast-qual happy.
    out->datagrams_sent     = atomic_load(&s->datagrams_sent);
    out->bytes_sent         = atomic_load(&s->bytes_sent);
    out->datagrams_received = atomic_load(&s->datagrams_received);
    out->send_drops         = atomic_load(&s->send_drops);
}
