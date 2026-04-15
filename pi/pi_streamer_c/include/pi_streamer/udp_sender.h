#ifndef PI_STREAMER_UDP_SENDER_H
#define PI_STREAMER_UDP_SENDER_H

// UDP send + control path. Two backends share this header:
//   Host / TDD:   src/udp_sender_mock.c   (records datagrams into an
//                   in-memory ring for test assertions)
//   Pi production: src/udp_sender_iouring.c  (compiled only when
//                   find_package(liburing) succeeds; uses io_uring with
//                   registered buffers and optional SQPOLL)
//
// Responsibilities:
//   - Enqueue outbound datagrams to a given (ip, port).
//   - Drive completions when using io_uring (no-op on mock).
//   - Receive inbound control messages (VID0/BEAT/PAWS/GONE/VACK) and
//     expose them to the state machine.
//
// Primary references: `man io_uring`, liburing/examples/, kernel.dk's
// io_uring PDF. No vault atomic-note citations.

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct pi_udp_sender pi_udp_sender_t;

typedef struct {
    const char *bind_ip;        // "0.0.0.0"
    uint16_t    bind_port;      // 3334
    size_t      sq_depth;       // io_uring SQ entries (256 is plenty)
    size_t      send_buf_bytes; // SO_SNDBUF target (e.g. 524288)
} pi_udp_sender_config_t;

pi_udp_sender_t *pi_udp_sender_create(const pi_udp_sender_config_t *cfg);
void             pi_udp_sender_destroy(pi_udp_sender_t *s);

// Enqueue a datagram for transmission. Returns 0 on enqueue,
// -1 on invalid argument or full queue.
int pi_udp_sender_send(pi_udp_sender_t *s,
                       const char      *dst_ip,
                       uint16_t         dst_port,
                       const uint8_t   *data,
                       size_t           data_len);

// Poll the backend for send completions and apply any per-datagram
// bookkeeping. No-op on the mock. Returns the number of completions
// processed on success, -1 on error.
int pi_udp_sender_poll(pi_udp_sender_t *s);

// Try to receive a pending control datagram. Non-blocking. Returns:
//    1  a datagram was copied to buf; *received, *src_ip, *src_port filled
//    0  no datagram available
//   -1  error
int pi_udp_sender_try_recv(pi_udp_sender_t *s,
                           uint8_t  *buf, size_t buf_cap, size_t *received,
                           char     *src_ip, size_t src_ip_cap,
                           uint16_t *src_port);

// Observability — cumulative counters. Safe to call from any thread.
typedef struct {
    uint64_t datagrams_sent;
    uint64_t bytes_sent;
    uint64_t datagrams_received;
    uint64_t send_drops;
} pi_udp_sender_stats_t;

void pi_udp_sender_stats(const pi_udp_sender_t *s,
                         pi_udp_sender_stats_t *out);

#endif // PI_STREAMER_UDP_SENDER_H
