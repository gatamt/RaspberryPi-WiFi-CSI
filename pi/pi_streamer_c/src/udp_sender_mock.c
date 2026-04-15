#include "pi_streamer/udp_sender.h"
#include "pi_streamer/_test_hooks.h"

#include <stdlib.h>
#include <string.h>

// Mock UDP sender. Records up to MOCK_MAX_SENDS datagrams in a simple
// in-memory ring so tests can assert on what was sent without needing a
// real socket. Inbound (RX) datagrams come from a process-global pending
// queue: tests pre-load (data, src_ip, src_port, deliver_at_iteration)
// entries, the pipeline owns the live sender, and try_recv consults the
// queue plus the pipeline's iteration counter to release entries at the
// right time.
//
// Why the pending queue is process-global:
//   pi_pipeline_run() creates and destroys its own UDP sender, so tests
//   cannot inject through a per-instance handle they don't own. The
//   global queue is drained at try_recv time, which is the only point
//   at which the pipeline reads from the sender anyway, so there is no
//   race. See include/pi_streamer/_test_hooks.h for the full design
//   rationale.
//
// Why the test-hook bodies are NOT `#ifdef PI_ENABLE_TEST_HOOKS`-guarded:
//   The mock backend is itself test-only — it is never linked into the
//   production binary (production uses udp_sender_iouring.c). Compiling
//   the bodies unconditionally means the static library exports stable
//   symbols regardless of how the test executable was built, which
//   sidesteps the per-target compile-definition propagation problem with
//   CMake static libraries.

#define MOCK_MAX_SENDS    32u
#define MOCK_MAX_DATA     1500u
#define MOCK_MAX_IP       64u
#define MOCK_PENDING_CAP  16u
#define MOCK_PENDING_DATA 64u

typedef struct {
    uint8_t  data[MOCK_MAX_DATA];
    size_t   data_len;
    char     dst_ip[MOCK_MAX_IP];
    uint16_t dst_port;
} mock_record_t;

struct pi_udp_sender {
    pi_udp_sender_config_t cfg;
    mock_record_t          ring[MOCK_MAX_SENDS];
    size_t                 head;          // next write slot
    size_t                 count;         // total sends recorded (saturates)
    pi_udp_sender_stats_t  stats;
};

// ---- process-global pending RX queue (test-only) -------------------------
typedef struct {
    uint8_t  data[MOCK_PENDING_DATA];
    size_t   data_len;
    char     src_ip[MOCK_MAX_IP];
    uint16_t src_port;
    uint64_t deliver_at_iteration;
    bool     in_use;
} pending_entry_t;

static pending_entry_t g_pending[MOCK_PENDING_CAP];

// ---- snapshot of the most recently destroyed sender (test-only) ----------
static struct {
    mock_record_t         ring[MOCK_MAX_SENDS];
    size_t                count;
    pi_udp_sender_stats_t stats;
} g_last_snapshot;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

pi_udp_sender_t *pi_udp_sender_create(const pi_udp_sender_config_t *cfg) {
    if (!cfg) return NULL;
    pi_udp_sender_t *s = calloc(1, sizeof *s);
    if (!s) return NULL;
    s->cfg = *cfg;
    return s;
}

void pi_udp_sender_destroy(pi_udp_sender_t *s) {
    if (!s) return;
    // Snapshot the send ring and stats into the process-global shadow
    // BEFORE freeing so tests can read what the pipeline transmitted
    // after pi_pipeline_run() has returned.
    g_last_snapshot.count = s->count;
    g_last_snapshot.stats = s->stats;
    memcpy(g_last_snapshot.ring, s->ring, sizeof s->ring);
    free(s);
}

int pi_udp_sender_send(pi_udp_sender_t *s,
                       const char      *dst_ip,
                       uint16_t         dst_port,
                       const uint8_t   *data,
                       size_t           data_len) {
    if (!s || !dst_ip || !data) return -1;
    if (data_len == 0u || data_len > MOCK_MAX_DATA) return -1;

    // Record into the ring slot `head`, then advance. We saturate `count`
    // at MOCK_MAX_SENDS so tests can distinguish "we sent exactly N" from
    // "we overflowed".
    mock_record_t *rec = &s->ring[s->head % MOCK_MAX_SENDS];
    memcpy(rec->data, data, data_len);
    rec->data_len = data_len;

    const size_t ip_len = strlen(dst_ip);
    const size_t copy_len = ip_len >= MOCK_MAX_IP ? MOCK_MAX_IP - 1u : ip_len;
    memcpy(rec->dst_ip, dst_ip, copy_len);
    rec->dst_ip[copy_len] = '\0';
    rec->dst_port = dst_port;

    s->head = (s->head + 1u) % MOCK_MAX_SENDS;
    if (s->count < MOCK_MAX_SENDS) s->count++;
    s->stats.datagrams_sent++;
    s->stats.bytes_sent += data_len;
    return 0;
}

int pi_udp_sender_poll(pi_udp_sender_t *s) {
    if (!s) return -1;
    // Mock has no pending completions.
    return 0;
}

int pi_udp_sender_try_recv(pi_udp_sender_t *s,
                           uint8_t  *buf, size_t buf_cap, size_t *received,
                           char     *src_ip, size_t src_ip_cap,
                           uint16_t *src_port) {
    if (!s) return -1;
    if (received) *received = 0u;
    if (src_port) *src_port = 0;

    // Drain one entry from the process-global pending queue whose
    // deliver_at_iteration is reached. We scan in slot order, which means
    // tests cannot rely on injection order across slots — but each test
    // injects entries with monotonically-increasing iteration indices, so
    // the natural FIFO ordering is preserved by slot allocation.
    const uint64_t now_iter = pi_pipeline_test_get_iteration_count();
    for (size_t i = 0; i < MOCK_PENDING_CAP; i++) {
        pending_entry_t *e = &g_pending[i];
        if (!e->in_use) continue;
        if (e->deliver_at_iteration > now_iter) continue;

        size_t n = e->data_len;
        if (buf && buf_cap > 0u) {
            if (n > buf_cap) n = buf_cap;
            memcpy(buf, e->data, n);
            if (received) *received = n;
        }
        if (src_ip && src_ip_cap > 0u) {
            const size_t srclen = strlen(e->src_ip);
            const size_t cpy = srclen >= src_ip_cap ? src_ip_cap - 1u : srclen;
            memcpy(src_ip, e->src_ip, cpy);
            src_ip[cpy] = '\0';
        }
        if (src_port) *src_port = e->src_port;

        e->in_use = false;
        s->stats.datagrams_received++;
        return 1;
    }

    return 0;  // no entries ready
}

void pi_udp_sender_stats(const pi_udp_sender_t *s,
                         pi_udp_sender_stats_t *out) {
    if (!out) return;
    if (!s) {
        memset(out, 0, sizeof *out);
        return;
    }
    *out = s->stats;
}

// ---------------------------------------------------------------------------
// Test-only hooks (declared in include/pi_streamer/_test_hooks.h)
// ---------------------------------------------------------------------------

void pi_udp_mock_pending_inject(const uint8_t *data,
                                size_t         n,
                                const char    *src_ip,
                                uint16_t       src_port,
                                uint64_t       deliver_at_iteration) {
    if (!data || n == 0u || n > MOCK_PENDING_DATA) return;
    for (size_t i = 0; i < MOCK_PENDING_CAP; i++) {
        pending_entry_t *e = &g_pending[i];
        if (e->in_use) continue;
        memcpy(e->data, data, n);
        e->data_len = n;
        if (src_ip) {
            const size_t ip_len = strlen(src_ip);
            const size_t cpy = ip_len >= MOCK_MAX_IP
                                   ? MOCK_MAX_IP - 1u
                                   : ip_len;
            memcpy(e->src_ip, src_ip, cpy);
            e->src_ip[cpy] = '\0';
        } else {
            e->src_ip[0] = '\0';
        }
        e->src_port             = src_port;
        e->deliver_at_iteration = deliver_at_iteration;
        e->in_use               = true;
        return;
    }
    // Queue full — drop silently. Tests that hit this should bump
    // MOCK_PENDING_CAP, not retry.
}

void pi_udp_mock_pending_clear(void) {
    memset(g_pending, 0, sizeof g_pending);
}

uint64_t pi_udp_mock_last_send_count(void) {
    return g_last_snapshot.count;
}

size_t pi_udp_mock_last_send_at(size_t   idx,
                                uint8_t *out,
                                size_t   out_cap) {
    if (!out || out_cap == 0u || idx >= g_last_snapshot.count) return 0u;
    const mock_record_t *rec = &g_last_snapshot.ring[idx];
    size_t n = rec->data_len;
    if (n > out_cap) n = out_cap;
    memcpy(out, rec->data, n);
    return n;
}

int pi_udp_mock_last_send_dst(size_t    idx,
                              char     *ip_out,
                              size_t    ip_cap,
                              uint16_t *port_out) {
    if (idx >= g_last_snapshot.count) return -1;
    const mock_record_t *rec = &g_last_snapshot.ring[idx];
    if (ip_out && ip_cap > 0u) {
        const size_t ip_len = strlen(rec->dst_ip);
        const size_t cpy = ip_len >= ip_cap ? ip_cap - 1u : ip_len;
        memcpy(ip_out, rec->dst_ip, cpy);
        ip_out[cpy] = '\0';
    }
    if (port_out) *port_out = rec->dst_port;
    return 0;
}

void pi_udp_mock_last_reset(void) {
    memset(&g_last_snapshot, 0, sizeof g_last_snapshot);
}
