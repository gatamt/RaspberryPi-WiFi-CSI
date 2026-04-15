#ifndef PI_STREAMER_STATE_MACHINE_H
#define PI_STREAMER_STATE_MACHINE_H

// Pipeline state machine — run-to-completion HSM in the Samek style, but
// without the full QEP event-dispatch hierarchy. A flat transition table
// is enough for our needs right now and is trivially auditable.
//
// States:
//   IDLE       initial — waiting for PI_EVENT_START
//   STARTING   camera/encoder/Hailo bootstrapping
//   STREAMING  steady state — frames flowing
//   PAUSED     client requested pause — workers idle but hot
//   ERROR      recoverable failure — RESET returns to IDLE
//   STOPPED    terminal — all events ignored
//
// Events: see pi_event_t below.
//
// Run-to-completion rule: an event that has no transition defined for the
// current state is silently IGNORED (not an error). This matches Samek's
// semantics and keeps client code from having to know every valid
// (state, event) pair.
//
// Primary reference: Samek, "Practical UML Statecharts in C/C++ 2e",
// Ch 2 (run-to-completion) and Ch 3 (HSM basics) — read directly from the
// source PDF, NOT from the vault atomic-note stub (see
// docs/Phase1-Atomic-Note-Verification.md).

#include <stdbool.h>

typedef enum {
    PI_STATE_IDLE = 0,
    PI_STATE_STARTING,
    PI_STATE_STREAMING,
    PI_STATE_PAUSED,
    PI_STATE_ERROR,
    PI_STATE_STOPPED,
    PI_STATE__COUNT,   // sentinel for "ignored" transitions + loop bound
} pi_state_t;

typedef enum {
    PI_EVENT_START = 0,
    PI_EVENT_START_OK,
    PI_EVENT_START_ERR,
    PI_EVENT_FRAME_READY,
    PI_EVENT_PAUSE,
    PI_EVENT_RESUME,
    PI_EVENT_STOP,
    PI_EVENT_ERROR,
    PI_EVENT_RESET,
    PI_EVENT__COUNT,   // sentinel for loop bound
} pi_event_t;

typedef struct pi_hsm pi_hsm_t;

// Transition observer. Fires exactly once per successful transition.
// NOT called for ignored or invalid dispatches.
typedef void (*pi_hsm_observer_fn)(pi_state_t from,
                                   pi_state_t to,
                                   pi_event_t event,
                                   void      *ctx);

// Construct a fresh HSM in state IDLE. `observer` may be NULL.
pi_hsm_t *pi_hsm_create(pi_hsm_observer_fn observer, void *ctx);

// Destroy. NULL-safe.
void pi_hsm_destroy(pi_hsm_t *hsm);

// Dispatch a single event.
// Return values:
//    1  transition took place; observer (if any) was called
//    0  event valid but ignored in this state (no-op)
//   -1  hsm is NULL or event is out of range
int pi_hsm_dispatch(pi_hsm_t *hsm, pi_event_t event);

// Observability.
pi_state_t pi_hsm_current_state(const pi_hsm_t *hsm);
bool       pi_hsm_is_terminal(const pi_hsm_t *hsm);

// String helpers for logging. Never return NULL.
const char *pi_state_name(pi_state_t state);
const char *pi_event_name(pi_event_t event);

#endif // PI_STREAMER_STATE_MACHINE_H
