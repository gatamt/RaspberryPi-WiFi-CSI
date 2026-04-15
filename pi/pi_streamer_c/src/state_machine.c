#include "pi_streamer/state_machine.h"

#include <stdlib.h>

// Guardrail: if someone adds a new state or event, this compile-time check
// will scream loudly until the table below is also updated. Without it, a
// mismatch would silently leave new cells as IDLE (0) because of C's
// designated-initializer fill rule.
_Static_assert(PI_STATE__COUNT == 6,
               "transition table must be updated if PI_STATE__COUNT changes");
_Static_assert(PI_EVENT__COUNT == 9,
               "transition table must be updated if PI_EVENT__COUNT changes");

struct pi_hsm {
    pi_state_t         state;
    pi_hsm_observer_fn observer;
    void              *ctx;
};

// Transition table. Rows = current state, columns = event, cell = next
// state (or PI_STATE__COUNT meaning "ignored in this state").
//
// Every cell is set explicitly to avoid relying on zero-init for "ignored",
// which would collide with PI_STATE_IDLE == 0.
static const pi_state_t kTransitions[PI_STATE__COUNT][PI_EVENT__COUNT] = {
    // ----- PI_STATE_IDLE -----
    [PI_STATE_IDLE] = {
        [PI_EVENT_START]       = PI_STATE_STARTING,
        [PI_EVENT_START_OK]    = PI_STATE__COUNT,
        [PI_EVENT_START_ERR]   = PI_STATE__COUNT,
        [PI_EVENT_FRAME_READY] = PI_STATE__COUNT,
        [PI_EVENT_PAUSE]       = PI_STATE__COUNT,
        [PI_EVENT_RESUME]      = PI_STATE__COUNT,
        [PI_EVENT_STOP]        = PI_STATE_STOPPED,
        [PI_EVENT_ERROR]       = PI_STATE_ERROR,
        [PI_EVENT_RESET]       = PI_STATE__COUNT,
    },
    // ----- PI_STATE_STARTING -----
    [PI_STATE_STARTING] = {
        [PI_EVENT_START]       = PI_STATE__COUNT,
        [PI_EVENT_START_OK]    = PI_STATE_STREAMING,
        [PI_EVENT_START_ERR]   = PI_STATE_ERROR,
        [PI_EVENT_FRAME_READY] = PI_STATE__COUNT,
        [PI_EVENT_PAUSE]       = PI_STATE__COUNT,
        [PI_EVENT_RESUME]      = PI_STATE__COUNT,
        [PI_EVENT_STOP]        = PI_STATE_STOPPED,
        [PI_EVENT_ERROR]       = PI_STATE_ERROR,
        [PI_EVENT_RESET]       = PI_STATE__COUNT,
    },
    // ----- PI_STATE_STREAMING -----
    [PI_STATE_STREAMING] = {
        [PI_EVENT_START]       = PI_STATE__COUNT,
        [PI_EVENT_START_OK]    = PI_STATE__COUNT,
        [PI_EVENT_START_ERR]   = PI_STATE__COUNT,
        // Self-transition is intentional: FRAME_READY is how the encoder
        // tells the state machine "I produced output". Observers fire.
        [PI_EVENT_FRAME_READY] = PI_STATE_STREAMING,
        [PI_EVENT_PAUSE]       = PI_STATE_PAUSED,
        [PI_EVENT_RESUME]      = PI_STATE__COUNT,
        [PI_EVENT_STOP]        = PI_STATE_STOPPED,
        [PI_EVENT_ERROR]       = PI_STATE_ERROR,
        [PI_EVENT_RESET]       = PI_STATE__COUNT,
    },
    // ----- PI_STATE_PAUSED -----
    [PI_STATE_PAUSED] = {
        [PI_EVENT_START]       = PI_STATE__COUNT,
        [PI_EVENT_START_OK]    = PI_STATE__COUNT,
        [PI_EVENT_START_ERR]   = PI_STATE__COUNT,
        [PI_EVENT_FRAME_READY] = PI_STATE__COUNT,
        [PI_EVENT_PAUSE]       = PI_STATE__COUNT,
        [PI_EVENT_RESUME]      = PI_STATE_STREAMING,
        [PI_EVENT_STOP]        = PI_STATE_STOPPED,
        [PI_EVENT_ERROR]       = PI_STATE_ERROR,
        [PI_EVENT_RESET]       = PI_STATE__COUNT,
    },
    // ----- PI_STATE_ERROR -----
    [PI_STATE_ERROR] = {
        [PI_EVENT_START]       = PI_STATE__COUNT,
        [PI_EVENT_START_OK]    = PI_STATE__COUNT,
        [PI_EVENT_START_ERR]   = PI_STATE__COUNT,
        [PI_EVENT_FRAME_READY] = PI_STATE__COUNT,
        [PI_EVENT_PAUSE]       = PI_STATE__COUNT,
        [PI_EVENT_RESUME]      = PI_STATE__COUNT,
        [PI_EVENT_STOP]        = PI_STATE_STOPPED,
        // Latch: another ERROR in ERROR is a self-transition, fires observer.
        [PI_EVENT_ERROR]       = PI_STATE_ERROR,
        [PI_EVENT_RESET]       = PI_STATE_IDLE,
    },
    // ----- PI_STATE_STOPPED (terminal) -----
    [PI_STATE_STOPPED] = {
        [PI_EVENT_START]       = PI_STATE__COUNT,
        [PI_EVENT_START_OK]    = PI_STATE__COUNT,
        [PI_EVENT_START_ERR]   = PI_STATE__COUNT,
        [PI_EVENT_FRAME_READY] = PI_STATE__COUNT,
        [PI_EVENT_PAUSE]       = PI_STATE__COUNT,
        [PI_EVENT_RESUME]      = PI_STATE__COUNT,
        [PI_EVENT_STOP]        = PI_STATE__COUNT,
        [PI_EVENT_ERROR]       = PI_STATE__COUNT,
        [PI_EVENT_RESET]       = PI_STATE__COUNT,
    },
};

pi_hsm_t *pi_hsm_create(pi_hsm_observer_fn observer, void *ctx) {
    pi_hsm_t *hsm = calloc(1, sizeof *hsm);
    if (!hsm) return NULL;
    hsm->state    = PI_STATE_IDLE;
    hsm->observer = observer;
    hsm->ctx      = ctx;
    return hsm;
}

void pi_hsm_destroy(pi_hsm_t *hsm) {
    free(hsm);
}

int pi_hsm_dispatch(pi_hsm_t *hsm, pi_event_t event) {
    if (!hsm) return -1;
    if ((unsigned)event >= (unsigned)PI_EVENT__COUNT) return -1;

    const pi_state_t from = hsm->state;
    const pi_state_t to   = kTransitions[from][event];
    if (to == PI_STATE__COUNT) {
        return 0;  // ignored — valid under run-to-completion
    }

    hsm->state = to;
    if (hsm->observer) {
        hsm->observer(from, to, event, hsm->ctx);
    }
    return 1;
}

pi_state_t pi_hsm_current_state(const pi_hsm_t *hsm) {
    return hsm ? hsm->state : PI_STATE__COUNT;
}

bool pi_hsm_is_terminal(const pi_hsm_t *hsm) {
    return hsm && hsm->state == PI_STATE_STOPPED;
}

const char *pi_state_name(pi_state_t state) {
    switch (state) {
        case PI_STATE_IDLE:      return "IDLE";
        case PI_STATE_STARTING:  return "STARTING";
        case PI_STATE_STREAMING: return "STREAMING";
        case PI_STATE_PAUSED:    return "PAUSED";
        case PI_STATE_ERROR:     return "ERROR";
        case PI_STATE_STOPPED:   return "STOPPED";
        case PI_STATE__COUNT:    return "<invalid>";
    }
    return "<unknown>";
}

const char *pi_event_name(pi_event_t event) {
    switch (event) {
        case PI_EVENT_START:       return "START";
        case PI_EVENT_START_OK:    return "START_OK";
        case PI_EVENT_START_ERR:   return "START_ERR";
        case PI_EVENT_FRAME_READY: return "FRAME_READY";
        case PI_EVENT_PAUSE:       return "PAUSE";
        case PI_EVENT_RESUME:      return "RESUME";
        case PI_EVENT_STOP:        return "STOP";
        case PI_EVENT_ERROR:       return "ERROR";
        case PI_EVENT_RESET:       return "RESET";
        case PI_EVENT__COUNT:      return "<invalid>";
    }
    return "<unknown>";
}
