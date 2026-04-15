#include "pi_streamer/pipeline.h"

#include <stdio.h>

// Thin wrapper around pi_pipeline_run. Keeps main.c trivial so tests can
// exercise the pipeline via pi_pipeline_run directly without spawning a
// subprocess.

int main(int argc, char **argv) {
    pi_pipeline_args_t args;
    pi_pipeline_args_defaults(&args);
    if (pi_pipeline_parse_args(argc, argv, &args) != 0) {
        fprintf(stderr, "pi_streamer: failed to parse arguments\n");
        return 2;
    }
    return pi_pipeline_run(&args);
}
