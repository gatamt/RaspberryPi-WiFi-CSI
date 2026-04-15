#include "pi_streamer/overlay.h"

#include <stdint.h>
#include <stddef.h>

// Clamp an int32 coordinate to [0, max-1]. Returns the clamped value.
// max is expected to be at least 1; for degenerate 0-width/height frames
// the caller never enters this helper (pi_overlay_draw_detections returns
// early).
static inline int32_t clamp_i32(int32_t v, int32_t lo, int32_t hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

// Draw one horizontal span of `thickness` rows: [x1..x2] at rows
// [y..y+thickness). The caller has already clamped x1/x2 to [0, width-1]
// and y to [0, height-thickness]. `thickness` is bounded by the caller.
static void draw_horizontal_bar(uint8_t *y_plane,
                                size_t   y_stride,
                                int32_t  x1,
                                int32_t  x2,
                                int32_t  y,
                                int32_t  thickness,
                                int32_t  height,
                                uint8_t  value) {
    if (x1 > x2 || thickness <= 0) return;
    const int32_t y_end = (y + thickness < height) ? (y + thickness) : height;
    for (int32_t row = y; row < y_end; row++) {
        uint8_t *line = y_plane + (size_t)row * y_stride;
        for (int32_t col = x1; col <= x2; col++) {
            line[col] = value;
        }
    }
}

// Draw one vertical span of `thickness` columns: rows [y1..y2] at
// columns [x..x+thickness). Caller has clamped y1/y2 and x.
static void draw_vertical_bar(uint8_t *y_plane,
                              size_t   y_stride,
                              int32_t  y1,
                              int32_t  y2,
                              int32_t  x,
                              int32_t  thickness,
                              int32_t  width,
                              uint8_t  value) {
    if (y1 > y2 || thickness <= 0) return;
    const int32_t x_end = (x + thickness < width) ? (x + thickness) : width;
    for (int32_t row = y1; row <= y2; row++) {
        uint8_t *line = y_plane + (size_t)row * y_stride;
        for (int32_t col = x; col < x_end; col++) {
            line[col] = value;
        }
    }
}

void pi_overlay_draw_detections(uint8_t              *y_plane,
                                int32_t               width,
                                int32_t               height,
                                size_t                y_stride,
                                const pi_detection_t *dets,
                                size_t                n_det,
                                int                   thickness,
                                uint8_t               value) {
    if (!y_plane || width <= 0 || height <= 0 || y_stride < (size_t)width) {
        return;
    }
    if (n_det == 0 || !dets) return;

    // Clamp thickness. 1..16 is generous: 16 pixels at 1280x720 is ~2% of
    // the frame width, already visually obtrusive.
    if (thickness < 1)  thickness = 1;
    if (thickness > 16) thickness = 16;

    for (size_t i = 0; i < n_det; i++) {
        const pi_detection_t *d = &dets[i];

        // Reject fully off-frame or zero-area boxes before clamping — a
        // degenerate box should draw NOTHING, not a single-pixel outline.
        if (d->x2 <= d->x1 || d->y2 <= d->y1) continue;
        if (d->x2 < 0 || d->y2 < 0) continue;
        if (d->x1 >= width || d->y1 >= height) continue;

        // Clamp to the frame. Note that x2/y2 are INCLUSIVE because the
        // bbox convention in pi_detection_t is (x1, y1) top-left and
        // (x2, y2) bottom-right of the filled region.
        const int32_t x1 = clamp_i32(d->x1, 0, width  - 1);
        const int32_t y1 = clamp_i32(d->y1, 0, height - 1);
        const int32_t x2 = clamp_i32(d->x2, 0, width  - 1);
        const int32_t y2 = clamp_i32(d->y2, 0, height - 1);

        // Top edge: rows [y1 .. y1+thickness)
        draw_horizontal_bar(y_plane, y_stride, x1, x2, y1,
                            thickness, height, value);

        // Bottom edge: rows ending at y2, so starting at y2-thickness+1.
        // Clamp bottom-edge start to 0 in case the box is so shallow that
        // the top and bottom bars would overlap — that is fine, we just
        // draw overlapping pixels.
        int32_t bottom_start = y2 - thickness + 1;
        if (bottom_start < 0) bottom_start = 0;
        draw_horizontal_bar(y_plane, y_stride, x1, x2, bottom_start,
                            thickness, height, value);

        // Left edge: columns [x1 .. x1+thickness)
        draw_vertical_bar(y_plane, y_stride, y1, y2, x1,
                          thickness, width, value);

        // Right edge: columns ending at x2, start at x2-thickness+1
        int32_t right_start = x2 - thickness + 1;
        if (right_start < 0) right_start = 0;
        draw_vertical_bar(y_plane, y_stride, y1, y2, right_start,
                          thickness, width, value);
    }
}
