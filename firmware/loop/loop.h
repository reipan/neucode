#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Control loop rate - single source of truth.
 *
 * main.cpp derives the HardwareTimer period from NC_CONTROL_LOOP_HZ.
 * Python exporters use NC_CONTROL_LOOP_DT_S to validate that the training
 * dt matches the hardware tick rate.
 */
#define NC_CONTROL_LOOP_HZ   1000
#define NC_CONTROL_LOOP_DT_S (1.0 / NC_CONTROL_LOOP_HZ)

/**
 * Size of the post-experiment telemetry buffer.
 * 512 frames x 16 B/frame = 8 KB; covers 5.12 s at 100 Hz dump rate.
 */
#define NC_LOOP_BUF_SIZE 512u

/** One buffered control-loop frame (t, sp, y, u). */
typedef struct {
    float t;
    float sp;
    float y;
    float u;
} nc_loop_frame_t;

void nc_loop_init(void);
void nc_loop_step(double now, double dt);

/**
 * Returns a pointer to the internal frame buffer and sets *out_count to the
 * number of valid frames recorded during the last experiment.
 *
 * Pointer is valid until the next exp start (which resets the count).
 */
const nc_loop_frame_t *nc_loop_get_buffer(uint16_t *out_count);

#ifdef __cplusplus
}
#endif