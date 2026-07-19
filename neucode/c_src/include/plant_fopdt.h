#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include "neucode_types.h"

/**
 * FOPDT plant:
 *   y_dot = (-y + K * u_delay)/tau,  y = state
 * Discretization (exact for the FO part):
 *   a = exp(-dt/tau),  b = K * (1 - a)
 *   y[k+1] = a*y[k] + b*u_delay[k]
 * Dead time theta handled by a circular buffer on u.
 */
typedef struct {
    // Parameters
    float K;            // gain
    float tau;          // time constant [s] > 0
    float theta;        // dead time [s] >= 0
    float dt;           // fixed sim step [s] > 0

    // Coulomb friction
    float friction;

    // Derived coefficients
    float a;            // exp(-dt/tau)
    float b;            // K*(1 - a)

    // Delay line for u (stores past inputs)
    float *ubuf;        // ring buffer of size buf_size
    size_t buf_size;      // >= ceil(theta/dt) + 2
    size_t head;        // next write index (u[k] goes here)

    // Dead time split into integer + fractional part
    size_t d_int;       // integer delay steps = floor(theta/dt)
    float d_alpha;      // fractional part in [0,1): theta/dt - d_int

    // State
    float y;            // output/state
} neucode_plant_fopdt_t;

/** Allocate + init delay buffer (pass a workspace you allocate) */
bool neucode_plant_fopdt_init(neucode_plant_fopdt_t* p, const neucode_fopdt_params_t* params, float dt, float* ubuf_workspace, size_t ubuf_len);

/** Reset plant state (y) to initial condition y0 */
void neucode_plant_fopdt_reset(neucode_plant_fopdt_t *p, float y0);

/** One step advance; returns y_true; y_meas_out adds disturbances/noise. */
void neucode_plant_fopdt_step(neucode_plant_fopdt_t* p, float u);

/** Get current output (y) */
float neucode_plant_fopdt_get_output(const neucode_plant_fopdt_t* p);

#ifdef __cplusplus
}
#endif