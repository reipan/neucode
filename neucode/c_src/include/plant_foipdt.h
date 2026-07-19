#pragma once
/*
 * First Order Integrating Plus Dead Time (FOIPDT) plant model.
 *
 * Transfer function: Y(s)/U(s) = Kv * e^{-theta*s} / (s * (tau*s + 1))
 *
 * State equations (discrete, exact ZOH for velocity, Euler for position):
 *   v[k+1] = a*v[k] + b*u_delayed[k]   (velocity dynamics)
 *   y[k+1] = y[k] + v[k]*dt            (position = integral of velocity)
 *
 * where: a = exp(-dt/tau),  b = Kv*(1 - a)
 *
 * Dead time is handled identically to plant_fopdt.c via a circular delay buffer
 * with linear interpolation between adjacent samples.
 */

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>
#include "neucode_types.h"

typedef struct {
    /* Parameters */
    float Kv;
    float tau;
    float theta;
    float dt;

    /* Velocity dynamics coefficients */
    float a;   /* exp(-dt/tau) */
    float b;   /* Kv*(1-a) */

    /* Coulomb friction */
    float friction;

    /* Plant state */
    float v;   /* current velocity [units/s] */
    float y;   /* current position [units] */

    /* Dead-time delay line (same pattern as plant_fopdt) */
    float*  ubuf;
    size_t  buf_size;
    size_t  head;
    size_t  d_int;   /* integer delay in samples */
    float   d_alpha; /* fractional delay in [0,1) */
} neucode_plant_foipdt_t;

bool  neucode_plant_foipdt_init(neucode_plant_foipdt_t* p, const neucode_foipdt_params_t* params,
                                float dt, float* ubuf_workspace, size_t ubuf_len);
void  neucode_plant_foipdt_reset(neucode_plant_foipdt_t* p, float y0);
void  neucode_plant_foipdt_step(neucode_plant_foipdt_t* p, float u);
float neucode_plant_foipdt_get_output(const neucode_plant_foipdt_t* p);

#ifdef __cplusplus
}
#endif
