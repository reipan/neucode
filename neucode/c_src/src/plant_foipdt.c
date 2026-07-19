#include "plant_foipdt.h"
#include <math.h>
#include <string.h>

/* Reuse the same wrap helper as plant_fopdt.c */
static inline size_t wrap(size_t x, size_t m) {
    return (x >= m) ? (x - m) : x;
}

/* Read delayed u from the circular delay buffer with linear interpolation */
static inline float read_delayed_u(const neucode_plant_foipdt_t *p) {
    size_t idx_now = (p->head == 0) ? (p->buf_size - 1) : (p->head - 1);
    size_t idx_d0  = (idx_now + p->buf_size - p->d_int) % p->buf_size;
    size_t idx_d1  = (idx_d0 == 0) ? (p->buf_size - 1) : (idx_d0 - 1);
    float u0 = p->ubuf[idx_d0];
    float u1 = p->ubuf[idx_d1];
    return (1.0f - p->d_alpha) * u0 + p->d_alpha * u1;
}

bool neucode_plant_foipdt_init(neucode_plant_foipdt_t* p, const neucode_foipdt_params_t* params,
                               float dt, float* ubuf_workspace, size_t ubuf_len) {
    if (!p || !params || !ubuf_workspace || ubuf_len < 2) return false;
    if (params->tau <= 0.0f || dt <= 0.0f || params->theta < 0.0f) return false;

    memset(p, 0, sizeof(neucode_plant_foipdt_t));

    p->Kv    = params->Kv;
    p->tau   = params->tau;
    p->theta = params->theta;
    p->friction = params->friction;
    p->dt    = dt;

    /* Exact ZOH discretisation of first-order velocity dynamics */
    p->a = expf(-dt / p->tau);
    p->b = p->Kv * (1.0f - p->a);

    /* Dead-time: split into integer + fractional parts for interpolation */
    float d_steps = p->theta / dt;
    p->d_int   = (size_t)floorf(d_steps);
    p->d_alpha = d_steps - (float)(p->d_int);

    if (p->d_int + 2 > ubuf_len) return false;

    p->buf_size = ubuf_len;
    p->ubuf     = ubuf_workspace;
    neucode_plant_foipdt_reset(p, 0.0f);
    return true;
}

void neucode_plant_foipdt_reset(neucode_plant_foipdt_t* p, float y0) {
    if (!p) return;
    p->v    = 0.0f;
    p->y    = y0;
    p->head = 0;
    memset(p->ubuf, 0, p->buf_size * sizeof(float));
}

void neucode_plant_foipdt_step(neucode_plant_foipdt_t* p, float u) {
    /* Push u into delay buffer */
    p->ubuf[p->head] = u;
    p->head = wrap(p->head + 1, p->buf_size);

    float u_delay = (p->theta > 0.0f) ? read_delayed_u(p) : u;

    /* Velocity dynamics: v[k+1] = a*v[k] + b*u_delay[k] */
    float v_next = p->a * p->v + p->b * u_delay;

    /* Coulomb friction: constant force opposing motion */
    if (p->friction > 0.0f) {
        if (p->v > 0.0f) {
            v_next -= p->friction;
            if (v_next < 0.0f) v_next = 0.0f;
        } else if (p->v < 0.0f) {
            v_next += p->friction;
            if (v_next > 0.0f) v_next = 0.0f;
        }
    }

    /* Position integration: y[k+1] = y[k] + v[k]*dt  (Euler, adequate at dt=1ms) */
    p->y += p->v * p->dt;

    p->v = v_next;
}

float neucode_plant_foipdt_get_output(const neucode_plant_foipdt_t* p) {
    if (!p) return 0.0f;
    return p->y;
}
