#include "plant_fopdt.h"
#include <math.h>
#include <string.h>

// Wrap index x in [0..m)
// We're using an array as a circular buffer, that's why we need this to wrap around when we reach the end
static inline size_t wrap(size_t x, size_t m) {
    return (x >= m) ? (x - m) : x;
}

// Read delayed u from the delay line with linear interpolation
static inline float read_delayed_u(const neucode_plant_fopdt_t *p) {
    // We need to figure out the indices of the two samples we need to interpolate between
    size_t idx_now = (p->head == 0) ? (p->buf_size - 1) : (p->head - 1);
    size_t idx_d0 = (idx_now + p->buf_size - p->d_int) % p->buf_size;
    size_t idx_d1 = (idx_d0 == 0) ? (p->buf_size - 1) : (idx_d0 - 1);

    // Those are the two samples we need to interpolate between
    float u0 = p->ubuf[idx_d0];
    float u1 = p->ubuf[idx_d1];

     // u_delay = (1 - alpha)*u[k - d_int] + alpha*u[k - d_int - 1]
    return (1.0f - p->d_alpha) * u0 + p->d_alpha * u1;
}

// Initialize FOPDT plant structure
bool neucode_plant_fopdt_init(neucode_plant_fopdt_t* p, const neucode_fopdt_params_t* params, float dt, float* ubuf_workspace, size_t ubuf_len) {
    // Bail out on invalid params
    if (!p || !params || !ubuf_workspace || ubuf_len < 2) return false;
    if (params->tau <= 0.0f || dt <= 0.0f || params->theta < 0.0f) return false;

    // Set plant to default state
    memset(p, 0, sizeof(neucode_plant_fopdt_t));

    // Set parameters
    p->K = params->K;
    p->tau = params->tau;
    p->theta = params->theta;
    p->friction = params->friction;
    p->dt = dt;

    // Define coefficients for exact discretization of FO part
    // a = exp(-dt/tau),  b = K*(1 - a)
    p->a = expf(-dt / p->tau);
    p->b = p->K * (1.0f - p->a);

    // Simulate continuous-time delays in a fixed-step discrete simulation.
    // Since the dead time is continuous, we need to split it into an integer and fractional part
    // This allows us to do a linear interpolation between two samples in the delay line
    float d_steps = p->theta / dt;

    // Integer part of dead time in steps
    // math: floor(theta/dt)
    p->d_int = (size_t)floorf(d_steps);
    // Fractional part of the dead time in [0,1)
    // math: theta/dt - d_int
    p->d_alpha = d_steps - (float)(p->d_int);

    // Check if the buffer is too small for the requested dead time
    if (p->d_int + 2 > ubuf_len) {
        return false;
    }

    p->buf_size = ubuf_len;
    p->ubuf = ubuf_workspace;
    neucode_plant_fopdt_reset(p, 0.0f);

    return true;
}

// Reset plant state (y) to initial condition y0
void neucode_plant_fopdt_reset(neucode_plant_fopdt_t *p, float y0) {
    if (p) {
        p->y = y0;
        memset(p->ubuf, 0, p->buf_size * sizeof(float));
        p->head = 0;
    }
}

// One step advance of the FOPDT plant
void neucode_plant_fopdt_step(neucode_plant_fopdt_t* p, float u) {
    // This logic is taken directly from your old function:
    p->ubuf[p->head] = u;
    p->head = wrap(p->head + 1, p->buf_size);

    float u_delay = (p->theta > 0.0f) ? read_delayed_u(p) : u;

    // y[k+1] = a*y[k] + b*u_delay[k]
    float y_next = p->a * p->y + p->b * u_delay;

    // Coulomb friction: constant force opposing output direction
    if (p->friction > 0.0f) {
        if (p->y > 0.0f) {
            y_next -= p->friction;
            if (y_next < 0.0f) y_next = 0.0f;
        } else if (p->y < 0.0f) {
            y_next += p->friction;
            if (y_next > 0.0f) y_next = 0.0f;
        }
    }

    p->y = y_next;
}

// Get output of previous step
float neucode_plant_fopdt_get_output(const neucode_plant_fopdt_t* p) {
    if (!p) return 0.0f;
    return p->y;
}