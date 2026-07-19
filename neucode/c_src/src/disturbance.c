#include "disturbance.h"
#include <string.h>
#include <math.h>

// Source: https://www.cec.uchile.cl/cinetica/pcordero/MC_libros/NumericalRecipesinC.pdf
// Chapter 7. Random Numbers
// "An Even Quicker Generator"
// Returns float in range [0..1)
static float lcg_rand(uint32_t* state) {
    *state = (1664525U * (*state) + 1013904223U);
    // "Correlation in k-space is not the only weakness of linear congruential generators."
    // "Such generators often have their low-order (least significant) bits much less random than their high-order bits"
    // So we use bits 8..31 (24 bits) for better randomness
    return (float)((*state >> 8) & 0x00FFFFFF) / 16777216.0f;
}

// Simple linear transformation
// Source: https://faculty.ksu.edu.sa/sites/default/files/introduction-to-probability-model-s.ross-math-cs.blog_.ir_.pdf
// Mapping [0..1) to [a..b)
static float uniform_rand(float a, float b, uint32_t* state) {
    return a + (b - a) * lcg_rand(state);
}

// Generate Gaussian random numbers using Marsaglia polar method
// Compared to Box-Muller this is slightly more efficient (no trig functions)
// Source: https://faculty.ksu.edu.sa/sites/default/files/introduction-to-probability-model-s.ross-math-cs.blog_.ir_.pdf
// 11.3 Special Techniques for Simulating Continuous Random Variables
// Page 683
static float gaussian_polar_rand(uint32_t* state, bool* has_spare, float* spare) {
    // Box-Muller transform
    if (*has_spare) {
        *has_spare = false;
        return *spare;
    }

    float u1, u2, v1, v2, s;
    do {
        u1 = uniform_rand(-1.0f, 1.0f, state);
        u2 = uniform_rand(-1.0f, 1.0f, state);
        v1 = 2.0f * u1 - 1.0f;
        v2 = 2.0f * u2 - 1.0f;
        s = v1 * v1 + v2 * v2;
    } while (s >= 1.0f || s == 0.0f);

    float m = sqrt(-2.0f * log(s) / s);

    // the method generates two independent standard normal variables
    // we return one and store the other for next time
    *spare = v2 * m;
    *has_spare = true;
    return v1 * m;
}

void neucode_disturbance_init(neucode_disturbance_t* p, const neucode_disturbance_config_t* config) {
    memset(p, 0, sizeof(neucode_disturbance_t));
    if (!config) {
        p->config.noise_type = NEUCODE_NOISE_NONE;
        p->rng_state = 123456789u;
        return;
    }
    p->config = *config; // Copy the configuration
    p->rng_state = config->seed ? config->seed : 123456789u;
    p->_bm_has_spare = false; // Reset the spare value cache.
}

void neucode_disturbance_apply(neucode_disturbance_t* p, float t, float* u_inout, float* y_inout) {
    // Apply input step disturbance (modifies controller output `u`)
    if (p->config.enable_input_step && t >= p->config.input_step_at_s) {
        *u_inout += p->config.input_step_value;
    }

    // Apply output step disturbance (modifies plant output `y`)
    if (p->config.enable_output_step && t >= p->config.output_step_at_s) {
        *y_inout += p->config.output_step_value;
    }

    // Apply measurement noise (modifies plant output `y`)
    switch (p->config.noise_type) {
        case NEUCODE_NOISE_UNIFORM:
            if (p->config.noise_amp > 0.0f) {
                *y_inout += uniform_rand(-p->config.noise_amp, p->config.noise_amp, &p->rng_state);
            }
            break;
        case NEUCODE_NOISE_GAUSSIAN:
            if (p->config.noise_std > 0.0f) {
                float z = gaussian_polar_rand(&p->rng_state, &p->_bm_has_spare, &p->_bm_spare); // N(0,1)
                *y_inout += p->config.noise_std * z;
            }
            break;
        case NEUCODE_NOISE_NONE:
        default:
            // No noise to add
            break;
    }

    // Cogging torque: first-harmonic Fourier approximation of the position-dependent
    // torque ripple in PMSM/BLDC motors, injected additively on u before the plant.
    // d(theta) = amp * sin(freq_mult * theta),  theta in radians.
    // freq_mult (cogging_freq_mult) = cogging periods/rev: use n_pp (approx) or LCM(N_slots, 2*n_pp) (exact).
    // Bianchi & Bolognani, Design techniques for reducing the cogging torque in surface-mounted PM motors, DOI: 10.1109/TIA.2002.802989
    if (p->config.enable_cogging_sine && p->config.cogging_sine_amp > 0.0f) {
        *u_inout += p->config.cogging_sine_amp *
            sinf((float)p->config.cogging_freq_mult * (*y_inout) * (3.14159265f / 180.0f));
    }
}