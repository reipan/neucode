#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include "neucode_types.h"

typedef struct {
    // A copy of the public configuration
    neucode_disturbance_config_t config;

    // Private internal state for the random number generator
    uint32_t rng_state;
    bool _bm_has_spare;
    float _bm_spare;

} neucode_disturbance_t;

/* Initialize the disturbance generator */
void neucode_disturbance_init(neucode_disturbance_t* p, const neucode_disturbance_config_t* config);

/* Apply disturbance to input and output signals */
void neucode_disturbance_apply(neucode_disturbance_t* p, float t, float* u_in, float* y_out);

#ifdef __cplusplus
}
#endif