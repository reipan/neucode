#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include "neucode_types.h"
#include <stdbool.h>
#include <stddef.h>

typedef struct {
    float t;
    float r;
    float y;
} neucode_ring_buffer_elem_t;

typedef struct {
    size_t cap;
    size_t head;
    size_t size;
    neucode_ring_buffer_elem_t *buf;
    float tail_s;
    size_t min_tail_samples;
} neucode_ring_buffer_t;

typedef struct {
    float t; // run time

    float iae; // integral of absolute error
    float ise; // integral of squared error
    float itae; // integral of time-weighted absolute error

    float isu; // integral of squared control effort

    float step_time; // time of step change
    float r_initial; // initial value
    float r_final; // final value

    float sse_val; // last computed steady-state error value
    
    bool step_mode;
    bool r_final_locked; // true if r_final has been set
    
    float y10, y90; // 10% and 90% of final value
    bool crossed_10, crossed_90; // flags for rise time detection
    float t_rise_start, t_rise_end; // times for rise time calculation
    
    float y_max, t_y_max;
    float y_min, t_y_min;
    
    neucode_ring_buffer_t sse_rb; // ring buffer for steady-state error
    neucode_metrics_config_t cfg; // copy of config for reference
} neucode_metrics_t;

/* Initializes the metrics state. Returns false if memory allocation fails. */
bool neucode_metrics_init(neucode_metrics_t* m, const neucode_metrics_config_t* cfg);

/* Frees any memory allocated by the metrics module. */
void neucode_metrics_destroy(neucode_metrics_t* m);

/* Resets all accumulated metrics to their initial state. */
void neucode_metrics_reset(neucode_metrics_t* m);

/* Updates all metrics for a single time step. */
void neucode_metrics_update(neucode_metrics_t* m, float dt, float r, float y, float u);

/* Calculates metrics for a batch of samples. */
void neucode_metrics_calc_batch(neucode_metrics_t* m, const float* t_arr, const float* r_arr, const float* y_arr, const float* u_arr, size_t n);

/* Calculates the final results and populates the public results struct. */
void neucode_metrics_finalize(const neucode_metrics_t* m, neucode_metrics_results_t* out);

#ifdef __cplusplus
}
#endif