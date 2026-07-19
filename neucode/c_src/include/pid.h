#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include "neucode_types.h"

typedef struct {
    // Gains
    double kp, ki, kd;

    // Limits & Tuning
    double u_min, u_max;
    double i_min, i_max;
    double kaw;
    double d_alpha;

    // State
    double integral;
    double prev_error;
    double prev_meas;
    double d_filt;
} neucode_pid_t;

/* Initializes the PID controller with given gains and output limits. */
void neucode_pid_init(neucode_pid_t* p, const neucode_pid_gains_t* gains, const neucode_pid_limits_t* limits);

/* Resets the PID controller state. */
void neucode_pid_reset(neucode_pid_t* p);

/* Executes a single PID control step. */
double neucode_pid_step(neucode_pid_t* p, double sp, double meas, double dt);

#ifdef __cplusplus
}
#endif