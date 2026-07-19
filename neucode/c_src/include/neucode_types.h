#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

/* Status codes */
typedef enum {
    NEUCODE_OK = 0,
    NEUCODE_ERROR_INVALID_ARG = 1,
    NEUCODE_ERROR_ALLOC_FAILED = 2,
} neucode_status_t;

/* Setpoint types */
typedef enum {
    NEUCODE_SP_STEP = 0,
    NEUCODE_SP_RAMP = 1,
    NEUCODE_SP_SIN  = 2
} neucode_setpoint_type_t;

/* Setpoint definition */
typedef struct {
    neucode_setpoint_type_t type;
    float step_time;    // s
    float v;            // step value
    float a, b;         // ramp start/end
    float time;         // ramp duration
    float amp;          // sine amplitude
    float freq;         // sine frequency (Hz)
} neucode_setpoint_def_t;

/* PID gains */
typedef struct {
    float kp;   // proportional gain
    float ki;   // integral gain
    // derivative gain, pre-scaled by the sample time (dt).
    float kd;
} neucode_pid_gains_t;

/* PID limits + filters */
typedef struct {
    float u_min;   // min controller output
    float u_max;   // max controller output
    float i_min;   // min integrator state
    float i_max;   // max integrator state
    float kaw;     // anti-windup back-calculation gain
    float d_alpha; // derivative LPF smoothing factor [0..1]
} neucode_pid_limits_t;

/* FOPDT plant parameters */
typedef struct {
    float K;     // gain
    float tau;   // time constant [s] > 0
    float theta; // dead time [s] >= 0
    float friction; // coulomb friction [output units], 0 = disabled
} neucode_fopdt_params_t;

/* FOIPDT plant parameters (First Order Integrating Plus Dead Time)
 * Models a velocity plant: G(s) = Kv * e^(-theta*s) / (s * (tau*s + 1))
 * Same field layout as neucode_fopdt_params_t so supervised-tuner observation
 * vectors are compatible between plant types. */
typedef struct {
    float Kv;    // velocity gain [units/s per V]
    float tau;   // velocity lag time constant [s] > 0
    float theta; // dead time [s] >= 0
    float friction; // coulomb friction [input units], 0 = disabled
} neucode_foipdt_params_t;

/* Plant type discriminator */
typedef enum {
    NEUCODE_PLANT_FOPDT  = 0,
    NEUCODE_PLANT_FOIPDT = 1,
} neucode_plant_type_t;

/* Noise types */
typedef enum {
    NEUCODE_NOISE_NONE    = 0,
    NEUCODE_NOISE_UNIFORM = 1,
    NEUCODE_NOISE_GAUSSIAN= 2
} neucode_noise_type_t;

/* Disturbance definition */
typedef struct {
    /* input (plant control) step disturbance */
    int enable_input_step;
    float input_step_at_s;
    float input_step_value;

    /* output (plant measurement) step disturbance */
    int enable_output_step;
    float output_step_at_s;
    float output_step_value;

    /* output measurement noise */
    neucode_noise_type_t noise_type;
    float noise_amp;    // uniform: amp
    float noise_std;    // gaussian: stddev
    uint32_t seed;      // RNG seed

    /* reserved for internal RNG cache (Box-Muller) */
    bool _bm_has_spare;
    float _bm_spare;
} neucode_disturbance_def_t;

/* Metrics / steady-state config (public subset) */
typedef struct {
    bool step_mode;             // true if we measure a step response
    float step_time;            // time of step change (s)
    float r_final;              // final setpoint value after step
    size_t min_tail_samples;    // minimum samples in tail window
    float tail_window_s;        // time window for steady-state error calc (s)
} neucode_metrics_cfg_t;

/* Result */
typedef struct {
    float iae;                          // Integral of Absolute Error
    float ise;                          // Integral of Squared Error
    float itae;                         // Integral of Time-weighted Abs Error
    float peak_value;                   // y peak (if available)
    float peak_time;                    // t at peak (if available)
    float overshoot_percent;            // % overshoot relative to final value
    float rise_time;                    // 10% -> 90% rise time
    float steady_state_error;           // absolute
    float steady_state_error_percent;   // percent (if available)
    float y_final;                      // final output sample
    size_t samples_written;             // number of simulated samples
} neucode_simulation_result_t;

/* Disturbance configuration */
typedef struct {
    // Input step disturbance
    bool enable_input_step;
    float input_step_at_s;
    float input_step_value;

    // Output step disturbance
    bool enable_output_step;
    float output_step_at_s;
    float output_step_value;

    // Simulates measurement noise on the plant output
    // Source: https://www.cds.caltech.edu/~murray/books/AM08/pdf/fbs-public_24Jul2020.pdf
    // Section 11.5 - Example 8.6 Modeling a noisy sinusoidal disturbance
    neucode_noise_type_t noise_type;
    float noise_amp; // for uniform (-amp .. +amp)
    float noise_std; // for Gaussian (mean=0, std)
    uint32_t seed;

    // Sinusoidal cogging disturbance: d = amp * sin(n_pp * y_deg * pi/180)
    // Injected on the input (u) before the plant - models periodic cogging torque
    // as a function of shaft position.  Enables sim-to-real cogging robustness study.
    bool enable_cogging_sine;
    float cogging_sine_amp;  // V_eq, peak amplitude
    int cogging_freq_mult; // cogging periods/rev: n_pp (approx) or LCM(N_slots, 2*n_pp) (exact)
} neucode_disturbance_config_t;

/* Metrics configuration */
typedef struct {
    bool step_mode;
    float step_time; // time of step change
    float r_final; // final value after step
    size_t min_tail_samples; // minimum samples in tail window for steady-state error calc
    float tail_window_s; // time window for steady-state error calc
    float max_rate_hz; // max expected rate for steady-state error ring buffer
} neucode_metrics_config_t;

/* Metrics results */
typedef struct {
    float iae;
    float ise;
    float itae;
    float isu;
    float peak_value;
    float peak_time;
    float overshoot_percent;
    float rise_time;
    float steady_state_error;
    float steady_state_error_percent;
} neucode_metrics_results_t;

typedef struct {
    size_t num_samples;
    float* time_out;
    float* setpoint_out;
    float* measurement_out;
    float* control_effort_out;
} neucode_timeseries_buffers_t;


/* External controller transport interface.
 *
 * Implement exchange() for the desired transport (SPI, UART, CAN, ...).
 * Called once per control step with the current setpoint and measurement.
 * Must return the control output u, or NAN on timeout / communication error.
 * The caller (external.c) will substitute the last valid u on NAN.
 *
 * ctx is an opaque handle owned by the transport implementation (e.g. a
 * pointer to an SPI peripheral descriptor).  It is never dereferenced by
 * the core controller layer. */
typedef struct {
    float    (*exchange)(float sp, float y, void *ctx);
    void      *ctx;          /* opaque transport handle                  */
    uint32_t   timeout_us;   /* hard per-step deadline (informational)   */
} nc_external_ctrl_t;

#ifdef __cplusplus
}
#endif