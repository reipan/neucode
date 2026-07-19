#include "ann.h"
#include "inference.h"
#include "model_data.h"
#include "profiler.h"

#include <stddef.h>
#include <stdio.h>

#ifndef SKIP_CONTROLLER_REGISTRATION
#include "controller.h"
#include "comm.h"
#endif

#ifndef NC_PID_ERR_DEADBAND_DEG
    #define NC_PID_ERR_DEADBAND_DEG 0.15f
#endif

#ifndef NC_ANN_U_MAX
    #define NC_ANN_U_MAX 1.0
#endif

#ifndef NC_ANN_U_MIN
    #define NC_ANN_U_MIN -1.0
#endif

/**
 * Initializes the ANN controller.
 *
 * There is no specific initialization needed for the ANN controller, everything we need
 * is already include in the model_data.h and inference.c files.
 *
 * @note This function is called internally by the ANN adapter.
 */
static void ann_init(void) {
    // No initialization needed for ANN controller.
    if (!ann_is_model_valid()) {
#ifndef SKIP_CONTROLLER_REGISTRATION
        nc_comm_send_log("warn: ANN controller using DUMMY model with zero weights");
#endif
    } else {
#ifndef SKIP_CONTROLLER_REGISTRATION
        nc_comm_send_log("info: ANN controller initialized with valid model");
#endif
    }
}


// Controller state (kept local to this module).
static double integral_error = 0.0;
static double prev_error = 0.0;
static double prev_output = 0.0;  // EMA filter state on output

// Output EMA alpha - default 0.0 (disabled), override via Makefile: -DNC_ANN_OUTPUT_EMA_ALPHA=x
// Disabled by default: the ANN produces sufficiently smooth outputs without filtering.
#ifndef NC_ANN_OUTPUT_EMA_ALPHA
    #define NC_ANN_OUTPUT_EMA_ALPHA 0.0
#endif

/**
 * Resets the ANN controller.
 *
 * @note This function is called internally by the ANN adapter.
 */
void ann_reset(void) {
    ann_inference_reset();

    // Reset controller state (kept local to this module).
    integral_error = 0.0;
    prev_error = 0.0;
    prev_output = 0.0;
}

// Runtime actuator limits for saturation.
static float ann_u_min = (float)NC_ANN_U_MIN;
static float ann_u_max = (float)NC_ANN_U_MAX;

void ann_set_output_limits(float u_min, float u_max) {
    ann_u_min = u_min;
    ann_u_max = u_max;
}

/**
 * Sets the ANN controller parameters.
 *
 * All parameters are embedded in the model data; there are no runtime parameters to set/use.
 *
 * @note This function is called internally by the ANN adapter.
 *
 * @param params Pointer to the parameters to set.
 * @param params_size Size of the parameters in bytes.
 */
static void ann_set_params(const void *params, size_t params_size) {
    // No runtime parameters to set for ANN controller.
    (void)params;
    (void)params_size;
}

/**
 * Performs an ANN control step.
 *
 * Passes the setpoint, measured value, error, integral error, and derivative error
 * to the neural network, runs inference, and returns the saturated control output.
 *
 * The ANN replacement controller is trained on 5 features:
 * [setpoint, measurement, error, integral_error, derivative_error]
 *
 * Integral feature: plain cumsum (no anti-windup). This matches the training
 * data generator which uses np.cumsum(error)*dt with no saturation gating.
 *
 * @note This function is called internally by the ANN adapter.
 *
 * @param sp Setpoint value.
 * @param y Measured value.
 * @param dt Time delta since last step.
 * @return Saturated control output value.
 */
double ann_step(double sp, double y, double dt) {
    double e = sp - y;
    if (e > -NC_PID_ERR_DEADBAND_DEG && e < NC_PID_ERR_DEADBAND_DEG) {
        return 0.0;
    }

    // Plain cumsum integral - matches training data generation; no anti-windup gating.
    integral_error += e * dt;

    // Clip integral to +/-5sigma of the training distribution to prevent out-of-distribution
    // runaway on hardware.
    {
        double int_mean  = (double)ann_get_integral_scaler_mean();
        double int_scale = (double)ann_get_integral_scaler_scale();
        double int_max   = int_mean + 5.0 * int_scale;
        double int_min   = int_mean - 5.0 * int_scale;
        if (integral_error > int_max) integral_error = int_max;
        if (integral_error < int_min) integral_error = int_min;
    }

    // Calculate derivative error.
    // The large derivative spike at t=0 (step onset) is clipped to DERIV_CLIP_VALUE but
    // remains within the training distribution. The network uses this as a step-onset signal.
    double derivative_error = (e - prev_error) / dt;
    prev_error = e;

    // Clamp derivative to training distribution (p99 clip) - prevents OOD spike at t=0
#ifdef DERIV_CLIP_VALUE
    if (derivative_error >  (double)DERIV_CLIP_VALUE) derivative_error =  (double)DERIV_CLIP_VALUE;
    if (derivative_error < -(double)DERIV_CLIP_VALUE) derivative_error = -(double)DERIV_CLIP_VALUE;
#endif

    // Build a float input array for the ANN
    // Features: [setpoint, measurement, error, integral_error, derivative_error]
    float input_floats[5] = {
        (float)sp,
        (float)y,
        (float)e,
        (float)integral_error,
        (float)derivative_error
    };

    int8_t raw_output = ann_inference(input_floats);

    // Scale the int8_t output back to float
    double scale_factor = (double)ann_get_output_scale();
    double output = (double)raw_output * scale_factor;

    // Output EMA - optional smoothing, disabled by default (alpha=0.0 bypasses the filter).
    if (NC_ANN_OUTPUT_EMA_ALPHA > 0.0) {
        output = NC_ANN_OUTPUT_EMA_ALPHA * output + (1.0 - NC_ANN_OUTPUT_EMA_ALPHA) * prev_output;
        prev_output = output;
    }

    // Saturate output
    if (output > (double)ann_u_max) {
        output = (double)ann_u_max;
    } else if (output < (double)ann_u_min) {
        output = (double)ann_u_min;
    }

    return output;
}

/**
 * Runs n inference calls with fixed synthetic inputs and reports cycle counts via UART log.
 *
 * Uses profiler_get_cycles() (DWT_CYCCNT) to measure each call. The motor and sensor
 * are not used; this function is safe to call with no hardware attached.
 *
 * @param n Number of benchmark iterations (recommended: 1000).
 */
void ann_benchmark(uint32_t n) {
#ifndef SKIP_CONTROLLER_REGISTRATION
    ann_reset();

    uint32_t cycles_sum = 0;
    uint32_t cycles_min = 0xFFFFFFFFu;
    uint32_t cycles_max = 0;

    for (uint32_t i = 0; i < n; i++) {
        uint32_t t_start = profiler_get_cycles();
        ann_step(22.5, 0.0, 0.001);
        uint32_t t_end = profiler_get_cycles();
        uint32_t cycles = t_end - t_start;
        cycles_sum += cycles;
        if (cycles < cycles_min) cycles_min = cycles;
        if (cycles > cycles_max) cycles_max = cycles;
    }

    /* Convert cycles to microseconds: cycles / (170 MHz) */
    uint32_t mean_cycles = cycles_sum / n;
    uint32_t mean_us     = mean_cycles / 170;
    uint32_t mean_us_frac = (mean_cycles % 170) * 10 / 170;
    uint32_t min_us      = cycles_min / 170;
    uint32_t max_us      = cycles_max / 170;

    char buf[128];
    snprintf(buf, sizeof(buf),
             "bench ann: mean=%lu.%luus min=%luus max=%luus (n=%lu)",
             (unsigned long)mean_us, (unsigned long)mean_us_frac,
             (unsigned long)min_us,  (unsigned long)max_us,
             (unsigned long)n);
    nc_comm_send_log(buf);
#endif
    (void)n;
}

/**
 * Registers the ANN controller with the abstract controller interface.
 *
 * @note This function should be called during system initialization.
 *
 * @see nc_controller_register_mode
 */
void nc_ann_controller_register(void) {
#ifndef SKIP_CONTROLLER_REGISTRATION
    nc_controller_register_mode(
        NC_CONTROLLER_MODE_ANN,
        ann_init,
        ann_reset,
        ann_set_params,
        ann_step
    );
#else
    (void)ann_init;
    (void)ann_set_params;
#endif
}