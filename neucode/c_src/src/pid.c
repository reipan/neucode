#include "pid.h"
#include <string.h>
#include <math.h>

/**
 * Clamps a double value within the specified range.
 *
 * Ensures that the input value x is not less than low and not greater than high.
 * If x is less than low, returns low. If x is greater than high, returns high.
 * Otherwise, returns x.
 *
 * @param x The value to clamp.
 * @param low The lower bound of the range.
 * @param high The upper bound of the range.
 * @return The clamped value.
 */
static inline double clampd(double x, double low, double high) {
    if (x < low) return low;
    if (x > high) return high;
    return x;
}

/**
 * Initializes a PID controller structure with given gains and limits.
 *
 *
 * @param p Pointer to the PID controller structure to initialize.
 * @param gains Pointer to the structure containing PID gain values.
 * @param limits Pointer to the structure containing PID output and integral limits.
 */
void neucode_pid_init(neucode_pid_t* p, const neucode_pid_gains_t* gains, const neucode_pid_limits_t* limits) {
    memset(p, 0, sizeof(neucode_pid_t));

    if (gains) {
        p->kp = gains->kp;
        p->ki = gains->ki;
        p->kd = gains->kd;
    }
    
    if (limits) {
        p->u_min = limits->u_min;
        p->u_max = limits->u_max;
        p->i_min = limits->i_min;
        p->i_max = limits->i_max;
        p->kaw = limits->kaw;
        p->d_alpha = limits->d_alpha;
    } else {
        p->u_min = -INFINITY;
        p->u_max = INFINITY;
        p->i_min = -INFINITY;
        p->i_max = INFINITY;
        p->d_alpha = 1.0; // No filter
        p->kaw = 0.0;     // No anti-windup
    }

    neucode_pid_reset(p);
}

/**
 * Resets the internal state of a PID controller.
 *
 * This function clears:
 * - previous error
 * - previous measurement
 * - integral term
 * - filtered derivative term
 *
 * @param p Pointer to the PID controller structure to reset.
 */
void neucode_pid_reset(neucode_pid_t* p) {
    if (!p) return;
    p->prev_error = 0.0;
    p->prev_meas = 0.0;
    p->integral = 0.0;
    p->d_filt = 0.0;
}

/**
 * Performs a single PID control step.
 *
 * Calculates the saturated control output based on the setpoint, measured value, and elapsed time.
 *
 * @param p Pointer to the PID controller structure.
 * @param sp Setpoint value (desired target).
 * @param meas Measured process value (current value).
 * @param dt Time interval since the last update (in seconds).
 * @return The computed control output.
 */
double neucode_pid_step(neucode_pid_t* p, double sp, double meas, double dt) {
    double error = sp - meas;
    double p_term = p->kp * error;

    // NOTE: kd is pre-scaled by the caller (kd = kd_textbook / dt), so we do not divide by dt here.
    // This makes kd independent of the control loop frequency and prevents derivative spikes when dt changes.
    // Better: This optimizes the calculation by absorbing the constant sample time (dt) into the tuning parameter, avoiding division on every step.
    const double d_meas_raw = (p->prev_meas - meas);
    p->d_filt = (1.0 - p->d_alpha) * p->d_filt + p->d_alpha * d_meas_raw;
    double d_term = p->kd * p->d_filt;

    const double u_raw = p_term + p->integral + d_term;
    const double u_sat = clampd(u_raw, p->u_min, p->u_max);

    double i_term_next = p->integral + (p->ki * error * dt) + (p->kaw * (u_sat - u_raw) * dt);
    p->integral = clampd(i_term_next, p->i_min, p->i_max);
    p->prev_error = error;
    p->prev_meas = meas;

    return u_sat;
}