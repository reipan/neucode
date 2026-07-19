#include "metrics.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>

/*
 * Metrics for control systems.
 *
 * Most of the implemented metrics are referenced in:
 * - Process Dynamics and Control by Seborg, Edgar, Mellichamp (Chapter 12.3.2)
 * - Feedback Systems by Astrom and Murray (Chapter 5.3)
 * - Modern Control Systems by Dorf and Bishop (Chapter 4.6)
 *
 * Implemented Metrics:
 * - IAE: Integral of Absolute Error (IAE) = integral |e(t)| dt
 * - ISE: Integral of Squared Error (ISE) = integral e(t)^2 dt
 * - ITAE: Integral of Time-weighted Absolute Error (ITAE) = integral t|e(t)| dt
 * - ISU: Integral of Squared Control Effort (ISU) = integral u(t)^2 dt
 * - Overshoot: Maximum peak value & time of the response curve as a percentage over the final value
 * - Rise Time: Time taken for the response to rise from 10% to 90% of the final value
 * - Steady-State Error: Difference between the final output and the desired setpoint value (right now only offline calc)
 */

/* Initialize ring buffer for steady-state error */
static bool sse_ring_buffer_init(neucode_ring_buffer_t* rb, const neucode_metrics_config_t* cfg) {
    if (!rb || !cfg) return false;
    rb->tail_s = (cfg->tail_window_s > 0.0f) ? cfg->tail_window_s : 1.0f;
    rb->min_tail_samples = (cfg->min_tail_samples > 0) ? cfg->min_tail_samples : 5;
    rb->cap = (size_t)(cfg->tail_window_s * cfg->max_rate_hz) + 8;
    if (rb->cap < cfg->min_tail_samples + 4) rb->cap = cfg->min_tail_samples + 4;
    rb->buf = (neucode_ring_buffer_elem_t*)calloc(rb->cap, sizeof(neucode_ring_buffer_elem_t));
    return rb->buf != NULL;
}

/* Free ring buffer */
static void sse_ring_buffer_reset(neucode_ring_buffer_t* rb) {
    if (rb && rb->buf) {
        free(rb->buf);
        *rb = (neucode_ring_buffer_t){0};
    }
}

/* Add new sample to ring buffer */
static void sse_ring_buffer_update(neucode_ring_buffer_t* rb, float t, float r, float y)
{
    if (!rb || !rb->buf || rb->cap == 0) return;
    rb->buf[rb->head].t = t;
    rb->buf[rb->head].r = r;
    rb->buf[rb->head].y = y;
    rb->head = (rb->head + 1) % rb->cap;
    if (rb->size < rb->cap) rb->size++;
}

/* Calculate average steady-state error over the tail window */
static float sse_ring_buffer_calc_avg(const neucode_ring_buffer_t* rb, float t_now, bool absolute)
{
    if (!rb || !rb->size || t_now < 0) return NAN;
    const float tail_start_t = t_now - rb->tail_s;
    size_t n = 0;
    float sum = 0.0f;
    for (size_t k = 0; k < rb->size; ++k) {
        size_t idx = (rb->head + rb->cap - 1 - k) % rb->cap; // reverse order
        // check if within tail window
        if (rb->buf[idx].t >= tail_start_t) {
            if (absolute) {
                sum += fabsf(rb->buf[idx].r - rb->buf[idx].y);
            } else {
                sum += (rb->buf[idx].r - rb->buf[idx].y);
            }
            n++;
        } else {
            break; // older than tail window
        }
    }
    // Ensure at least min_tail_samples samples
    if (n < rb->min_tail_samples) {
        return NAN;
    }
    return (float)(sum / (double)n);
}

/* Get Overshoot value (percentage) */
static float calc_overshoot_percent(const neucode_metrics_t* m) {
    if (!(m->step_mode && m->r_final_locked)) {
        return 0.0f;
    }
    float amp = fabsf(m->r_final - m->r_initial);
    if (!(amp > 0.0f) || !isfinite(amp)) {
        return 0.0f; // avoid division by zero
    }
    bool rising = (m->r_final - m->r_initial) >= 0.0f;
    float os;
    if (rising) {
        // peak above final
        os = 100.0f * (m->y_max - m->r_final) / amp;
    } else {
        // trough below final
        os = 100.0f * (m->r_final - m->y_min) / amp;
    }
    // never report negative overshoot
    return (os > 0.0f && isfinite(os)) ? os : 0.0f;
}

bool neucode_metrics_init(neucode_metrics_t* m, const neucode_metrics_config_t* cfg) {
    memset(m, 0, sizeof(neucode_metrics_t));
    m->y_max = -INFINITY;
    m->t_y_max = NAN;
    m->y_min = INFINITY;
    m->t_y_min = NAN;
    m->y10 = NAN;
    m->y90 = NAN;
    m->t_rise_start = NAN;
    m->t_rise_end = NAN;

    // Init config if provided or use defaults
    if (cfg) {
        m->cfg = *cfg;
    } else {
        // Apply default configuration if none is provided
        m->cfg.tail_window_s = 1.0f;
        m->cfg.max_rate_hz = 100.0f;
        m->cfg.min_tail_samples = 5;
    }

    m->step_mode = m->cfg.step_mode;
    m->step_time = m->cfg.step_time;
    m->r_final = m->cfg.r_final;
    m->r_final_locked = isfinite(m->r_final);

    // Init ring buffer for steady-state error
    return sse_ring_buffer_init(&m->sse_rb, &m->cfg);
}

/* Destroy metrics */
void neucode_metrics_destroy(neucode_metrics_t* m) {
    if (m) {
        sse_ring_buffer_reset(&m->sse_rb);
        memset(m, 0, sizeof(neucode_metrics_t));
    }
}

/* Reset metrics to initial state */
void neucode_metrics_reset(neucode_metrics_t* m) {
    if (m) {
        neucode_metrics_config_t tmp_cfg = m->cfg;
        neucode_metrics_destroy(m);
        neucode_metrics_init(m, &tmp_cfg);
    }
}

/* Update metrics with new sample */
void neucode_metrics_update(neucode_metrics_t* m, float dt, float r, float y, float u) {
    m->t += dt; // next time step

    float e = r - y; // calc error
    m->iae += fabsf(e) * dt; // update iae
    m->ise += (e * e) * dt; // update ise
    m->isu += (u * u) * dt; // update isu
    
    // weight = time since step (0 before the step)
    // This should work for all setpoint types
    // @todo: rename step_time to reference_time or event_time?
    float w = (m->t >= m->step_time) ? (m->t - m->step_time) : 0.0f;
    m->itae += w * fabsf(e) * dt;

    // m->itae += m->t * fabsf(e) * dt; // update itae

    if (m->step_mode) {
        // Remember the value of r before the step
        if (m->t < m->step_time) {
            m->r_initial = r;
        } else if (!m->r_final_locked) {
            m->r_final = r;
            m->r_final_locked = true;
        }

        // Prepare for rise time calculation
        if (m->step_mode && m->r_final_locked && isnan(m->y10)) {
            float amp = m->r_final - m->r_initial;
            m->y10 = m->r_initial + 0.1f * amp;
            m->y90 = m->r_initial + 0.9f * amp;
        }

        // Check for step change (only if in step mode)
        if (m->step_mode && m->t >= m->step_time) {
            if (y > m->y_max) {
                m->y_max = y;
                m->t_y_max = m->t;
            }
            if (y < m->y_min) {
                m->y_min = y;
                m->t_y_min = m->t;
            }
        }

        // Rise time detection (only if in step mode and r_final is known)
        if (m->step_mode && m->t >= m->step_time && m->r_final_locked) {
            bool rising = (m->r_final > m->r_initial);
            if (!m->crossed_10) {
                if ((rising && y >= m->y10) || (!rising && y <= m->y10)) {
                    m->crossed_10 = true;
                    m->t_rise_start = m->t;
                }
            }
            if (m->crossed_10 && !m->crossed_90) {
                if ((rising && y >= m->y90) || (!rising && y <= m->y90)) {
                    m->crossed_90 = true;
                    m->t_rise_end = m->t;
                }
            }
        }
    }

    sse_ring_buffer_update(&m->sse_rb, m->t, r, y);
}

/**
 * Calculate metrics for a batch of samples.
 * The arrays t_arr, r_arr, y_arr, u_arr must be of length n
 */
void neucode_metrics_calc_batch(neucode_metrics_t* m, const float* t_arr, const float* r_arr, const float* y_arr, const float* u_arr, size_t n) {
    if (!t_arr || !r_arr || !y_arr || !u_arr || !m || n == 0) return;

    for (size_t i = 1; i < n; ++i) {
        float dt = t_arr[i] - t_arr[i - 1];
        if (dt < 0.0f) {
            // Non-monotonic time, skip this sample
            continue;
        }
        neucode_metrics_update(m, dt, r_arr[i], y_arr[i], u_arr[i]);
    }
}

/* Finalize metrics and compute results */
void neucode_metrics_finalize(const neucode_metrics_t* m, neucode_metrics_results_t* out) {
    out->iae = m->iae;
    out->ise = m->ise;
    out->itae = m->itae;
    out->isu = m->isu;
    out->peak_value = m->y_max;
    out->peak_time = m->t_y_max;
    out->overshoot_percent = calc_overshoot_percent(m);
    if (m->crossed_10 && m->crossed_90) {
        out->rise_time = m->t_rise_end - m->t_rise_start;
    } else {
        out->rise_time = NAN;
    }

    // Calc absolute steady-state error from ring buffer
    out->steady_state_error = sse_ring_buffer_calc_avg(&m->sse_rb, m->t, true);
    // Calc percentage steady-state error
    float amp = fabsf(m->r_final - m->r_initial);
    // If the amplitude is too small or not finite, use a fallback value
    if (!(amp > 1e-12f) || !isfinite(amp)) {
        amp = fmaxf(fabsf(m->r_final), 1.0f); // avoid division by zero
    }
    // Compute percentage if absolute value is valid
    if (isfinite(out->steady_state_error)) {
        out->steady_state_error_percent = 100.0f * out->steady_state_error / amp;
    } else {
        out->steady_state_error_percent = NAN;
    }
}