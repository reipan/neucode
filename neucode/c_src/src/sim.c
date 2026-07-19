#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "sim.h"
#include "disturbance.h"
#include "metrics.h"
#include "neucode_types.h"
#include "pid.h"
#include "plant_fopdt.h"
#include "plant_foipdt.h"
#include "sp.h"

/* Private Simulation Struct */
struct neucode_sim_t {
    /* Time */
    float dt;
    float T;
    size_t N; // computed num samples

    /* Step-Mode */
    float current_time;
    float last_y_meas;
    float last_u;

    neucode_pid_t pid_state;

    /* Plant - only one is active at a time, selected by plant_type */
    neucode_plant_type_t  plant_type;
    neucode_plant_fopdt_t  plant_state;
    neucode_plant_foipdt_t plant_foipdt_state;

    neucode_sp_t sp_state;
    neucode_disturbance_t disturbance_state;
    neucode_metrics_t metrics_state;

    /* Sensor IIR low-pass filter (matches firmware sensor.c) */
    float sensor_iir_alpha;
    float sensor_iir_prev;
    bool  sensor_iir_enabled;

    /* Shared delay buffer (allocated for whichever plant is active) */
    float *plant_ubuf;
    size_t plant_ubuf_len;

    bool is_pid_set;
    bool is_plant_set;
    bool is_sp_set;
    bool is_metrics_set;
};

/* We keep this out of the header because it's only used here */
static void _neucode_sim_do_step(neucode_sim_t* sim, float t, float setpoint, float u_in, bool use_external_u, float* y_meas_out, float* u_out);

/* API */
NEUCODE_API neucode_status_t neucode_sim_create(neucode_sim_t** out) {
    if (!out) return NEUCODE_ERROR_INVALID_ARG;
    *out = (neucode_sim_t*)calloc(1, sizeof(neucode_sim_t));
    if (!*out) return NEUCODE_ERROR_ALLOC_FAILED;
    return NEUCODE_OK;
}

NEUCODE_API void neucode_sim_destroy(neucode_sim_t* sim) {
    if (!sim) return;
    if (sim->plant_ubuf) free(sim->plant_ubuf);
    neucode_metrics_destroy(&sim->metrics_state);
    /* finally free the struct itself */
    free(sim);
}

/* time step + total time */
NEUCODE_API neucode_status_t neucode_sim_set_time_step(neucode_sim_t* sim, float dt, float T) {
    if (!sim || !(dt > 0.f) || !(T > 0.f)) return NEUCODE_ERROR_INVALID_ARG;
    sim->dt = dt;
    sim->T = T;
    sim->N = (size_t)floorf(T / dt) + 1;
    return NEUCODE_OK;
}

/* PID gains + limits */
NEUCODE_API neucode_status_t neucode_sim_set_pid(neucode_sim_t* sim, const neucode_pid_gains_t* gains, const neucode_pid_limits_t* limits) {
    if (!sim) return NEUCODE_ERROR_INVALID_ARG;
    neucode_pid_init(&sim->pid_state, gains, limits);
    sim->is_pid_set = true;
    return NEUCODE_OK;
}

/* FOPDT plant params */
NEUCODE_API neucode_status_t neucode_sim_set_fopdt(neucode_sim_t* sim, const neucode_fopdt_params_t* params) {
    if (!sim || !params || !(params->tau > 0.f) || !(params->theta >= 0.f)) return NEUCODE_ERROR_INVALID_ARG;

    // Let the setter handle delay buffer and plant initialization
    if (sim->plant_ubuf) free(sim->plant_ubuf);
    sim->plant_ubuf_len = (size_t)ceilf(params->theta / sim->dt) + 2;
    sim->plant_ubuf = (float*)calloc(sim->plant_ubuf_len, sizeof(float));
    if (!sim->plant_ubuf) return NEUCODE_ERROR_ALLOC_FAILED;

    bool ok = neucode_plant_fopdt_init(&sim->plant_state, params, sim->dt, sim->plant_ubuf, sim->plant_ubuf_len);
    if (ok) sim->plant_type = NEUCODE_PLANT_FOPDT;
    sim->is_plant_set = ok;
    return ok ? NEUCODE_OK : NEUCODE_ERROR_INVALID_ARG;
}

/* FOIPDT plant params */
NEUCODE_API neucode_status_t neucode_sim_set_foipdt(neucode_sim_t* sim, const neucode_foipdt_params_t* params) {
    if (!sim || !params || !(params->tau > 0.f) || !(params->theta >= 0.f)) return NEUCODE_ERROR_INVALID_ARG;

    if (sim->plant_ubuf) free(sim->plant_ubuf);
    sim->plant_ubuf_len = (size_t)ceilf(params->theta / sim->dt) + 2;
    sim->plant_ubuf = (float*)calloc(sim->plant_ubuf_len, sizeof(float));
    if (!sim->plant_ubuf) return NEUCODE_ERROR_ALLOC_FAILED;

    bool ok = neucode_plant_foipdt_init(&sim->plant_foipdt_state, params, sim->dt, sim->plant_ubuf, sim->plant_ubuf_len);
    if (ok) sim->plant_type = NEUCODE_PLANT_FOIPDT;
    sim->is_plant_set = ok;
    return ok ? NEUCODE_OK : NEUCODE_ERROR_INVALID_ARG;
}

/* Setpoint */
NEUCODE_API neucode_status_t neucode_sim_set_setpoint(neucode_sim_t* sim, const neucode_setpoint_def_t* def) {
    if (!sim || !def) return NEUCODE_ERROR_INVALID_ARG;
    neucode_sp_init(&sim->sp_state, def);
    sim->is_sp_set = true;
    return NEUCODE_OK;
}

/* Disturbance (input + output + noise) */
NEUCODE_API neucode_status_t neucode_sim_set_disturbance(neucode_sim_t* sim, const neucode_disturbance_config_t* config) {
    if (!sim) return NEUCODE_ERROR_INVALID_ARG;
    // Call the init function from our new module.
    neucode_disturbance_init(&sim->disturbance_state, config);
    return NEUCODE_OK;
}

/* Sensor IIR filter */
NEUCODE_API neucode_status_t neucode_sim_set_sensor_filter(neucode_sim_t* sim, float alpha) {
    if (!sim) return NEUCODE_ERROR_INVALID_ARG;
    if (alpha <= 0.0f || alpha > 1.0f) {
        sim->sensor_iir_enabled = false;
        return NEUCODE_OK;
    }
    sim->sensor_iir_alpha   = alpha;
    sim->sensor_iir_prev    = 0.0f;
    sim->sensor_iir_enabled = true;
    return NEUCODE_OK;
}

/* Metrics */
NEUCODE_API neucode_status_t neucode_sim_set_metrics(neucode_sim_t* sim, const neucode_metrics_config_t* config) {
    if (!sim) return NEUCODE_ERROR_INVALID_ARG;
    if (!config) {
        sim->is_metrics_set = 0;
        return NEUCODE_OK;
    }

    neucode_metrics_destroy(&sim->metrics_state); // free any existing state
    bool ok = neucode_metrics_init(&sim->metrics_state, config);
    sim->is_metrics_set = ok;
    return ok ? NEUCODE_OK : NEUCODE_ERROR_ALLOC_FAILED;
}

/* A unified reset function */
NEUCODE_API neucode_status_t neucode_sim_reset(neucode_sim_t* sim) {
    if (!sim) return NEUCODE_ERROR_INVALID_ARG;
    neucode_pid_reset(&sim->pid_state);
    if (sim->plant_type == NEUCODE_PLANT_FOIPDT) {
        neucode_plant_foipdt_reset(&sim->plant_foipdt_state, 0.f);
    } else {
        neucode_plant_fopdt_reset(&sim->plant_state, 0.f);
    }
    if (sim->is_metrics_set) {
        neucode_metrics_reset(&sim->metrics_state);
    }

    // Re-initialize disturbance
    // neucode_disturbance_init zeroes the complete struct, so we need to save and restore the config.
    neucode_disturbance_config_t saved_config = sim->disturbance_state.config;
    neucode_disturbance_init(&sim->disturbance_state, &saved_config);

    /* Reset Step-Mode variables */
    sim->current_time = 0.0f;
    sim->last_y_meas = 0.0f;
    sim->last_u = 0.0f;
    sim->sensor_iir_prev = 0.0f;

    return NEUCODE_OK;
}

/* RUN THE SIMULATION */
NEUCODE_API neucode_status_t neucode_sim_run(
    neucode_sim_t* sim,
    neucode_simulation_result_t* result,
    const neucode_timeseries_buffers_t* buffers
) {
    if (!sim) return NEUCODE_ERROR_INVALID_ARG;
    // It's easier to just check N (has been already been computed) instead of dt and T again.
    if (!sim->is_pid_set || !sim->is_plant_set || !sim->is_sp_set || sim->N == 0) return NEUCODE_ERROR_INVALID_ARG;

    /* Reset states (initialized in the setters) */
    neucode_sim_reset(sim);

    /* Main loop */
    for (size_t k = 0; k < sim->N; ++k) {
        float t = (float)k * sim->dt;
        float setpoint = neucode_sp_eval(&sim->sp_state, t);

        // In a full run, the control variable is calculated internally by the PID.
        _neucode_sim_do_step(sim, t, setpoint, 0.0f, false, &sim->last_y_meas, &sim->last_u);

        /* Store time-series data if buffers are provided */
        if (buffers && k < buffers->num_samples) {
            if (buffers->time_out) {
                buffers->time_out[k] = t;
            }
            if (buffers->setpoint_out) {
                buffers->setpoint_out[k] = setpoint;
            }
            if (buffers->measurement_out) {
                buffers->measurement_out[k] = sim->last_y_meas;
            }
            if (buffers->control_effort_out) {
                buffers->control_effort_out[k] = sim->last_u;
            }
        }
    }

    /* Finalize result */
    if (result) {
        memset(result, 0, sizeof(neucode_simulation_result_t));
        result->y_final = sim->last_y_meas;
        result->samples_written = sim->N;
    }

    return NEUCODE_OK;
}

/* Step-Mode */
NEUCODE_API neucode_status_t neucode_sim_step(neucode_sim_t* sim, float control_variable) {
    if (!sim) return NEUCODE_ERROR_INVALID_ARG;
    if (!sim->is_plant_set || !sim->is_sp_set) return NEUCODE_ERROR_INVALID_ARG;

    float setpoint = neucode_sp_eval(&sim->sp_state, sim->current_time);

    // Execute one step using the external control variable
    _neucode_sim_do_step(sim, sim->current_time, setpoint, control_variable, true, &sim->last_y_meas, &sim->last_u);

    // Advance time for the next step
    sim->current_time += sim->dt;

    return NEUCODE_OK;
}

/* Retrieve metrics results after a run */
NEUCODE_API neucode_status_t neucode_sim_get_metrics_results(neucode_sim_t* sim, neucode_metrics_results_t* results) {
    if (!sim || !results) {
        return NEUCODE_ERROR_INVALID_ARG;
    }
    if (!sim->is_metrics_set) {
        // Return an error if metrics were not configured for this run.
        memset(results, 0, sizeof(neucode_metrics_results_t));
        return NEUCODE_ERROR_INVALID_ARG;
    }

    // DELEGATE: The metrics module is responsible for all final calculations.
    neucode_metrics_finalize(&sim->metrics_state, results);

    return NEUCODE_OK;
}

/* Get the simulation state after the last executed step */
NEUCODE_API size_t neucode_sim_get_state_vector(neucode_sim_t* sim, float* out_state) {
    if (!sim || !out_state) return 0;

    // The state vector for the controller is based on the state AFTER the last step,
    // sim->current_time is the time for the upcoming step.
    float setpoint_next = neucode_sp_eval(&sim->sp_state, sim->current_time);
    float error_next = setpoint_next - sim->last_y_meas;

    out_state[0] = error_next;
    out_state[1] = sim->last_y_meas;
    out_state[2] = setpoint_next;

    // Return size
    return 3;
}

/* Internal Step Helper */
static void _neucode_sim_do_step(neucode_sim_t* sim, float t, float setpoint, float u_in, bool use_external_u, float* y_meas_out, float* u_out) {
    float u_final;
    float y_true;

    /* Get current plant output */
    if (sim->plant_type == NEUCODE_PLANT_FOIPDT) {
        y_true = neucode_plant_foipdt_get_output(&sim->plant_foipdt_state);
    } else {
        y_true = neucode_plant_fopdt_get_output(&sim->plant_state);
    }
    float y_meas = y_true;

    /* Compute control signal */
    if (use_external_u) {
        u_final = u_in;
    } else {
        u_final = (float)neucode_pid_step(&sim->pid_state, setpoint, sim->last_y_meas, sim->dt);
    }

    neucode_disturbance_apply(&sim->disturbance_state, t, &u_final, &y_meas);

    /* Sensor IIR low-pass: y_filtered = alpha * y_raw + (1-alpha) * y_prev */
    if (sim->sensor_iir_enabled) {
        y_meas = sim->sensor_iir_alpha * y_meas
               + (1.0f - sim->sensor_iir_alpha) * sim->sensor_iir_prev;
        sim->sensor_iir_prev = y_meas;
    }

    /* Advance the active plant */
    if (sim->plant_type == NEUCODE_PLANT_FOIPDT) {
        neucode_plant_foipdt_step(&sim->plant_foipdt_state, u_final);
    } else {
        neucode_plant_fopdt_step(&sim->plant_state, u_final);
    }

    if (sim->is_metrics_set) {
        neucode_metrics_update(&sim->metrics_state, sim->dt, setpoint, y_meas, u_final);
    }

    *y_meas_out = y_meas;
    *u_out = u_final;
}