#pragma once
#ifdef __cplusplus
extern "C" {
#endif

#include "neucode_types.h"

/* Export macro (ELF; no-op on static builds)
 * @todo: further check: https://gcc.gnu.org/wiki/Visibility
 */
#define NEUCODE_API __attribute__((visibility("default")))

/* Opaque handle 
 * Acts as a pass-through for all simulation parameters and state
 */
typedef struct neucode_sim_t neucode_sim_t;

/* Lifecycle */
NEUCODE_API neucode_status_t neucode_sim_create(neucode_sim_t** out_sim);
NEUCODE_API neucode_status_t neucode_sim_reset(neucode_sim_t* sim);
NEUCODE_API void neucode_sim_destroy(neucode_sim_t* sim);

/* Configuration */
NEUCODE_API neucode_status_t neucode_sim_set_time_step(neucode_sim_t* sim, float dt, float total_time);
NEUCODE_API neucode_status_t neucode_sim_set_pid(neucode_sim_t* sim, const neucode_pid_gains_t* gains, const neucode_pid_limits_t* limits);
NEUCODE_API neucode_status_t neucode_sim_set_fopdt(neucode_sim_t* sim, const neucode_fopdt_params_t* plant_params);
NEUCODE_API neucode_status_t neucode_sim_set_foipdt(neucode_sim_t* sim, const neucode_foipdt_params_t* plant_params);
NEUCODE_API neucode_status_t neucode_sim_set_setpoint(neucode_sim_t* sim, const neucode_setpoint_def_t* setpoint_def);
NEUCODE_API neucode_status_t neucode_sim_set_disturbance(neucode_sim_t* sim, const neucode_disturbance_config_t* disturbance_config);
NEUCODE_API neucode_status_t neucode_sim_set_sensor_filter(neucode_sim_t* sim, float alpha);
NEUCODE_API neucode_status_t neucode_sim_set_metrics(neucode_sim_t* sim, const neucode_metrics_config_t* metrics_cfg);

/* Execution */
NEUCODE_API neucode_status_t neucode_sim_run(
    neucode_sim_t* sim,
    neucode_simulation_result_t* result,
    const neucode_timeseries_buffers_t* buffers
);

/* Stepped execution */
NEUCODE_API neucode_status_t neucode_sim_step(neucode_sim_t* sim, float control_variable_input);
NEUCODE_API size_t neucode_sim_get_state_vector(neucode_sim_t* sim, float* out_state);

/* Retrieve metrics results after a run */
NEUCODE_API neucode_status_t neucode_sim_get_metrics_results(neucode_sim_t* sim, neucode_metrics_results_t* results);

#ifdef __cplusplus
}
#endif