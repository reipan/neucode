# todo: check if using bint is safe for stdbool

import numpy as np
cimport numpy as np
from libc.stdint cimport uint32_t
from libc.stdlib cimport malloc, free
from libc.string cimport memset

cdef extern from "neucode_types.h":
    ctypedef struct neucode_pid_gains_t:
        float kp
        float ki
        float kd

    ctypedef struct neucode_pid_limits_t:
        float u_min
        float u_max
        float i_min
        float i_max
        float kaw
        float d_alpha

    ctypedef struct neucode_fopdt_params_t:
        float K
        float tau
        float theta
        float friction

    ctypedef struct neucode_foipdt_params_t:
        float Kv
        float tau
        float theta
        float friction

    ctypedef enum neucode_setpoint_type_t:
        NEUCODE_SP_STEP
        NEUCODE_SP_RAMP
        NEUCODE_SP_SIN

    ctypedef struct neucode_setpoint_def_t:
        neucode_setpoint_type_t type
        float step_time
        float v
        float a
        float b
        float time
        float amp
        float freq

    ctypedef struct neucode_timeseries_buffers_t:
        size_t num_samples
        float* time_out
        float* setpoint_out
        float* measurement_out
        float* control_effort_out

    ctypedef enum neucode_noise_type_t:
        NEUCODE_NOISE_NONE
        NEUCODE_NOISE_UNIFORM
        NEUCODE_NOISE_GAUSSIAN

    ctypedef struct neucode_disturbance_config_t:
        bint enable_input_step
        float input_step_at_s
        float input_step_value
        bint enable_output_step
        float output_step_at_s
        float output_step_value
        neucode_noise_type_t noise_type
        float noise_amp
        float noise_std
        uint32_t seed
        bint enable_cogging_sine
        float cogging_sine_amp
        int cogging_freq_mult

    ctypedef struct neucode_metrics_config_t:
        bint step_mode
        float step_time
        float r_final
        float tail_window_s
        float max_rate_hz
        size_t min_tail_samples

    ctypedef struct neucode_simulation_result_t:
        size_t samples_written
        float y_final

    ctypedef struct neucode_metrics_results_t:
        float iae
        float ise
        float itae
        float isu
        float peak_value
        float peak_time
        float overshoot_percent
        float rise_time
        float steady_state_error
        float steady_state_error_percent

cdef extern from "pid.h":
    ctypedef struct neucode_pid_t:
        double kp, ki, kd
        double u_min, u_max
        double i_min, i_max
        double kaw, d_alpha
        double integral, prev_error, prev_meas, d_filt

    void neucode_pid_init(neucode_pid_t* p, const neucode_pid_gains_t* gains,
                          const neucode_pid_limits_t* limits)
    void neucode_pid_reset(neucode_pid_t* p)
    double neucode_pid_step(neucode_pid_t* p, double sp, double meas, double dt)

cdef extern from "metrics.h":
    ctypedef struct neucode_ring_buffer_elem_t:
        float t
        float r
        float y

    ctypedef struct neucode_ring_buffer_t:
        size_t cap;
        size_t head;
        size_t size;
        neucode_ring_buffer_elem_t *buf;
        float tail_s;
        size_t min_tail_samples;

    ctypedef struct neucode_metrics_t:
        float iae
        float ise
        float itae
        float isu
        float step_time
        float r_initial
        float r_final
        float sse_val
        bint step_mode
        bint r_final_locked
        float y10
        float y90
        bint crossed_10
        bint crossed_90
        float t_rise_start
        float t_rise_end
        float y_max
        float t_y_max
        float y_min
        float t_y_min
        neucode_ring_buffer_t sse_rb
        neucode_metrics_config_t config

    bint neucode_metrics_init(neucode_metrics_t* metrics, const neucode_metrics_config_t* metrics_cfg);
    void neucode_metrics_destroy(neucode_metrics_t* metrics);
    void neucode_metrics_reset(neucode_metrics_t* metrics);
    void neucode_metrics_calc_batch(neucode_metrics_t* metrics, const float* t_arr, const float* r_arr, const float* y_arr, const float* u_arr, size_t n);
    void neucode_metrics_finalize(const neucode_metrics_t* metrics, neucode_metrics_results_t* out);

cdef class StandaloneMetrics:
    """
    Standalone wrapper for the neucode_metrics_t C struct and its associated functions.
    It allows calculating PID control metrics independently from the simulation core.
    """
    cdef neucode_metrics_t* _metrics

    def __cinit__(self):
        self._metrics = <neucode_metrics_t*> malloc(sizeof(neucode_metrics_t))
        if self._metrics is NULL:
            raise MemoryError("Failed to allocate memory for metrics struct")

    def __init__(self, metrics_config=None):
        cdef neucode_metrics_config_t config

        if metrics_config is None:
            memset(&config, 0, sizeof(neucode_metrics_config_t))
            if not neucode_metrics_init(self._metrics, &config):
                raise RuntimeError("Failed to initialize metrics struct")
            return

        memset(&config, 0, sizeof(neucode_metrics_config_t))
        config.step_mode = metrics_config.get('step_mode', False)
        config.step_time = metrics_config.get('step_time', 0.0)
        config.r_final = metrics_config.get('r_final', 0.0)
        config.tail_window_s = metrics_config.get('tail_window_s', 1.0)
        config.max_rate_hz = metrics_config.get('max_rate_hz', 1000.0)
        config.min_tail_samples = metrics_config.get('min_tail_samples', 5)

        if not neucode_metrics_init(self._metrics, &config):
            raise RuntimeError("Failed to initialize metrics struct")

    def __dealloc__(self):
        if self._metrics != NULL:
            neucode_metrics_destroy(self._metrics)
            free(self._metrics)
            self._metrics = NULL

    def reset(self):
        """
        Resets the metrics calculation state.
        """
        neucode_metrics_reset(self._metrics)

    def process_telemetry(self, float[:] t, float[:] r, float[:] y, float[:] u):
        """
        Processes a batch of telemetry data to update metrics.
        """
        cdef size_t n = t.shape[0]
        if r.shape[0] != n or y.shape[0] != n or u.shape[0] != n:
            raise ValueError("All input arrays must have the same length.")

        if n == 0:
            return  # Nothing to process

        neucode_metrics_calc_batch(self._metrics,
                                   &t[0],
                                   &r[0],
                                   &y[0],
                                   &u[0],
                                   n)

    def get_results(self):
        """
        Retrieves the final calculated metrics.
        """
        cdef neucode_metrics_results_t c_metrics_result
        neucode_metrics_finalize(self._metrics, &c_metrics_result)

        results = {
            'iae': c_metrics_result.iae,
            'ise': c_metrics_result.ise,
            'itae': c_metrics_result.itae,
            'isu': c_metrics_result.isu,
            'overshoot_percent': c_metrics_result.overshoot_percent,
            'rise_time': c_metrics_result.rise_time,
            'peak_value': c_metrics_result.peak_value,
            'peak_time': c_metrics_result.peak_time,
            'steady_state_error': c_metrics_result.steady_state_error,
            'steady_state_error_percent': c_metrics_result.steady_state_error_percent,
        }
        return results

cdef class PIDState:
    """
    Standalone wrapper around the C-core PID controller (neucode_pid_step).

    Exposes the same PID algorithm used in both the simulation engine and the
    MCU firmware, callable step-by-step from Python.
    """
    cdef neucode_pid_t _state

    def __init__(self, gains, limits=None):
        cdef neucode_pid_gains_t c_gains
        c_gains.kp = gains.get('kp', 0.0)
        c_gains.ki = gains.get('ki', 0.0)
        c_gains.kd = gains.get('kd', 0.0)

        cdef const neucode_pid_limits_t* c_limits_ptr = NULL
        cdef neucode_pid_limits_t c_limits
        if limits is not None:
            c_limits.u_min   = limits.get('u_min', -1.0)
            c_limits.u_max   = limits.get('u_max',  1.0)
            c_limits.i_min   = limits.get('i_min', c_limits.u_min)
            c_limits.i_max   = limits.get('i_max', c_limits.u_max)
            c_limits.kaw     = limits.get('kaw', 0.0)
            c_limits.d_alpha = limits.get('d_alpha', 0.1)
            c_limits_ptr = &c_limits
        neucode_pid_init(&self._state, &c_gains, c_limits_ptr)

    def reset(self):
        neucode_pid_reset(&self._state)

    def step(self, double sp, double meas, double dt):
        return neucode_pid_step(&self._state, sp, meas, dt)

cdef extern from "sim.h":
    ctypedef enum neucode_status_t:
        NEUCODE_OK
        NEUCODE_ERROR_INVALID_ARG
        NEUCODE_ERROR_ALLOC_FAILED

    ctypedef struct neucode_sim_t:
        pass

    int neucode_sim_create(neucode_sim_t** sim)
    void neucode_sim_destroy(neucode_sim_t* sim)
    neucode_status_t neucode_sim_set_time_step(neucode_sim_t* sim, float dt, float total_time)
    neucode_status_t neucode_sim_set_pid(neucode_sim_t* sim, const neucode_pid_gains_t* gains, const neucode_pid_limits_t* limits)
    neucode_status_t neucode_sim_set_fopdt(neucode_sim_t* sim, const neucode_fopdt_params_t* plant_params)
    neucode_status_t neucode_sim_set_foipdt(neucode_sim_t* sim, const neucode_foipdt_params_t* plant_params)
    neucode_status_t neucode_sim_set_setpoint(neucode_sim_t* sim, const neucode_setpoint_def_t* setpoint_def)
    neucode_status_t neucode_sim_set_disturbance(neucode_sim_t* sim, const neucode_disturbance_config_t* disturbance_config)
    neucode_status_t neucode_sim_set_sensor_filter(neucode_sim_t* sim, float alpha)
    neucode_status_t neucode_sim_set_metrics(neucode_sim_t* sim, const neucode_metrics_config_t* metrics_cfg)
    neucode_status_t neucode_sim_run(neucode_sim_t* sim, neucode_simulation_result_t* result, const neucode_timeseries_buffers_t* buffers)
    neucode_status_t neucode_sim_get_metrics_results(neucode_sim_t* sim, neucode_metrics_results_t* results)
    neucode_status_t neucode_sim_reset(neucode_sim_t* sim)
    neucode_status_t neucode_sim_step(neucode_sim_t* sim, float control_variable)
    size_t neucode_sim_get_state_vector(neucode_sim_t* sim, float* out_state)   

cdef class Simulation:
    """
    Full wrapper for the neucode_sim_t C struct and its associated functions.
    It allows setting up and running a PID control simulation with FOPDT plant,
    setpoints, disturbances, and metrics calculation.
    """
    cdef neucode_sim_t* _sim

    def __cinit__(self):
        self._sim = NULL
        status = neucode_sim_create(&self._sim)
        if status != 0:
            raise MemoryError("Failed to create simulation object")

    def __dealloc__(self):
        if self._sim != NULL:
            neucode_sim_destroy(self._sim)
            self._sim = NULL

    def set_time_step(self, float dt, float total_time):
        """
        Set the time step and total simulation time.
        """
        status = neucode_sim_set_time_step(self._sim, dt, total_time)
        if status != NEUCODE_OK:
            raise ValueError("Invalid arguments for setting time step (status={})...".format(status))

    def set_pid(self, gains, limits=None):
        """
        Set the PID controller parameters.
        """
        cdef neucode_pid_gains_t c_gains
        c_gains.kp = gains.get('kp', 0.0)
        c_gains.ki = gains.get('ki', 0.0)
        c_gains.kd = gains.get('kd', 0.0)

        cdef const neucode_pid_limits_t* c_limits_ptr = NULL
        cdef neucode_pid_limits_t c_limits
        if limits is not None:
            c_limits.u_min = limits.get('u_min', -1.0)
            c_limits.u_max = limits.get('u_max', 1.0)
            c_limits.i_min = limits.get('i_min', c_limits.u_min)
            c_limits.i_max = limits.get('i_max', c_limits.u_max)
            c_limits.kaw = limits.get('kaw', 0.0)
            c_limits.d_alpha = limits.get('d_alpha', 0.1)
            c_limits_ptr = &c_limits
        status = neucode_sim_set_pid(self._sim, &c_gains, c_limits_ptr)
        if status != NEUCODE_OK:
            raise ValueError("Invalid arguments for setting PID parameters.")

    def set_fopdt(self, params):
        """
        Set the FOPDT plant parameters.
        """
        cdef neucode_fopdt_params_t c_params
        c_params.K = params.get('K', 1.0)
        c_params.tau = params.get('tau', 1.0)
        c_params.theta = params.get('theta', 0.0)
        c_params.friction = params.get('friction', 0.0)

        status = neucode_sim_set_fopdt(self._sim, &c_params)
        if status != NEUCODE_OK:
            raise ValueError("Invalid arguments for setting FOPDT parameters.")

    def set_foipdt(self, params):
        """
        Set the FOIPDT plant parameters.
        K_v is the velocity gain, tau the velocity lag, theta the dead time.
        """
        cdef neucode_foipdt_params_t c_params
        c_params.Kv    = params.get('Kv', 1.0)
        c_params.tau   = params.get('tau', 1.0)
        c_params.theta = params.get('theta', 0.0)
        c_params.friction = params.get('friction', 0.0)

        status = neucode_sim_set_foipdt(self._sim, &c_params)
        if status != NEUCODE_OK:
            raise ValueError("Invalid arguments for setting FOIPDT parameters.")


    def set_setpoint(self, setpoint_config):
        """
        Set the setpoint definition.
        """
        cdef neucode_setpoint_def_t c_sp_def
        sp_type = setpoint_config.get('type', 'step').lower()

        if sp_type == 'step':
            c_sp_def.type = NEUCODE_SP_STEP
            c_sp_def.step_time = setpoint_config.get('step_time', 0.0)
            c_sp_def.v = setpoint_config.get('v', 1.0)
        elif sp_type == 'ramp':
            c_sp_def.type = NEUCODE_SP_RAMP
            c_sp_def.step_time = setpoint_config.get('step_time', 0.0)
            c_sp_def.a = setpoint_config.get('a', 0.0)
            c_sp_def.b = setpoint_config.get('b', 0.0)
            c_sp_def.time = setpoint_config.get('time', 1.0)
        elif sp_type == 'sin':
            c_sp_def.type = NEUCODE_SP_SIN
            c_sp_def.step_time = setpoint_config.get('step_time', 0.0)
            c_sp_def.amp = setpoint_config.get('amp', 1.0)
            c_sp_def.freq = setpoint_config.get('freq', 1.0)
        else:
            raise ValueError("Invalid setpoint type: {}. Must be 'step', 'ramp', or 'sin'.".format(sp_type))

        status = neucode_sim_set_setpoint(self._sim, &c_sp_def)
        if status != NEUCODE_OK:
            raise ValueError("Invalid arguments for setpoint definition.")

    def set_disturbance(self, disturbance_config):
        """
        Set the disturbance configuration.
        """
        if disturbance_config is None:
            status = neucode_sim_set_disturbance(self._sim, NULL)
            if status != NEUCODE_OK:
                raise ValueError("Failed to disable disturbances.")
            return

        cdef neucode_disturbance_config_t c_disturbance

        c_disturbance.enable_input_step = disturbance_config.get('enable_input_step', False)
        c_disturbance.input_step_at_s = disturbance_config.get('input_step_at_s', 0.0)
        c_disturbance.input_step_value = disturbance_config.get('input_step_value', 0.0)
        c_disturbance.enable_output_step = disturbance_config.get('enable_output_step', False)
        c_disturbance.output_step_at_s = disturbance_config.get('output_step_at_s', 0.0)
        c_disturbance.output_step_value = disturbance_config.get('output_step_value', 0.0)
        c_disturbance.noise_amp = disturbance_config.get('noise_amp', 0.0)
        c_disturbance.noise_std = disturbance_config.get('noise_std', 0.0)
        c_disturbance.seed = disturbance_config.get('seed', 0)
        
        noise_type = disturbance_config.get('noise_type', 'none').lower()
        if noise_type == 'none':
            c_disturbance.noise_type = NEUCODE_NOISE_NONE
        elif noise_type == 'uniform':
            c_disturbance.noise_type = NEUCODE_NOISE_UNIFORM
        elif noise_type == 'gaussian':
            c_disturbance.noise_type = NEUCODE_NOISE_GAUSSIAN
        else:
            raise ValueError("Invalid noise type: {}. Must be 'none', 'uniform', or 'gaussian'.".format(noise_type))

        c_disturbance.enable_cogging_sine = disturbance_config.get('enable_cogging_sine', False)
        c_disturbance.cogging_sine_amp = disturbance_config.get('cogging_sine_amp', 0.0)
        c_disturbance.cogging_freq_mult = disturbance_config.get('cogging_freq_mult', 1)

        status = neucode_sim_set_disturbance(self._sim, &c_disturbance)
        if status != NEUCODE_OK:
            raise ValueError("Invalid arguments for disturbance configuration.")

    def set_sensor_filter(self, float alpha):
        """
        Enable IIR low-pass on the measurement signal.

        Matches the firmware sensor filter: y = alpha*raw + (1-alpha)*prev.
        Pass alpha <= 0 to disable.
        """
        neucode_sim_set_sensor_filter(self._sim, alpha)

    def set_metrics(self, metrics_config):
        """
        Set the metrics configuration.
        """
        if metrics_config is None:
            status = neucode_sim_set_metrics(self._sim, NULL)
            if status != NEUCODE_OK:
                raise ValueError("Failed to disable metrics.")
            return

        cdef neucode_metrics_config_t c_metrics

        c_metrics.step_mode = metrics_config.get('step_mode', False)
        c_metrics.step_time = metrics_config.get('step_time', 0.0)
        c_metrics.r_final = metrics_config.get('r_final', 0.0)
        c_metrics.tail_window_s = metrics_config.get('tail_window_s', 1.0)
        c_metrics.max_rate_hz = metrics_config.get('max_rate_hz', 1000.0)
        c_metrics.min_tail_samples = metrics_config.get('min_tail_samples', 5)

        status = neucode_sim_set_metrics(self._sim, &c_metrics)
        if status == NEUCODE_ERROR_ALLOC_FAILED:
            raise MemoryError("Failed to allocate memory for metrics ring buffer.")
        if status != NEUCODE_OK:
            raise ValueError("Invalid arguments for metrics configuration.")

    def run(self,
        np.ndarray[np.float32_t, ndim=1] time_out=None,
        np.ndarray[np.float32_t, ndim=1] sp_out=None,
        np.ndarray[np.float32_t, ndim=1] y_out=None,
        np.ndarray[np.float32_t, ndim=1] u_out=None):
        """
        Run the simulation.
        """
        cdef neucode_simulation_result_t c_result
        cdef neucode_metrics_results_t c_metrics_result
       
        # If time_out is provided, we assume time series output is enabled
        cdef bool time_series_enabled = time_out is not None

        # But better safe than sorry (check all array present + same length)
        if time_series_enabled:
            if time_out is None or sp_out is None or y_out is None or u_out is None:
                raise ValueError("If time_out is provided, all other output arrays (sp_out, y_out, u_out) must also be provided.")
            num_samples = time_out.shape[0]
            if sp_out.shape[0] != num_samples or y_out.shape[0] != num_samples or u_out.shape[0] != num_samples:
                raise ValueError("All time-series output arrays must have the same length.")

        cdef neucode_timeseries_buffers_t c_buffers
        cdef const neucode_timeseries_buffers_t* c_buffers_ptr = NULL

        if time_series_enabled:
            c_buffers.time_out = <float*> time_out.data
            c_buffers.setpoint_out = <float*> sp_out.data
            c_buffers.measurement_out = <float*> y_out.data
            c_buffers.control_effort_out = <float*> u_out.data
            c_buffers.num_samples = num_samples
            c_buffers_ptr = &c_buffers

        status = neucode_sim_run(self._sim, &c_result, c_buffers_ptr)
        if status != NEUCODE_OK:
            raise RuntimeError("Simulation run failed with status code: {}".format(status))

        results = {
            'samples_written': c_result.samples_written,
            'y_final': c_result.y_final
        }

        if status == NEUCODE_OK:
            status = neucode_sim_get_metrics_results(self._sim, &c_metrics_result)
            if status == NEUCODE_OK:
                results.update({
                    'iae': c_metrics_result.iae,
                    'ise': c_metrics_result.ise,
                    'itae': c_metrics_result.itae,
                    'isu': c_metrics_result.isu,
                    'overshoot_percent': c_metrics_result.overshoot_percent,
                    'rise_time': c_metrics_result.rise_time,
                    'peak_value': c_metrics_result.peak_value,
                    'peak_time': c_metrics_result.peak_time,
                    'steady_state_error': c_metrics_result.steady_state_error,
                    'steady_state_error_percent': c_metrics_result.steady_state_error_percent,
                })
        return results

    def get_metrics_results(self):
        """
        Retrieves the final calculated metrics after a simulation run.
        This is typically called after a stepped execution loop.
        """
        cdef neucode_metrics_results_t c_metrics_result
        results = {}

        status = neucode_sim_get_metrics_results(self._sim, &c_metrics_result)
        if status == NEUCODE_OK:
            results.update({
                'iae': c_metrics_result.iae,
                'ise': c_metrics_result.ise,
                'itae': c_metrics_result.itae,
                'isu': c_metrics_result.isu,
                'overshoot_percent': c_metrics_result.overshoot_percent,
                'rise_time': c_metrics_result.rise_time,
                'peak_value': c_metrics_result.peak_value,
                'peak_time': c_metrics_result.peak_time,
                'steady_state_error': c_metrics_result.steady_state_error,
                'steady_state_error_percent': c_metrics_result.steady_state_error_percent,
            })
        return results

    def reset(self):
        """
        Resets the simulation state, including time, plant state, PID state, etc.
        This is necessary before starting a stepped execution loop.
        """
        status = neucode_sim_reset(self._sim)
        if status != NEUCODE_OK:
            raise RuntimeError("Failed to reset simulation state (status={}).".format(status))

    def step(self, float control_variable):
        """
        Advances the simulation by a single time step using an externally
        provided control variable.
        """
        status = neucode_sim_step(self._sim, control_variable)
        if status != NEUCODE_OK:
            raise RuntimeError("Simulation step failed (status={}).".format(status))

    def get_state_vector(self, np.ndarray[np.float32_t, ndim=1] out_state):
        """
        Fills a pre-allocated NumPy array with the current state vector
        [error, measurement, setpoint] needed for the controller's next decision.
        """
        if out_state.shape[0] < 3:
            raise ValueError("Output array for state vector must have a size of at least 3.")

        cdef size_t size_written = neucode_sim_get_state_vector(self._sim, <float*> out_state.data)
        if size_written == 0:
            raise RuntimeError("Failed to get state vector from simulation core.")