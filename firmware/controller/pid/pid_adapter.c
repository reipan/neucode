#include "cmd_handler.h"
#include "controller.h"
#include "pid_adapter.h"
#include "pid.h"

//well that explains the bang-bang controller behavior
// #ifndef NC_PID_U_TUNE_CLAMP
//     #define NC_PID_U_TUNE_CLAMP 0.2
// #endif

#ifndef NC_PID_ERR_DEADBAND_DEG
    #define NC_PID_ERR_DEADBAND_DEG 0.15f
#endif

static neucode_pid_t pid;
static bool initialized = false;

/**
 * Initializes the PID controller.
 *
 * @note This function is called internally by the PID adapter.
 */
static void pid_init(void) {
    neucode_pid_gains_t gains = {0};
    neucode_pid_limits_t limits = {
        .u_min=-1.0f,
        .u_max=1.0f,
        .i_min=-1.0f,
        .i_max=1.0f,
        .kaw=0.0f,
        .d_alpha=0.0f
    };
    
    neucode_pid_init(&pid, &gains, &limits);
    initialized = true;
}

/**
 * Resets the PID controller state.
 *
 * @note This function is called internally by the PID adapter.
 */
static void pid_reset(void) {
    if (!initialized) {
        pid_init();
    }
    neucode_pid_reset(&pid);
}

/**
 * Sets the PID controller parameters.
 *
 * @note This function is called internally by the PID adapter.
 *
 * @param params Pointer to the parameters to set.
 * @param params_size Size of the parameters in bytes.
 */
static void pid_set_params(const void *params, size_t params_size) {
    if (!params || params_size < sizeof(nc_comm_cmd_params_t)) {
        return;
    }

    (void)params_size;
    if (!initialized) {
        pid_init();
    }

    // we need to cast this to the expected nc_comm_cmd_params_t
    const nc_comm_cmd_params_t *p = (const nc_comm_cmd_params_t *)params;

    pid.kp = p->kp;
    pid.ki = p->ki;
    pid.kd = p->kd;

    pid.d_alpha = (double)p->d_alpha;
    pid.kaw = (double)p->kaw;

    // clamp to sane limits
    if (pid.d_alpha < 0.0) pid.d_alpha = 0.0;
    if (pid.d_alpha > 1.0) pid.d_alpha = 1.0;
    if (pid.kaw < 0.0) pid.kaw = 0.0;
}

/**
 * Performs a PID control step.
 *
 * @note This function is called internally by the PID adapter.
 *
 * @param sp Setpoint value.
 * @param y Measured value.
 * @param dt Time delta since last step.
 * @return Control output value.
 */
static double pid_step(double sp, double y, double dt) {
    if (!initialized) {
        pid_init();
    }

    double e = sp - y;
    if (e > -NC_PID_ERR_DEADBAND_DEG && e < NC_PID_ERR_DEADBAND_DEG) {
        return 0.0;
    }

    double u = neucode_pid_step(&pid, sp, y, dt);
    // if (u >  NC_PID_U_TUNE_CLAMP) u =  NC_PID_U_TUNE_CLAMP;
    // if (u < -NC_PID_U_TUNE_CLAMP) u = -NC_PID_U_TUNE_CLAMP;

    return u;
}

/**
 * Registers the PID controller with the abstract controller interface.
 *
 * @note This function should be called during system initialization.
 *
 * @see nc_controller_register_mode
 */
void nc_pid_controller_register(void) {
    nc_controller_register_mode(
        NC_CONTROLLER_MODE_PID,
        pid_init,
        pid_reset,
        pid_set_params,
        pid_step
    );
}