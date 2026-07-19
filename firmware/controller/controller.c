#include "controller.h"

// Necessary to allow switching between controllers
typedef struct {
    void (*init)(void);
    void (*reset)(void);
    void (*set_params)(const void *params, size_t params_size);
    double (*step)(double sp, double y, double dt);
    int initialized;
} nc_controller_impl_t;

static nc_controller_impl_t available_controllers[NC_CONTROLLER_MODE_COUNT] = {0};
static nc_controller_mode_t active_controller = NC_CONTROLLER_MODE_PID;

/**
 * Retrieves the currently active controller implementation.
 *
 * @return Pointer to the active controller implementation structure.
 */
static nc_controller_impl_t* get_active_controller(void) {
    return &available_controllers[(int)active_controller];
}

/**
 * Forces initialization of the controller if not already initialized.
 *
 * @param impl Pointer to the controller implementation structure.
 */
static void force_init(nc_controller_impl_t *impl) {
    if (!impl->initialized && impl->init) {
        impl->init();
        impl->initialized = 1;
    }
}

/**
 * Registers controller callbacks.
 * 
 * @param mode The controller mode to register.
 * @param init_callback Function to initialize the controller.
 * @param reset_callback Function to reset the controller state.
 * @param set_params_callback Function to set controller parameters.
 * @param step_callback Function to perform a control step.
 */
void nc_controller_register_mode(
    nc_controller_mode_t mode,
    void (*init_callback)(void),
    void (*reset_callback)(void),
    void (*set_params_callback)(const void *params, size_t params_size),
    double(*step_callback)(double sp, double y, double dt)  
) {
    if (mode >= NC_CONTROLLER_MODE_COUNT) {
        return;
    }

    available_controllers[mode] = (nc_controller_impl_t){
        .init = init_callback,
        .reset = reset_callback,
        .set_params = set_params_callback,
        .step = step_callback,
        .initialized = 0
    };
}

/**
 * Initializes the controller by calling the registered init callback.
 */
void nc_controller_init(void) {
    nc_controller_impl_t *impl = get_active_controller();
    force_init(impl);
}

/**
 * Resets the controller state by calling the registered reset callback.
 */
void nc_controller_reset(void) {
    nc_controller_impl_t *impl = get_active_controller();
    force_init(impl);

    if (impl->reset) {
        impl->reset();
    }
}

/**
 * Sets the controller mode.
 *
 * @param mode The desired controller mode.
 */
void nc_controller_set_mode(nc_controller_mode_t mode) {
    if (mode < 0 || mode >= NC_CONTROLLER_MODE_COUNT) {
        return;
    }

    if (active_controller == mode) {
        return;
    }

    active_controller = mode;

    // pointer to the controller implementation to use
    nc_controller_impl_t *impl = &available_controllers[active_controller];

    // run initialization if not yet done
    force_init(impl);
    
    // reset for safety reasons
    if (impl->reset) {
        impl->reset();
    }
}

/**
 * Gets the current controller mode.
 *
 * @return The current controller mode.
 */
nc_controller_mode_t nc_controller_get_mode(void)
{
    return active_controller;
}

/**
 * Sets the controller parameters by calling the registered set_params callback.
 *
 * @param params Pointer to the parameters to set.
 * @param params_size Size of the parameters in bytes.
 */
void nc_controller_set_params(const void *params, size_t params_size) {
    nc_controller_impl_t *impl = get_active_controller();
    force_init(impl);

    if (impl->set_params) {
        impl->set_params(params, params_size);
    }
}

/**
 * Performs a control step by calling the registered step callback.
 *
 * @param sp Setpoint value.
 * @param y Measured value.
 * @param dt Time delta since last step.
 * @return Control output value.
 */
double nc_controller_step(double sp, double y, double dt)
{
    nc_controller_impl_t *impl = get_active_controller();
    force_init(impl);

    if (impl->step) {
        return impl->step(sp, y, dt);
    }
    return 0.0;
}
