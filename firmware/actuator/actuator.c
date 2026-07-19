#include "actuator.h"
#include <stddef.h>

static void (*actuator_init_callback)(void)        = NULL;
static void (*actuator_enable_callback)(bool enable) = NULL;
static void (*actuator_set_callback)(float u)      = NULL;
static void (*actuator_loop_callback)(void)        = NULL;

/**
 * Sets the callback functions for the actuator module.
 *
 * loop_callback is optional (pass NULL if not needed). When provided it is
 * called by nc_actuator_loop() on every main loop() iteration to keep the
 * inner FOC commutation running at the highest possible rate, independently
 * of the slower control-decision tick.
 *
 * @param init_callback   Initialises the actuator hardware.
 * @param enable_callback Enables or disables the actuator output.
 * @param set_callback    Applies a new control value u.
 * @param loop_callback   Fast inner-loop tick (e.g. SimpleFOC loopFOC()).
 */
void nc_actuator_register(
    void (*init_callback)(void),
    void (*enable_callback)(bool enable),
    void (*set_callback)(float u),
    void (*loop_callback)(void)
) {
    actuator_init_callback   = init_callback;
    actuator_enable_callback = enable_callback;
    actuator_set_callback    = set_callback;
    actuator_loop_callback   = loop_callback;
}

/**
 * Initializes the actuator module.
 *
 * @note Needs to be called once during system startup before using any actuator functions.
 */
void nc_actuator_init(void) {
    if (actuator_init_callback) {
        actuator_init_callback();
    }
}

/**
 * Enables or disables the actuator.
 *
 * @param enable True to enable the actuator, false to disable it.
 */
void nc_actuator_enable(bool enable) {
    if (actuator_enable_callback) {
        actuator_enable_callback(enable);
    }
}

/**
 * Sets the control signal for the actuator.
 *
 * @param u The control signal value to be set.
 */
void nc_actuator_set(float u) {
    if (actuator_set_callback) {
        actuator_set_callback(u);
    }
}

/**
 * Fast inner-loop tick - call every main loop() iteration.
 *
 * Dispatches to the registered loop_callback (e.g. SimpleFOC loopFOC())
 * so the FOC commutation runs at the highest possible rate, decoupled from
 * the slower control-decision rate governed by the hardware timer ISR.
 * No-op if no loop_callback was registered.
 */
void nc_actuator_loop(void) {
    if (actuator_loop_callback) {
        actuator_loop_callback();
    }
}