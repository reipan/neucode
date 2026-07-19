#include "actuator.h"
#include "actuator_dummy.h"

static bool actuator_dummy_enabled = false;

static void actuator_dummy_init(void) {
    // Just a stub for now
}

static void actuator_dummy_enable(bool enable) {
    actuator_dummy_enabled = enable;
}

static void actuator_dummy_set(float u) {
    if (actuator_dummy_enabled) {
        // Just a stub for now
        return;
    }

    (void)u;  // Suppress unused variable warning
}

void nc_actuator_dummy_register(void) {
    nc_actuator_register(
        actuator_dummy_init,
        actuator_dummy_enable,
        actuator_dummy_set
    );
}

