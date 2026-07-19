#include "sensor.h"
#include "sensor_dummy.h"

static void sensor_dummy_init(void) {
    // Just a stub for now
}

static float sensor_dummy_read(void) {
    return 0.0f;
}

static void sensor_dummy_zero(void) {
    // No-op
}

void nc_sensor_dummy_register(void) {
    nc_sensor_register(
        sensor_dummy_init,
        sensor_dummy_read,
        sensor_dummy_zero
    );
}