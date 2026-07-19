#include "sensor.h"
#include <stddef.h>

static float (*sensor_read_callback)(void) = NULL;
static float (*sensor_read_raw_callback)(void) = NULL;
static void (*sensor_init_callback)(void) = NULL;
static void (*sensor_zero_callback)(void) = NULL;

/**
 * Sets the initialization and read callbacks for the sensor.
 *
 * This function allows the user to specify custom callback functions for sensor initialization
 * and sensor reading operations.
 *
 * @param init_callback Pointer to a function that initializes the sensor. 
 * @param read_callback Pointer to a function that reads the sensor value.
 */
void nc_sensor_register(
    void (*init_callback)(void),
    float (*read_callback)(void),
    float (*read_raw_callback)(void),
    void (*zero_callback)(void)
) {
    sensor_init_callback = init_callback;
    sensor_read_callback = read_callback;
    sensor_read_raw_callback = read_raw_callback;
    sensor_zero_callback = zero_callback;
}

/**
 * Initializes the sensor hardware and related configurations.
 *
 * @note: Needs to be called once during system startup before using any sensor functions.
 */
void nc_sensor_init(void) {
    if (sensor_init_callback) {
        sensor_init_callback();
    }
}

/**
 * Reads the current value from the sensor.
 *
 * @return The sensor value as a double. If no read callback is set, returns 0.0.
 */
float nc_sensor_read(void) {
    if (sensor_read_callback) {
        return sensor_read_callback();
    }
    return 0.0f;
}

float nc_sensor_read_raw(void) {
    if (sensor_read_raw_callback) {
        return sensor_read_raw_callback();
    }
    return 0.0f;
}

/**
 * Zeros or calibrates the sensor.
 *
 * This function calls the registered zero callback to perform sensor zeroing or calibration.
 */
void nc_sensor_zero(void) {
    if (sensor_zero_callback) {
        sensor_zero_callback();
    }
}