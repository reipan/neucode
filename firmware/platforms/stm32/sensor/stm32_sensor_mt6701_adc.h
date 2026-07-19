#pragma once

#include <stdbool.h>

// Register
void nc_sensor_mt6701_adc_register(void);

// Raw
unsigned int nc_sensor_mt6701_get_raw_adc(void);
float nc_sensor_mt6701_get_angle_deg_raw(void);

// Filtered
float nc_sensor_mt6701_get_angle_deg_filtered(void);
float nc_sensor_mt6701_get_angle_deg_unwrapped(void);

// Initialization status
bool nc_sensor_mt6701_is_initialized(void);