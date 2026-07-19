#pragma once

#include <stdbool.h>

void nc_sensor_init(void);
float nc_sensor_read(void); // IIR-filtered - use in controller
float nc_sensor_read_raw(void); // unfiltered - use in FOC inner loop
void nc_sensor_zero(void);

void nc_sensor_register(
    void (*init_callback)(void),
    float (*read_callback)(void),
    float (*read_raw_callback)(void), // unfiltered, for FOC inner loop
    void (*zero_callback)(void)
);