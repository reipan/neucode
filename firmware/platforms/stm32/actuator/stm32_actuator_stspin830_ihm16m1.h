#pragma once

#include <stdint.h>

void nc_actuator_stspin830_ihm16m1_register(void);
float nc_actuator_stspin830_ihm16m1_calibrate_offset(float amp, uint32_t settle_ms);