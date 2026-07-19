#pragma once

#include <stdint.h>

void nc_snn_controller_register(void);

void snn_reset(void);
void snn_set_output_limits(float u_min, float u_max);
float snn_step(float sp, float y, float dt);
void snn_benchmark(uint32_t n);