#pragma once

#include <stdint.h>

void nc_ann_controller_register(void);

// Functions exposed for direct testing use when compiling with -DSKIP_CONTROLLER_REGISTRATION.
void ann_reset(void);
void ann_set_output_limits(float u_min, float u_max);
double ann_step(double sp, double y, double dt);
void ann_benchmark(uint32_t n);