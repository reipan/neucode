#pragma once

#include <stddef.h>
#include "controller_types.h"

void nc_controller_init(void);
void nc_controller_reset(void);
void nc_controller_set_mode(nc_controller_mode_t mode);
nc_controller_mode_t nc_controller_get_mode(void);
void nc_controller_set_params(const void *params, size_t params_size);
double nc_controller_step(double sp, double y, double dt);

void nc_controller_register_mode(
    nc_controller_mode_t mode,
    void (*init_callback)(void),
    void (*reset_callback)(void),
    void (*set_params_callback)(const void *params, size_t params_size),
    double(*step_callback)(double sp, double y, double dt)
);