#pragma once

#include <stdbool.h>

void nc_actuator_init(void);
void nc_actuator_enable(bool enable);
void nc_actuator_set(float u);
void nc_actuator_loop(void);   /* call every main loop() for fast inner FOC tick */

void nc_actuator_register(
    void (*init_callback)(void),
    void (*enable_callback)(bool enable),
    void (*set_callback)(float u),
    void (*loop_callback)(void)  /* optional: NULL if not needed */
);