#pragma once

#include <stddef.h>
#include <stdbool.h>
#include "controller_types.h"
#include "setpoint.h"

// Control mode parameters structure
typedef struct {
    float kp;
    float ki;
    float kd;
    float sp;
    float d_alpha;
    float kaw;
    nc_controller_mode_t mode;
} nc_comm_cmd_params_t;

void nc_comm_cmd_handle_line(const char *line, size_t len);
const nc_comm_cmd_params_t *nc_comm_cmd_get_params(void);
bool nc_comm_cmd_experiment_running(void);
bool nc_comm_cmd_consume_nozero(void);
const nc_setpoint_def_t* nc_comm_cmd_get_sp_def(void);
const nc_setpoint_def_t* nc_comm_cmd_get_u_def(void);