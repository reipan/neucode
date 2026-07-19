#pragma once

// structs are duplicates (just prefix changed to not popluate namespace)
// of the simcore definition for setpoint in neucode_types.h
typedef enum {
    NC_SP_STEP = 0,
    NC_SP_RAMP = 1,
    NC_SP_SIN  = 2
} nc_setpoint_type_t;

typedef struct {
    nc_setpoint_type_t type;
    float step_time;   // s (relative to exp start)
    float v;           // step value
    float a, b;        // ramp start/end
    float time;        // ramp duration
    float amp;         // sine amplitude
    float freq;        // sine frequency (Hz)
} nc_setpoint_def_t;

float nc_setpoint_eval(const nc_setpoint_def_t *def, float t_since_exp_start);