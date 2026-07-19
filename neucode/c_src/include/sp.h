#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include "neucode_types.h"

typedef struct {
    neucode_setpoint_def_t def;
} neucode_sp_t;

/* Initialize setpoint struct from public definition */
void neucode_sp_init(neucode_sp_t* p, const neucode_setpoint_def_t* def);

/* Evaluate setpoint at time t */
double neucode_sp_eval(const neucode_sp_t* sp, double t);

#ifdef __cplusplus
}
#endif