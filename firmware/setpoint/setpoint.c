#include "setpoint.h"
#include <math.h>

/**
 * Evaluates the setpoint definition at a given time since experiment start.
 *
 * @note Exact logical duplicate of neucode_sp_eval in neucode/src/sp.c
 *
 * @param def Pointer to the setpoint definition structure.
 * @param t_since_exp_start Time since experiment start in seconds.
 * @return The evaluated setpoint value.
 */
float nc_setpoint_eval(const nc_setpoint_def_t *def, float t_since_exp_start) {
    if (!def) return 0.0f;

    switch (def->type) {
        case NC_SP_STEP:
            return (t_since_exp_start >= def->step_time) ? def->v : 0.0f;
        case NC_SP_RAMP:
            if (t_since_exp_start < def->step_time) return def->a;
            if (t_since_exp_start >= def->step_time + def->time) return def->b;
            {
                float alpha = (t_since_exp_start - def->step_time) / def->time;
                return def->a + alpha * (def->b - def->a);
            }
        case NC_SP_SIN:
            if (t_since_exp_start < def->step_time) return 0.0f;
            return def->amp * sinf(2.0f * 3.14159265f * def->freq * (t_since_exp_start - def->step_time));
        default:
            return 0.0f;
    }
}