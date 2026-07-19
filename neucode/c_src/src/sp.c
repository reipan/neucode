#include "math_constants.h"
#include "neucode_types.h"
#include "sp.h"
#include <string.h>
#include <math.h>

void neucode_sp_init(neucode_sp_t *p, const neucode_setpoint_def_t *def) {
    memset(p, 0, sizeof(neucode_sp_t));
    if (def) {
        p->def = *def;
    }
}

double neucode_sp_eval(const neucode_sp_t* p, double t) {
    switch(p->def.type) {
        case NEUCODE_SP_STEP:
            return (t >= p->def.step_time) ? p->def.v: 0.0;

        case NEUCODE_SP_RAMP:
            if (t < p->def.step_time) return p->def.a;
            if (t >= p->def.step_time + p->def.time) return p->def.b;
            {
                double alpha = (t - p->def.step_time) / p->def.time;
                return p->def.a + alpha * (p->def.b - p->def.a);
            }

        case NEUCODE_SP_SIN:
            if (t < p->def.step_time) return 0.0;
            return p->def.amp * sin(2.0 * M_PI * p->def.freq * (t - p->def.step_time));
    }
    return 0.0;
}