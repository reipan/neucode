#include "external.h"
#include "controller.h"
#include "controller_types.h"

#include <math.h>    /* isnan() */
#include <string.h>  /* memcpy  */

/* -- Static state ----------------------------------------------------------- */

static nc_external_ctrl_t s_cfg   = {0};
static float              s_last_u = 0.0f;

/* -- Forward declarations --------------------------------------------------- */

static void   ext_init(void);
static void   ext_reset(void);
static void   ext_set_params(const void *params, size_t params_size);
static double ext_step(double sp, double y, double dt);

/* -- Public API ------------------------------------------------------------- */

/**
 * Register a transport and wire up NC_CONTROLLER_MODE_EXTERNAL.
 *
 * Copies cfg into internal storage and registers the controller callbacks.
 * Must be called before nc_loop_init().
 */
void nc_external_ctrl_register(const nc_external_ctrl_t *cfg) {
    memcpy(&s_cfg, cfg, sizeof(nc_external_ctrl_t));

    nc_controller_register_mode(
        NC_CONTROLLER_MODE_EXTERNAL,
        ext_init,
        ext_reset,
        ext_set_params,
        ext_step
    );
}

/* -- Callbacks -------------------------------------------------------------- */

static void ext_init(void) {
    s_last_u = 0.0f;
}

static void ext_reset(void) {
    s_last_u = 0.0f;
}

static void ext_set_params(const void *params, size_t params_size) {
    /* No parameters for the external controller - transport is configured
     * at registration time via nc_external_ctrl_register(). */
    (void)params;
    (void)params_size;
}

/**
 * Perform one control step via the registered transport.
 *
 * Calls exchange(sp, y, ctx). If exchange() returns NAN (timeout or
 * communication error) the last valid output is held - no jerk on dropout.
 *
 * @param sp  Setpoint [same units as y].
 * @param y   Plant measurement.
 * @param dt  Time elapsed since last step (seconds) - passed for completeness;
 *            the external inference engine manages its own timing.
 * @return    Control output u.
 */
static double ext_step(double sp, double y, double dt) {
    (void)dt;

    if (!s_cfg.exchange) {
        return (double)s_last_u;
    }

    float u = s_cfg.exchange((float)sp, (float)y, s_cfg.ctx);

    if (isnan(u)) {
        /* Timeout or error - hold last valid output. */
        return (double)s_last_u;
    }

    s_last_u = u;
    return (double)u;
}
