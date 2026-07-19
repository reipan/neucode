#include "loop.h"
#include "neucode_types.h"
#include "comm.h"
#include "setpoint.h"
#include "stdbool.h"
#include "cmd_handler.h"
#include "controller.h"
#include "sensor.h"
#include "actuator.h"
#include <stdint.h>
#include <stdio.h>
#include <math.h>

static nc_comm_telemetry_t telemetry;
static bool prev_exp_running = false;
static double t0 = 0.0;
static uint32_t loop_tick = 0;
static nc_setpoint_def_t sp_active;
static nc_setpoint_def_t u_active;
static int dbg_steps = 0;

// Post-experiment dump buffer - filled every control tick during an experiment.
static nc_loop_frame_t s_buf[NC_LOOP_BUF_SIZE];
static uint16_t s_buf_count = 0;

// Anti-stiction integral: only active for ANN/SNN modes.
// Accumulates when motor is stuck (error large, position not changing).
// Resets when experiment starts or motor is near target.
#if NC_ANTI_STICTION
static float s_stuck_integral = 0.0f;
static float s_y_prev = 0.0f;
#endif

// Send telemetry every N steps
const uint32_t TELEMETRY_INTERVAL_STEPS = 100;

/**
 * Initializes loop module.
 * @note Needs to be called before nc_loop_step.
 */
void nc_loop_init(void) {
    nc_controller_init();
    t0 = 0.0;
    prev_exp_running = false;
    loop_tick = 0;
}

/**
 * Performs a single control step of the main loop.
 * @param now The current time in seconds.
 * @param dt  The time elapsed since the last step, in seconds.
 */
void nc_loop_step(double now, double dt) {
    loop_tick++;
    bool exp_running = nc_comm_cmd_experiment_running();
    const nc_comm_cmd_params_t *params = nc_comm_cmd_get_params();

    if (!exp_running) {
        if (prev_exp_running) {
            // Experiment just ended, set actuator to zero
            nc_actuator_set(0.0f);
        }
        prev_exp_running = false;
        // Keep sensor unwrap accumulator current while idle.
        if (params->mode == NC_CONTROLLER_MODE_SYSID) {
            nc_sensor_read_raw();
        } else {
            nc_sensor_read();
        }
        return;
    }

    // check if a new experiment has started
    if (exp_running && !prev_exp_running) {
        // reset controller and time
        nc_controller_reset();
        if (!nc_comm_cmd_consume_nozero()) {
            nc_sensor_zero();
        }
        t0 = now;
        sp_active = *nc_comm_cmd_get_sp_def();
        u_active = *nc_comm_cmd_get_u_def();
        dbg_steps = 0;
        s_buf_count = 0;  // reset dump buffer for new experiment
#if NC_ANTI_STICTION
        s_stuck_integral = 0.0f;
        s_y_prev         = 0.0f;
#endif

    }
    prev_exp_running = exp_running;

    nc_controller_set_mode(params->mode);
    nc_controller_set_params(params, sizeof(*params));

    double sp = nc_setpoint_eval(&sp_active, (float)(now - t0));

    // SYSID: bypass IIR to capture true mechanical response.
    double y;
    if (params->mode == NC_CONTROLLER_MODE_SYSID) {
        y = (double)nc_sensor_read_raw();
    } else {
        y = (double)nc_sensor_read();
    }

    double u;
    if (params->mode == NC_CONTROLLER_MODE_OPEN_LOOP ||
        params->mode == NC_CONTROLLER_MODE_SYSID) {
        u = nc_setpoint_eval(&u_active, (float)(now - t0));
    } else {
        u = nc_controller_step(sp, y, dt);
    }

#if NC_ANTI_STICTION
    // Anti-stiction integral for ANN: when the motor is stuck despite large
    // error, accumulate a small integral to push through cogging.
    // PID has its own ki; SNN/Akida learn integral via DAgger training.
    // Needs to be enabled on build time (NC_ANTI_STICTION=1)
    if (params->mode == NC_CONTROLLER_MODE_ANN) {
        float error_f    = (float)(sp - y);
        float moved      = (float)(y) - s_y_prev;
        if (moved < 0.0f) moved = -moved;
        if (error_f < 0.0f) error_f = -error_f;
        float signed_err = (float)(sp - y);

        if (error_f > 0.5f && error_f < 44.0f && moved < 0.15f) {
            s_stuck_integral += 0.0016f * signed_err;
            if (s_stuck_integral >  1.0f) s_stuck_integral =  1.0f;
            if (s_stuck_integral < -1.0f) s_stuck_integral = -1.0f;
        } else if (error_f > 0.5f && moved >= 0.15f) {
            s_stuck_integral *= 0.95f;
        } else if (error_f < 0.5f) {
            s_stuck_integral = 0.0f;
        }
        u += (double)s_stuck_integral;
        if (u >  1.0) u =  1.0;
        if (u < -1.0) u = -1.0;
    }
    s_y_prev = (float)y;
#endif


    // Guard: SPI timeout in external mode returns NAN - clamp to zero to protect the driver
    if (!isfinite(u)) u = 0.0;
    nc_actuator_set((float)u);

    // Store frame in dump buffer at ~100 Hz (every 10th tick at 1 kHz).
    if (loop_tick % (NC_CONTROL_LOOP_HZ / 100) == 0 &&
        s_buf_count < NC_LOOP_BUF_SIZE) {
        s_buf[s_buf_count].t  = (float)(now - t0);
        s_buf[s_buf_count].sp = (float)sp;
        s_buf[s_buf_count].y  = (float)y;
        s_buf[s_buf_count].u  = (float)u;
        s_buf_count++;
    }

    // Debug burst: log sp/y/u/raw for first 5 steps of each experiment.
    if (dbg_steps < 5) {
        char dbg[96];
        float raw = nc_sensor_read_raw();
        snprintf(dbg, sizeof(dbg),
                 "info: loop dbg step=%d sp=%.2f y=%.3f u=%.4f raw=%.3f",
                 dbg_steps, (double)sp, (double)y, (double)u, (double)raw);
        nc_comm_send_log(dbg);
        dbg_steps++;
    }

    telemetry.t = (float)(now - t0);
    telemetry.sp = (float)sp;
    telemetry.y = (float)y;
    telemetry.u = (float)u;

    bool send_now = (loop_tick % TELEMETRY_INTERVAL_STEPS == 0);
    if (exp_running && send_now) {
        nc_comm_send_telemetry(&telemetry);
    }
}

/**
 * Returns the dump buffer recorded during the last experiment.
 *
 * @param out_count Set to the number of valid frames in the buffer.
 * @return Pointer to the frame array (valid until next exp start).
 */
const nc_loop_frame_t *nc_loop_get_buffer(uint16_t *out_count) {
    *out_count = s_buf_count;
    return s_buf;
}