#include "controller_types.h"
#include "cmd_handler.h"
#include "comm.h"
#include "loop.h"
#include "setpoint.h"
#include "sensor.h"
#include "actuator.h"
#include "ann.h"
#include "snn.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/**
 * Intialize internal command parameters with default values.
 */
static nc_comm_cmd_params_t nc_cmd_params = {
    .kp = 1.0f,
    .ki = 0.0f,
    .kd = 0.0f,
    .sp = 0.0f,
    .d_alpha = 0.1f,
    .kaw = 0.0f,
    .mode = NC_CONTROLLER_MODE_PID
};

/**
 * Default setpoint definition (step at 0s to 0.0).
 */
static nc_setpoint_def_t g_sp = {
    .type = NC_SP_STEP,
    .v = 0.0f,
    .step_time = 0.0f
};

/**
 * Default open-loop input definition u(t) (step at 0s to 0.0).
 * Reuses the same waveform struct as setpoints (step/ramp/sin).
 */
static nc_setpoint_def_t g_u = {
    .type = NC_SP_STEP,
    .v = 0.0f,
    .step_time = 0.0f
};

/**
 * Indicates whether an experiment is currently running.
 */
static bool experiment_running = false;

/**
 * When true, the next experiment start will skip nc_sensor_zero().
 * Consumed and reset automatically on first read.
 */
static bool g_skip_sensor_zero = false;

/**
 * Logs a message for communication command handling.
 *
 * @param msg Pointer to a null-terminated string containing the message to log.
 */
static void nc_comm_cmd_log(const char *msg) {
    nc_comm_send_log(msg);
}

/**
 * Trims leading and trailing whitespace from the input string.
 *
 * @param str Pointer to the input string to be trimmed.
 * @return Pointer to the trimmed string.
 */
static char *nc_comm_cmd_trim_whitespace(char *str) {
    if (!str) return str;

    // Trim leading whitespace
    while (*str == ' ' || *str == '\t' || *str == '\r' || *str == '\n') {
        str++;
    }

    // Trim trailing whitespace
    size_t len = strlen(str);
    while (len > 0) {
        char c = str[len - 1];
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
            str[len - 1] = '\0';
            len--;
        } else {
            break;
        }
    }
    return str;
}

/**
 * Parses a float value from a string.
 *
 * @param str Pointer to the input string containing the float representation.
 * @param out_value Pointer to store the parsed float value.
 * @return 0 on success, -1 on failure.
 */
static int nc_comm_cmd_parse_float(const char *str, float *out_value) {
    if (!str || !out_value) return -1;

    char *end = NULL;
    float v = strtof(str, &end);

    if (end == str) {
        // No conversion performed
        return -1;
    }

    *out_value = v;
    return 0;
}

/**
 * Safely copies a string into a destination buffer with size checking.
 *
 * @param dst Pointer to the destination buffer.
 * @param dst_size Size of the destination buffer.
 * @param src Pointer to the source string to copy.
 */
static void nc_comm_cmd_safe_copy(char *dst, size_t dst_size, const char *src)
{
    if (!dst || dst_size == 0) return;

    if (!src) {
        dst[0] = '\0';
        return;
    }

    size_t len = strlen(src);
    if (len >= dst_size) len = dst_size - 1;

    memcpy(dst, src, len);
    dst[len] = '\0';
}

/**
 * Handles a simple 'ping' command and responds with 'pong'.
 */
static void nc_comm_cmd_handle_ping(void) {
    nc_comm_send_log("pong");
}

/**
 * Handles a 'reset' command by performing a software reset of the MCU.
 *
 * Logs an acknowledgement, flushes any pending UART data with a short busy-wait,
 * then triggers a Cortex-M4 SYSRESETREQ via the SCB AIRCR register.
 */
static void nc_comm_cmd_handle_reset(void) {
    nc_comm_send_log("ok: resetting");
    for (volatile int i = 0; i < 500000; i++) {}
    /* Software reset via Cortex-M4 SCB AIRCR - VECTKEY=0x05FA, SYSRESETREQ=bit2 */
    (*((volatile unsigned int *)0xE000ED0C)) = (0x05FAU << 16U) | (1U << 2U);
    while (1) {}
}

// helper to return setpoint type and its parameters buffer
static void nc_comm_cmd_get_sp_str(char *buf, size_t buf_size) {
    if (!buf || buf_size == 0) return;

    switch (g_sp.type) {
        case NC_SP_STEP:
            snprintf(buf, buf_size, "step time=%.3f v=%.3f",
                     (double)g_sp.step_time,
                     (double)g_sp.v);
            break;
        case NC_SP_RAMP:
            snprintf(buf, buf_size, "ramp time=%.3f duration=%.3f a=%.3f b=%.3f",
                     (double)g_sp.step_time,
                     (double)g_sp.time,
                     (double)g_sp.a,
                     (double)g_sp.b);
            break;
        case NC_SP_SIN:
            snprintf(buf, buf_size, "sin time=%.3f amp=%.3f freq=%.3f",
                     (double)g_sp.step_time,
                     (double)g_sp.amp,
                     (double)g_sp.freq);
            break;
        default:
            snprintf(buf, buf_size, "unknown");
            break;
    }
}

/**
 * Handles a 'show' command to display current parameters.
 */
static void nc_comm_cmd_handle_show(void) {
    char sp_buf[96];
    nc_comm_cmd_get_sp_str(sp_buf, sizeof(sp_buf));

    char buf[128];
    int n = snprintf(buf, sizeof(buf), "mode: %s, params: kp=%.4f, ki=%.4f, kd=%.4f / d_alpha=%.3f, kaw=%.3f / setpoint=%s",
                     nc_controller_mode_to_str(nc_cmd_params.mode),
                     (double)nc_cmd_params.kp,
                     (double)nc_cmd_params.ki,
                     (double)nc_cmd_params.kd,
                     (double)nc_cmd_params.d_alpha,
                     (double)nc_cmd_params.kaw,
                     sp_buf);
    if (n > 0) {
        nc_comm_cmd_log(buf);
    }
}

/**
 * Handler for the 'pid' command.
 *
 * This is a atomic way of setting all three PID parameters at once.
 *
 * @param rest Pointer to the remaining command string after the PID command keyword.
 */
static void nc_comm_cmd_handle_pid(const char *rest)
{
    if (!rest || *rest == '\0') {
        nc_comm_cmd_log("err: pid requires 3 values");
        return;
    }

    if (nc_cmd_params.mode != NC_CONTROLLER_MODE_PID) {
        nc_comm_cmd_log("err: pid ignored (not in PID mode)");
        return;
    }

    char buf[64];
    nc_comm_cmd_safe_copy(buf, sizeof(buf), rest);

    char *s = nc_comm_cmd_trim_whitespace(buf);

    // Tokenize
    char *kp_str = s;
    char *ki_str = NULL;
    char *kd_str = NULL;

    // Find first space
    char *space1 = strchr(s, ' ');
    if (!space1) {
        nc_comm_cmd_log("err: pid requires 3 values");
        return;
    }
    // Null-terminate first token, that is kp_str set now
    *space1 = '\0';

    // Repeat for ki and kd
    ki_str = nc_comm_cmd_trim_whitespace(space1 + 1);
    char *space2 = strchr(ki_str, ' ');
    if (!space2) {
        nc_comm_cmd_log("err: pid requires 3 values");
        return;
    }
    // Null-terminate second token, that is ki_str set now
    *space2 = '\0';
    // kd_str is rest
    kd_str = nc_comm_cmd_trim_whitespace(space2 + 1);

    float kp, ki, kd;
    if (nc_comm_cmd_parse_float(kp_str, &kp) != 0 ||
        nc_comm_cmd_parse_float(ki_str, &ki) != 0 ||
        nc_comm_cmd_parse_float(kd_str, &kd) != 0) {
        nc_comm_cmd_log("err: invalid numeric value in pid");
        return;
    }

    nc_cmd_params.kp = kp;
    nc_cmd_params.ki = ki;
    nc_cmd_params.kd = kd;

    nc_comm_cmd_log("ok: set pid");
}

/**
 * Handles a 'set' command to set individual parameters.
 *
 * @note Supported fields: kp, ki, kd, sp
 * @todo we need to probably add a extra cmd for setpoint types and their parameters and maybe even remove the set cmd completely.
 *
 * @param rest Pointer to the remaining command string after the set command keyword.
 */
static void nc_comm_cmd_handle_set(const char *rest) {
    if (!rest || *rest == '\0') {
        nc_comm_cmd_log("err: set requires field and value");
        return;
    }

    char buf[64];
    nc_comm_cmd_safe_copy(buf, sizeof(buf), rest);

    char *field = buf;

    // Find space separating field and value
    char *value_str = strchr(buf, ' ');
    if (!value_str) {
        nc_comm_cmd_log("err: set requires value");
        return;
    }
    // Null-terminate field string
    *value_str = '\0';
    
    // Move to value string
    value_str++;

    // Trim whitespace for both
    field = nc_comm_cmd_trim_whitespace(field);
    value_str = nc_comm_cmd_trim_whitespace(value_str);

    // Parse float value
    float value = 0.0f;
    if (nc_comm_cmd_parse_float(value_str, &value) != 0) {
        nc_comm_cmd_log("err: invalid numeric value in set");
        return;
    }

    // Validate and set field
    if (strcmp(field, "kp") == 0) {
        nc_cmd_params.kp = value;
        nc_comm_send_log("ok: set kp");
    } else if (strcmp(field, "ki") == 0) {
        nc_cmd_params.ki = value;
        nc_comm_send_log("ok: set ki");
    } else if (strcmp(field, "kd") == 0) {
        nc_cmd_params.kd = value;
        nc_comm_send_log("ok: set kd");
    } else if (strcmp(field, "sp") == 0) {
        nc_cmd_params.sp = value;
        nc_comm_send_log("ok: set sp");
    } else if (strcmp(field, "d_alpha") == 0) {
        nc_cmd_params.d_alpha = value;
        nc_comm_send_log("ok: set d_alpha");
    } else if (strcmp(field, "kaw") == 0) {
        nc_cmd_params.kaw = value;
        nc_comm_send_log("ok: set kaw");
    } else {
        nc_comm_cmd_log("err: unknown field in set");
        return;
    }
}

/**
 * Handles 'exp' commands to control experiment workflow.
 *
 * @note Supported arguments: start, stop
 *
 * @param rest Pointer to a string containing the remaining arguments after the exp command keyword.
 */
static void nc_comm_cmd_handle_exp(const char *rest) {
    if (!rest || *rest == '\0') {
        nc_comm_cmd_log("err: exp requires argument (start|stop|dump)");
        return;
    }

    char buf[32];
    nc_comm_cmd_safe_copy(buf, sizeof(buf), rest);

    char *arg = nc_comm_cmd_trim_whitespace(buf);
    if (strncmp(arg, "start", 5) == 0) {
        g_skip_sensor_zero = (strstr(arg, "nozero") != NULL);
        experiment_running = true;
        if (g_skip_sensor_zero) {
            nc_comm_send_log("ok: experiment started (nozero)");
        } else {
            nc_comm_send_log("ok: experiment started");
        }
    } else if (strcmp(arg, "stop") == 0) {
        experiment_running = false;
        nc_comm_send_log("ok: experiment stopped");
    } else if (strcmp(arg, "dump") == 0) {
        /* Stream the dump buffer as T, telemetry lines so Python can ingest
         * them with the existing TelemetryLine parser. */
        uint16_t count = 0;
        const nc_loop_frame_t *frames = nc_loop_get_buffer(&count);
        char hdr[48];
        snprintf(hdr, sizeof(hdr), "ok: exp dump %u frames", (unsigned int)count);
        nc_comm_send_log(hdr);
        nc_comm_telemetry_t t_frame;
        for (uint16_t i = 0; i < count; i++) {
            t_frame.t  = frames[i].t;
            t_frame.sp = frames[i].sp;
            t_frame.y  = frames[i].y;
            t_frame.u  = frames[i].u;
            nc_comm_send_telemetry(&t_frame);
        }
        nc_comm_send_log("ok: dump complete");
    } else {
        nc_comm_cmd_log("err: invalid argument for exp (start|stop|dump)");
    }
}

/**
 * Handles 'mode' commands to set the controller mode.
 *
 * @note Supported arguments: pid, ann, snn, open, sysid, external
 *
 * @param rest Pointer to a string containing the remaining arguments after the mode command keyword.
 */
static void nc_comm_cmd_handle_mode(const char *rest) {
    if (!rest || *rest == '\0') {
        nc_comm_cmd_log("err: mode requires argument (pid|ann|snn|open|sysid|external)");
        return;
    }

    char buf[32];
    nc_comm_cmd_safe_copy(buf, sizeof(buf), rest);

    char *arg = nc_comm_cmd_trim_whitespace(buf);
    if (strcmp(arg, "pid") == 0) {
        // Set mode to PID
        nc_cmd_params.mode = NC_CONTROLLER_MODE_PID;
        nc_comm_send_log("ok: mode set to PID");
    } else if (strcmp(arg, "ann") == 0) {
        // Set mode to ANN
        nc_cmd_params.mode = NC_CONTROLLER_MODE_ANN;
        nc_comm_send_log("ok: mode set to ANN");
    } else if (strcmp(arg, "snn") == 0) {
        // Set mode to SNN
        nc_cmd_params.mode = NC_CONTROLLER_MODE_SNN;
        nc_comm_send_log("ok: mode set to SNN");
    } else if (strcmp(arg, "open") == 0) {
        // Set mode to OPEN_LOOP
        nc_cmd_params.mode = NC_CONTROLLER_MODE_OPEN_LOOP;
        nc_comm_send_log("ok: mode set to OPEN_LOOP");
    } else if (strcmp(arg, "sysid") == 0) {
        // System identification: open-loop control + raw (unfiltered) sensor output
        // + 100 Hz telemetry (every controller tick, no decimation).
        // Use: mode sysid -> u step <t> <v> -> exp start
        nc_cmd_params.mode = NC_CONTROLLER_MODE_SYSID;
        nc_comm_send_log("ok: mode set to SYSID (raw sensor, 100 Hz telemetry)");
    } else if (strcmp(arg, "external") == 0) {
        nc_cmd_params.mode = NC_CONTROLLER_MODE_EXTERNAL;
        nc_comm_send_log("ok: mode set to EXTERNAL (SPI inference)");
    } else {
        nc_comm_cmd_log("err: invalid argument for mode (pid|ann|snn|open|sysid|external)");
    }
}

/**
 * Handles 'u' commands to define the open-loop input waveform u(t).
 *
 * Syntax mirrors the 'sp' command:
 * - u step <step_time_s> <value>
 * - u ramp <step_time_s> <duration_s> <a> <b>
 * - u sin  <step_time_s> <amp> <freq>
 */
static void nc_comm_cmd_handle_u(const char *rest) {
    if (!rest || *rest == '\0') {
        nc_comm_cmd_log("err: u requires args (type and params)");
        return;
    }

    char buf[96];
    nc_comm_cmd_safe_copy(buf, sizeof(buf), rest);

    char *rest_str = nc_comm_cmd_trim_whitespace(buf);
    // Parse type
    char *type_str = strtok(rest_str, " ");
    if (!type_str) {
        nc_comm_cmd_log("err: u requires type");
        return;
    }

    if (strcmp(type_str, "step") == 0) {
        g_u.type = NC_SP_STEP;
        char *t_str = strtok(NULL, " ");
        char *v_str = strtok(NULL, " ");

        float step_time = 0.0f;
        float v = 0.0f;
        if (!t_str || nc_comm_cmd_parse_float(t_str, &step_time) ||
            !v_str || nc_comm_cmd_parse_float(v_str, &v)) {
            nc_comm_cmd_log("err: u step requires step_time and v");
            return;
        }

        if (step_time < 0.0f) {
            nc_comm_cmd_log("err: u step_time must be >= 0");
            return;
        }

        g_u.step_time = step_time;
        g_u.v = v;
        nc_comm_cmd_log("ok: u step set");
        return;
    } else if (strcmp(type_str, "ramp") == 0) {
        g_u.type = NC_SP_RAMP;
        char *t_str = strtok(NULL, " ");
        char *duration_str = strtok(NULL, " ");
        char *a_str = strtok(NULL, " ");
        char *b_str = strtok(NULL, " ");

        float step_time = 0.0f;
        float duration = 0.0f;
        float a = 0.0f;
        float b = 0.0f;

        if (!t_str || nc_comm_cmd_parse_float(t_str, &step_time) ||
            !duration_str || nc_comm_cmd_parse_float(duration_str, &duration) ||
            !a_str || nc_comm_cmd_parse_float(a_str, &a) ||
            !b_str || nc_comm_cmd_parse_float(b_str, &b)) {
            nc_comm_cmd_log("err: u ramp requires step_time, duration, a, b");
            return;
        }

        if (step_time < 0.0f || duration <= 0.0f) {
            nc_comm_cmd_log("err: u ramp step_time must be >= 0 and duration > 0");
            return;
        }

        g_u.step_time = step_time;
        g_u.time = duration;
        g_u.a = a;
        g_u.b = b;
        nc_comm_cmd_log("ok: u ramp set");
        return;
    } else if (strcmp(type_str, "sin") == 0) {
        g_u.type = NC_SP_SIN;
        char *t_str = strtok(NULL, " ");
        char *amp_str = strtok(NULL, " ");
        char *freq_str = strtok(NULL, " ");

        float step_time = 0.0f;
        float amp = 0.0f;
        float freq = 0.0f;

        if (!t_str || nc_comm_cmd_parse_float(t_str, &step_time) ||
            !amp_str || nc_comm_cmd_parse_float(amp_str, &amp) ||
            !freq_str || nc_comm_cmd_parse_float(freq_str, &freq)) {
            nc_comm_cmd_log("err: u sin requires step_time, amp, freq");
            return;
        }

        if (step_time < 0.0f || freq <= 0.0f) {
            nc_comm_cmd_log("err: u sin step_time must be >= 0 and freq > 0");
            return;
        }

        g_u.step_time = step_time;
        g_u.amp = amp;
        g_u.freq = freq;
        nc_comm_cmd_log("ok: u sin set");
        return;
    } else {
        nc_comm_cmd_log("err: u unknown type (step|ramp|sin)");
        return;
    }
}

static void nc_comm_cmd_handle_sp(const char *rest) {
    if (!rest || *rest == '\0') {
        nc_comm_cmd_log("err: sp requires args (type and params)");
        return;
    }

    char buf[96];
    nc_comm_cmd_safe_copy(buf, sizeof(buf), rest);

    char *rest_str = nc_comm_cmd_trim_whitespace(buf);
    // Parse type
    char *type_str = strtok(rest_str, " ");
    if (!type_str) {
        nc_comm_cmd_log("err: sp requires type");
        return;
    }

    if (strcmp(type_str, "step") == 0) {
        g_sp.type = NC_SP_STEP;
        // Parse step parameters here
        char *t_str = strtok(NULL, " ");
        char *v_str = strtok(NULL, " ");

        float step_time = 0.0f;
        float v = 0.0f;
        if (!t_str || nc_comm_cmd_parse_float(t_str, &step_time) ||
            !v_str || nc_comm_cmd_parse_float(v_str, &v)) {
            nc_comm_cmd_log("err: sp step requires step_time and v");
            return;
        }

        if (step_time < 0.0f) {
            nc_comm_cmd_log("err: sp step_time must be >= 0");
            return;
        }

        g_sp.step_time = step_time;
        g_sp.v = v;
        nc_comm_cmd_log("ok: sp step set");
        return;
    } else if (strcmp(type_str, "ramp") == 0) {
        g_sp.type = NC_SP_RAMP;
        // Parse ramp parameters here
        char *t_str = strtok(NULL, " ");
        char *duration_str = strtok(NULL, " ");
        char *a_str = strtok(NULL, " ");
        char *b_str = strtok(NULL, " ");

        float step_time = 0.0f;
        float duration = 0.0f;
        float a = 0.0f;
        float b = 0.0f;

        if (!t_str || nc_comm_cmd_parse_float(t_str, &step_time) ||
            !duration_str || nc_comm_cmd_parse_float(duration_str, &duration) ||
            !a_str || nc_comm_cmd_parse_float(a_str, &a) ||
            !b_str || nc_comm_cmd_parse_float(b_str, &b)) {
            nc_comm_cmd_log("err: sp ramp requires step_time, duration, a, b");
            return;
        }

        if (step_time < 0.0f || duration <= 0.0f) {
            nc_comm_cmd_log("err: sp ramp step_time must be >= 0 and duration > 0");
            return;
        }

        g_sp.step_time = step_time;
        g_sp.time = duration;
        g_sp.a = a;
        g_sp.b = b;
        nc_comm_cmd_log("ok: sp ramp set");
        return;
    } else if (strcmp(type_str, "sin") == 0) {
        g_sp.type = NC_SP_SIN;
        // Parse sine parameters here
        char *t_str = strtok(NULL, " ");
        char *amp_str = strtok(NULL, " ");
        char *freq_str = strtok(NULL, " ");

        float step_time = 0.0f;
        float amp = 0.0f;
        float freq = 0.0f;

        if (!t_str || nc_comm_cmd_parse_float(t_str, &step_time) ||
            !amp_str || nc_comm_cmd_parse_float(amp_str, &amp) ||
            !freq_str || nc_comm_cmd_parse_float(freq_str, &freq)) {
            nc_comm_cmd_log("err: sp sin requires step_time, amp, freq");
            return;
        }

        if (step_time < 0.0f || freq <= 0.0f) {
            nc_comm_cmd_log("err: sp sin step_time must be >= 0 and freq > 0");
            return;
        }

        g_sp.step_time = step_time;
        g_sp.amp = amp;
        g_sp.freq = freq;
        nc_comm_cmd_log("ok: sp sin set");
        return;
    } else {
        nc_comm_cmd_log("err: sp unknown type (step|ramp|sin)");
        return;
    }
}

/**
 * Handles a 'bench' command that measures inference latency using the DWT cycle counter.
 *
 * Syntax: bench ann <N> | bench snn <N>
 *
 * @param rest Pointer to the remaining command string after the 'bench' keyword.
 */
static void nc_comm_cmd_handle_bench(const char *rest) {
    if (!rest || *rest == '\0') {
        nc_comm_cmd_log("err: bench requires controller and count (bench ann 1000 | bench snn 1000)");
        return;
    }

    char buf[32];
    nc_comm_cmd_safe_copy(buf, sizeof(buf), rest);
    char *s = nc_comm_cmd_trim_whitespace(buf);

    char *space = strchr(s, ' ');
    if (!space) {
        nc_comm_cmd_log("err: bench requires count (bench ann 1000 | bench snn 1000)");
        return;
    }
    *space = '\0';
    char *count_str = nc_comm_cmd_trim_whitespace(space + 1);

    float fv = 0.0f;
    if (nc_comm_cmd_parse_float(count_str, &fv) != 0 || fv < 1.0f) {
        nc_comm_cmd_log("err: bench count must be a positive integer");
        return;
    }
    uint32_t n = (uint32_t)fv;

    if (strcmp(s, "ann") == 0) {
        ann_benchmark(n);
    } else if (strcmp(s, "snn") == 0) {
        snn_benchmark(n);
    } else {
        nc_comm_cmd_log("err: bench controller must be ann or snn");
    }
}

/**
 * Handles a 'sensor' command - one-shot read of the current sensor value.
 *
 * Useful for diagnosing ADC/DMA without starting a full experiment.
 * Responds with: L,sensor: <angle_deg> deg
 */
static void nc_comm_cmd_handle_sensor(void) {
    float angle = nc_sensor_read();
    char buf[48];
    snprintf(buf, sizeof(buf), "sensor: %.3f deg", (double)angle);
    nc_comm_cmd_log(buf);
}

/**
 * Handles a single command line received by the communication interface.
 *
 * This function processes the input command line, parses it, and performs
 * the corresponding action based on the command received.
 *
 * @note Implemented commands: ping, show, pid
 *
 * @param line Pointer to the command line string.
 * @param len Length of the command line string.
 */
void nc_comm_cmd_handle_line(const char *line, size_t len) {
    if (!line || !len) return;

    // Prepare a buffer, make sure line content fits, copy into it and null-terminate
    char buf[64];
    nc_comm_cmd_safe_copy(buf, sizeof(buf), line);
    
    // Trim whitespaces and bail out if empty
    char *s = nc_comm_cmd_trim_whitespace(buf);
    if (*s == '\0') {
        return; // Empty line
    }

    // Split into first token and rest
    char *space_pos = strchr(s, ' ');
    char *cmd = s;
    char *rest = NULL;
    if (space_pos) {
        *space_pos = '\0';
        rest = space_pos + 1;
        rest = nc_comm_cmd_trim_whitespace(rest);
    }

    // Handle commands
    if (strcmp(cmd, "ping") == 0) {
        nc_comm_cmd_handle_ping();
    } else if (strcmp(cmd, "show") == 0) {
        nc_comm_cmd_handle_show();
    } else if (strcmp(cmd, "set") == 0) {
        nc_comm_cmd_handle_set(rest);
    } else if (strcmp(cmd, "pid") == 0) {
        nc_comm_cmd_handle_pid(rest);
    } else if (strcmp(cmd, "exp") == 0) {
        nc_comm_cmd_handle_exp(rest);
    } else if (strcmp(cmd, "mode") == 0) {
        nc_comm_cmd_handle_mode(rest);
    } else if (strcmp(cmd, "sp") == 0) {
        nc_comm_cmd_handle_sp(rest);
    } else if (strcmp(cmd, "u") == 0) {
        nc_comm_cmd_handle_u(rest);
    } else if (strcmp(cmd, "bench") == 0) {
        nc_comm_cmd_handle_bench(rest);
    } else if (strcmp(cmd, "sensor") == 0) {
        nc_comm_cmd_handle_sensor();
    } else if (strcmp(cmd, "reset") == 0) {
        nc_comm_cmd_handle_reset();
    } else {
        nc_comm_cmd_log("ERR,Unknown command");
    }
}

/**
 * Retrieves the current command parameters.
 *
 * @return Pointer to the current command parameters structure.
 */
const nc_comm_cmd_params_t *nc_comm_cmd_get_params(void) {
    return &nc_cmd_params;
}

/**
 * Indicates whether an experiment is currently running.
 *
 * @return true if an experiment is running, false otherwise.
 */
bool nc_comm_cmd_experiment_running(void) {
    return experiment_running;
}

/**
 * Returns whether the next experiment start should skip sensor zeroing.
 * Consuming this flag resets it - subsequent exp starts will zero again.
 */
bool nc_comm_cmd_consume_nozero(void) {
    bool val = g_skip_sensor_zero;
    g_skip_sensor_zero = false;
    return val;
}

/**
 * Retrieves the current setpoint definition.
 *
 * @return Pointer to the current setpoint definition structure.
 */
const nc_setpoint_def_t* nc_comm_cmd_get_sp_def(void) {
    return &g_sp;
}

const nc_setpoint_def_t* nc_comm_cmd_get_u_def(void) {
    return &g_u;
}