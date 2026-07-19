#include "comm.h"
#include <string.h>
#include <stdio.h>

#define NC_CMD_BUFFER_SIZE 128

static nc_comm_tx_fn_t nc_comm_tx_callback = NULL;
static nc_comm_cmd_fn_t nc_comm_cmd_callback = NULL;
static uint8_t nc_comm_cmd_buffer[NC_CMD_BUFFER_SIZE];
static size_t nc_comm_cmd_buffer_len = 0;

/**
 * Initializes the communication module by setting up callback functions and buffers.
 *
 * @param tx_callback Pointer to the transmit callback function.
 * @param cmd_callback Pointer to the command callback function.
 */
void nc_comm_init(nc_comm_tx_fn_t tx_callback, nc_comm_cmd_fn_t cmd_callback) {
    nc_comm_tx_callback = tx_callback;
    nc_comm_cmd_callback = cmd_callback;
    nc_comm_cmd_buffer_len = 0;
    nc_comm_cmd_buffer[0] = '\0';
}

/**
 * Transmits a null-terminated string over the communication interface.
 *
 * @param str Pointer to the null-terminated string to transmit.
 */
static void nc_comm_tx_str(const char *str) {
    if (!nc_comm_tx_callback || !str) return;

    size_t len = strlen(str);
    if (len == 0) return;
    nc_comm_tx_callback((const uint8_t *)str, len);
}

/**
 * Sends a log message over the communication interface.
 * Log messages are prefixed with "L," and terminated with a newline.
 *
 * @param msg Pointer to a null-terminated string containing the log message to send.
 */
void nc_comm_send_log(const char *msg) {
    if (!msg) return;

    char buf[160];
    int n = snprintf(buf, sizeof(buf), "L,%s\r\n", msg);
    if (n < 0) {
        return;
    }
    nc_comm_tx_str(buf);
}

/**
 * Sends telemetry data (metrics) over the communication interface.
 * Telemetry messages are prefixed with "T," and contain time, setpoint, output, and control signal values.
 *
 * @param telemetry Pointer to a nc_comm_telemetry_t structure containing the telemetry data to send.
 */
void nc_comm_send_telemetry(const nc_comm_telemetry_t *telemetry) {
    if (!telemetry) return;

    char buf[128];
    int n = snprintf(buf, sizeof(buf), "T,%.6f,%.6f,%.6f,%.6f\r\n",
                     telemetry->t,
                     telemetry->sp,
                     telemetry->y,
                     telemetry->u);
    if (n < 0) {
        return;
    }
    nc_comm_tx_str(buf);
}

/**
 * Processes received bytes from the communication interface.
 *
 * @param data Pointer to the buffer containing received bytes.
 * @param len Number of bytes received in the buffer.
 */
void nc_comm_rx_bytes(const uint8_t *data, size_t len) {
    if (!data || len == 0) return;

    for (size_t i = 0; i < len; i++) {
        // Read char
        char c = (char)data[i];

        // Newline and/or carriage return ends command
        if (c == '\n' || c == '\r') {
            // Terminate command string
            if (nc_comm_cmd_callback && nc_comm_cmd_buffer_len > 0) {
                nc_comm_cmd_callback((const char *)nc_comm_cmd_buffer, nc_comm_cmd_buffer_len);
            }
            // Reset buffer
            nc_comm_cmd_buffer_len = 0;
            nc_comm_cmd_buffer[0] = '\0';
        } else {
            // Add to buffer
            if (nc_comm_cmd_buffer_len < NC_CMD_BUFFER_SIZE - 1) {
                nc_comm_cmd_buffer[nc_comm_cmd_buffer_len++] = c;
                nc_comm_cmd_buffer[nc_comm_cmd_buffer_len] = '\0';
            } else {
                // Buffer overflow, reset
                nc_comm_cmd_buffer_len = 0;
                nc_comm_cmd_buffer[0] = '\0';
            }
        }
    }
}