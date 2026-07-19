#pragma once

#include <stdint.h>
#include <stddef.h>

typedef struct {
    float t;
    float sp;
    float y;
    float u;
} nc_comm_telemetry_t;

// Callback function types
typedef void (*nc_comm_tx_fn_t)(const uint8_t *data, size_t len);
typedef void (*nc_comm_cmd_fn_t)(const char *cmd, size_t len);

// Initialize the communication module
void nc_comm_init(nc_comm_tx_fn_t tx_cb, nc_comm_cmd_fn_t cmd_cb);

// The "sink" for incoming bytes
void nc_comm_rx_bytes(const uint8_t *data, size_t len);

// Send telemetry data
void nc_comm_send_telemetry(const nc_comm_telemetry_t *telemetry);

// Send log message
void nc_comm_send_log(const char *msg);