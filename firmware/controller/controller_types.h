#pragma once

typedef enum {
    NC_CONTROLLER_MODE_PID = 0,
    NC_CONTROLLER_MODE_ANN = 1,
    NC_CONTROLLER_MODE_SNN = 2,
    NC_CONTROLLER_MODE_OPEN_LOOP = 3,
    NC_CONTROLLER_MODE_SYSID = 4,  // open-loop + raw sensor + 100 Hz telemetry
    NC_CONTROLLER_MODE_EXTERNAL = 5,  // transport-agnostic external inference (e.g. Akida via SPI)
    NC_CONTROLLER_MODE_COUNT
} nc_controller_mode_t;

const char* nc_controller_mode_to_str(nc_controller_mode_t mode);