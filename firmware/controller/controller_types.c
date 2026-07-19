#include "controller_types.h"

/**
 * Converts a controller mode enum value to its corresponding string representation.
 *
 * @param mode The controller mode to convert.
 * @return A pointer to a string representing the controller mode.
 */
const char* nc_controller_mode_to_str(nc_controller_mode_t mode) {
    switch (mode) {
        case NC_CONTROLLER_MODE_PID:
            return "PID";
        case NC_CONTROLLER_MODE_ANN:
            return "ANN";
        case NC_CONTROLLER_MODE_SNN:
            return "SNN";
        case NC_CONTROLLER_MODE_OPEN_LOOP:
            return "OPEN_LOOP";
        case NC_CONTROLLER_MODE_SYSID:
            return "SYSID";
        case NC_CONTROLLER_MODE_EXTERNAL:
            return "EXTERNAL";
        default:
            return "UNKNOWN";
    }
}