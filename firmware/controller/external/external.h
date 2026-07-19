#pragma once

#include "neucode_types.h"

/**
 * Register a transport and wire up NC_CONTROLLER_MODE_EXTERNAL.
 *
 * Must be called before nc_loop_init(). The cfg struct is copied internally;
 * the caller does not need to keep it alive after this call.
 *
 * @param cfg  Pointer to the populated transport descriptor.
 */
void nc_external_ctrl_register(const nc_external_ctrl_t *cfg);
