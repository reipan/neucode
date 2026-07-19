// This is just a dummy file to allow compilation when no exported model data is
// present.
#pragma once

// Actuator limits (volts). Overwritten by SNNExporter with model-specific
// values.
#define NC_SNN_U_MIN -1.0f
#define NC_SNN_U_MAX 1.0f

// EMA smoothing coefficient for the SNN output filter (firmware 1 kHz rate).
#define NC_SNN_OUTPUT_EMA_ALPHA 0.9f