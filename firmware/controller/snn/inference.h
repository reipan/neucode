#pragma once

#include <stdint.h>
#include <stdbool.h>

// We currently use 5 + 1 (4 context and 1 spike) so this should be enough
#define SNN_MAX_INPUT_SIZE 32

// Standalone API using plain C implementation
void snn_inference_reset(void);
int32_t snn_inference(int32_t* inputs);

// Helper to get output scale factor
float snn_get_output_scale_factor(void);

// Helper to check if model data is valid (non-dummy)
bool snn_is_model_valid(void);

// Helper to get configuration parameters from model data
int snn_get_context_size(void);
int snn_get_spike_size(void);
int snn_get_input_frac_bits(void);
int32_t snn_get_encoder_delta_threshold(void);
float snn_get_output_scale(void);
float snn_get_integral_window(void);
float snn_get_windup_limit(void);
