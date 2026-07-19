#pragma once 

#include <stdint.h>
#include <stdbool.h>

// Standalone API using plain C implementation
void ann_inference_reset(void);
int8_t ann_inference(float* input_floats);

// CMSIS-NN API (if available) - Used for comparison
void ann_inference_reset_cmsisnn(void);
int8_t ann_generic_inference_cmsisnn(float* input_floats);

// Helper to check if model data is valid (non-dummy)
bool ann_is_model_valid(void);

// Helper to get configuration parameters from model data
float ann_get_output_scale(void);
int ann_get_input_size(void);
int ann_get_output_size(void);

// Integral feature scaler accessors (index 3) - used for training-range clip in ann.c
float ann_get_integral_scaler_mean(void);
float ann_get_integral_scaler_scale(void);
