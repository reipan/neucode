// This is just a dummy file to allow compilation when no exported model data is present.
#pragma once
#include <stdint.h>
#include <stdbool.h>

// Model Architecture
#define NUM_LAYERS 3
#define INPUT_CONTEXT_SIZE 1
#define INPUT_SPIKE_SIZE 1
#define HIDDEN_SIZE 1

// Dummy MinMaxScaler Parameters
#define SCALER_DIM INPUT_CONTEXT_SIZE
const float SCALER_MIN[SCALER_DIM] = {0.0f};
const float SCALER_RANGE[SCALER_DIM] = {1.0f};

// Quantization and Safe Limits
#define INPUT_FRAC_BITS 0
#define ENCODER_DELTA_THRESHOLD 2147483647 // Max Int: Encoder never generates spikes
#define OUTPUT_WEIGHT_SHIFT 0
#define OUTPUT_SCALE 0.0f
#define LIF_BETA_MULTIPLIER 0 // No decay
#define LIF_BETA_SHIFT 0
#define LIF_THRESHOLD 2147483647 // Max Int: Neuron never fires
// No derivative clipping in dummy state (FLT_MAX = effectively disabled)
#define DERIV_CLIP_VALUE 3.40282347e+38f

// Per-feature clip bounds (effectively disabled in dummy state)
const float FEATURE_CLIP_MIN[SCALER_DIM] = {-3.40282347e+38f};
const float FEATURE_CLIP_MAX[SCALER_DIM] = {3.40282347e+38f};

// Dummy Weights and Biases (Zeroed)
// Layer: fc_context (standard) to int8 weights
static const int8_t DUMMY_CONTEXT_WEIGHTS[1] = {0};
static const int32_t DUMMY_CONTEXT_BIAS[1] = {0};

// Layer: fc_input_spikes (input_spikes) to int32 weights
static const int32_t DUMMY_SPIKE_WEIGHTS[1] = {0};

// Layer: output_scale (standard) to int8 weights
static const int8_t DUMMY_OUTPUT_WEIGHTS[1] = {0};

// Pointer Tables
static const int8_t* const WEIGHT_POINTERS[NUM_LAYERS] = {
    DUMMY_CONTEXT_WEIGHTS,
    (const int8_t*)DUMMY_SPIKE_WEIGHTS,
    DUMMY_OUTPUT_WEIGHTS
};

static const int32_t* const BIAS_POINTERS[NUM_LAYERS] = {
    DUMMY_CONTEXT_BIAS, // Context layer has bias
    0, // Spike inputs have no bias
    0  // Output layer has no bias
};

// Layer Metadata
static const int32_t LAYER_THRESHOLDS[NUM_LAYERS] = {
    LIF_THRESHOLD, // Context
    LIF_THRESHOLD, // Spikes
    2147483647     // Output
};

static const int32_t LAYER_BETA_MULTIPLIERS[NUM_LAYERS] = {0, 0, 0};
static const int32_t LAYER_BETA_SHIFTS[NUM_LAYERS] = {0, 0, 0};

static const int LAYER_IN_DIMS[NUM_LAYERS] = {
    INPUT_CONTEXT_SIZE, 
    INPUT_SPIKE_SIZE, 
    HIDDEN_SIZE
};

static const int LAYER_OUT_DIMS[NUM_LAYERS] = {
    HIDDEN_SIZE, 
    HIDDEN_SIZE, 
    1
};