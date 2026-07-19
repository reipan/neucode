// This is just a dummy file to allow compilation when no exported model data is present.
#pragma once
#include <stdint.h>
#include <stdbool.h>

#define INPUT_FRAC_BITS 4
#define OUTPUT_FRAC_BITS 4
#define NUM_LAYERS 1
#define MAX_LAYER_DIM 1

// Dummy StandardScaler Parameters
#define SCALER_DIM 1
static const float SCALER_MEAN[] = { 0.0f };
static const float SCALER_SCALE[] = { 1.0f };
// No derivative clipping in dummy state (FLT_MAX = effectively disabled)
#define DERIV_CLIP_VALUE 3.40282347e+38f

static const int8_t DUMMY_WEIGHTS[] = {0}; 
static const int32_t DUMMY_BIAS[] = {0};

static const int8_t* const WEIGHT_POINTERS[] = { DUMMY_WEIGHTS };
static const int32_t* const BIAS_POINTERS[] = { DUMMY_BIAS };
static const int LAYER_SHIFTS[] = { 0 };
static const int LAYER_IN_DIMS[] = { 1 };
static const int LAYER_OUT_DIMS[] = { 1 };
static const bool LAYER_HAS_RELU[] = { false };