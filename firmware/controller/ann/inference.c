#include "inference.h"
#include "model_data.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

static int8_t buf0[MAX_LAYER_DIM];
static int8_t buf1[MAX_LAYER_DIM];

float ann_get_integral_scaler_mean(void)  { return SCALER_MEAN[3]; }
float ann_get_integral_scaler_scale(void) { return SCALER_SCALE[3]; }

/**
 * Heuristic check to determine if the model data is valid (non-dummy).
 *
 * The idea is to sum up a few weights from the first layer and see if they are all zero,
 * which would indicate the dummy model is in use. There is no reason to iterate over all weights.
 *
 * @return true if the model is valid and ready for inference, false otherwise.
 */
bool ann_is_model_valid(void) {
    int32_t weight_sum = 0;
    const int8_t* layer1_weights_pointer = WEIGHT_POINTERS[0];
    int layer1_total_weights = LAYER_IN_DIMS[0] * LAYER_OUT_DIMS[0];
    
    if (layer1_total_weights > 10) {
        layer1_total_weights = 10;
    }

    for (int i = 0; i < layer1_total_weights; i++) {
        weight_sum += abs(layer1_weights_pointer[i]);
    }

    return (weight_sum != 0);
}

/**
 * Helper to return the output scale factor based on OUTPUT_FRAC_BITS.
 * This factor is used to convert the int8_t output back to float.
 *
 * @return float The scale factor for converting int8_t output to float.
 */
float ann_get_output_scale(void) {
    float scale;
    
    // Determine the output scale safely, handling negative fractional bits
#if OUTPUT_FRAC_BITS >= 0
        scale = 1.0f / (float)(1 << OUTPUT_FRAC_BITS);
#else
        scale = (float)(1 << -OUTPUT_FRAC_BITS);
#endif
    
    return scale;
}

/**
 * Applies StandardScaler normalization to raw input data.
 * Normalization: normalized[i] = (raw[i] - mean[i]) / scale[i]
 * 
 * This must be applied BEFORE quantization to preserve signal resolution.
 *
 * @param raw_input Pointer to the raw input float array.
 * @param normalized_output Pointer to the normalized output float array.
 * @param num_elements Number of elements to normalize.
 */
static void normalize_input(const float* raw_input, float* normalized_output, int num_elements) {
    for (int i = 0; i < num_elements; i++) {
        normalized_output[i] = (raw_input[i] - SCALER_MEAN[i]) / SCALER_SCALE[i];
    }
}

/**
 * Quantizes floating-point input data to int8 format using the defined input fractional bits.
 * 
 * INPUT_FRAC_BITS is defined in inference.h and determines the Q-format for input data.
 * It is based on the calibration statistics during model export and is a power-of-two scaling factor.
 * input_data[i] is multiplied by 2^(INPUT_FRAC_BITS) and then cast to int8.
 * The shift operation is a efficient way to calculate the float value of 2^(INPUT_FRAC_BITS).
 *
 * Values are clamped to the int8 range of -128 to 127.
 *
 * @param input_data Pointer to the input float array.
 * @param output_buffer Pointer to the output int8_t array.
 * @param num_elements Number of elements to quantize.
 */
static void quantize_input(const float* input_data, int8_t* output_buffer, int num_elements) {
    float scale;

#if INPUT_FRAC_BITS >= 0
        scale = (float)(1 << INPUT_FRAC_BITS);
#else
        scale = 1.0f / (float)(1 << -INPUT_FRAC_BITS);
#endif

    for (int i = 0; i < num_elements; i++) {
        float scaled = input_data[i] * scale;
        if (scaled > 127.0f) {
            scaled = 127.0f;
        }
        if (scaled < -128.0f) {
            scaled = -128.0f;
        }
        output_buffer[i] = (int8_t)roundf(scaled);
    }
}

/**
 * Resets the inference buffers to zero.
 *
 * @note: Weights are constant and do not require resetting.
 */
void ann_inference_reset(void) {
    memset(buf0, 0, MAX_LAYER_DIM * sizeof(int8_t));
    memset(buf1, 0, MAX_LAYER_DIM * sizeof(int8_t));
}

/**
 * Executes a dense (fully connected) neural network layer
 * 
 * Bare-bone implementation to replace "arm_fully_connected_s8" from CMSIS-NN without any optimizations.
 * Allowing for dependency-free builds and easier understanding of the operations while maintaining the strict
 * compatibility with CMSIS-NN data structures.
 *
 * Just like the exporter, this implementation is based on:
 * - Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference", https://arxiv.org/pdf/1712.05877
 * - Lai et al., "CMSIS-NN: Efficient Neural Network Kernels for Arm Cortex-M CPUs", https://arxiv.org/pdf/1801.06601
 * 
 * @note Due to the missing (DSP-)optimizations, this is only suitable for small networks in control tasks.
 * @note Right now the only activation supported is ReLU.
 *
 * @param input  Pointer to the input vector (int8_t).
 * @param output Pointer to the output vector (int8_t) where the result will be stored.
 */
static void dense_layer_run(const int8_t* input, int8_t* output, 
                            const int8_t* weights, const int32_t* bias,
                            int in_dim, int out_dim, int rshift, bool relu) {
    for (int i = 0; i < out_dim; i++) {
        int32_t acc = bias[i];

        // Dot product (bias is already added)
        for (int j = 0; j < in_dim; j++) {
            acc += input[j] * weights[i * in_dim + j];
        }

        // Do the Re-Quantization
        // "Meanwhile, multiplication by 2^-n can be implemented with an efficient bitshift,
        // albeit one that needs to have correct round-to-nearest behavior ..."
        // Jacob et al. https://arxiv.org/pdf/1712.05877
        if (rshift > 0) {
            // This calcs the rounding bias for right shifts
            acc = acc + (1 << (rshift - 1));
            // Do the multiplication of 2^-n via right shift (actually divides by 2^n)
            acc = acc >> rshift;
        } else {
            // Just in case: negative shift means left shift 
            acc = acc << (-rshift);
        }

        // Clamp
        if (acc > 127) {
            acc = 127;
        }
        if (acc < -128) {
            acc = -128;
        }

        // Activation (ReLU)
        if (relu && acc < 0) {
            acc = 0;
        }

        output[i] = (int8_t)acc;
    }
}

/**
 * Generic Inference Function using memory-efficient buffer swapping.
 * 
 * Pipeline:
 * 1. Apply StandardScaler normalization to raw input
 * 2. Quantize normalized input to int8
 * 3. Run inference through quantized layers
 * 
 * @param input_floats Pointer to input float array (raw sensor values)
 * @return int8_t The output value after inference
 */
int8_t ann_inference(float* input_floats) {
    // bail out early for dummy model
    if (!ann_is_model_valid()) {
        return 0;
    }

    // Clamp derivative_error (feature index 4) before scaling to match training p99 clip.
    // Step-boundary spikes (Delta_sp / dt) are out-of-distribution without this clamp.
#if defined(DERIV_CLIP_VALUE) && (SCALER_DIM > 4)
    if (input_floats[4] >  DERIV_CLIP_VALUE) input_floats[4] =  DERIV_CLIP_VALUE;
    if (input_floats[4] < -DERIV_CLIP_VALUE) input_floats[4] = -DERIV_CLIP_VALUE;
#endif

    // Normalize input (apply StandardScaler)
    float normalized_input[LAYER_IN_DIMS[0]];
    normalize_input(input_floats, normalized_input, LAYER_IN_DIMS[0]);

    // Quantize normalized input -> buf0
    quantize_input(normalized_input, buf0, LAYER_IN_DIMS[0]);

    // Use two buffers for input and output swapping
    int8_t* input_ptr = buf0;
    int8_t* output_ptr = buf1;

    // Layer Loop
    for (int i = 0; i < NUM_LAYERS; i++) {
        dense_layer_run(
            input_ptr, 
            output_ptr,
            WEIGHT_POINTERS[i], 
            BIAS_POINTERS[i], 
            LAYER_IN_DIMS[i], 
            LAYER_OUT_DIMS[i], 
            LAYER_SHIFTS[i],
            LAYER_HAS_RELU[i]
        );

        // Swap Buffers
        int8_t* temp = input_ptr;
        input_ptr = output_ptr;
        output_ptr = temp;
    }

    return input_ptr[0];
}

/**
 * Returns the expected input size, which is determined by the first layer's input dimension.
 *
 * @return int The number of input features expected.
 */
int ann_get_input_size(void) {
    return LAYER_IN_DIMS[0];
}

/**
 * Returns the expected output size, which is determined by the last layer's output dimension.
 *
 * @return int The number of output features produced.
 */
int ann_get_output_size(void) {
    return LAYER_OUT_DIMS[NUM_LAYERS - 1];
}