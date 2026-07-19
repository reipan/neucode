#include "inference.h"
#include "model_data.h"
#include <string.h>
#include <stdlib.h>

typedef struct {
    int32_t membrane[HIDDEN_SIZE];
    int8_t output_spikes[HIDDEN_SIZE];
} nc_snn_state_t;

static nc_snn_state_t snn_state;
static bool is_initialized = false;
static bool model_valid = false;

/**
 * Applies decay to the membrane potential based on beta parameters.
 * 
 * This function is necessary due to the quantized representation of beta.
 * Uses 64-bit intermediate to avoid overflow.
 *
 * @param value Current membrane potential.
 * @param beta_multiplier Multiplier for decay.
 * @param beta_shift Shift for decay.
 *
 * @return Decayed membrane potential.
 */
static inline int32_t apply_decay(int32_t value, int32_t beta_multiplier, int32_t beta_shift) {
    return (int32_t)( ((int64_t)value * beta_multiplier) >> beta_shift );
}

/**
 * Initializes the SNN inference state.
 */
void snn_inference_reset(void) {
    memset(&snn_state, 0, sizeof(snn_state));
    model_valid = snn_is_model_valid();
    is_initialized = true;
}

/**
 * Heuristic check to distinguish between the dummy model (all zeros)
 * and a trained model.
 *
 * @return true if the model is valid, false otherwise.
 */
bool snn_is_model_valid(void) {
    int32_t weight_sum = 0;

#if INPUT_CONTEXT_SIZE > 0
    // Layer 0 is the Context Layer (int8 weights)
    const int8_t* w = WEIGHT_POINTERS[0];
    int check_count = LAYER_IN_DIMS[0] * LAYER_OUT_DIMS[0];
    if (check_count > 10) check_count = 10;
    for (int i = 0; i < check_count; i++) {
        weight_sum += abs(w[i]);
    }
#else
    // Spike-only model: layer 0 is int32 spike weights
    const int32_t* w = (const int32_t*)WEIGHT_POINTERS[0];
    int check_count = LAYER_IN_DIMS[0] * LAYER_OUT_DIMS[0];
    if (check_count > 10) check_count = 10;
    for (int i = 0; i < check_count; i++) {
        weight_sum += (w[i] != 0);
    }
#endif

    return (weight_sum > 0);
}

/**
 * Performs inference on the given input data using the SNN model.
 *
 * These inputs should already be scaled to INPUT_FRAC_BITS fixed-point format.
 * Indices 0 to (INPUT_CONTEXT_SIZE-1) are quantized context inputs.
 * Indices [INPUT_CONTEXT_SIZE .. INPUT_CONTEXT_SIZE+INPUT_SPIKE_SIZE-1] are spike inputs.
 * Channel layout is defined by INPUT_CHANNEL_TYPE/FEATURE/POLARITY arrays in
 * model_config.h (populated by snn.c from the exported channel descriptors).
 *
 * @param inputs Pointer to an array of quantized int32_t input values.
 * @return int32_t The result of the inference computation.
 */
int32_t snn_inference(int32_t* inputs) {
    if (!is_initialized) {
        snn_inference_reset();
    }

    if (!model_valid) {
        return 0;
    }

    // Extract Context Inputs (may be empty for spike-only models)
#if INPUT_CONTEXT_SIZE > 0
    int32_t q_context[INPUT_CONTEXT_SIZE];
    for (int i = 0; i < INPUT_CONTEXT_SIZE; i++) {
        q_context[i] = inputs[i];
    }
    const int8_t* context_weights = (const int8_t*)WEIGHT_POINTERS[0];
    const int32_t* context_bias = (const int32_t*)BIAS_POINTERS[0];
#endif

    // Spike inputs begin after context and span INPUT_SPIKE_SIZE elements.
    const int32_t* spike_inputs = &inputs[INPUT_CONTEXT_SIZE];

    // Spike weights: int32, aligned to common accumulation shift.
#if INPUT_CONTEXT_SIZE > 0
    const int32_t* spike_weights = (const int32_t*)WEIGHT_POINTERS[1];
#else
    const int32_t* spike_weights = (const int32_t*)WEIGHT_POINTERS[0];
#endif

    // Output layer -- float32 for precision. OUTPUT_SCALE_WEIGHTS is accessed
    // by name (not via WEIGHT_POINTERS) and emitted by SNNExporter for all architectures.
    const float* float_output_weights = OUTPUT_SCALE_WEIGHTS;

    // LIF Dynamics
    int32_t threshold = LAYER_THRESHOLDS[0];
    int32_t beta_multiplier = LAYER_BETA_MULTIPLIERS[0];
    int32_t beta_shift = LAYER_BETA_SHIFTS[0];

    float output_sum = 0.0f;

    // Iterate over neurons first to match PyTorch row-major weight layout (flattened)
    for (int neuron = 0; neuron < HIDDEN_SIZE; neuron++) {
#if INPUT_CONTEXT_SIZE > 0
        int64_t acc = (int64_t)context_bias[neuron];
        for (int context_idx = 0; context_idx < INPUT_CONTEXT_SIZE; context_idx++) {
            acc += (int64_t)q_context[context_idx] * (*context_weights++);
        }
        int32_t current = (int32_t)acc;
#else
        int32_t current = 0;
#endif

        for (int s = 0; s < INPUT_SPIKE_SIZE; s++) {
            const int32_t spike = spike_inputs[s];
            if (spike != 0) {
                current += spike * spike_weights[neuron * INPUT_SPIKE_SIZE + s];
            }
        }

        // LIF dynamics
        int32_t membrane = snn_state.membrane[neuron];
        membrane = apply_decay(membrane, beta_multiplier, beta_shift);
        membrane += current;

        // Threshold check
        if (membrane >= threshold) {
            // Fire!
            snn_state.output_spikes[neuron] = 1;
            // Soft reset: subtract threshold from membrane potential
            membrane -= threshold; 

            // Fused output layer accumulation (float32 - preserves precision of trained model)
            output_sum += float_output_weights[neuron];
        } else {
            snn_state.output_spikes[neuron] = 0;
            // Clamp negative potential to match PyTorch's non-negative LIF
            if (membrane < 0) membrane = 0;
        }

        // Update membrane potential state
        snn_state.membrane[neuron] = membrane;
    }

    // Scale float output to int32 for backward-compat with snn.c caller.
    // Caller: out_float = raw_int * (OUTPUT_SCALE / 2^OUTPUT_WEIGHT_SHIFT)
    //                   = (output_sum * 2^16) * (OUTPUT_SCALE / 65536)
    //                   = output_sum * OUTPUT_SCALE   (identical to float32 sim)
    return (int32_t)(output_sum * 65536.0f);
}

/**
 * Returns the number of context inputs expected.
 *
 * @return int Number of context inputs.
 */
int snn_get_context_size(void) {
    return INPUT_CONTEXT_SIZE;
}

/**
 * Returns the number of spike inputs expected.
 *
 * @return int Number of spike inputs.
 */
int snn_get_spike_size(void) {
    return INPUT_SPIKE_SIZE;
}

/**
 * Returns the number of fractional bits used for fixed-point representation of inputs.
 *
 * @return int Number of fractional bits.
 */
int snn_get_input_frac_bits(void) {
    return INPUT_FRAC_BITS;
}

/**
 * Returns the delta threshold for the spike encoder.
 * 
 * @return int32_t Delta threshold for spike encoding.
 */
int32_t snn_get_encoder_delta_threshold(void) {
    return ENCODER_DELTA_THRESHOLD;
}

/**
 * Returns the output de-quantization scale factor.
 *
 * @return float Output scale factor.
 */
float snn_get_output_scale(void) {
    return OUTPUT_SCALE;
}

/**
 * Returns a single scale factor to convert the raw int32 sum returned by snn_inference()
 * into physical output units.
 *
 * @return float Output scale factor for converting raw int32 output to physical units.
 */
float snn_get_output_scale_factor(void) {
    return OUTPUT_SCALE / (float)(1 << OUTPUT_WEIGHT_SHIFT);
}

/**
 * Returns the integral window size for the hybrid integral component.
 *
 * @return float Integral window size in error units. Disabled if zero.
*/
float snn_get_integral_window(void) {
#ifdef INTEGRAL_WINDOW
    return INTEGRAL_WINDOW;
#else
    return 0.0f;
#endif
}

/**
 * Returns the integrator limit for the hybrid integral component.
 *
 * @return float Integrator limit. Disabled if zero.
 */
float snn_get_windup_limit(void) {
#ifdef WINDUP_LIMIT
    return WINDUP_LIMIT;
#else
    return 0.0f;
#endif
}