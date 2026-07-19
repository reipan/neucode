#include "sensor.h"
#include "stm32_sensor_mt6701_adc.h"
#include "main.h"
#include <stdbool.h>
#include <stdint.h>

extern ADC_HandleTypeDef hadc1;

static volatile uint16_t adc_raw = 0;
static volatile bool initialized = false;

volatile uint16_t adc_dma_buffer = 0;

static float angle_deg_raw = 0.0f;
static float angle_deg_prev = 0.0f;
static float angle_deg_filtered = 0.0f;
static float angle_deg_unwrapped = 0.0f;
static float angle_deg_unwrapped_filt = 0.0f;  // software IIR on final output
static bool first_reading = true;

static float zero_offset_deg = 0.0f;

// Software IIR low-pass filter applied to the unwrapped angle output.
// Reduces ADC quantisation noise while preserving motor dynamics.
// The ANN/SNN were trained on Gaussian sensor noise; this filter brings the
// hardware noise characteristics closer to the training distribution.
#define SENSOR_IIR_ALPHA 0.04f

/**
 * Initializes the DMA (Direct Memory Access) for the MT6701 ADC sensor.
 *
 * @note This function is never called directly. It is registered as a callback using nc_sensor_init().
 */
static void mt6701_adc_dma_init(void) {
    HAL_ADCEx_Calibration_Start(&hadc1, ADC_SINGLE_ENDED);

    // Start ADC in DMA mode to continuously read into adc_dma_buffer
    if(HAL_ADC_Start_DMA(&hadc1, (uint32_t*)&adc_dma_buffer, 1) == HAL_OK) {
        initialized = true;
    }
}

/**
 * Reads the angular position from the MT6701 sensor using ADC with DMA and returns the value in degrees.
 *
 * @return The measured angle in degrees as a floating-point value.
 */
static float mt6701_adc_dma_read_deg(void) {
    if (!initialized) {
        return 0.0f;
    }

    uint16_t raw = adc_dma_buffer;
    adc_raw = raw;
    float current_angle = (raw * 360.0f) / 4096.0f;

    angle_deg_raw = current_angle;
    // Hardware oversampling (16x) in the G4 ADC acts as our LPF, so we copy raw to filtered.
    angle_deg_filtered = current_angle;

    if (first_reading) {
        angle_deg_prev = current_angle;
        angle_deg_unwrapped = current_angle;
        angle_deg_unwrapped_filt = current_angle;
        first_reading = false;
        return current_angle;
    }

    float delta = current_angle - angle_deg_prev;

    // Handle wrap-around
    if (delta > 180.0f) {
        delta -= 360.0f;
    }

    if (delta < -180.0f) {
        delta += 360.0f;
    }

    angle_deg_unwrapped += delta;
    angle_deg_prev = current_angle;

    // Apply software IIR low-pass filter to the unwrapped output
    angle_deg_unwrapped_filt = SENSOR_IIR_ALPHA * angle_deg_unwrapped
                               + (1.0f - SENSOR_IIR_ALPHA) * angle_deg_unwrapped_filt;

    // Apply zero offset
    return angle_deg_unwrapped_filt - zero_offset_deg;
}

/**
 * Zeros the MT6701 sensor by setting the current filtered angle as the zero reference.
 *
 * @note This function is never called directly. It is registered as a callback using nc_sensor_zero().
 */
static void mt6701_adc_dma_zero(void) {
    zero_offset_deg = angle_deg_unwrapped_filt;
}

/**
 * Registers the MT6701 ADC-based sensor with the system.
 *
 * @note Ensure that the ADC and any required hardware interfaces are properly configured before calling this function.
 */
void nc_sensor_mt6701_adc_register(void) {
    nc_sensor_register(
        mt6701_adc_dma_init,
        mt6701_adc_dma_read_deg,
        mt6701_adc_dma_zero
    );
}

/**
 * Retrieves the raw ADC value from the MT6701 sensor.
 *
 * @return unsigned int The raw ADC value read from the MT6701 sensor.
 */
unsigned int nc_sensor_mt6701_get_raw_adc(void) {
    return adc_raw;
}

/**
 * Retrieves the raw angle measurement from the MT6701 sensor in degrees.
 *
 * @return float The raw angle in degrees as measured by the MT6701 sensor.
 */
float nc_sensor_mt6701_get_angle_deg_raw(void) {
    return angle_deg_raw;
}

/**
 * Retrieves the filtered angle measurement from the MT6701 sensor in degrees.
 *
 * @deprecated Use nc_sensor_mt6701_get_angle_deg_raw() instead.
 * @return The filtered angle in degrees as a float.
 */
float nc_sensor_mt6701_get_angle_deg_filtered(void) {
    return angle_deg_filtered;
}

/**
 * Retrieves the unwrapped angle in degrees from the MT6701 sensor.
 *
 * This function reads the current angle measurement from the MT6701 sensor,
 * processes it to account for multiple rotations (unwrapping), and returns
 * the resulting angle in degrees as a floating-point value.
 *
 * @return The unwrapped angle in degrees.
 */
float nc_sensor_mt6701_get_angle_deg_unwrapped(void) {
    return angle_deg_unwrapped;
}

/**
 * Checks if the MT6701 sensor has been initialized.
 *
 * @return true if the sensor is initialized, false otherwise.
 */
bool nc_sensor_mt6701_is_initialized(void) {
    return initialized;
}
