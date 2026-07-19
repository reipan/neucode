#include "stm32_actuator_stspin830_ihm16m1.h"
#include "stm32_sensor_mt6701_adc.h"
#include "actuator.h"
#include "sensor.h"
#include "main.h"
#include "comm.h"
#include <stdbool.h>
#include <stdint.h>
#include <math.h>
#include <stdio.h>

extern TIM_HandleTypeDef htim1;

#define NC_PI 3.14159265358979323846f
#define NC_TWO_PI (2.0f * NC_PI)

#ifndef NC_IHM16M1_POLE_PAIRS
  #define NC_IHM16M1_POLE_PAIRS 7u
#endif

#define NC_MT6701_ADC_MIN_VALID 20u
#define NC_MT6701_ADC_MAX_VALID (4095u - 20u)

#ifndef NC_IHM16M1_HAS_DRV_EN
    #define NC_IHM16M1_HAS_DRV_EN 0
#endif

// Optional global driver enable pin (not used on IHM16M1 / STSPIN830 by default)
#if NC_IHM16M1_HAS_DRV_EN
    #define NC_DRV_EN_GPIO_Port GPIOB
    #define NC_DRV_EN_Pin GPIO_PIN_0
#endif

#ifndef NC_IHM16M1_SINE_AMP_CAP
    #define NC_IHM16M1_SINE_AMP_CAP 0.15f
#endif

#ifndef NC_IHM16M1_CAL_AMP_CAP
    #define NC_IHM16M1_CAL_AMP_CAP 0.08f
#endif

#define NC_ENU_GPIO_Port GPIOB
#define NC_ENU_Pin GPIO_PIN_13

#define NC_ENV_GPIO_Port GPIOB
#define NC_ENV_Pin GPIO_PIN_14

#define NC_ENW_GPIO_Port GPIOB
#define NC_ENW_Pin GPIO_PIN_15

#define NC_ENFAULT_GPIO_Port GPIOB
#define NC_ENFAULT_Pin GPIO_PIN_12

#ifndef NC_IHM16M1_OMEGA_STEP
    #define NC_IHM16M1_OMEGA_STEP 0.005f
#endif

static bool enabled = false;
static float omega_step = NC_IHM16M1_OMEGA_STEP;
static float theta_e = 0.0f; // electrical angle for smooth rotation in radians
static float elec_offset_rad = 0.0f;  // runtime electrical offset (rad)

typedef enum {
    NC_IHM16M1_MODE_OPENLOOP = 0,
    NC_IHM16M1_MODE_SENSORED_TORQUE = 1,
} nc_ihm16m1_mode_t;

#ifndef NC_IHM16M1_MODE
    #define NC_IHM16M1_MODE NC_IHM16M1_MODE_SENSORED_TORQUE
#endif

static nc_ihm16m1_mode_t mode = (nc_ihm16m1_mode_t)NC_IHM16M1_MODE;

/**
 * Wraps an angle in radians to the range [0, 2PI).
 *
 * @param x The angle in radians.
 * @return The wrapped angle in the range [0, 2PI).
 */
static float wrap_0_2pi(float x) {
    while (x >= NC_TWO_PI) {
        x -= NC_TWO_PI;
    }
    while (x < 0.0f) {
        x += NC_TWO_PI;
    }
    return x;
}

/**
 * Simple logger function to send messages via the communication port.
 *
 * @param msg The message string to log.
 */
static void logger(const char *msg) {
   nc_comm_send_log(msg);
}

/**
 * Disables the STSPIN830 motor driver stage.
 */
static void stspin830_stage_disable(void) {
    HAL_GPIO_WritePin(NC_ENFAULT_GPIO_Port, NC_ENFAULT_Pin, GPIO_PIN_RESET);
    HAL_GPIO_WritePin(NC_ENU_GPIO_Port, NC_ENU_Pin, GPIO_PIN_RESET);
    HAL_GPIO_WritePin(NC_ENV_GPIO_Port, NC_ENV_Pin, GPIO_PIN_RESET);
    HAL_GPIO_WritePin(NC_ENW_GPIO_Port, NC_ENW_Pin, GPIO_PIN_RESET);
}

/**
 * Enables all channels of the STSPIN830 motor driver stage.
 */
static void stspin830_stage_enable_all(void) {
    HAL_GPIO_WritePin(NC_ENU_GPIO_Port, NC_ENU_Pin, GPIO_PIN_SET);
    HAL_GPIO_WritePin(NC_ENV_GPIO_Port, NC_ENV_Pin, GPIO_PIN_SET);
    HAL_GPIO_WritePin(NC_ENW_GPIO_Port, NC_ENW_Pin, GPIO_PIN_SET);
    HAL_GPIO_WritePin(NC_ENFAULT_GPIO_Port, NC_ENFAULT_Pin, GPIO_PIN_SET);
}

/** 
 * Clamp helper, to make sure duty is between 0.0 and 1.0
 */
static float clampf(float x, float low, float high) {
    if (x < low) return low;
    if (x > high) return high;
    return x;
}

// Local helper: only used within this actuator module.
static bool mt6701_is_valid() {
    if (!nc_sensor_mt6701_is_initialized()) {
        return false;
    }

    unsigned int adc = nc_sensor_mt6701_get_raw_adc();
    if (adc < NC_MT6701_ADC_MIN_VALID || adc > NC_MT6701_ADC_MAX_VALID) {
        return false;
    }

    return true;
}

/**
 * Set the duty cycle for a specific PWM channel.
 */
static void set_duty(uint32_t channel, float duty) {
    duty = clampf(duty, 0.0f, 1.0f);
    uint32_t arr = __HAL_TIM_GET_AUTORELOAD(&htim1);
    uint32_t ccr = (uint32_t)(duty * (float)(arr + 1));
    __HAL_TIM_SET_COMPARE(&htim1, channel, ccr);
}

/**
 * Set the duty cycle for all three PWM channels.
 *
 * @param a Duty cycle for channel 1 (Phase U).
 * @param b Duty cycle for channel 2 (Phase V).
 * @param c Duty cycle for channel 3 (Phase W).
 */
static void set_all_duty(float a, float b, float c) {
    set_duty(TIM_CHANNEL_1, a);
    set_duty(TIM_CHANNEL_2, b);
    set_duty(TIM_CHANNEL_3, c);
}

/**
 * Enables or disables the STSPIN830 motor driver stage.
 *
 * @param enable Set to true to enable the driver, false to disable it.
 * @note This function uses an optional global driver enable pin if defined otherwise it's a no-op.
 */
static void driver_enable(bool enable) {
    #if NC_IHM16M1_HAS_DRV_EN
        HAL_GPIO_WritePin(NC_DRV_EN_GPIO_Port, NC_DRV_EN_Pin,
                      enable ? GPIO_PIN_SET : GPIO_PIN_RESET);
    #else
        (void)enable;
    #endif
}

/**
 * Initializes PWM actuator.
 */
static void init_callback(void) {
    HAL_TIM_PWM_Start(&htim1, TIM_CHANNEL_1);
    HAL_TIM_PWM_Start(&htim1, TIM_CHANNEL_2);
    HAL_TIM_PWM_Start(&htim1, TIM_CHANNEL_3);
    driver_enable(false);
    stspin830_stage_disable();
    set_all_duty(0.0f, 0.0f, 0.0f);
    enabled = false;
    logger("actuator_stspin830_ihm16m1: initialized\r\n");
}

/**
 * Enables or disables the actuator PWM.
 *
 * @param enable Set to true to enable the callback, false to disable it.
 */
static void enable_callback(bool enable) {
    enabled = enable;
    if (!enable) {
        driver_enable(false);
        set_all_duty(0.0f, 0.0f, 0.0f);
        stspin830_stage_disable();
        return;
    }

    driver_enable(true);
    set_all_duty(0.0f, 0.0f, 0.0f);
    stspin830_stage_disable();
    HAL_Delay(20);
    stspin830_stage_enable_all();
    HAL_Delay(5);
}

/**
 * Sets all PWM channels to neutral (50% duty cycle).
 */
static void set_neutral(void) {
    set_all_duty(0.5f, 0.5f, 0.5f);
}

/**
 * Applies a static magnetic field to the motor by setting fixed PWM duty cycles.
 *
 * @param theta_e The electrical angle in radians.
 * @param amp The amplitude of the sine wave (0.0 to 0.25).
 */
static void apply_static_field(float theta_e, float amp) {
    theta_e = wrap_0_2pi(theta_e);
    // math safety cap
    amp = clampf(amp, -0.5f, 0.5f);

    float val_u = 0.5f + (amp * sinf(theta_e));
    float val_v = 0.5f + (amp * sinf(theta_e + (2.0f * NC_PI / 3.0f)));
    float val_w = 0.5f + (amp * sinf(theta_e + (4.0f * NC_PI / 3.0f)));

    set_all_duty(val_u, val_v, val_w);
}

/**
 * Applies three-phase sine wave commutation PWM to the motor driver based on the control effort.
 *
 * Has two modes controlled by NC_IHM16M1_MODE:
 * - Open-loop: increments electrical angle by fixed step each call.
 * - Sensored torque control: uses MT6701 sensor feedback to determine electrical angle.
 *
 * @param u The control signal/effort value.
 */
static void set_callback(float u) {
    if (!enabled) return;

    // Map effort (-1.0 to 1.0) to electrical angle and amplitude
    float effort = clampf(u, -1.0f, 1.0f);
    
    // Signed amplitude scaling using duty cap
    float amp = effort * NC_IHM16M1_SINE_AMP_CAP;
    
    if (mode == NC_IHM16M1_MODE_SENSORED_TORQUE) {
        // Closed-loop: get electrical angle from MT6701 sensor
        if (!mt6701_is_valid()) {
            // disable stage and set 0 duty
            set_all_duty(0.0f, 0.0f, 0.0f);
            stspin830_stage_disable();
            return;
        }

        // get deg angle from sensor
        float mech_angle_deg = nc_sensor_mt6701_get_angle_deg_filtered();
        // convert to radians
        float mech_angle_rad = mech_angle_deg * (NC_PI / 180.0f);
        // convert to electrical angle (multiply by pole pairs + offset)
        theta_e = (mech_angle_rad * (float)NC_IHM16M1_POLE_PAIRS) + elec_offset_rad;
        theta_e = wrap_0_2pi(theta_e + (NC_PI * 0.5f));
    } else {
        // Open-loop: increment electrical angle based on fixed step
        if (fabsf(effort) > 0.001f) {
            theta_e += (effort > 0.0f) ? omega_step : -omega_step;
            theta_e = wrap_0_2pi(theta_e);
        }
        theta_e = wrap_0_2pi(theta_e);
    }

    stspin830_stage_enable_all();
    
    // Phase duty calculations for three-phase sine wave commutation
    apply_static_field(theta_e, amp);
}

/**
 * Calibrates the electrical offset of the motor by applying a static field and measuring the position.
 *
 * @param amp The amplitude of the sine wave to apply during calibration (-0.5 to 0.5).
 * @param settle_ms The time in milliseconds to wait for settling before taking the measurement.
 * @return The calculated electrical offset in radians.
 */
float nc_actuator_stspin830_ihm16m1_calibrate_offset(float amp, uint32_t settle_ms) {
    if (!enabled) {
        logger("actuator_stspin830_ihm16m1: cannot calibrate offset when disabled\r\n");
        return 0.0f;
    }

    // Prime DMA pipeline so adc_raw is not stuck at 0
    // (guards against a startup race where the first reads return a stale 0).
    for (int i = 0; i < 50; i++) {
        (void)nc_sensor_read();
        HAL_Delay(2);
        if (nc_sensor_mt6701_get_raw_adc() != 0) break;
    }

    if (!mt6701_is_valid()) {
        char buf[96];
        snprintf(buf, sizeof(buf),
            "actuator_stspin830_ihm16m1: MT6701 invalid -> init=%d adc=%u deg=%.2f\r\n",
            (int)nc_sensor_mt6701_is_initialized(),
            (unsigned)nc_sensor_mt6701_get_raw_adc(),
            (double)nc_sensor_mt6701_get_angle_deg_filtered()
        );
        logger(buf);
        return 0.0f;
    }

    stspin830_stage_enable_all();
    set_neutral();
    HAL_Delay(20);

    const float theta_e_start = 0.0f;
    float cal_amp = clampf(amp, -NC_IHM16M1_CAL_AMP_CAP, NC_IHM16M1_CAL_AMP_CAP);
    apply_static_field(theta_e_start, cal_amp);

    uint32_t t0 = HAL_GetTick();
    while ((HAL_GetTick() - t0) < settle_ms) {
        // wait for settling
        (void)nc_sensor_read();
        HAL_Delay(2);
    }

    float mech_angle_deg = nc_sensor_mt6701_get_angle_deg_filtered();
    float mech_angle_rad = mech_angle_deg * (NC_PI / 180.0f);

    float offset = theta_e_start - (mech_angle_rad * (float)NC_IHM16M1_POLE_PAIRS);;
    offset = wrap_0_2pi(offset);
    elec_offset_rad = offset;

    char buf[64];
    snprintf(buf, sizeof(buf), "cal: elec_offset=%.4f rad\r\n", (double)elec_offset_rad);
    logger(buf);

    return offset;
}

/**
 * Registers the actuator PWM functionality.
 */
void nc_actuator_stspin830_ihm16m1_register(void)
{
    nc_actuator_register(
        init_callback,
        enable_callback,
        set_callback
    );
}