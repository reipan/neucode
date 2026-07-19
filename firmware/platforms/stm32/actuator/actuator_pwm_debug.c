// This is just to debug the PWM stuff and check if TIM1 CH1/CH2/CH3 are working correctly.
#include "actuator_pwm_debug.h"
#include "actuator.h"
#include "main.h"
#include "comm.h"
#include "stm32g4xx_hal_tim.h"
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

extern TIM_HandleTypeDef htim1;

#ifndef NC_PWM_DEBUG_HAS_ENABLE
    #define NC_PWM_DEBUG_HAS_ENABLE 0
#endif

// If the driver enable pin is available, define it here
// Not done yet.
#if NC_PWM_DEBUG_HAS_ENABLE
    #define NC_DRV_EN_GPIO_Port GPIOB
    #define NC_DRV_EN_Pin GPIO_PIN_0
#endif

#ifndef NC_PWM_DEBUG_DUTY_CAP
    #define NC_PWM_DEBUG_DUTY_CAP 0.10f
#endif

#define NC_ENU_GPIO_Port GPIOB
#define NC_ENU_Pin GPIO_PIN_13

#define NC_ENV_GPIO_Port GPIOB
#define NC_ENV_Pin GPIO_PIN_14

#define NC_ENW_GPIO_Port GPIOB
#define NC_ENW_Pin GPIO_PIN_15

#define NC_ENFAULT_GPIO_Port GPIOB
#define NC_ENFAULT_Pin GPIO_PIN_12

static bool enabled = false;

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
 */
static void set_all_duty(float a, float b, float c) {
    set_duty(TIM_CHANNEL_1, a);
    set_duty(TIM_CHANNEL_2, b);
    set_duty(TIM_CHANNEL_3, c);
}

/**
 * Enables or disables the STSPIN830 motor driver stage.
 */
static void driver_enable(bool enable) {
    #if NC_PWM_DEBUG_HAS_ENABLE
        HAL_GPIO_WritePin(NC_DRV_EN_GPIO_Port, NC_DRV_EN_Pin,
                      enable ? GPIO_PIN_SET : GPIO_PIN_RESET);
    #else
        (void)enable;
    #endif
}


/**
 * Initializes PWM actuator debugging.
 */
static void init_callback(void) {
    HAL_TIM_PWM_Start(&htim1, TIM_CHANNEL_1);
    HAL_TIM_PWM_Start(&htim1, TIM_CHANNEL_2);
    HAL_TIM_PWM_Start(&htim1, TIM_CHANNEL_3);
    driver_enable(false);
    stspin830_stage_disable();
    set_all_duty(0.0f, 0.0f, 0.0f);
    enabled = false;
    logger("actuator_pwm_debug: initialized\r\n");
}

/**
 * Enables or disables the actuator PWM debugging.
 *
 * @param enable Set to true to enable the callback, false to disable it.
 */
static void enable_callback(bool enable) {
    enabled = enable;
    if (!enable) {
        driver_enable(false);
        set_all_duty(0.0f, 0.0f, 0.0f);
        stspin830_stage_disable();
        // logger("actuator_pwm_debug: disabled\r\n");
        return;
    }

    driver_enable(true);
    set_all_duty(0.0f, 0.0f, 0.0f);
    stspin830_stage_enable_all();
    // logger("actuator_pwm_debug: enabled\r\n");
}

/**
 * Sets the PWM duty cycles based on the provided control signal.
 *
 * @param u The callback value to be set, typically representing a control signal or duty cycle.
 */
static void set_callback(float u) {
    if (!enabled) return;
    
    // a bit of security here
    float mag = (u < 0.0f) ? -u : u;
    mag = clampf(mag, 0.0f, 1.0f);
    float d = clampf(mag, 0.0f, NC_PWM_DEBUG_DUTY_CAP);

    float au = mag;

    if (au < 0.33f) {
        set_all_duty(d, 0.0f, 0.0f);
    } else if (au < 0.66f) {
        set_all_duty(0.0f, d, 0.0f);
    } else {
        set_all_duty(0.0f, 0.0f, d);
    }

    uint32_t c1 = __HAL_TIM_GET_COMPARE(&htim1, TIM_CHANNEL_1);
    uint32_t c2 = __HAL_TIM_GET_COMPARE(&htim1, TIM_CHANNEL_2);
    uint32_t c3 = __HAL_TIM_GET_COMPARE(&htim1, TIM_CHANNEL_3);
    char buf[48];
    snprintf(buf, sizeof(buf), "CCR:%lu,%lu,%lu\r\n",
            (unsigned long)c1,
            (unsigned long)c2,
            (unsigned long)c3);
    logger(buf);
}

/**
 * Registers the actuator PWM debug functionality.
 */
void nc_actuator_pwm_debug_register(void)
{
    nc_actuator_register(
        init_callback,
        enable_callback,
        set_callback
    );
}