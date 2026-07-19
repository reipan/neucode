/**
 * MT6701 analog sensor - STM32Duino platform.
 * Uses LL API (not HAL) to avoid HAL_ADC_MspInit() conflict with STM32Duino's
 * wiring_analog.c.
 *
 * ADC1_IN1 (PA0), 12-bit, 16x HW oversampling (OVSS=4), DMA1 Ch1 circular.
 * Diagnostic build: prints DMA/ADC state at init time to isolate root cause.
 */

#include <Arduino.h>
#include <stm32g4xx_ll_adc.h>
#include <stm32g4xx_ll_bus.h>
#include <stm32g4xx_ll_dma.h>
#include <stm32g4xx_ll_dmamux.h>
#include <stm32g4xx_ll_gpio.h>

extern "C" {
#include "sensor.h"
}

#define NC_ADC_INSTANCE ADC1
#define NC_ADC_CHANNEL LL_ADC_CHANNEL_1 // PA0 = ADC1_IN1 = A0
#define NC_GPIO_PORT GPIOA
#define NC_GPIO_PIN LL_GPIO_PIN_0
#define NC_DMA_INSTANCE DMA1
#define NC_DMA_CHANNEL LL_DMA_CHANNEL_1
#define NC_DMAMUX_REQ LL_DMAMUX_REQ_ADC1

#define SENSOR_IIR_ALPHA 0.04f
#define ADC_RAIL_MARGIN 20u
#define ADC_FULL_SCALE 4095u

// DMA destination - must survive driver.init()
static volatile uint16_t s_adc_buf = 0;
static volatile bool s_initialized = false;

static float s_angle_prev = 0.0f;
static float s_angle_unwrapped = 0.0f;
static float s_angle_unwrapped_filt = 0.0f;
static float s_zero_offset = 0.0f;
static bool s_first_reading = true;

static void mt6701_init(void) {
  LL_AHB2_GRP1_EnableClock(LL_AHB2_GRP1_PERIPH_ADC12);
  LL_AHB2_GRP1_EnableClock(LL_AHB2_GRP1_PERIPH_GPIOA);
  LL_AHB1_GRP1_EnableClock(LL_AHB1_GRP1_PERIPH_DMA1);
  LL_AHB1_GRP1_EnableClock(LL_AHB1_GRP1_PERIPH_DMAMUX1);

  // STM32Duino leaves ADC1 in deep power-down (DEEPPWD, CR bit 29 = 1).
  // While DEEPPWD=1, LL_ADC_Enable() is a no-op and the ADC never converts.
  // Force-disable first so LL_ADC_Init/LL_ADC_REG_Init can apply settings
  // (they silently return ERROR if ADEN=1).
  if (LL_ADC_REG_IsConversionOngoing(NC_ADC_INSTANCE)) {
    LL_ADC_REG_StopConversion(NC_ADC_INSTANCE);
    while (LL_ADC_REG_IsStopConversionOngoing(NC_ADC_INSTANCE)) {}
  }
  if (LL_ADC_IsEnabled(NC_ADC_INSTANCE)) {
    LL_ADC_Disable(NC_ADC_INSTANCE);
    while (LL_ADC_IsDisableOngoing(NC_ADC_INSTANCE)) {}
  }
  LL_ADC_ClearFlag_ADRDY(NC_ADC_INSTANCE);

  // PA0: analog, no pull
  LL_GPIO_SetPinMode(NC_GPIO_PORT, NC_GPIO_PIN, LL_GPIO_MODE_ANALOG);
  LL_GPIO_SetPinPull(NC_GPIO_PORT, NC_GPIO_PIN, LL_GPIO_PULL_NO);

  // DMA1 Ch1: periph->mem, circular, half-word
  LL_DMA_SetPeriphRequest(NC_DMA_INSTANCE, NC_DMA_CHANNEL, NC_DMAMUX_REQ);
  LL_DMA_SetDataTransferDirection(NC_DMA_INSTANCE, NC_DMA_CHANNEL,
                                  LL_DMA_DIRECTION_PERIPH_TO_MEMORY);
  LL_DMA_SetChannelPriorityLevel(NC_DMA_INSTANCE, NC_DMA_CHANNEL,
                                 LL_DMA_PRIORITY_HIGH);
  LL_DMA_SetMode(NC_DMA_INSTANCE, NC_DMA_CHANNEL, LL_DMA_MODE_CIRCULAR);
  LL_DMA_SetPeriphIncMode(NC_DMA_INSTANCE, NC_DMA_CHANNEL,
                          LL_DMA_PERIPH_NOINCREMENT);
  LL_DMA_SetMemoryIncMode(NC_DMA_INSTANCE, NC_DMA_CHANNEL,
                          LL_DMA_MEMORY_NOINCREMENT);
  LL_DMA_SetPeriphSize(NC_DMA_INSTANCE, NC_DMA_CHANNEL,
                       LL_DMA_PDATAALIGN_HALFWORD);
  LL_DMA_SetMemorySize(NC_DMA_INSTANCE, NC_DMA_CHANNEL,
                       LL_DMA_MDATAALIGN_HALFWORD);
  LL_DMA_SetDataLength(NC_DMA_INSTANCE, NC_DMA_CHANNEL, 1);
  LL_DMA_SetPeriphAddress(NC_DMA_INSTANCE, NC_DMA_CHANNEL,
      LL_ADC_DMA_GetRegAddr(NC_ADC_INSTANCE, LL_ADC_DMA_REG_REGULAR_DATA));
  LL_DMA_SetMemoryAddress(NC_DMA_INSTANCE, NC_DMA_CHANNEL,
                          (uint32_t)&s_adc_buf);

  // ADC common clock: PCLK/4 = 42.5 MHz
  LL_ADC_SetCommonClock(__LL_ADC_COMMON_INSTANCE(NC_ADC_INSTANCE),
                        LL_ADC_CLOCK_SYNC_PCLK_DIV4);

  {
    LL_ADC_InitTypeDef init = {};
    init.Resolution = LL_ADC_RESOLUTION_12B;
    init.DataAlignment = LL_ADC_DATA_ALIGN_RIGHT;
    init.LowPowerMode = LL_ADC_LP_MODE_NONE;
    LL_ADC_Init(NC_ADC_INSTANCE, &init);
  }

  {
    LL_ADC_REG_InitTypeDef reg = {};
    reg.TriggerSource = LL_ADC_REG_TRIG_SOFTWARE;
    reg.SequencerLength = LL_ADC_REG_SEQ_SCAN_DISABLE;
    reg.SequencerDiscont = LL_ADC_REG_SEQ_DISCONT_DISABLE;
    reg.ContinuousMode = LL_ADC_REG_CONV_CONTINUOUS;
    reg.DMATransfer = LL_ADC_REG_DMA_TRANSFER_UNLIMITED;
    reg.Overrun = LL_ADC_REG_OVR_DATA_OVERWRITTEN;
    LL_ADC_REG_Init(NC_ADC_INSTANCE, &reg);
  }

  LL_ADC_SetOverSamplingScope(NC_ADC_INSTANCE,
                              LL_ADC_OVS_GRP_REGULAR_CONTINUED);
  LL_ADC_ConfigOverSamplingRatioShift(NC_ADC_INSTANCE, LL_ADC_OVS_RATIO_16,
                                      LL_ADC_OVS_SHIFT_RIGHT_4);

  LL_ADC_REG_SetSequencerRanks(NC_ADC_INSTANCE, LL_ADC_REG_RANK_1,
                               NC_ADC_CHANNEL);
  LL_ADC_SetChannelSamplingTime(NC_ADC_INSTANCE, NC_ADC_CHANNEL,
                                LL_ADC_SAMPLINGTIME_247CYCLES_5);
  LL_ADC_SetChannelSingleDiff(NC_ADC_INSTANCE, NC_ADC_CHANNEL,
                              LL_ADC_SINGLE_ENDED);

  LL_ADC_DisableDeepPowerDown(NC_ADC_INSTANCE);
  LL_ADC_EnableInternalRegulator(NC_ADC_INSTANCE);
  delayMicroseconds(100);  // conservative TADCVREG_STUP after DEEPPWD exit; 20us is datasheet min at max temp


  LL_ADC_StartCalibration(NC_ADC_INSTANCE, LL_ADC_SINGLE_ENDED);
  while (LL_ADC_IsCalibrationOnGoing(NC_ADC_INSTANCE)) {
  }

  LL_ADC_Enable(NC_ADC_INSTANCE);
  while (!LL_ADC_IsActiveFlag_ADRDY(NC_ADC_INSTANCE)) {
  }

  // Enable DMA before starting ADC - order matters
  LL_DMA_EnableChannel(NC_DMA_INSTANCE, NC_DMA_CHANNEL);
  LL_ADC_REG_StartConversion(NC_ADC_INSTANCE);

  // Allow at least one full 16x oversampled conversion before reads (~100 us)
  delayMicroseconds(200);

  s_initialized = true;
}

static float mt6701_read(void) {
  if (!s_initialized)
    return 0.0f;

  // DMA keeps s_adc_buf continuously fresh - no blocking read needed.
  uint16_t raw = s_adc_buf;

  if (raw < ADC_RAIL_MARGIN || raw > (ADC_FULL_SCALE - ADC_RAIL_MARGIN)) {
    return s_angle_unwrapped_filt - s_zero_offset;
  }

  float current_angle = (float)raw * (360.0f / 4096.0f);

  if (s_first_reading) {
    s_angle_prev = current_angle;
    s_angle_unwrapped = current_angle;
    s_angle_unwrapped_filt = current_angle;
    s_first_reading = false;
    return current_angle - s_zero_offset;
  }

  float delta = current_angle - s_angle_prev;
  if (delta > 180.0f)
    delta -= 360.0f;
  if (delta < -180.0f)
    delta += 360.0f;
  s_angle_unwrapped += delta;
  s_angle_prev = current_angle;

  s_angle_unwrapped_filt = SENSOR_IIR_ALPHA * s_angle_unwrapped +
                           (1.0f - SENSOR_IIR_ALPHA) * s_angle_unwrapped_filt;

  return s_angle_unwrapped_filt - s_zero_offset;
}

static void mt6701_zero(void) {
  // Do NOT reset s_first_reading - unwrap state must continue uninterrupted.
  s_zero_offset = s_angle_unwrapped_filt;
}

static float mt6701_read_raw(void) {
  if (!s_initialized)
    return 0.0f;

  uint16_t raw = s_adc_buf;

  if (raw < ADC_RAIL_MARGIN || raw > (ADC_FULL_SCALE - ADC_RAIL_MARGIN)) {
    return s_angle_unwrapped - s_zero_offset;
  }

  float current_angle = (float)raw * (360.0f / 4096.0f);

  if (s_first_reading) {
    s_angle_prev            = current_angle;
    s_angle_unwrapped       = current_angle;
    s_angle_unwrapped_filt  = current_angle;
    s_first_reading         = false;
    return current_angle - s_zero_offset;
  }

  float delta = current_angle - s_angle_prev;
  if (delta >  180.0f) delta -= 360.0f;
  if (delta < -180.0f) delta += 360.0f;
  s_angle_unwrapped += delta;
  s_angle_prev = current_angle;

  return s_angle_unwrapped - s_zero_offset;
}

extern "C" float nc_sensor_read_physical(void) {
  mt6701_read_raw();
  return s_angle_unwrapped;
}

extern "C" void nc_sensor_mt6701_arduino_register(void) {
  nc_sensor_register(mt6701_init, mt6701_read, mt6701_read_raw, mt6701_zero);
}
