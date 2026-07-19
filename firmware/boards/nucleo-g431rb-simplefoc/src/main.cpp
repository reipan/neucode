/**
 * main.cpp - PlatformIO / STM32Duino entry point
 * NUCLEO-G431RB + X-NUCLEO-IHM16M1 + SimpleFOC
 *
 * Control loop: NC_CONTROL_LOOP_HZ (defined in loop.h)
 *   - Sensor:      MT6701 via analogRead (arduino_sensor.cpp)
 *   - Actuator:    SimpleFOC closed-loop angle (simplefoc_actuator.cpp)
 *   - Comm:        STLINK UART Serial (arduino_comm.cpp)
 *   - Controllers: PID / ANN / SNN (shared firmware/controller/)
 *
 * Real-time strategy:
 *   TIM6 fires an ISR at NC_CONTROL_LOOP_HZ and sets a volatile flag.
 *   loop() services the flag so nc_loop_step() runs in thread context
 *   (avoids ISR / SimpleFOC thread-safety issues). SimpleFOC's loopFOC()
 *   runs every loop() iteration for maximum FOC update rate.
 */

#include <Arduino.h>
#include <HardwareTimer.h>
#include <math.h>
#include <string.h>
#include <stm32g4xx_hal.h>

extern "C" {
#include "actuator.h"
#include "ann.h"
#include "cmd_handler.h"
#include "controller.h"
#include "controller_types.h"
#include "external.h"
#include "loop.h"
#include "pid_adapter.h"
#include "sensor.h"
#include "snn.h"
#include "stm32_ext_spi.h"
}

// Forward declarations for platform-specific init functions
extern "C" void nc_sensor_mt6701_arduino_register(void);
extern "C" void nc_actuator_simplefoc_register(void);
extern void arduino_comm_init(void);
extern void arduino_comm_poll(void);

static volatile bool control_tick = false;
static double time_now_s = 0.0;

static void control_loop_isr() {
    control_tick = true;
}


void setup() {
    arduino_comm_init();
    Serial.println("=== NeuCoDe (SimpleFOC) ready ===");

    nc_sensor_mt6701_arduino_register();
    nc_sensor_init();

    nc_actuator_simplefoc_register();
    nc_actuator_init();
    nc_actuator_enable(true);

    nc_pid_controller_register();
    nc_ann_controller_register();
    nc_snn_controller_register();

    static SPI_HandleTypeDef hspi3 = { .Instance = SPI3 };
    static nc_ext_spi_config_t spi_cfg = {
        .hspi       = &hspi3,
        .spi_port   = GPIOC,
        .sck_pin    = GPIO_PIN_10,
        .miso_pin   = GPIO_PIN_11,
        .mosi_pin   = GPIO_PIN_12,
        .spi_af     = GPIO_AF6_SPI3,
        .drdy_port  = GPIOD,
        .drdy_pin   = GPIO_PIN_2,
        .timeout_ms = 20,
    };
    nc_ext_spi_hw_init(&spi_cfg);
    static nc_external_ctrl_t ext_ctrl = nc_ext_spi_make(&spi_cfg);
    nc_external_ctrl_register(&ext_ctrl);

    nc_loop_init();

    // TIM7 - basic timer with dedicated IRQ (TIM7_IRQn), avoids TIM6/DAC shared interrupt.
    HardwareTimer *ctrl_timer = new HardwareTimer(TIM7);
    ctrl_timer->setOverflow(NC_CONTROL_LOOP_HZ, HERTZ_FORMAT);
    ctrl_timer->attachInterrupt(control_loop_isr);
    ctrl_timer->resume();

    Serial.print("Control loop: ");
    Serial.print(NC_CONTROL_LOOP_HZ);
    Serial.println(" Hz (TIM7, hard real-time)");
}


void loop() {
    // SimpleFOC inner loop - runs as fast as possible for low-latency FOC.
    nc_actuator_loop();

    // Comm RX - non-blocking poll.
    arduino_comm_poll();

    // Control step - rate governed by TIM6 ISR, logic runs here (thread-safe).
    if (control_tick) {
        control_tick = false;
        time_now_s  += NC_CONTROL_LOOP_DT_S;
        nc_loop_step(time_now_s, NC_CONTROL_LOOP_DT_S);
    }
}
