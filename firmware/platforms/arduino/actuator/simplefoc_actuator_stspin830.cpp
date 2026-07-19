/**
 * SimpleFOC actuator - NUCLEO-G431RB + X-NUCLEO-IHM16M1 (STSPIN830).
 *
 * Pin mapping:
 *   TIM1 CH1/2/3  -> PA8 / PA9 / PA10  (phase PWM)
 *   EN_U/V/W      -> PB13 / PB14 / PB15 (half-bridge enable)
 */

#include <Arduino.h>
#include <SimpleFOC.h>

// STSPIN830 IC enable - must be HIGH before driver.init()
#define ENFAULT_PIN PB12

extern "C" {
#include "actuator.h"
#include "sensor.h"
float nc_sensor_read_physical(void);
}

// Motor and driver objects - GBM2804H-100T (7 pole pairs), STSPIN830 on IHM16M1

// GBM2804H-100T: 14-pole -> 7 pole pairs
static BLDCMotor motor = BLDCMotor(7);

// STSPIN830 on IHM16M1: 3 independent PWM + 3 enable pins
// EN pins are active-HIGH (same as original firmware)
static BLDCDriver3PWM driver = BLDCDriver3PWM(
    PA8,  PA9,  PA10,   // phase PWM  (TIM1 CH1/2/3)
    PB13, PB14, PB15    // enable U/V/W
);

// Bridge existing nc_sensor_read() into SimpleFOC's sensor interface.
// nc_sensor_read() returns UNWRAPPED degrees (continuously accumulating).
// GenericSensor expects the RAW wrapped angle in radians (0-2pi);
// SimpleFOC handles full-rotation counting internally.
// FOC uses the unfiltered angle: no IIR lag so initFOC() detects motor
// movement correctly and loopFOC() commutates with minimum latency.
// nc_sensor_read() (IIR-filtered) is used by the neucode controller only.
static GenericSensor foc_sensor([]() -> float {
    float deg = nc_sensor_read_physical();      // physical angle, ignores zero offset
    float wrapped = fmod(deg, 360.0f);          // wrap to 0-360deg
    if (wrapped < 0.0f) wrapped += 360.0f;
    return wrapped * (float)(PI / 180.0f);      // 0-2pi rad for SimpleFOC
}, []() {
    // No extra init needed - sensor is already initialised via nc_sensor_init()
});

// Actuator callbacks registered with actuator.c

static void init_cb(void) {
    // Enable STSPIN830 IC
    pinMode(ENFAULT_PIN, OUTPUT);
    digitalWrite(ENFAULT_PIN, HIGH);
    delay(100);

    foc_sensor.init();

    driver.voltage_power_supply = 12.0f;
    driver.voltage_limit = 6.0f;
    driver.pwm_frequency = 40000;
    driver.init();

    motor.linkDriver(&driver);
    motor.linkSensor(&foc_sensor);

    motor.voltage_sensor_align = 3.0f;
    motor.voltage_limit   = 3.0f;
    motor.velocity_limit  = 50.0f;
    motor.controller = MotionControlType::torque;

    motor.useMonitoring(Serial);
    motor.init();
    motor.initFOC();
}

static void enable_cb(bool en) {
    if (en) {
        motor.enable();
    } else {
        motor.disable();
    }
}

static void loop_cb(void) {
    // Called every main loop() iteration for fast FOC commutation.
    motor.loopFOC();
}

static void set_cb(float u) {
    // u is normalised [-1, 1] from the neucode controller.
    // SimpleFOC torque mode: move(v) applies v as Vq (volts).
    // Scale u by motor.voltage_limit so u=+/-1 uses full available torque.
    motor.move(-u * motor.voltage_limit);
}


extern "C" void nc_actuator_simplefoc_register(void) {
    nc_actuator_register(init_cb, enable_cb, set_cb, loop_cb);
}
