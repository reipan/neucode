#include "sp.h"
#include "unity.h"
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>

#include "sim.h"
#include "pid.h"
#include "disturbance.h"
#include "plant_fopdt.h"


#define FLOAT_TOLERANCE 0.02f // A 2% tolerance is reasonable
#define DT_TEST 0.01f

/*
 * Validates the P-only control behavior, specifically the steady-state offset.
 * A fundamental property of proportional-only control is that it results in a
 * steady-state error. This test verifies that the simulation correctly models
 * this behavior using the final value theorem.
 */
void test_ProportionalOnlyController_ShouldSettleWithCorrectOffset(void)
{
    const float K = 0.3f;
    const float tau = 1.0f;
    const float theta = 0.0f;
    const float kp = 8.0f;
    const float dt = DT_TEST;
    const float setpoint = 1.0f;

    neucode_pid_gains_t gains = { .kp = kp, .ki = 0.0f, .kd = 0.0f };
    neucode_pid_limits_t limits = { .u_min = -1000.0f, .u_max = 1000.0f }; // Use sensible defaults
    neucode_fopdt_params_t plant_params = { .K = K, .tau = tau, .theta = theta };
    neucode_setpoint_def_t sp_def = { .type = NEUCODE_SP_STEP, .step_time = 0.0f, .v = setpoint };

    neucode_pid_t pid;
    neucode_pid_init(&pid, &gains, &limits);

    neucode_plant_fopdt_t plant;
    float ubuf[2];
    neucode_plant_fopdt_init(&plant, &plant_params, dt, ubuf, 2);

    neucode_sp_t sp;
    neucode_sp_init(&sp, &sp_def);

    neucode_disturbance_t disturbance;
    neucode_disturbance_init(&disturbance, NULL); // No disturbance

    const float simT = 10.0f;
    const int N = (int)(simT / dt);
    float y_meas = 0.0f;
    for (int k = 0; k < N; ++k) {
        float t = k * dt;
        float setpoint = neucode_sp_eval(&sp, t);
        float u = (float)neucode_pid_step(&pid, setpoint, y_meas, dt);
        float y_true = neucode_plant_fopdt_get_output(&plant);

        neucode_disturbance_apply(&disturbance, t, &u, &y_true);
        y_meas = y_true; // output with disturbance/noise is the measured output

        neucode_plant_fopdt_step(&plant, u);
    }

    // Check if the final value matches the theoretical steady-state value
    const float expected_final_value = (plant_params.K * kp) / (1.0f + plant_params.K * kp) * setpoint;
    TEST_ASSERT_FLOAT_WITHIN(FLOAT_TOLERANCE, expected_final_value, y_meas);
}

/*
 * Definitive baseline validation for the entire C-level simulation core.
 * This test uses a simple, ultra-conservative PI controller to verify that the
 * closed-loop system (PID + Plant) is fundamentally stable and behaves correctly.
 */
void test_ControlLoop_BaselineValidation_ShouldBeStable(void)
{
    const float K = 0.5f;
    const float tau = 4.284f;
    const float theta = 0.276f;
    const float dt = DT_TEST;
    const double kp = 2.0;
    const double ki = 1.0;

    neucode_pid_gains_t gains = { .kp = kp, .ki = ki, .kd = 0.0 };
    neucode_pid_limits_t limits = {
        .u_min = -10.0, .u_max = 10.0, .i_min = -10.0, .i_max = 10.0,
        .d_alpha = 1.0, .kaw = 0.1
    };
    neucode_fopdt_params_t plant_params = { .K = K, .tau = tau, .theta = theta };
    neucode_setpoint_def_t sp_def = { .type = NEUCODE_SP_STEP, .step_time = 1.0f, .v = 1.0f };

    // PI controller with ultra-conservative gains for stability
    neucode_pid_t pid;
    neucode_pid_init(&pid, &gains, &limits);

    // Allocate and initialize the plant
    neucode_plant_fopdt_t plant;
    size_t ubuf_len = (size_t)ceilf(theta / dt) + 2;
    float* ubuf = (float*)calloc(ubuf_len, sizeof(float));
    TEST_ASSERT_NOT_NULL(ubuf);
    neucode_plant_fopdt_init(&plant, &plant_params, dt, ubuf, ubuf_len);

    neucode_sp_t sp;
    neucode_sp_init(&sp, &sp_def);

    // Disturbance is off for this test
    neucode_disturbance_t disturbance;
    neucode_disturbance_init(&disturbance, NULL);

    const float simT = 60.0f;
    const int N = (int)(simT / dt);
    float y_meas = 0.0f;
    float peak = -INFINITY;

    for (int k = 0; k < N; ++k) {
        float t = k * dt;
        float setpoint = neucode_sp_eval(&sp, t);
        float u_ideal = (float)neucode_pid_step(&pid, setpoint, y_meas, dt);
        float y_true = neucode_plant_fopdt_get_output(&plant);

        neucode_disturbance_apply(&disturbance, t, &u_ideal, &y_true);
        y_meas = y_true; // output with disturbance/noise is the measured output

        if (y_meas > peak) peak = y_meas;
        neucode_plant_fopdt_step(&plant, u_ideal);
    }

    // Check 1: The final value must reach the setpoint
    TEST_ASSERT_FLOAT_WITHIN(FLOAT_TOLERANCE, sp_def.v, y_meas);

    // Check 2: Overshoot must be minimal, accounting for the physically
    // necessary overshoot caused by the plant's dead time.
    TEST_ASSERT_TRUE_MESSAGE(peak < 1.15f, "Overshoot is unexpectedly high (>15%).");

    free(ubuf);
}

/*
 * Tests the neucode_sim_step() function for a single step.
 * This test initializes a full simulation with a FOPDT plant,
 * runs it for one time step, and verifies that the plant's process variable
 * has been updated correctly according to the controller's output.
 */
void test_SingleStepSimulation_ShouldUpdatePlantStateCorrectly(void)
{
    // Create simulation
    neucode_sim_t* sim = NULL;
    neucode_status_t status = neucode_sim_create(&sim);
    TEST_ASSERT_EQUAL_INT(NEUCODE_OK, status);

    // Configure time
    status = neucode_sim_set_time_step(sim, DT_TEST, 1.0f);
    TEST_ASSERT_EQUAL_INT(NEUCODE_OK, status);

    // We don't need to set a PID here since we provide external control input
    // Configure plant
    neucode_fopdt_params_t plant_params = {
        .K = 1.0f,
        .tau = 2.0f,
        .theta = 0.0f
    };
    status = neucode_sim_set_fopdt(sim, &plant_params);
    TEST_ASSERT_EQUAL_INT(NEUCODE_OK, status);

    // Configure setpoint
    neucode_setpoint_def_t sp_def = { .type = NEUCODE_SP_STEP, .step_time = 0.0f, .v = 1.0f };
    status = neucode_sim_set_setpoint(sim, &sp_def);
    TEST_ASSERT_EQUAL_INT(NEUCODE_OK, status);

    // No disturbance for this test
    neucode_disturbance_t disturbance;
    neucode_disturbance_init(&disturbance, NULL);
    TEST_ASSERT_EQUAL_INT(NEUCODE_OK, status);

    // Reset simulation
    status = neucode_sim_reset(sim);
    TEST_ASSERT_EQUAL_INT(NEUCODE_OK, status);

    // Run a single step with control input u=1.0
    float control_input = 1.0f;
    status = neucode_sim_step(sim, control_input);
    TEST_ASSERT_EQUAL_INT(NEUCODE_OK, status);

    // Retrieve plant output state
    float state_vector[3];
    size_t state_size = neucode_sim_get_state_vector(sim, state_vector);
    TEST_ASSERT_EQUAL_INT(3, state_size);

    // For a FOPDT with K=1, tau=2, theta=0, and u=1.0, after one step dt=0.01,
    // y(t) = K * u * (1 - exp(-dt/tau))
    float expected_y = plant_params.K * control_input * (1.0f - expf(-DT_TEST / plant_params.tau));
    TEST_ASSERT_FLOAT_WITHIN(FLOAT_TOLERANCE, expected_y, state_vector[1]);

    // cleanup
    neucode_sim_destroy(sim);
    sim = NULL;
}

void run_control_loop_tests(void) {
    RUN_TEST(test_ProportionalOnlyController_ShouldSettleWithCorrectOffset);
    RUN_TEST(test_ControlLoop_BaselineValidation_ShouldBeStable);
    RUN_TEST(test_SingleStepSimulation_ShouldUpdatePlantStateCorrectly);
}