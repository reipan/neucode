#include "unity.h"

/*
 * EXPLICIT DECLARATIONS
 */
extern void run_control_loop_tests(void);


void setUp(void) {}
void tearDown(void){}

/*
 * The main entry point for the entire test suite.
 * It calls the test suite "runner" from each individual test file.
 */
int main(void) {
    UNITY_BEGIN();

    // Run the test suites
    run_control_loop_tests();

    return UNITY_END();
}
