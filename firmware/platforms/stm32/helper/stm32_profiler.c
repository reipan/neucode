#include "profiler.h"

/**
 * ARM Cortex-M4 Internal Registers
 *
 * Cycle Counter Register (DWT_CYCCNT @ 0xE0001004)
 * Control Register (DWT_CONTROL @ 0xE0001000)
 * Debug Exception and Monitor Control Register (SCB_DEMCR @ 0xE000EDFC)
 *
 * @see https://documentation-service.arm.com/static/5fce431be167456a35b36ade
 */
#define DWT_CYCCNT (*(volatile uint32_t*)0xE0001004)
#define DWT_CONTROL (*(volatile uint32_t*)0xE0001000)
#define SCB_DEMCR (*(volatile uint32_t*)0xE000EDFC)

/**
 * Initialize the profiler.
 *
 * Set bit 24 of DEMCR register called "TRCENA" to enable the trace and debug blocks.
 * Set cycle count DWT_CYCCNT to zero.
 * Enable the cycle counter by setting bit 0 (CYCCNTENA) of the DWT_CONTROL register.
 * 
 * @see https://developer.arm.com/documentation/ddi0403/d/Debug-Architecture/ARMv7-M-Debug/The-Data-Watchpoint-and-Trace-unit
 */
void profiler_init(void) {
    SCB_DEMCR |= 0x01000000;
    DWT_CYCCNT = 0;
    DWT_CONTROL |= 0x00000001;
}

/**
 * Get the current value of the cycle counter (DWT_CYCCNT).
 */
uint32_t profiler_get_cycles(void) {
    return DWT_CYCCNT;
}