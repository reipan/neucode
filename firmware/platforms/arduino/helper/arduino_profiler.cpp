// DWT cycle-counter profiler - PlatformIO / STM32Duino
// Do NOT include profiler.h here (no extern "C" guards -> linkage conflict).
// C callers (ann.c, snn.c) include profiler.h and see C-linkage declarations.
// We define with extern "C" so the symbol names match.
#include <Arduino.h>

extern "C" void profiler_init(void) {
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0;
    DWT->CTRL  |= DWT_CTRL_CYCCNTENA_Msk;
}

extern "C" uint32_t profiler_get_cycles(void) {
    return DWT->CYCCNT;
}
