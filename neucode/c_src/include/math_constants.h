#pragma once

#ifdef __cplusplus
extern "C" {
#endif
    
// Math constants (not in C11 standard)
#ifndef M_PI
#define M_PI acos(-1.0)
#endif

#ifndef M_TAU
#define M_TAU (2.0 * M_PI)
#endif

#ifndef M_E
#define M_E 2.71828182845904523536
#endif

#ifdef __cplusplus
}
#endif