// Portable build for hosts without AVX-512 VNNI, via SIMDe (https://github.com/simd-everywhere/simde).
// Same results as the native build; not for timing.
#pragma once
#define SIMDE_ENABLE_NATIVE_ALIASES
#include <simde/x86/avx512.h>
#include <stdint.h>
static inline simde__m512i ygg_sra_epi32(simde__m512i a, simde__m128i c) {  // not in SIMDe
    int32_t t[16]; simde_mm512_storeu_si512(t, a); const int s = simde_mm_cvtsi128_si32(c);
    for (int i = 0; i < 16; i++) t[i] >>= s;  // arithmetic on negative int32 with every supported compiler (C++20: always)
    return simde_mm512_loadu_si512(t);
}
#undef _mm512_sra_epi32
#define _mm512_sra_epi32 ygg_sra_epi32
static inline int ygg_mask_reduce_add_epi32(simde__mmask16 k, simde__m512i a) {  // not in SIMDe
    return simde_mm512_reduce_add_epi32(simde_mm512_maskz_mov_epi32(k, a));
}
#undef _mm512_mask_reduce_add_epi32
#define _mm512_mask_reduce_add_epi32 ygg_mask_reduce_add_epi32
