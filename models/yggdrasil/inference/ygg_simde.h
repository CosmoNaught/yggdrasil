// Portable build of ygg.cpp for hosts without AVX-512 VNNI: SIMDe emulates the
// AVX-512 intrinsics with exact integer / IEEE semantics. Correctness checks only; timings of this build are not the
// AVX-512 kernel's. Build: clang++ -DYGG_SIMDE -I<simde> -O3 -march=native -ffp-contract=off -std=c++17 ...
// (SIMDe: https://github.com/simd-everywhere/simde, header-only)
#pragma once
#define SIMDE_ENABLE_NATIVE_ALIASES
#include <simde/x86/avx512.h>
#include <stdint.h>
static inline simde__m512i ygg_sra_epi32(simde__m512i a, simde__m128i c) {  // not in SIMDe
    int32_t t[16]; simde_mm512_storeu_si512(t, a); const int s = simde_mm_cvtsi128_si32(c);
    for (int i = 0; i < 16; i++) t[i] >>= s;
    return simde_mm512_loadu_si512(t);
}
#undef _mm512_sra_epi32
#define _mm512_sra_epi32 ygg_sra_epi32
