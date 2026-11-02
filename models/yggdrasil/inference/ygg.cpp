// Yggdrasil int8 inference: single stream, single core, AVX-512 VNNI. Model files come from export.py.
//
// Usage: ygg logits model.bin x.i32 nwin T out.f32 [--blocked] [--prefetch]
//        ygg score  model.bin x.i32 y.i32 nwin T [--blocked] [--prefetch]
//        ygg time   model.bin x.i32 [--seg S] [--warm K] [--blocked] [--prefetch]
//        ygg bench  model.bin [--secs S] [--rounds K] [--tokens x.i32] [--no-logits] [--blocked] [--prefetch] [--randleaf K]
// --blocked walks 4 tree levels per step; --prefetch prefetches the leaf record. YGG_THP=0 disables huge pages.
#define _GNU_SOURCE 1
#ifdef YGG_SIMDE
#include "ygg_simde.h"
#else
#include <immintrin.h>
#endif
#include <math.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <algorithm>
#include <map>
#include <string>
#include <vector>

// exp(x) = 2^n p(f), t = x log2(e) = n + f
static inline __m512 vexp(__m512 x) {
    x = _mm512_min_ps(_mm512_max_ps(x, _mm512_set1_ps(-87.f)), _mm512_set1_ps(88.f));
    __m512 t = _mm512_mul_ps(x, _mm512_set1_ps(1.4426950408889634f));
    __m512 n = _mm512_roundscale_ps(t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m512 f = _mm512_sub_ps(t, n);
    __m512 p = _mm512_set1_ps(1.5353362e-4f);
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(1.3398874e-3f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(9.6181291e-3f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(5.5504109e-2f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(2.4022651e-1f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(6.9314718e-1f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(1.0f));
    return _mm512_scalef_ps(p, n);
}
static inline __mmask16 vmask(int n) { return n >= 16 ? (__mmask16)0xFFFF : n <= 0 ? (__mmask16)0 : (__mmask16)((1u << n) - 1); }
static float softmax(const float *lg, float *prob, int V) {  // returns log sum exp
    __m512 mx = _mm512_set1_ps(-INFINITY);
    for (int i = 0; i < V; i += 16) mx = _mm512_mask_max_ps(mx, vmask(V - i), mx, _mm512_loadu_ps(lg + i));
    const float mxs = _mm512_reduce_max_ps(mx); const __m512 mv = _mm512_set1_ps(mxs); __m512 sm = _mm512_setzero_ps();
    for (int i = 0; i < V; i += 16) {
        __m512 e = _mm512_maskz_mov_ps(vmask(V - i), vexp(_mm512_sub_ps(_mm512_maskz_loadu_ps(vmask(V - i), lg + i), mv)));
        _mm512_mask_storeu_ps(prob + i, vmask(V - i), e); sm = _mm512_add_ps(sm, e);
    }
    const float tot = _mm512_reduce_add_ps(sm); const __m512 inv = _mm512_set1_ps(1.f / tot);
    for (int i = 0; i < V; i += 16) _mm512_mask_storeu_ps(prob + i, vmask(V - i), _mm512_mul_ps(_mm512_maskz_loadu_ps(vmask(V - i), prob + i), inv));
    return mxs + logf(tot);
}

static double now() { timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

struct Lay {
    char *rec; int RS, oA, oU, oP, oT, TS;   // record bytes, offsets of A / U_low / P / T (-1: absent), T row stride
    const uint8_t *P; size_t PS;             // readout of leaf r at P + r * PS (shared_out: PS = 0)
    const uint8_t *Q; float mz, sq;          // class map, scales
    const uint8_t *w; const float *thr;      // branch: node rows (u8, w + 128) [2^D][n], thresholds [V][2^D]
    const uint8_t *vl; const float *mv; float eps, mul;
    const float *mulv, *r2b, *mzl;           // group scales (null: per tensor): m_ul per coordinate,
                                             // r^2 of the layer below per 16 coordinates, m_z per leaf and row
    const int8_t *vt; const float *mT; int oF, oC, oS;  // factored input (null: table T)
    const float *sqv;                                    // head scale per class (null: one sq)
    uint8_t *wb; float *thb; int NB, cbase[4];  // blocked walk layout, built at load
    int ck, cbits; uint32_t cmod;               // root context: k (0 = none), hash bits (0 = exact), V^(k-1)
};
struct Model {
    int L, n, b, V, D, R, R2, shared, F, rank, sh, NL, VP;
    float *head_b; Lay lay[8]; size_t bytes;
};

static void *amalloc2m(size_t sz, bool thp) {
    size_t al = (sz + (2u << 20) - 1) / (2u << 20) * (2u << 20);
    void *p; if (posix_memalign(&p, 2u << 20, al)) { fprintf(stderr, "out of memory\n"); exit(1); }
    if (thp) madvise(p, al, MADV_HUGEPAGE);
    return p;
}
#ifndef MADV_COLLAPSE
#define MADV_COLLAPSE 25
#endif
// Collapse into huge pages now (Linux >= 6.1) instead of waiting for khugepaged.
static void collapse2m(void *p, size_t sz, bool thp) {
    if (!thp) return;
    const size_t al = (sz + (2u << 20) - 1) / (2u << 20) * (2u << 20);
    madvise(p, al, MADV_COLLAPSE);
}
static long anon_huge_kb() {
    FILE *f = fopen("/proc/self/smaps_rollup", "r"); if (!f) return -1;
    char line[256]; long v = -1;
    while (fgets(line, sizeof line, f)) if (!strncmp(line, "AnonHugePages:", 14)) v = atol(line + 14);
    fclose(f); return v;
}

// Blocked layout: the nodes of each 4-level subtree (heap order) in 16 consecutive slots.
// rows wb[(blk * 16 + o) * n], thresholds thb[(x * NB + blk) * 16 + o]; chunk c starts at level 4c.
static void build_blocked(const Model &m, Lay &y, bool thp) {
    const int D = m.D, NL = m.NL, nch = (D + 3) / 4, N = m.n, V = m.V;
    int NB = 0;
    for (int c = 0; c < nch; c++) { y.cbase[c] = NB; NB += 1 << (4 * c); }
    y.NB = NB;
    const size_t wsz = (size_t)NB * 16 * N, tsz = (size_t)V * NB * 16;
    y.wb = (uint8_t *)amalloc2m(wsz, thp); y.thb = (float *)amalloc2m(tsz * 4, thp);
    memset(y.wb, 128, wsz);
    for (size_t i = 0; i < tsz; i++) y.thb[i] = INFINITY;  // pad slots: never taken
    for (int c = 0; c < nch; c++) {
        const int lk = D - 4 * c < 4 ? D - 4 * c : 4;
        for (int rt = 0; rt < (1 << (4 * c)); rt++) {
            const int idx = (1 << (4 * c)) + rt, b = y.cbase[c] + rt;
            for (int j = 0, o = 0; j < lk; j++)
                for (int i = 0; i < (1 << j); i++, o++) {
                    const int node = (idx << j) + i;
                    memcpy(y.wb + ((size_t)b * 16 + o) * N, y.w + (size_t)node * N, N);
                    for (int x = 0; x < V; x++) y.thb[((size_t)x * NB + b) * 16 + o] = y.thr[(size_t)x * NL + node];
                }
        }
    }
}

static Model load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    char mg[4]; int32_t h[12];
    if (fread(mg, 1, 4, f) != 4 || memcmp(mg, "YGG1", 4) || fread(h, 4, 12, f) != 12) { fprintf(stderr, "bad model file\n"); exit(1); }
    Model m{};
    m.L = h[0]; m.n = h[1]; m.b = h[2]; m.V = h[3]; m.D = h[4]; m.R = h[5]; m.R2 = h[6]; m.shared = h[7];
    m.F = h[8]; m.rank = h[9]; m.sh = h[10];
    const int na = h[11];
    m.NL = 1 << m.D; m.VP = (m.V + 31) / 32 * 32;
    if ((m.n != 64 && m.n != 128 && m.n != 256) || (m.b != 8 && m.b != 16) || (m.R != 8 && m.R != 16 && m.R != 32) || m.R2 > 64 ||
        m.R2 % 4 || m.L < 1 || m.L > 8 || m.V > 256 || m.D > 16 || (m.L > 1 && m.D < 1)) {
        fprintf(stderr, "unsupported model\n"); exit(1);
    }
    struct E { char name[32]; int64_t off, nb; };
    std::vector<E> tab(na);
    for (auto &e : tab) if (fread(&e, sizeof(E), 1, f) != 1) exit(1);
    long pos = ftell(f); pos += (64 - pos % 64) % 64;
    fseek(f, 0, SEEK_END); long end = ftell(f); fseek(f, pos, SEEK_SET);
    const char *thpv = getenv("YGG_THP");
    const bool thp = !thpv || strcmp(thpv, "0");
    const size_t sz = (size_t)(end - pos);
    char *blob = (char *)amalloc2m(sz + 128, thp);
    memset(blob + sz, 0, 128);
    for (size_t o = 0; o < sz;) {
        size_t c = fread(blob + o, 1, sz - o < (64u << 20) ? sz - o : (64u << 20), f);
        if (!c) { fprintf(stderr, "short model file\n"); exit(1); }
        o += c;
    }
    fclose(f);
    collapse2m(blob, sz + 128, thp);
    m.bytes = sz;
    std::map<std::string, char *> arr;
    for (auto &e : tab) arr[e.name] = blob + e.off;
    auto g = [&](const std::string &s) -> char * { auto it = arr.find(s); return it == arr.end() ? nullptr : it->second; };
    auto need = [&](const std::string &s) -> char * { char *p = g(s); if (!p) { fprintf(stderr, "missing %s\n", s.c_str()); exit(1); } return p; };
    m.head_b = (float *)need("head_b");
    for (int l = 0; l < m.L; l++) {
        const std::string P = "l" + std::to_string(l) + "_";
        Lay &y = m.lay[l];
        const int32_t *mt = (const int32_t *)need(P + "meta");
        y.RS = mt[0]; y.oA = mt[1]; y.oU = mt[2]; y.oP = mt[3]; y.oT = mt[4];
        y.vt = (const int8_t *)g(P + "vt");
        if (y.vt) {
            if (m.rank % 4 || m.rank > 64 || m.rank < 4) { fprintf(stderr, "factored: rank %d\n", m.rank); exit(1); }
            y.oF = mt[5]; y.oC = mt[6]; y.oS = mt[7]; y.mT = (const float *)need(P + "mT");
        }
        if (l == 0 && g(P + "ctx")) {
            const int32_t *cx = (const int32_t *)g(P + "ctx");
            y.ck = cx[0]; y.cmod = (uint32_t)cx[1]; y.cbits = cx[2];
        }
        y.TS = l == 0 && !y.ck ? 0 : 2 * m.n;
        y.rec = need(P + "rec");
        if (m.shared) { y.P = (const uint8_t *)need(P + "P"); y.PS = 0; }
        else { y.P = (const uint8_t *)y.rec + y.oP; y.PS = y.RS; }
        y.Q = (const uint8_t *)need(P + "Q");
        y.mzl = (const float *)g(P + "mzl"); y.sqv = (const float *)g(P + "sqv");
        const float *sc = (const float *)need(P + "scal");  // mz, sq, eps, mul
        y.mz = sc[0]; y.sq = sc[1]; y.eps = sc[2]; y.mul = sc[3];
        if (l > 0) {
            y.w = (const uint8_t *)need(P + "w"); y.thr = (const float *)need(P + "thr");
            if (m.R2) { y.vl = (const uint8_t *)need(P + "vl"); y.mv = (const float *)need(P + "mv"); }
            y.mulv = (const float *)g(P + "mulv"); y.r2b = (const float *)g(P + "r2b");
            build_blocked(m, y, thp);
            collapse2m(y.wb, (size_t)y.NB * 16 * m.n, thp);
            collapse2m(y.thb, (size_t)m.V * y.NB * 16 * 4, thp);
        }
    }
    return m;
}

// ---------------------------------------------------------------------------------------------------- primitives

template <int N> static inline int32_t sumN(const int8_t *q) {  // sum of N int8 (N a multiple of 64)
    __m512i v = _mm512_sad_epu8(_mm512_xor_si512(_mm512_loadu_si512(q), _mm512_set1_epi8((char)0x80)), _mm512_setzero_si512());
    for (int c = 64; c < N; c += 64)
        v = _mm512_add_epi64(v, _mm512_sad_epu8(_mm512_xor_si512(_mm512_loadu_si512(q + c), _mm512_set1_epi8((char)0x80)), _mm512_setzero_si512()));
    return (int32_t)_mm512_reduce_add_epi64(v) - 128 * N;
}
// 16 vectors of 16 int32 -> one vector, lane i = sum of v[i]
static inline __m512i reduce16(const __m512i *v) {
    __m512i t[8], u[4], w0, w1;
    for (int i = 0; i < 8; i++) t[i] = _mm512_add_epi32(_mm512_unpacklo_epi32(v[2 * i], v[2 * i + 1]), _mm512_unpackhi_epi32(v[2 * i], v[2 * i + 1]));
    for (int i = 0; i < 4; i++) u[i] = _mm512_add_epi32(_mm512_unpacklo_epi64(t[2 * i], t[2 * i + 1]), _mm512_unpackhi_epi64(t[2 * i], t[2 * i + 1]));
    w0 = _mm512_add_epi32(_mm512_shuffle_i32x4(u[0], u[1], 0x88), _mm512_shuffle_i32x4(u[0], u[1], 0xDD));
    w1 = _mm512_add_epi32(_mm512_shuffle_i32x4(u[2], u[3], 0x88), _mm512_shuffle_i32x4(u[2], u[3], 0xDD));
    return _mm512_add_epi32(_mm512_shuffle_i32x4(w0, w1, 0x88), _mm512_shuffle_i32x4(w0, w1, 0xDD));
}
// 16 partial sums of dot(w + 128, q) over N features (caller subtracts 128 sum q)
template <int N> static inline __m512i dotv(const uint8_t *w, const int8_t *q) {
    __m512i a = _mm512_dpbusd_epi32(_mm512_setzero_si512(), _mm512_loadu_si512(w), _mm512_loadu_si512(q));
    for (int c = 64; c < N; c += 64) a = _mm512_dpbusd_epi32(a, _mm512_loadu_si512(w + c), _mm512_loadu_si512(q + c));
    return a;
}
// rho = 1 / sqrt(mean(q^2) + eps), exact integer sum of squares
template <int N> static inline float rmsr(const int8_t *q, float eps) {
    __m512i a = _mm512_setzero_si512();
    for (int c = 0; c < N; c += 32) {
        const __m512i v = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i *)(q + c)));
        a = _mm512_add_epi32(a, _mm512_madd_epi16(v, v));
    }
    return 1.0f / sqrtf((float)_mm512_reduce_add_epi32(a) * (1.0f / N) + eps);
}

// Group scales: rho = 1 / sqrt(sum_g r2[g] S_g / N + eps), S_g = exact sum of q^2 over group g.
// The f32 sum runs in group order, as in export.int_forward.
template <int N> static inline float rmsr_g(const int8_t *q, const float *r2, float eps) {
    float s = 0.0f;
    for (int c = 0; c < N; c += 32) {
        const __m512i v = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i *)(q + c)));
        const __m512i a = _mm512_madd_epi16(v, v);  // lanes 0-7: coordinates c .. c+15, lanes 8-15: c+16 .. c+31
        s = s + r2[c / 16] * (float)_mm512_mask_reduce_add_epi32(0x00FF, a);
        s = s + r2[c / 16 + 1] * (float)_mm512_mask_reduce_add_epi32(0xFF00, a);
    }
    return 1.0f / sqrtf(s * (1.0f / N) + eps);
}

// ---------------------------------------------------------------------------------------------------- the walk

template <int N> static inline int walk(const Model &m, const Lay &y, int x, const int8_t *qb, float rho) {
    const int32_t corr = 128 * sumN<N>(qb);
    const float *th = y.thr + (size_t)x * m.NL;
    int ix = 1;
    for (int d = 0; d < m.D; d++) {
        const int32_t dot = _mm512_reduce_add_epi32(dotv<N>(y.w + (size_t)ix * N, qb)) - corr;
        ix = 2 * ix + ((float)dot * rho > th[ix]);
    }
    return ix - m.NL;
}
template <int N> static inline int walk_blocked(const Model &m, const Lay &y, int x, const int8_t *qb, float rho) {
    const __m512i corr = _mm512_set1_epi32(128 * sumN<N>(qb));
    const __m512 rv = _mm512_set1_ps(rho);
    int ix = 1;
    for (int c = 0, d = 0; d < m.D; c++, d += 4) {
        const int lk = m.D - d < 4 ? m.D - d : 4, nn = (1 << lk) - 1;
        const int b = y.cbase[c] + ix - (1 << d);
        const uint8_t *W = y.wb + (size_t)b * 16 * N;
        __m512i v[16];
        if (nn == 15) { for (int o = 0; o < 15; o++) v[o] = dotv<N>(W + N * o, qb); }
        else for (int o = 0; o < 15; o++) v[o] = o < nn ? dotv<N>(W + N * o, qb) : _mm512_setzero_si512();
        v[15] = _mm512_setzero_si512();
        const __m512i sum = _mm512_sub_epi32(reduce16(v), corr);
        const float *tb = y.thb + ((size_t)x * y.NB + b) * 16;
        const uint32_t mk = _mm512_cmp_ps_mask(_mm512_mul_ps(_mm512_cvtepi32_ps(sum), rv), _mm512_loadu_ps(tb), _CMP_GT_OQ);
        int p = 0;
        for (int j = 0; j < lk; j++) p = 2 * p + 1 + ((mk >> p) & 1);
        ix = (ix << lk) + p - nn;
    }
    return ix - m.NL;
}
template <int N> static inline void pf_leaf(const Lay &y, int r, int x) {
    const char *b = y.rec + (size_t)r * y.RS;
    if (y.vt) { for (int o = 0; o < y.RS; o += 64) _mm_prefetch(b + o, _MM_HINT_T0); return; }
    for (int o = 0; o < y.oT; o += 64) _mm_prefetch(b + o, _MM_HINT_T0);
    const char *t = b + y.oT + (size_t)x * y.TS;
    for (int o = 0; o < 2 * N; o += 64) _mm_prefetch(t + o, _MM_HINT_T0);
}

// ---------------------------------------------------------------------------------------------------- leaf update

// zl[0 .. 16 RG) = sat8(rint(f32(V_low q_below) m_v rho)); vl u8 [RG][N/4][16][4], RG = ceil(R2 / 16)
template <int N> static inline void lowin(const Model &m, const Lay &y, const int8_t *qb, int8_t *zl, float rho) {
    const int RG = (m.R2 + 15) / 16;
    const int32_t corr = 128 * sumN<N>(qb);
    for (int gi = 0; gi < RG; gi++) {
        __m512i a = _mm512_setzero_si512(), b = a, e = a, f = a;
        const uint8_t *v = y.vl + (size_t)gi * N * 16;
        for (int kc = 0; kc < N / 4; kc += 4) {
            a = _mm512_dpbusd_epi32(a, _mm512_loadu_si512(v + kc * 64), _mm512_set1_epi32(*(const int32_t *)(qb + 4 * kc)));
            b = _mm512_dpbusd_epi32(b, _mm512_loadu_si512(v + kc * 64 + 64), _mm512_set1_epi32(*(const int32_t *)(qb + 4 * kc + 4)));
            e = _mm512_dpbusd_epi32(e, _mm512_loadu_si512(v + kc * 64 + 128), _mm512_set1_epi32(*(const int32_t *)(qb + 4 * kc + 8)));
            f = _mm512_dpbusd_epi32(f, _mm512_loadu_si512(v + kc * 64 + 192), _mm512_set1_epi32(*(const int32_t *)(qb + 4 * kc + 12)));
        }
        a = _mm512_sub_epi32(_mm512_add_epi32(_mm512_add_epi32(a, b), _mm512_add_epi32(e, f)), _mm512_set1_epi32(corr));
        __m512 zf = _mm512_mul_ps(_mm512_cvtepi32_ps(a), _mm512_loadu_ps(y.mv + 16 * gi));
        zf = _mm512_mul_ps(zf, _mm512_set1_ps(rho));
        __m512i zi = _mm512_cvtps_epi32(zf);
        zi = _mm512_max_epi32(_mm512_min_epi32(zi, _mm512_set1_epi32(127)), _mm512_set1_epi32(-127));
        _mm_storeu_si128((__m128i *)(zl + 16 * gi), _mm512_cvtepi32_epi8(zi));
    }
}

// vpermd indices for b = 8: lane i of 16-row group gg reads the q pair (cols 2p, 2p+1) of its own block
struct Idx { __m512i v[2][8]; };
static Idx make_idx8() {
    Idx r;
    for (int h = 0; h < 2; h++) for (int p = 0; p < 8; p++) {
        alignas(64) int32_t t[16];
        for (int i = 0; i < 16; i++) t[i] = 8 * h + (i / 8) * 4 + (p % 4);
        r.v[h][p] = _mm512_load_si512(t);
    }
    return r;
}
static const Idx IDX8 = make_idx8();

// q (N int8) <- sat8(A16_r q >> sh (rounded) + T_r[x] + rint(f32(U_r zl) m_ul)); A16 [N/16][BLK/2][16][2] int16,
// U_low [R2/4][N][4] u8 (u + 128)
template <int BLK, int LOW, int N>
static inline void update(const Model &m, const Lay &y, int r, int x, int8_t *q, const int8_t *zl) {
    constexpr int NG = N / 16, NQ = N / 32;
    const char *rec = y.rec + (size_t)r * y.RS;
    __m512i acc[NG], q16[NQ];
    for (int i = 0; i < NQ; i++) q16[i] = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i *)(q + 32 * i)));
    const int16_t *A = (const int16_t *)(rec + y.oA);
    alignas(64) int16_t qh[N];
    if (BLK == 16) for (int i = 0; i < NQ; i++) _mm512_store_si512(qh + 32 * i, q16[i]);
    const __m512i rnd = _mm512_set1_epi32(1 << (m.sh - 1));
    const __m128i shc = _mm_cvtsi32_si128(m.sh);
    for (int gg = 0; gg < NG; gg++) {  // 4 independent accumulator chains per group
        __m512i a[4] = {_mm512_setzero_si512(), _mm512_setzero_si512(), _mm512_setzero_si512(), _mm512_setzero_si512()};
        const __m512i qs = q16[gg >> 1];
        for (int p = 0; p < BLK / 2; p++) {
            const __m512i x0 = BLK == 16 ? _mm512_set1_epi32(*(const int32_t *)(qh + 16 * gg + 2 * p))
                                         : _mm512_permutexvar_epi32(IDX8.v[gg & 1][p], qs);
            a[p & 3] = _mm512_dpwssd_epi32(a[p & 3], _mm512_loadu_si512(A + (gg * BLK / 2 + p) * 32), x0);
        }
        acc[gg] = _mm512_sra_epi32(_mm512_add_epi32(_mm512_add_epi32(_mm512_add_epi32(a[0], a[1]), _mm512_add_epi32(a[2], a[3])), rnd), shc);
    }
    if (y.vt) {  // factored: T_r[x] = rint(f32(U_r v(x)) f32(su_r mT)) + c_r; U_r [rank/4][N][4] u8 (u + 128)
        const uint8_t *uf = (const uint8_t *)(rec + y.oF);
        const int8_t *v = y.vt + (size_t)x * m.rank;
        int32_t vs = 0; for (int j = 0; j < m.rank; j++) vs += v[j];
        __m512i lt[NG];
        for (int b = 0; b < NG; b++) lt[b] = _mm512_setzero_si512();
        for (int kc = 0; kc < m.rank / 4; kc++) {
            const __m512i vb = _mm512_set1_epi32(*(const int32_t *)(v + 4 * kc));
            for (int b = 0; b < NG; b++) lt[b] = _mm512_dpbusd_epi32(lt[b], _mm512_loadu_si512(uf + ((size_t)kc * NG + b) * 64), vb);
        }
        const int16_t *C = (const int16_t *)(rec + y.oC);
        const __m512 su = _mm512_set1_ps(*(const float *)(rec + y.oS)); const __m512i cv = _mm512_set1_epi32(128 * vs);
        for (int b = 0; b < NG; b++) {
            const __m512 mul = _mm512_mul_ps(su, _mm512_loadu_ps(y.mT + 16 * b));
            const __m512i t = _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(lt[b], cv)), mul));
            acc[b] = _mm512_add_epi32(acc[b], _mm512_add_epi32(t, _mm512_cvtepi16_epi32(_mm256_loadu_si256((const __m256i *)(C + 16 * b)))));
        }
    } else {
        const int16_t *T = (const int16_t *)(rec + y.oT + (size_t)x * y.TS);
        for (int b = 0; b < NG; b++) acc[b] = _mm512_add_epi32(acc[b], _mm512_cvtepi16_epi32(_mm256_loadu_si256((const __m256i *)(T + 16 * b))));
    }
    if (LOW) {
        const uint8_t *u = (const uint8_t *)(rec + y.oU);
        int32_t zs = 0; for (int j = 0; j < m.R2; j++) zs += zl[j];
        __m512i la[NG];
        for (int b = 0; b < NG; b++) la[b] = _mm512_setzero_si512();
        for (int kc = 0; kc < m.R2 / 4; kc++) {
            const __m512i zb = _mm512_set1_epi32(*(const int32_t *)(zl + 4 * kc));
            for (int b = 0; b < NG; b++) la[b] = _mm512_dpbusd_epi32(la[b], _mm512_loadu_si512(u + ((size_t)kc * NG + b) * 64), zb);
        }
        const __m512i cz = _mm512_set1_epi32(128 * zs);
        if (y.mulv) {  // m_ul per coordinate
            for (int b = 0; b < NG; b++) acc[b] = _mm512_add_epi32(acc[b], _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(la[b], cz)), _mm512_loadu_ps(y.mulv + 16 * b))));
        } else {
            const __m512 mu = _mm512_set1_ps(y.mul);
            for (int b = 0; b < NG; b++) acc[b] = _mm512_add_epi32(acc[b], _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(la[b], cz)), mu)));
        }
    }
    for (int b = 0; b < NG; b++) _mm_storeu_si128((__m128i *)(q + 16 * b), _mm512_cvtsepi32_epi8(acc[b]));
}

// ---------------------------------------------------------------------------------------------------- readout

// lg[0 .. VP) = base + f32(Q zq) sq, zq = sat8(rint(f32(P_r q) m_z)); P u8 (p + 128): R = 8 [N/8][16 lanes][4] (lane
// 2r + e holds P[r][8c + 4e .. +3]), R >= 16 blocks of 16 rows [N/4][16][4]; Q u8 [R/4][VP][4]
template <int R, int N>
static inline void readout(const Model &m, const Lay &y, int r, const int8_t *q, float *lg, const float *base) {
    const uint8_t *P = y.P + (size_t)r * y.PS;
    const float *mz = y.mzl ? y.mzl + (m.shared ? 0 : (size_t)r * (R == 8 ? 16 : R)) : nullptr;
    alignas(64) int8_t zq[32];
    const int32_t corr = 128 * sumN<N>(q);
    if (R == 8) {
        __m512i a = _mm512_setzero_si512(), b = a;
        for (int c = 0; c < N / 8; c += 2) {
            a = _mm512_dpbusd_epi32(a, _mm512_loadu_si512(P + 64 * c), _mm512_set1_epi64(*(const int64_t *)(q + 8 * c)));
            b = _mm512_dpbusd_epi32(b, _mm512_loadu_si512(P + 64 * c + 64), _mm512_set1_epi64(*(const int64_t *)(q + 8 * c + 8)));
        }
        a = _mm512_add_epi32(a, b);
        a = _mm512_add_epi32(a, _mm512_shuffle_epi32(a, (_MM_PERM_ENUM)0xB1));
        a = _mm512_sub_epi32(a, _mm512_set1_epi32(corr));
        __m512i zi = _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(a), mz ? _mm512_loadu_ps(mz) : _mm512_set1_ps(y.mz)));
        zi = _mm512_max_epi32(_mm512_min_epi32(zi, _mm512_set1_epi32(127)), _mm512_set1_epi32(-127));
        _mm_storel_epi64((__m128i *)zq, _mm512_cvtepi32_epi8(_mm512_maskz_compress_epi32(0x5555, zi)));
    } else {
        for (int rb = 0; rb < R / 16; rb++) {
            const uint8_t *Pb = P + (size_t)rb * 16 * N;
            __m512i a = _mm512_setzero_si512(), b = a, e = a, f = a;
            for (int c = 0; c < N / 4; c += 4) {
                a = _mm512_dpbusd_epi32(a, _mm512_loadu_si512(Pb + 64 * c), _mm512_set1_epi32(*(const int32_t *)(q + 4 * c)));
                b = _mm512_dpbusd_epi32(b, _mm512_loadu_si512(Pb + 64 * c + 64), _mm512_set1_epi32(*(const int32_t *)(q + 4 * c + 4)));
                e = _mm512_dpbusd_epi32(e, _mm512_loadu_si512(Pb + 64 * c + 128), _mm512_set1_epi32(*(const int32_t *)(q + 4 * c + 8)));
                f = _mm512_dpbusd_epi32(f, _mm512_loadu_si512(Pb + 64 * c + 192), _mm512_set1_epi32(*(const int32_t *)(q + 4 * c + 12)));
            }
            a = _mm512_sub_epi32(_mm512_add_epi32(_mm512_add_epi32(a, b), _mm512_add_epi32(e, f)), _mm512_set1_epi32(corr));
            __m512i zi = _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(a), mz ? _mm512_loadu_ps(mz + 16 * rb) : _mm512_set1_ps(y.mz)));
            zi = _mm512_max_epi32(_mm512_min_epi32(zi, _mm512_set1_epi32(127)), _mm512_set1_epi32(-127));
            _mm_store_si128((__m128i *)(zq + 16 * rb), _mm512_cvtepi32_epi8(zi));
        }
    }
    int32_t zs = 0; for (int j = 0; j < R; j++) zs += zq[j];
    const __m512i cz = _mm512_set1_epi32(128 * zs); const __m512 s = _mm512_set1_ps(y.sq);
    for (int c = 0; c < m.VP; c += 32) {  // two 16-class chunks per pass
        __m512i a0 = _mm512_setzero_si512(), a1 = a0;
        for (int kc = 0; kc < R / 4; kc++) {
            const __m512i zb = _mm512_set1_epi32(*(const int32_t *)(zq + 4 * kc));
            a0 = _mm512_dpbusd_epi32(a0, _mm512_loadu_si512(y.Q + ((size_t)kc * m.VP + c) * 4), zb);
            a1 = _mm512_dpbusd_epi32(a1, _mm512_loadu_si512(y.Q + ((size_t)kc * m.VP + c + 16) * 4), zb);
        }
        const __m512 s0 = y.sqv ? _mm512_loadu_ps(y.sqv + c) : s, s1 = y.sqv ? _mm512_loadu_ps(y.sqv + c + 16) : s;
        _mm512_store_ps(lg + c, _mm512_add_ps(_mm512_load_ps(base + c), _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(a0, cz)), s0)));
        _mm512_store_ps(lg + c + 16, _mm512_add_ps(_mm512_load_ps(base + c + 16), _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(a1, cz)), s1)));
    }
}

// ---------------------------------------------------------------------------------------------------- one token

enum { F_NOLOGITS = 1, F_PFL = 2, F_WBLK = 4, F_RANDLEAF = 8 };
struct State { int8_t *q; float *lg; uint64_t t; int randK; uint32_t c; };  // c: rolling context of the root

static inline int randleaf(const State &st, int l, int x, int r) {
    uint64_t h = st.t * 0x9E3779B97F4A7C15ull + (uint64_t)l * 0xD6E8FEB86659FD93ull + x;
    h ^= h >> 29; h *= 0xBF58476D1CE4E5B9ull; h ^= h >> 32;
    return (int)((r + h) & (uint64_t)(st.randK - 1));  // keeps the data dependency on the walked leaf
}

template <int BLK, int R, int LOW, int N>
static void step(const Model &m, State &st, int x, int fl) {
    for (int l = 0; l < m.L; l++) {
        const Lay &y = m.lay[l];
        int8_t *q = st.q + l * N;
        const int8_t *qb = q - N;
        int r = x; float rho = 1.0f;
        if (l == 0 && y.ck) {
            st.c = (st.c % y.cmod) * (uint32_t)m.V + (uint32_t)x;
            r = y.cbits ? (int)((st.c * 2654435761u) >> (32 - y.cbits)) : (int)st.c;
        }
        if (l > 0) {
            rho = y.r2b ? rmsr_g<N>(qb, y.r2b, y.eps) : rmsr<N>(qb, y.eps);
            r = (fl & F_WBLK) ? walk_blocked<N>(m, y, x, qb, rho) : walk<N>(m, y, x, qb, rho);
            if (fl & F_RANDLEAF) r = randleaf(st, l, x, r);
        }
        if (fl & F_PFL) pf_leaf<N>(y, r, x);
        if (LOW && l > 0) {
            alignas(16) int8_t zl[64];
            lowin<N>(m, y, qb, zl, rho);
            update<BLK, 1, N>(m, y, r, x, q, zl);
        } else update<BLK, 0, N>(m, y, r, x, q, nullptr);
        if (!(fl & F_NOLOGITS)) readout<R, N>(m, y, r, q, st.lg, l == 0 ? m.head_b : st.lg);
    }
    st.t++;
}
typedef void (*Fn)(const Model &, State &, int, int);
template <int BLK, int R, int N> static Fn p3(int low) { return low ? step<BLK, R, 1, N> : step<BLK, R, 0, N>; }
template <int BLK, int N> static Fn p2(int R, int low) { return R == 32 ? p3<BLK, 32, N>(low) : R == 16 ? p3<BLK, 16, N>(low) : p3<BLK, 8, N>(low); }
template <int N> static Fn p1(int blk, int R, int low) { return blk == 16 ? p2<16, N>(R, low) : p2<8, N>(R, low); }
static Fn pick(const Model &m) {
    return m.n == 256 ? p1<256>(m.b, m.R, m.R2 > 0) : m.n == 128 ? p1<128>(m.b, m.R, m.R2 > 0) : p1<64>(m.b, m.R, m.R2 > 0);
}

static State mkstate(const Model &m) {
    State st{nullptr, nullptr, 0, m.NL, 0};
    if (posix_memalign((void **)&st.q, 64, (size_t)m.L * m.n + 64) || posix_memalign((void **)&st.lg, 64, (size_t)m.VP * 4)) exit(1);
    memset(st.q, 0, (size_t)m.L * m.n + 64);
    return st;
}
static std::vector<int32_t> readi32(const char *path, size_t cnt) {
    FILE *f = fopen(path, "rb"); std::vector<int32_t> v(cnt);
    if (!f || fread(v.data(), 4, cnt, f) != cnt) { fprintf(stderr, "short %s\n", path); exit(1); }
    fclose(f); return v;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: ygg logits|score|bench model.bin ... (see ygg.cpp)\n"); return 1; }
    const std::string mode = argv[1];
    Model m = load(argv[2]);
    const Fn fn = pick(m);
    State st = mkstate(m);
    const int V = m.V;
    if (mode == "logits" || mode == "score") {
        const bool sc = mode == "score";
        if (argc < 7) { fprintf(stderr, "usage\n"); return 1; }
        const int a0 = sc ? 5 : 4, nw = atoi(argv[a0]), T = atoi(argv[a0 + 1]);
        int fl = 0;
        for (int i = sc ? 7 : 8; i < argc; i++) fl |= !strcmp(argv[i], "--blocked") ? F_WBLK : !strcmp(argv[i], "--prefetch") ? F_PFL : 0;
        std::vector<int32_t> tok = readi32(argv[3], (size_t)nw * T), tgt;
        if (sc) tgt = readi32(argv[4], (size_t)nw * T);
        std::vector<float> out(sc ? 0 : (size_t)nw * T * V);
        double nll = 0; const double t0 = now();
        for (int s = 0; s < nw; s++) {
            memset(st.q, 0, (size_t)m.L * m.n); st.c = 0;
            for (int t = 0; t < T; t++) {
                const size_t i = (size_t)s * T + t;
                if (tok[i] < 0 || tok[i] >= V) { fprintf(stderr, "token out of range\n"); return 1; }
                fn(m, st, tok[i], fl);
                if (!sc) { memcpy(&out[i * V], st.lg, 4 * V); continue; }
                double mx = -1e300; for (int c = 0; c < V; c++) mx = st.lg[c] > mx ? st.lg[c] : mx;
                double se = 0; for (int c = 0; c < V; c++) se += exp((double)st.lg[c] - mx);
                nll += log(se) + mx - (double)st.lg[tgt[i]];
            }
        }
        if (sc) { printf("ygg score windows=%d T=%d bpc=%.6f secs=%.2f\n", nw, T, nll / ((double)nw * T) / log(2.0), now() - t0); return 0; }
        FILE *o = fopen(argv[6], "wb");
        if (!o || fwrite(out.data(), 4, out.size(), o) != out.size()) { perror(argv[6]); return 1; }
        fclose(o);
        printf("ygg logits windows=%d T=%d secs=%.2f\n", nw, T, now() - t0);
        return 0;
    }
    if (mode == "time") {
        if (argc < 4) { fprintf(stderr, "usage: ygg time model.bin x.i32 [--seg S] [--warm K] [--prefetch] [--blocked]\n"); return 1; }
        int seg = 10000, fl = 0; long warm = -1;
        for (int i = 4; i < argc; i++) {
            if (!strcmp(argv[i], "--seg") && i + 1 < argc) seg = atoi(argv[++i]);
            else if (!strcmp(argv[i], "--warm") && i + 1 < argc) warm = atol(argv[++i]);
            else if (!strcmp(argv[i], "--prefetch")) fl |= F_PFL;
            else if (!strcmp(argv[i], "--blocked")) fl |= F_WBLK;
            else { fprintf(stderr, "unknown option %s\n", argv[i]); return 1; }
        }
        FILE *f = fopen(argv[3], "rb"); if (!f) { perror(argv[3]); return 1; }
        fseek(f, 0, SEEK_END); const size_t NT = ftell(f) / 4; fclose(f);
        std::vector<int32_t> tok = readi32(argv[3], NT);
        for (auto v : tok) if (v < 0 || v >= V) { fprintf(stderr, "token out of range\n"); return 1; }
        float *prob; if (posix_memalign((void **)&prob, 64, (size_t)m.VP * 4 + 64)) return 1;
        double chk = 0;
        size_t T0 = 0;  // first timed token
        if (warm < 0) {  // untimed pass over the stream, then reset
            for (size_t t = 0; t < NT; t++) { fn(m, st, tok[t], fl); chk += softmax(st.lg, prob, V); }
            memset(st.q, 0, (size_t)m.L * m.n); st.c = 0;
        } else {  // untimed first K tokens, state carried
            if ((size_t)warm >= NT) { fprintf(stderr, "--warm >= stream length\n"); return 1; }
            for (size_t t = 0; t < (size_t)warm; t++) { fn(m, st, tok[t], fl); chk += softmax(st.lg, prob, V); }
            T0 = (size_t)warm;
        }
        std::vector<double> ns; const double t0 = now();
        for (size_t s0 = T0; s0 < NT; s0 += seg) {
            const size_t s1 = std::min(NT, s0 + (size_t)seg); const double a = now();
            for (size_t t = s0; t < s1; t++) { fn(m, st, tok[t], fl); chk += softmax(st.lg, prob, V); }
            ns.push_back((now() - a) * 1e9 / (s1 - s0));
        }
        const double tot = now() - t0; std::vector<double> so = ns; std::sort(so.begin(), so.end());
        printf("ygg time L=%d n=%d b=%d V=%d D=%d R=%d R2=%d shared=%d tokens=%zu seg=%d prefetch=%d blocked=%d "
               "ns/token median=%.1f mean=%.1f min=%.1f max=%.1f bytes=%zu thp_kB=%ld cpu=%d chk=%.6g\n",
               m.L, m.n, m.b, m.V, m.D, m.R, m.R2, m.shared, NT, seg, !!(fl & F_PFL), !!(fl & F_WBLK), so[so.size() / 2],
               tot * 1e9 / (NT - T0), so[0], so.back(), m.bytes, anon_huge_kb(), sched_getcpu(), chk);
        return 0;
    }
    if (mode != "bench") { fprintf(stderr, "unknown mode %s\n", argv[1]); return 1; }
    double secs = 1.0; int rounds = 5, fl = 0; const char *tokf = nullptr;
    for (int i = 3; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--secs" && i + 1 < argc) secs = atof(argv[++i]);
        else if (a == "--rounds" && i + 1 < argc) rounds = atoi(argv[++i]);
        else if (a == "--tokens" && i + 1 < argc) tokf = argv[++i];
        else if (a == "--no-logits") fl |= F_NOLOGITS;
        else if (a == "--prefetch") fl |= F_PFL;
        else if (a == "--blocked") fl |= F_WBLK;
        else if (a == "--randleaf" && i + 1 < argc) {
            fl |= F_RANDLEAF; st.randK = atoi(argv[++i]);
            if (st.randK < 1 || st.randK > m.NL || (st.randK & (st.randK - 1))) { fprintf(stderr, "--randleaf K: power of two <= 2^D\n"); return 1; }
        } else { fprintf(stderr, "unknown option %s\n", argv[i]); return 1; }
    }
    if ((fl & F_RANDLEAF) && m.L < 2) { fprintf(stderr, "--randleaf needs a branch layer\n"); return 1; }
    std::vector<int32_t> tok;
    if (tokf) {
        FILE *f = fopen(tokf, "rb"); if (!f) { perror(tokf); return 1; }
        fseek(f, 0, SEEK_END); const long nt = ftell(f) / 4; fseek(f, 0, SEEK_SET);
        tok.resize(nt); if (fread(tok.data(), 4, nt, f) != (size_t)nt) return 1; fclose(f);
        for (auto v : tok) if (v < 0 || v >= V) { fprintf(stderr, "token out of range\n"); return 1; }
    } else {
        tok.resize(1 << 16);
        uint64_t r = 88172645463325252ull;
        for (auto &v : tok) { r ^= r << 13; r ^= r >> 7; r ^= r << 17; v = (int32_t)(r % V); }
    }
    const size_t NT = tok.size();
    size_t t = 0;
    const double w0 = now();
    while (now() - w0 < 0.2) for (int i = 0; i < 64; i++, t++) fn(m, st, tok[t % NT], fl);  // untimed warm-up
    std::vector<double> ns;
    for (int rep = 0; rep < rounds; rep++) {
        long cnt = 0; const double t0 = now(); double el;
        do { for (int i = 0; i < 64; i++, t++, cnt++) fn(m, st, tok[t % NT], fl); } while ((el = now() - t0) < secs);
        ns.push_back(el * 1e9 / cnt);
    }
    std::sort(ns.begin(), ns.end());
    printf("ygg bench L=%d n=%d b=%d V=%d D=%d R=%d R2=%d shared=%d logits=%d prefetch=%d blocked=%d randleaf=%d tok=%s "
           "ns/token median=%.1f min=%.1f bytes=%zu thp_kB=%ld cpu=%d chk=%d\n",
           m.L, m.n, m.b, m.V, m.D, m.R, m.R2, m.shared, !(fl & F_NOLOGITS), !!(fl & F_PFL), !!(fl & F_WBLK),
           (fl & F_RANDLEAF) ? st.randK : 0, tokf ? "text" : "random", ns[ns.size() / 2], ns[0], m.bytes, anon_huge_kb(),
           sched_getcpu(), (int)st.q[0] + (int)st.lg[0]);
    return 0;
}
