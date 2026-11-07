// Yggdrasil int8 inference: one stream, one core, AVX-512 VNNI. Model files come from export.py, whose FORMAT
// section defines the layout; the integer scheme is in README.md.
//
// YGG_THP=0 disables huge pages.
#define _GNU_SOURCE 1
#ifdef YGG_SIMDE
#include "ygg_simde.h"
#else
#include <immintrin.h>
#endif
#include <math.h>
#include <sched.h>
#include <stdarg.h>
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

static const char *kUsage =
    "usage: ygg logits model.bin x.i32 nwin T out.f32        logits (nwin, T, V) f32, zero state at every window\n"
    "       ygg score  model.bin x.i32 y.i32 nwin T          bits per token of the targets y\n"
    "       ygg time   model.bin x.i32 [--seg S] [--warm K]  ns per token with the full output distribution, state "
    "carried\n";

constexpr int kMaxLayers = 8;               // export.LIMITS
constexpr int kMaxRouterDepth = 16;
constexpr int kMaxRank = 64;                // input_rank and layer_input_rank
constexpr int kRankMultiple = 4;
constexpr int kClassPad = 32;               // export.CLASS_PAD
constexpr size_t kHugePage = 2u << 20;
constexpr size_t kReadChunk = 64u << 20;    // bytes per fread of the model
constexpr size_t kBlobPad = 128;            // zero bytes after the blob: full-vector loads may read past an array
static const char *kOutputBias = "head_b";  // export.OUTPUT_BIAS

[[noreturn]] static void die(const char *fmt, ...) {
    va_list ap; va_start(ap, fmt); vfprintf(stderr, fmt, ap); va_end(ap); fputc('\n', stderr); exit(1);
}

// exp(x) = 2^n p(f) with x log2(e) = n + f
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

// ---------------------------------------------------------------------------------------------------- model file

#pragma pack(push, 1)
struct Header {  // export.HEADER, then the array count
    int32_t n_layers, d_state, block_size, vocab, router_depth, readout_rank, layer_input_rank, reserved,
        router_token_dim, input_rank, shift, n_arrays;
};
struct ArrayEntry { char name[32]; int64_t offset, bytes; };
#pragma pack(pop)
static_assert(sizeof(Header) == 48, "Header mirrors export.HEADER");

struct GroupScales {                   // null: one scale per tensor (Layer scalars)
    const float *layer_input_scale;   // (n) m_ul per coordinate
    const float *state_r2;            // (n / 16) squared relative scales of the layer below
    const float *readout_scale;       // per expert and kernel lane
    const float *unembed_scale;       // (VP) per class
};
struct FactoredInput {                 // token_code null: table storage
    const int8_t *token_code;         // (V, input_rank)
    const float *token_scale;         // (n)
    int off_input_proj, off_bias, off_input_scale;  // in the expert record
};
struct Layer {
    char *record; int record_bytes;   // expert r at record + r * record_bytes
    int off_transition, off_layer_input_proj, off_readout, off_table;  // in the record, -1: absent
    int table_stride;                 // bytes per token row of T_r; 0 for the token layer (its own row only)
    const uint8_t *readout; size_t readout_stride;  // readout of expert r at readout + r * readout_stride
    const uint8_t *unembed;           // u8 [R/4][VP][4]
    float readout_scale, unembed_scale, rms_eps, layer_input_scale;  // export.SCALARS
    const uint8_t *router_weight;     // routed: u8 (w + 128) [2^D][n], heap order from node 1
    const float *router_threshold;    // routed: [V][2^D]
    const uint8_t *layer_input_down;  // u8 [RG][n/4][16][4]
    const float *layer_input_down_scale;  // (16 RG)
    GroupScales group;
    FactoredInput factored;
};
struct Model {
    int n_layers, n, block, vocab, depth, readout_rank, layer_input_rank, input_rank, shift;
    int n_experts, vocab_pad;         // 2^depth, vocab padded to kClassPad
    float *output_bias; Layer layers[kMaxLayers]; size_t bytes;
};

static void *alloc_huge(size_t sz, bool thp) {
    const size_t al = (sz + kHugePage - 1) / kHugePage * kHugePage;
    void *p; if (posix_memalign(&p, kHugePage, al)) die("out of memory (%zu bytes)", al);
    if (thp) madvise(p, al, MADV_HUGEPAGE);
    return p;
}
#ifndef MADV_COLLAPSE
#define MADV_COLLAPSE 25
#endif
static void collapse_huge(void *p, size_t sz, bool thp) {  // huge pages now (Linux >= 6.1), not when khugepaged runs
    if (thp) madvise(p, (sz + kHugePage - 1) / kHugePage * kHugePage, MADV_COLLAPSE);
}
static long anon_huge_kb() {
    FILE *f = fopen("/proc/self/smaps_rollup", "r"); if (!f) return -1;
    char line[256]; long v = -1;
    while (fgets(line, sizeof line, f)) if (!strncmp(line, "AnonHugePages:", 14)) v = atol(line + 14);
    fclose(f); return v;
}

static Header read_header(FILE *f, const char *path) {
    char magic[4]; Header h;
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "YGG1", 4) || fread(&h, sizeof h, 1, f) != 1)
        die("%s: not a model file", path);
    const bool ok = (h.d_state == 64 || h.d_state == 128 || h.d_state == 256) && (h.block_size == 8 || h.block_size == 16) &&
                    (h.readout_rank == 8 || h.readout_rank == 16 || h.readout_rank == 32) &&
                    h.layer_input_rank <= kMaxRank && h.layer_input_rank % kRankMultiple == 0 &&
                    h.n_layers >= 1 && h.n_layers <= kMaxLayers && h.vocab <= 256 && h.router_depth <= kMaxRouterDepth &&
                    (h.n_layers == 1 || h.router_depth >= 1) && h.reserved == 0;
    if (!ok) die("%s: unsupported model (export.LIMITS)", path);
    return h;
}

// The arrays after the table, read into one huge-page blob; returns name -> address.
static std::map<std::string, char *> read_blob(FILE *f, const char *path, int n_arrays, size_t *bytes) {
    std::vector<ArrayEntry> tab(n_arrays);
    if (fread(tab.data(), sizeof(ArrayEntry), n_arrays, f) != (size_t)n_arrays) die("%s: short array table", path);
    long pos = ftell(f); pos += (64 - pos % 64) % 64;
    fseek(f, 0, SEEK_END); const long end = ftell(f); fseek(f, pos, SEEK_SET);
    const char *thpv = getenv("YGG_THP");
    const bool thp = !thpv || strcmp(thpv, "0");
    const size_t sz = (size_t)(end - pos);
    char *blob = (char *)alloc_huge(sz + kBlobPad, thp);
    memset(blob + sz, 0, kBlobPad);
    for (size_t o = 0; o < sz;) {
        const size_t c = fread(blob + o, 1, std::min(sz - o, kReadChunk), f);
        if (!c) die("%s: short model file (%zu of %zu array bytes)", path, o, sz);
        o += c;
    }
    collapse_huge(blob, sz + kBlobPad, thp);
    *bytes = sz;
    std::map<std::string, char *> arr;
    for (auto &e : tab) arr[e.name] = blob + e.offset;
    return arr;
}

static void bind_layer(Model &m, int l, const std::map<std::string, char *> &arr) {
    const std::string P = "l" + std::to_string(l) + "_";
    auto get = [&](const char *s) -> char * { auto it = arr.find(P + s); return it == arr.end() ? nullptr : it->second; };
    auto need = [&](const char *s) -> char * { char *p = get(s); if (!p) die("model file: missing %s%s", P.c_str(), s); return p; };
    Layer &y = m.layers[l];
    const int32_t *meta = (const int32_t *)need("meta");  // export.RECORD order
    y.record_bytes = meta[0]; y.off_transition = meta[1]; y.off_layer_input_proj = meta[2]; y.off_readout = meta[3];
    y.off_table = meta[4];
    y.factored.token_code = (const int8_t *)get("vt");
    if (y.factored.token_code) {
        if (m.input_rank % kRankMultiple || m.input_rank > kMaxRank || m.input_rank < kRankMultiple)
            die("model file: factored input_rank %d", m.input_rank);
        y.factored.off_input_proj = meta[5]; y.factored.off_bias = meta[6]; y.factored.off_input_scale = meta[7];
        y.factored.token_scale = (const float *)need("mT");
    }
    y.table_stride = l == 0 ? 0 : 2 * m.n;
    y.record = need("rec");
    y.readout = (const uint8_t *)y.record + y.off_readout; y.readout_stride = y.record_bytes;
    y.unembed = (const uint8_t *)need("Q");
    y.group.readout_scale = (const float *)get("mzl"); y.group.unembed_scale = (const float *)get("sqv");
    const float *sc = (const float *)need("scal");  // export.SCALARS order
    y.readout_scale = sc[0]; y.unembed_scale = sc[1]; y.rms_eps = sc[2]; y.layer_input_scale = sc[3];
    if (l > 0) {
        y.router_weight = (const uint8_t *)need("w"); y.router_threshold = (const float *)need("thr");
        if (m.layer_input_rank) {
            y.layer_input_down = (const uint8_t *)need("vl"); y.layer_input_down_scale = (const float *)need("mv");
        }
        y.group.layer_input_scale = (const float *)get("mulv"); y.group.state_r2 = (const float *)get("r2b");
    }
}

static Model load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    const Header h = read_header(f, path);
    Model m{};
    m.n_layers = h.n_layers; m.n = h.d_state; m.block = h.block_size; m.vocab = h.vocab; m.depth = h.router_depth;
    m.readout_rank = h.readout_rank; m.layer_input_rank = h.layer_input_rank;
    m.input_rank = h.input_rank; m.shift = h.shift;  // h.router_token_dim: folded into the thresholds, not read
    m.n_experts = 1 << m.depth; m.vocab_pad = (m.vocab + kClassPad - 1) / kClassPad * kClassPad;
    const auto arr = read_blob(f, path, h.n_arrays, &m.bytes);
    fclose(f);
    auto it = arr.find(kOutputBias);
    if (it == arr.end()) die("%s: missing %s", path, kOutputBias);
    m.output_bias = (float *)it->second;
    for (int l = 0; l < m.n_layers; l++) bind_layer(m, l, arr);
    return m;
}

// ---------------------------------------------------------------------------------------------------- primitives

// u8 x s8 products (dpbusd) store weights as w + 128 and subtract 128 sum(q) after the dot product.
template <int N> static inline int32_t sum_s8(const int8_t *q) {  // sum of N int8 (N a multiple of 64)
    __m512i v = _mm512_sad_epu8(_mm512_xor_si512(_mm512_loadu_si512(q), _mm512_set1_epi8((char)0x80)), _mm512_setzero_si512());
    for (int c = 64; c < N; c += 64)
        v = _mm512_add_epi64(v, _mm512_sad_epu8(_mm512_xor_si512(_mm512_loadu_si512(q + c), _mm512_set1_epi8((char)0x80)), _mm512_setzero_si512()));
    return (int32_t)_mm512_reduce_add_epi64(v) - 128 * N;
}
template <int N> static inline __m512i dot_lanes(const uint8_t *w, const int8_t *q) {  // 16 partial sums of dot(w + 128, q)
    __m512i a = _mm512_dpbusd_epi32(_mm512_setzero_si512(), _mm512_loadu_si512(w), _mm512_loadu_si512(q));
    for (int c = 64; c < N; c += 64) a = _mm512_dpbusd_epi32(a, _mm512_loadu_si512(w + c), _mm512_loadu_si512(q + c));
    return a;
}
// 16 rows of w u8 [N/4][16][4] times q: dot(w_i + 128, q) in lane i, 4 accumulator chains to hide the dpbusd latency
template <int N> static inline __m512i dot16x4(const uint8_t *w, const int8_t *q) {
    __m512i a = _mm512_setzero_si512(), b = a, e = a, f = a;
    for (int kc = 0; kc < N / 4; kc += 4) {
        a = _mm512_dpbusd_epi32(a, _mm512_loadu_si512(w + kc * 64), _mm512_set1_epi32(*(const int32_t *)(q + 4 * kc)));
        b = _mm512_dpbusd_epi32(b, _mm512_loadu_si512(w + kc * 64 + 64), _mm512_set1_epi32(*(const int32_t *)(q + 4 * kc + 4)));
        e = _mm512_dpbusd_epi32(e, _mm512_loadu_si512(w + kc * 64 + 128), _mm512_set1_epi32(*(const int32_t *)(q + 4 * kc + 8)));
        f = _mm512_dpbusd_epi32(f, _mm512_loadu_si512(w + kc * 64 + 192), _mm512_set1_epi32(*(const int32_t *)(q + 4 * kc + 12)));
    }
    return _mm512_add_epi32(_mm512_add_epi32(a, b), _mm512_add_epi32(e, f));
}
// rho = 1 / sqrt(mean(q^2) + eps), exact integer sum of squares
template <int N> static inline float rms_rho(const int8_t *q, float eps) {
    __m512i a = _mm512_setzero_si512();
    for (int c = 0; c < N; c += 32) {
        const __m512i v = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i *)(q + c)));
        a = _mm512_add_epi32(a, _mm512_madd_epi16(v, v));
    }
    return 1.0f / sqrtf((float)_mm512_reduce_add_epi32(a) * (1.0f / N) + eps);
}
// group scales: rho = 1 / sqrt(sum_g r2[g] S_g / N + eps), S_g = sum of q^2 over group g, f32 sum in group order
template <int N> static inline float rms_rho_group(const int8_t *q, const float *r2, float eps) {
    float s = 0.0f;
    for (int c = 0; c < N; c += 32) {
        const __m512i v = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i *)(q + c)));
        const __m512i a = _mm512_madd_epi16(v, v);  // lanes 0-7: coordinates c .. c+15, lanes 8-15: c+16 .. c+31
        s = s + r2[c / 16] * (float)_mm512_mask_reduce_add_epi32(0x00FF, a);
        s = s + r2[c / 16 + 1] * (float)_mm512_mask_reduce_add_epi32(0xFF00, a);
    }
    return 1.0f / sqrtf(s * (1.0f / N) + eps);
}

// ---------------------------------------------------------------------------------------------------- routing

// heap order from node 1: right iff f32(w_ix . q) rho > threshold[x][ix], the token part of the test folded in
template <int N> static inline int route(const Model &m, const Layer &y, int x, const int8_t *qb, float rho) {
    const int32_t corr = 128 * sum_s8<N>(qb);
    const float *th = y.router_threshold + (size_t)x * m.n_experts;
    int ix = 1;
    for (int d = 0; d < m.depth; d++) {
        const int32_t dot = _mm512_reduce_add_epi32(dot_lanes<N>(y.router_weight + (size_t)ix * N, qb)) - corr;
        ix = 2 * ix + ((float)dot * rho > th[ix]);
    }
    return ix - m.n_experts;
}

// ---------------------------------------------------------------------------------------------------- expert update

// zl[0 .. 16 RG) = sat8(rint(f32(V' q_below) m_v rho)), RG = ceil(R2 / 16)
template <int N> static inline void layer_input(const Model &m, const Layer &y, const int8_t *qb, int8_t *zl, float rho) {
    const int RG = (m.layer_input_rank + 15) / 16;
    const __m512i corr = _mm512_set1_epi32(128 * sum_s8<N>(qb));
    for (int gi = 0; gi < RG; gi++) {
        const __m512i a = _mm512_sub_epi32(dot16x4<N>(y.layer_input_down + (size_t)gi * N * 16, qb), corr);
        __m512 zf = _mm512_mul_ps(_mm512_cvtepi32_ps(a), _mm512_loadu_ps(y.layer_input_down_scale + 16 * gi));
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

// acc[g] = (A16_r q >> shift) rounded, for 16-row groups g; A16 [N/16][BLK/2][16][2] int16
template <int BLK, int N>
static inline void add_transition(const Model &m, const char *rec, const Layer &y, const int8_t *q, __m512i *acc) {
    constexpr int NG = N / 16, NQ = N / 32;
    __m512i q16[NQ];
    for (int i = 0; i < NQ; i++) q16[i] = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i *)(q + 32 * i)));
    const int16_t *A = (const int16_t *)(rec + y.off_transition);
    alignas(64) int16_t qh[N];
    if (BLK == 16) for (int i = 0; i < NQ; i++) _mm512_store_si512(qh + 32 * i, q16[i]);
    const __m512i rnd = _mm512_set1_epi32(1 << (m.shift - 1));
    const __m128i shc = _mm_cvtsi32_si128(m.shift);
    for (int gg = 0; gg < NG; gg++) {  // 4 accumulator chains per group to hide the dpwssd latency
        __m512i a[4] = {_mm512_setzero_si512(), _mm512_setzero_si512(), _mm512_setzero_si512(), _mm512_setzero_si512()};
        const __m512i qs = q16[gg >> 1];
        for (int p = 0; p < BLK / 2; p++) {
            const __m512i x0 = BLK == 16 ? _mm512_set1_epi32(*(const int32_t *)(qh + 16 * gg + 2 * p))
                                         : _mm512_permutexvar_epi32(IDX8.v[gg & 1][p], qs);
            a[p & 3] = _mm512_dpwssd_epi32(a[p & 3], _mm512_loadu_si512(A + (gg * BLK / 2 + p) * 32), x0);
        }
        acc[gg] = _mm512_sra_epi32(_mm512_add_epi32(_mm512_add_epi32(_mm512_add_epi32(a[0], a[1]), _mm512_add_epi32(a[2], a[3])), rnd), shc);
    }
}
// acc += T_r[x], int16
template <int N> static inline void add_table_input(const char *rec, const Layer &y, int x, __m512i *acc) {
    const int16_t *T = (const int16_t *)(rec + y.off_table + (size_t)x * y.table_stride);
    for (int b = 0; b < N / 16; b++) acc[b] = _mm512_add_epi32(acc[b], _mm512_cvtepi16_epi32(_mm256_loadu_si256((const __m256i *)(T + 16 * b))));
}
// acc += rint(f32(U_r v(x)) f32(su_r token_scale)) + c_r; U_r [rank/4][N][4] u8 (u + 128)
template <int N> static inline void add_factored_input(const Model &m, const char *rec, const Layer &y, int x, __m512i *acc) {
    constexpr int NG = N / 16;
    const FactoredInput &fc = y.factored;
    const uint8_t *uf = (const uint8_t *)(rec + fc.off_input_proj);
    const int8_t *v = fc.token_code + (size_t)x * m.input_rank;
    int32_t vs = 0; for (int j = 0; j < m.input_rank; j++) vs += v[j];
    __m512i lt[NG];
    for (int b = 0; b < NG; b++) lt[b] = _mm512_setzero_si512();
    for (int kc = 0; kc < m.input_rank / 4; kc++) {
        const __m512i vb = _mm512_set1_epi32(*(const int32_t *)(v + 4 * kc));
        for (int b = 0; b < NG; b++) lt[b] = _mm512_dpbusd_epi32(lt[b], _mm512_loadu_si512(uf + ((size_t)kc * NG + b) * 64), vb);
    }
    const int16_t *C = (const int16_t *)(rec + fc.off_bias);
    const __m512 su = _mm512_set1_ps(*(const float *)(rec + fc.off_input_scale)); const __m512i cv = _mm512_set1_epi32(128 * vs);
    for (int b = 0; b < NG; b++) {
        const __m512 mul = _mm512_mul_ps(su, _mm512_loadu_ps(fc.token_scale + 16 * b));
        const __m512i t = _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(lt[b], cv)), mul));
        acc[b] = _mm512_add_epi32(acc[b], _mm512_add_epi32(t, _mm512_cvtepi16_epi32(_mm256_loadu_si256((const __m256i *)(C + 16 * b)))));
    }
}
// acc += rint(f32(U'_r zl) m_ul); U'_r [R2/4][N][4] u8 (u + 128)
template <int N> static inline void add_layer_input(const Model &m, const char *rec, const Layer &y, const int8_t *zl, __m512i *acc) {
    constexpr int NG = N / 16;
    const uint8_t *u = (const uint8_t *)(rec + y.off_layer_input_proj);
    int32_t zs = 0; for (int j = 0; j < m.layer_input_rank; j++) zs += zl[j];
    __m512i la[NG];
    for (int b = 0; b < NG; b++) la[b] = _mm512_setzero_si512();
    for (int kc = 0; kc < m.layer_input_rank / 4; kc++) {
        const __m512i zb = _mm512_set1_epi32(*(const int32_t *)(zl + 4 * kc));
        for (int b = 0; b < NG; b++) la[b] = _mm512_dpbusd_epi32(la[b], _mm512_loadu_si512(u + ((size_t)kc * NG + b) * 64), zb);
    }
    const __m512i cz = _mm512_set1_epi32(128 * zs);
    if (y.group.layer_input_scale) {
        for (int b = 0; b < NG; b++) acc[b] = _mm512_add_epi32(acc[b], _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(la[b], cz)), _mm512_loadu_ps(y.group.layer_input_scale + 16 * b))));
    } else {
        const __m512 mu = _mm512_set1_ps(y.layer_input_scale);
        for (int b = 0; b < NG; b++) acc[b] = _mm512_add_epi32(acc[b], _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(la[b], cz)), mu)));
    }
}

// q (N int8) <- sat8(A16_r q >> shift + T_r[x] + rint(f32(U'_r zl) m_ul))
template <int BLK, int LOW, int N>
static inline void update(const Model &m, const Layer &y, int r, int x, int8_t *q, const int8_t *zl) {
    const char *rec = y.record + (size_t)r * y.record_bytes;
    __m512i acc[N / 16];
    add_transition<BLK, N>(m, rec, y, q, acc);
    if (y.factored.token_code) add_factored_input<N>(m, rec, y, x, acc);
    else add_table_input<N>(rec, y, x, acc);
    if (LOW) add_layer_input<N>(m, rec, y, zl, acc);
    for (int b = 0; b < N / 16; b++) _mm_storeu_si128((__m128i *)(q + 16 * b), _mm512_cvtsepi32_epi8(acc[b]));
}

// ---------------------------------------------------------------------------------------------------- readout

// lg[0 .. VP) = base + f32(Q zq) sq, zq = sat8(rint(f32(P_r q) m_z)); P u8 (p + 128): R = 8 [N/8][16 lanes][4] (lane
// 2r + e holds P[r][8c + 4e .. +3]), R >= 16 blocks of 16 rows [N/4][16][4]; Q u8 [R/4][VP][4]
template <int R, int N>
static inline void readout(const Model &m, const Layer &y, int r, const int8_t *q, float *lg, const float *base) {
    const uint8_t *P = y.readout + (size_t)r * y.readout_stride;
    const float *mz = y.group.readout_scale ? y.group.readout_scale + (size_t)r * (R == 8 ? 16 : R) : nullptr;
    alignas(64) int8_t zq[32];
    const int32_t corr = 128 * sum_s8<N>(q);
    if (R == 8) {
        __m512i a = _mm512_setzero_si512(), b = a;
        for (int c = 0; c < N / 8; c += 2) {
            a = _mm512_dpbusd_epi32(a, _mm512_loadu_si512(P + 64 * c), _mm512_set1_epi64(*(const int64_t *)(q + 8 * c)));
            b = _mm512_dpbusd_epi32(b, _mm512_loadu_si512(P + 64 * c + 64), _mm512_set1_epi64(*(const int64_t *)(q + 8 * c + 8)));
        }
        a = _mm512_add_epi32(a, b);
        a = _mm512_add_epi32(a, _mm512_shuffle_epi32(a, (_MM_PERM_ENUM)0xB1));
        a = _mm512_sub_epi32(a, _mm512_set1_epi32(corr));
        __m512i zi = _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(a), mz ? _mm512_loadu_ps(mz) : _mm512_set1_ps(y.readout_scale)));
        zi = _mm512_max_epi32(_mm512_min_epi32(zi, _mm512_set1_epi32(127)), _mm512_set1_epi32(-127));
        _mm_storel_epi64((__m128i *)zq, _mm512_cvtepi32_epi8(_mm512_maskz_compress_epi32(0x5555, zi)));
    } else {
        for (int rb = 0; rb < R / 16; rb++) {
            const __m512i a = _mm512_sub_epi32(dot16x4<N>(P + (size_t)rb * 16 * N, q), _mm512_set1_epi32(corr));
            __m512i zi = _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(a), mz ? _mm512_loadu_ps(mz + 16 * rb) : _mm512_set1_ps(y.readout_scale)));
            zi = _mm512_max_epi32(_mm512_min_epi32(zi, _mm512_set1_epi32(127)), _mm512_set1_epi32(-127));
            _mm_store_si128((__m128i *)(zq + 16 * rb), _mm512_cvtepi32_epi8(zi));
        }
    }
    int32_t zs = 0; for (int j = 0; j < R; j++) zs += zq[j];
    const __m512i cz = _mm512_set1_epi32(128 * zs); const __m512 s = _mm512_set1_ps(y.unembed_scale);
    const float *sv = y.group.unembed_scale;
    for (int c = 0; c < m.vocab_pad; c += 32) {  // two 16-class chunks per pass
        __m512i a0 = _mm512_setzero_si512(), a1 = a0;
        for (int kc = 0; kc < R / 4; kc++) {
            const __m512i zb = _mm512_set1_epi32(*(const int32_t *)(zq + 4 * kc));
            a0 = _mm512_dpbusd_epi32(a0, _mm512_loadu_si512(y.unembed + ((size_t)kc * m.vocab_pad + c) * 4), zb);
            a1 = _mm512_dpbusd_epi32(a1, _mm512_loadu_si512(y.unembed + ((size_t)kc * m.vocab_pad + c + 16) * 4), zb);
        }
        const __m512 s0 = sv ? _mm512_loadu_ps(sv + c) : s, s1 = sv ? _mm512_loadu_ps(sv + c + 16) : s;
        _mm512_store_ps(lg + c, _mm512_add_ps(_mm512_load_ps(base + c), _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(a0, cz)), s0)));
        _mm512_store_ps(lg + c + 16, _mm512_add_ps(_mm512_load_ps(base + c + 16), _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(a1, cz)), s1)));
    }
}

// ---------------------------------------------------------------------------------------------------- one token

struct State { int8_t *q; float *lg; };  // q: int8 states of all layers, lg: logits

template <int BLK, int R, int LOW, int N>
static void step(const Model &m, State &st, int x) {
    for (int l = 0; l < m.n_layers; l++) {
        const Layer &y = m.layers[l];
        int8_t *q = st.q + l * N;
        int r = x;
        if (l > 0) {
            const int8_t *qb = q - N;  // state of the layer below
            const float rho = y.group.state_r2 ? rms_rho_group<N>(qb, y.group.state_r2, y.rms_eps) : rms_rho<N>(qb, y.rms_eps);
            r = route<N>(m, y, x, qb, rho);
            if (LOW) {
                alignas(16) int8_t zl[kMaxRank];
                layer_input<N>(m, y, qb, zl, rho);
                update<BLK, 1, N>(m, y, r, x, q, zl);
            } else update<BLK, 0, N>(m, y, r, x, q, nullptr);
        } else update<BLK, 0, N>(m, y, r, x, q, nullptr);
        readout<R, N>(m, y, r, q, st.lg, l == 0 ? m.output_bias : st.lg);
    }
}
typedef void (*StepFn)(const Model &, State &, int);
template <int BLK, int R, int N> static StepFn dispatch_low(int low) { return low ? step<BLK, R, 1, N> : step<BLK, R, 0, N>; }
template <int BLK, int N> static StepFn dispatch_readout(int R, int low) {
    return R == 32 ? dispatch_low<BLK, 32, N>(low) : R == 16 ? dispatch_low<BLK, 16, N>(low) : dispatch_low<BLK, 8, N>(low);
}
template <int N> static StepFn dispatch_block(int blk, int R, int low) {
    return blk == 16 ? dispatch_readout<16, N>(R, low) : dispatch_readout<8, N>(R, low);
}
static StepFn dispatch_state(const Model &m) {
    const int low = m.layer_input_rank > 0;
    return m.n == 256 ? dispatch_block<256>(m.block, m.readout_rank, low)
         : m.n == 128 ? dispatch_block<128>(m.block, m.readout_rank, low) : dispatch_block<64>(m.block, m.readout_rank, low);
}

static State make_state(const Model &m) {
    State st{nullptr, nullptr};
    if (posix_memalign((void **)&st.q, 64, (size_t)m.n_layers * m.n + 64) || posix_memalign((void **)&st.lg, 64, (size_t)m.vocab_pad * 4))
        die("out of memory (state)");
    memset(st.q, 0, (size_t)m.n_layers * m.n + 64);
    return st;
}
static void reset(const Model &m, State &st) { memset(st.q, 0, (size_t)m.n_layers * m.n); }

static std::vector<int32_t> read_i32(const char *path, size_t cnt) {
    FILE *f = fopen(path, "rb"); std::vector<int32_t> v(cnt);
    if (!f || fread(v.data(), 4, cnt, f) != cnt) die("%s: fewer than %zu int32", path, cnt);
    fclose(f); return v;
}
static std::vector<int32_t> read_i32_all(const char *path) {
    FILE *f = fopen(path, "rb"); if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END); const size_t n = ftell(f) / 4; fclose(f);
    return read_i32(path, n);
}
static void check_tokens(const std::vector<int32_t> &tok, int V) {
    for (auto v : tok) if (v < 0 || v >= V) die("token %d out of range [0, %d)", v, V);
}

// ---------------------------------------------------------------------------------------------------- commands

// logits: argv = model x nwin T out; score: argv = model x y nwin T
static int cmd_logits_score(const Model &m, StepFn fn, State &st, bool score, char **argv) {
    const int nw = atoi(argv[score ? 3 : 2]), T = atoi(argv[score ? 4 : 3]), V = m.vocab;
    const std::vector<int32_t> tok = read_i32(argv[1], (size_t)nw * T);
    const std::vector<int32_t> tgt = score ? read_i32(argv[2], (size_t)nw * T) : std::vector<int32_t>();
    check_tokens(tok, V);
    std::vector<float> out(score ? 0 : (size_t)nw * T * V);
    double nll = 0; const double t0 = now();
    for (int s = 0; s < nw; s++) {
        reset(m, st);
        for (int t = 0; t < T; t++) {
            const size_t i = (size_t)s * T + t;
            fn(m, st, tok[i]);
            if (!score) { memcpy(&out[i * V], st.lg, 4 * V); continue; }
            // double precision: score is the reference number, softmax() the fast f32 path that time measures
            double mx = -1e300; for (int c = 0; c < V; c++) mx = st.lg[c] > mx ? st.lg[c] : mx;
            double se = 0; for (int c = 0; c < V; c++) se += exp((double)st.lg[c] - mx);
            nll += log(se) + mx - (double)st.lg[tgt[i]];
        }
    }
    if (score) { printf("ygg score windows=%d T=%d bpc=%.6f secs=%.2f\n", nw, T, nll / ((double)nw * T) / log(2.0), now() - t0); return 0; }
    FILE *o = fopen(argv[4], "wb");
    if (!o || fwrite(out.data(), 4, out.size(), o) != out.size()) { perror(argv[4]); return 1; }
    fclose(o);
    printf("ygg logits windows=%d T=%d secs=%.2f\n", nw, T, now() - t0);
    return 0;
}

// argv = model x [--seg S] [--warm K], argc counted from model
static int cmd_time(const Model &m, StepFn fn, State &st, int argc, char **argv) {
    int seg = 10000; long warm = -1;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--seg") && i + 1 < argc) seg = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warm") && i + 1 < argc) warm = atol(argv[++i]);
        else { fprintf(stderr, "unknown option %s\n%s", argv[i], kUsage); return 1; }
    }
    const std::vector<int32_t> tok = read_i32_all(argv[1]);
    const size_t NT = tok.size();
    check_tokens(tok, m.vocab);
    float *prob; if (posix_memalign((void **)&prob, 64, (size_t)m.vocab_pad * 4 + 64)) die("out of memory");
    double chk = 0;  // printed so that the softmax is not optimised away
    size_t T0 = 0;   // first timed token
    if (warm < 0) {  // untimed pass over the stream, then reset
        for (size_t t = 0; t < NT; t++) { fn(m, st, tok[t]); chk += softmax(st.lg, prob, m.vocab); }
        reset(m, st);
    } else {         // untimed first K tokens, state carried
        if ((size_t)warm >= NT) die("--warm %ld >= stream length %zu", warm, NT);
        for (size_t t = 0; t < (size_t)warm; t++) { fn(m, st, tok[t]); chk += softmax(st.lg, prob, m.vocab); }
        T0 = (size_t)warm;
    }
    std::vector<double> ns; const double t0 = now();
    for (size_t s0 = T0; s0 < NT; s0 += seg) {
        const size_t s1 = std::min(NT, s0 + (size_t)seg); const double a = now();
        for (size_t t = s0; t < s1; t++) { fn(m, st, tok[t]); chk += softmax(st.lg, prob, m.vocab); }
        ns.push_back((now() - a) * 1e9 / (s1 - s0));
    }
    const double tot = now() - t0; std::vector<double> so = ns; std::sort(so.begin(), so.end());
    printf("ygg time n_layers=%d d_state=%d block_size=%d vocab=%d router_depth=%d readout_rank=%d layer_input_rank=%d "
           "tokens=%zu seg=%d ns/token median=%.1f mean=%.1f min=%.1f max=%.1f bytes=%zu thp_kB=%ld "
           "cpu=%d chk=%.6g\n",
           m.n_layers, m.n, m.block, m.vocab, m.depth, m.readout_rank, m.layer_input_rank, NT, seg,
           so[so.size() / 2], tot * 1e9 / (NT - T0), so[0], so.back(), m.bytes, anon_huge_kb(), sched_getcpu(), chk);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 3) { fputs(kUsage, stderr); return 1; }
    const std::string mode = argv[1];
    const bool score = mode == "score";
    if ((mode == "logits" || score) && argc != 7) { fputs(kUsage, stderr); return 1; }
    if (mode != "logits" && !score && (mode != "time" || argc < 4)) { fputs(kUsage, stderr); return 1; }
    const Model m = load(argv[2]);
    const StepFn fn = dispatch_state(m);
    State st = make_state(m);
    return mode == "time" ? cmd_time(m, fn, st, argc - 2, argv + 2) : cmd_logits_score(m, fn, st, score, argv + 2);
}
