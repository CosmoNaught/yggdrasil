# Yggdrasil int8 CPU kernel

This int8 kernel is the CPU backend: one stream on one core with AVX-512 VNNI. It approximates the float model of
`yggdrasil/inference.py`.

    make -C cpu ygg                          # AVX-512 VNNI build for this host (ARCH=native)
    make -C cpu ygg ARCH=sapphirerapids      # the build of the paper's timings
    make -C cpu ygg_simde SIMDE=<simde>      # portable build (https://github.com/simd-everywhere/simde), not for timing

    python cpu/export.py --npz results/<id>.npz --out model.bin [--storage factored]
    python cpu/check.py  --npz results/<id>.npz --model model.bin    # bits per token, kernel against float, 32 test windows
    python -c "import numpy as np, tasks.text as t; np.asarray(t.streams('enwik8')[2][:100000], np.int32).tofile('tokens.i32')"
    taskset -c 0 cpu/ygg time model.bin tokens.i32 --seg 2000 --warm 1000

`export.py` and `check.py` read the task and window length from the run record `<id>.json` next to the checkpoint
(or `--task` and `--seq_len`), and the data from `$YGGDRASIL_DATA` (default `data/`, or `--data`). `tokens.i32` is
the first 100k test bytes of enwik8 as int32.

`export.py` calibrates the int8 scales on training windows and writes `model.bin` and `model.bin.json`. `--storage
table` stores the input rows `T_r[x]` of every expert, `--storage factored` stores `U_r` in int8 and computes the rows
at every step. `export.int_forward` is a NumPy model of the kernel, equal to it bit for bit. On the first 32 test
windows the kernel stays within 0.02 bits per token of the float model.

    make -C cpu test                                      # kernel == export.int_forward on random models
    python -m unittest cpu/tests/test_checkpoints.py      # kernel within tolerance of the float model, paper checkpoints
    python -m unittest cpu/tests/test_triton.py           # Triton step == JAX step (GPU, jax + torch + triton)

The checkpoint tests read `$YGG_RUNS/enwik8_small.npz` and `$YGG_RUNS/hg38_medium.npz` (default `results/`) and the
data from `$YGGDRASIL_DATA`.

## Integer scheme

Per layer l the state is `h_l ~ s_l q_l` with `q_l` int8, one scale per layer or, with `--state_scale group`, one per
16 coordinates (then the readout scales are per expert and row, the unembedding scales per class).

    transition   p = (A16_r q + 2^13) >> 14,  A16 = rint(2^14 A_r)
    input        T_r[x] = rint((U_r v(x) + c_r) / s_l), int16
    RMS          rho = 1 / sqrt(mean(q_{l-1}^2) + 1e-6 / s_{l-1}^2)
    routing      right iff f32(dot(w, q_{l-1})) rho > threshold[x][node]
    from below   zl = sat8(rint(f32(V' q_{l-1}) m_v rho)),  drive += rint(f32(U'_r zl) m_ul)
    readout      zq = sat8(rint(f32(P_r q_l) m_z)),  logits = output_bias + sum_l f32(Q zq) sq_l

With `--storage factored` the routed experts store `U_r` int8 and `c_r` int16, and
`T_r[x] = rint(f32(U_r v[x]) f32(su_r s_e)) + c_r`. The file layout (header, array table, per-expert records) is the
FORMAT section of `export.py`; `ygg.cpp` mirrors it.

## Inference backends

`yggdrasil.inference.Yggdrasil` runs the float model in JAX. `backend="triton"` runs one token step for B streams in
one Triton kernel launch (`yggdrasil/inference_triton.py`): every layer, the hard routing, the expert reads, the state
update and the readout, with the unembedding as one GEMM in torch. It needs torch and triton (the `[gpu]` extra) and
has the same interface:

    from yggdrasil.inference import Yggdrasil
    model = Yggdrasil.load("results/<id>.npz", backend="triton", cuda_graph=True)
    state = model.init_state(64)                 # (B, layers, n) on the GPU
    logits, state = model.step(tokens, state)
    bits = model.score(windows)

The number of warps and the router levels read per round of loads are tuned per batch size at the first step.
`cuda_graph=True` captures one step per batch size and replays it.
