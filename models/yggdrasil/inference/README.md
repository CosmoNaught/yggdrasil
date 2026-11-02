# Yggdrasil int8 CPU runtime

Single-stream int8 inference of the model in `models/yggdrasil/training/train.py` on one CPU core (AVX-512 VNNI).
A step reads the token's root leaf, walks each branch tree, reads one branch leaf per layer and computes the logits.
Supported: n 64 / 128 / 256, block 8 or 16, up to 8 layers, D <= 16, V <= 256, `--branch dyn`.

## Build

```
make -C models/yggdrasil/inference ygg                         # AVX-512 kernel (-march=sapphirerapids)
make -C models/yggdrasil/inference ygg_simde SIMDE=<simde>     # portable SIMDe build, same numbers, for correctness only
```

## Export

```
python models/yggdrasil/inference/export.py --npz results/<run>.npz --out model.bin   # task and T from <run>.json
```

Scales are calibrated on 128 training windows. `--storage factored` stores U_r and c_r instead of the T_r[x] table.

## Check

```
python models/yggdrasil/inference/check.py --npz results/<run>.npz --model model.bin --int_ref
python models/yggdrasil/inference/test_kernel.py                                      # kernel vs NumPy integer model
```

`check.py` prints the float and int8 bits per token on the test windows; `--int_ref` adds the bit-exact NumPy model
and its max logit difference to the kernel (0 when exact). `--ygg` selects the binary (e.g. `ygg_simde`).

## Time

```
python models/yggdrasil/inference/export.py --random 128,16,12,256 --out m.bin       # n,b,D,V[,layers,out_rank,lower_rank,shared_out]
taskset -c 0 models/yggdrasil/inference/ygg bench m.bin [--prefetch] [--blocked]
```
