# Yggdrasil int8 CPU inference

Single-stream int8 inference on one CPU core with AVX-512 VNNI.

```
make -C models/yggdrasil/inference ygg                       # AVX-512 build
make -C models/yggdrasil/inference ygg_simde SIMDE=<simde>   # portable build, not for timing

python models/yggdrasil/inference/export.py --npz results/<run>.npz --out model.bin
python models/yggdrasil/inference/check.py --npz results/<run>.npz --model model.bin --int_ref
python models/yggdrasil/inference/test_kernel.py

python models/yggdrasil/inference/export.py --random 128,16,12,256 --out m.bin
taskset -c 0 models/yggdrasil/inference/ygg bench m.bin [--prefetch] [--blocked]
```
