# Search protocol of the baselines

Every baseline number in the paper comes from this protocol. `sweeps/configs/{enwik8,hg38}/{small,medium,large}/`
and `sweeps/configs/state/<task>/` hold the W&B sweep of every cell (search space, method, metric, trial length),
`sweeps/registry.csv` the sweeps we ran for it, and `baselines/configs/` the selected setting of every reported cell,
with its sweep, best trial and final run.

**Training.** Every run uses AdamW with beta1 0.9, eps 1e-8 and decoupled weight decay on matrices only, gradient
clipping at global norm 1, a linear warm-up over 10% of the steps and cosine decay to 1% of the peak learning rate, and
skips any update with a non-finite loss or gradient. Training runs in fp32 without TF32, except the Transformer and
DeltaProduct, which run under bf16 autocast (`AMP` in each model module). The loss is the cross-entropy averaged over
every labelled position of the batch.

| Task | Batch | Final run | Validation |
|---|---|---|---|
| enwik8 | 64 windows of 256 bytes | 80k steps | every 5k steps, first 512 validation windows |
| hg38 | 16 windows of 1024 bases | 80k steps | every 5k steps, first 512 validation windows |
| state tasks | 256 sequences of lengths 3 to 40 | 100k steps | every 4k steps, 4096 sequences of lengths 40 to 256 |

Each run keeps its best validation checkpoint and tests it on the full test split in windows of the training length,
or on the 8192 test sequences of lengths 40 to 256 (`tasks`).

**Search.** One W&B Bayesian sweep per model and cell, 8 trials on one seed (0 on text and DNA, 160 on the state
tasks), each validated every 2k steps and scored by its best validation checkpoint. The objective is minus the
validation bits per token on enwik8 and hg38, and the validation final-answer accuracy, scaled to 0 at chance and 1 at
perfect, on the state tasks.

| Cell | Trial | Searched |
|---|---|---|
| enwik8, hg38, each size | 10k steps | `lr` log-uniform [5e-4, 2e-2], `weight_decay` {0, 0.01, 0.1}, `beta2` {0.95, 0.99, 0.999}; shape fixed by the size, plus `pkm_heads` {1, 2, 4} and `pkm_topk` {8, 16, 32} (`lr` up to 8e-2 for GRU on enwik8 small and hg38 small and medium, LSTM on enwik8 medium, Mamba-2 and Mamba-2 + PKM on enwik8 large and hg38 medium, Gated DeltaNet on enwik8 small and DeltaProduct-4 on enwik8 medium) |
| state tasks | 10k steps | the same, plus `n_layers` {1, 2, 4}, `d_model` {64, 128, 256}, `head_dim` {16, 32} (Transformer, Gated DeltaNet, DeltaProduct), `d_state` {16, 32, 64} (Mamba-2); for PD-SSM one `layout` in place of `n_layers` and `d_model` |
| S5, A5, wide (`*_wide.yaml`) | 40k steps | `lr` log-uniform [1e-4, 2e-2], `weight_decay` {0, 1e-6, 0.01, 0.1}, `beta2` as above, `n_layers` 1 to 4, one `layout` (Transformer, Mamba-2, Gated DeltaNet, DeltaProduct-2/4) |

A `layout` such as `d384_h12_p32` sets `d_model` 384, `n_heads` 12 and `head_dim` 32; `d128_p16` (Mamba-2) sets
`d_model` 128 and `head_dim` 16; `L2_d128` (PD-SSM) sets `n_layers` 2 and `d_model` 128, with 1, 2 or 4 layers at
`d_model` 64 and 128 and 1 or 2 at 256.

**Best trial.** Among the first 8 finished trials of a cell by start time, pooling its sweep with any top-up sweep
that replaced crashed trials, the one with the highest objective, ties going to the lower final training loss. In the
wide search, a cell whose best objective is below 0.02 is decided by the accuracy on the training lengths 3 to 40.

**Finals and reported score.** The best trial retrains for the final budget on 3 seeds, 0, 1 and 2 on text and DNA
and 160, 161 and 162 on the state tasks. A table cell is the test score of the seed with the best validation score,
ties going to the lowest seed, with the mean and standard error over the 3 seeds. On S5 and A5 the parallel-trainable
baselines report the search, main or wide, whose best seed has the better validation score.
