#!/usr/bin/env bash
# Fast experimental preset for sparsity=0.7.
# Changes vs baseline:
#   - sparsity 0.1 -> 0.7 (target test point)
#   - batch_size 4 -> 16 (~2x wall-clock speedup expected)
#   - lr 3e-5 -> 6e-5 (sqrt-scaling for 4x batch)
#   - epochs 10 -> 13 (3 post-prune recovery epochs; pruning still ends at ep10)
# If bs=16 OOMs, fall back to bs=8 with lr=4.5e-5.
set -e
cd "$(dirname "$0")"
python run_experiment.py \
  --sparsity 0.7 \
  --epochs 13 \
  --batch_size 16 \
  --lr 6e-5 \
  --lr_quant 1e-4 \
  --bit_reduction 1 \
  --pruning_periods 6 \
  --projection_periods 6 \
  --start_pruning_epoch 5 \
  --pruning_end_epoch 10 \
  --lr_scheduler linear \
  --warmup_ratio 0.1 \
  --seed 42 \
  --out_root ./results \
  --exp_tag fast_sp70
