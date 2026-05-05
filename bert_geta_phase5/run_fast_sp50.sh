#!/usr/bin/env bash
# Fast preset for sparsity=0.5, extended fine-tune.
# Same as run_fast_sp70.sh but:
#   - sparsity 0.7 -> 0.5
#   - epochs 13 -> 16 (6 post-prune recovery epochs; sp70 logs showed F1 still climbing at ep13)
# Pruning still ends at ep10, so 6 fine-tune epochs after compression is reached.
# Predicted wall time: 16 * ~67min = ~18h on the same GPU.
set -e
cd "$(dirname "$0")"
python run_experiment.py \
  --sparsity 0.5 \
  --epochs 16 \
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
  --exp_tag fast_sp50
