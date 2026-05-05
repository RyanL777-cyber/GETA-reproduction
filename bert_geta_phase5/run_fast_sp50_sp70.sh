#!/usr/bin/env bash
# Combined sweep: sp=0.5 and sp=0.7 with extended fine-tune (epochs=16).
# Reasoning:
#   - sp=0.7 @ 13ep showed F1 still climbing at ep13 (82.02 -> 83.84, +1.83 across 3 post-prune epochs).
#   - 16 epochs gives 6 post-prune recovery epochs; pruning still ends at ep10.
# Output folders: ./results/combined16_sp50/ and ./results/combined16_sp70/
#   (won't collide with existing ./results/fast_sp70_sp70/ from the 13ep run)
# Predicted wall time: 2 * 16 * ~67min = ~36h on the same GPU.
set -e
cd "$(dirname "$0")"
python run_experiment.py \
  --sparsity 0.5 0.7 \
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
  --exp_tag combined16
