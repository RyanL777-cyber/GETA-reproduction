#!/usr/bin/env bash
# Frozen snapshot of the 2026-05-02 baseline run (results/sp10, 25.5h, F1=84.45 @ sp10).
# Use this to reproduce the original config; switch back anytime with: bash run_baseline.sh
set -e
cd "$(dirname "$0")"
python run_experiment.py \
  --sparsity 0.1 \
  --epochs 10 \
  --batch_size 4 \
  --lr 3e-5 \
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
  --exp_tag baseline_repro
