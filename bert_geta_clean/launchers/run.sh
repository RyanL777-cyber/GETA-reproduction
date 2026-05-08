#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# General-purpose launcher template for BERT × GETA SQuAD.
# Edit the variables in the "KNOBS" section below; everything else is plumbing.
# For ready-made preset configs (best result, paper-faithful, our actual runs),
# see PRESETS.md in this directory.
# -----------------------------------------------------------------------------
set -e
cd "$(dirname "$0")/.."

# === KNOBS — change these for your experiment ================================

# Target group sparsity. Pass one or multiple to sweep:
#   SPARSITY=(0.7)            # single point
#   SPARSITY=(0.1 0.3 0.5 0.7) # full sweep (~4× wall time)
SPARSITY=(0.5)

EPOCHS=10                # paper used various; 10 is a reasonable starting point
BATCH_SIZE=16            # paper=4; 16 trains ~2× faster on a 24GB GPU
LR=6e-5                  # sqrt-scale from paper 3e-5 when bs goes 4→16
LR_QUANT=1e-4

BIT_REDUCTION=2          # paper br=2; set 1 for faster/less aggressive quantization
START_PRUNE_EPOCH=5      # epoch when group pruning begins
END_PRUNE_EPOCH=10       # epoch when target sparsity is reached

LR_SCHEDULER=linear      # "linear" (warmup+decay) or "none"
WARMUP_RATIO=0.1

EXP_TAG=my_run           # output dir prefix: results/${EXP_TAG}_sp<NN>/

# === PLUMBING — usually no need to touch ====================================

python src/run_experiment.py \
  --sparsity "${SPARSITY[@]}" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --lr_quant "$LR_QUANT" \
  --bit_reduction "$BIT_REDUCTION" \
  --pruning_periods 6 \
  --projection_periods 6 \
  --start_pruning_epoch "$START_PRUNE_EPOCH" \
  --pruning_end_epoch "$END_PRUNE_EPOCH" \
  --lr_scheduler "$LR_SCHEDULER" \
  --warmup_ratio "$WARMUP_RATIO" \
  --seed 42 \
  --out_root ./results \
  --exp_tag "$EXP_TAG"
