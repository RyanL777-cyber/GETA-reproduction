#!/usr/bin/env bash
# =========================================================================
# Phase 4.9 EXPERIMENT matrix (NOT the final reproduction)
# =========================================================================
# Diagnostic + recovery experiments only. The final 4-sparsity 10-epoch
# reproduction belongs in a separate script when we're ready to commit.
#
# Stage A: 7 diagnostic runs (sp10, 3 epoch, 10% data, 1000 val)
#          Goal: identify the intervention that fixes the epoch-2 F1 crash.
#          Wall time: ~3h on 4 GPUs in parallel.
#
# Stage B: 3 recovery runs (sp10, 10 epoch, 10% data, 1000 val)
#          Goal: pick the winning recipe before final reproduction.
#          Wall time: ~5.5h on 3 GPUs in parallel.
#
# All numbers are TREND ONLY — not comparable to paper Table 3 (subset data).
#
# =========================================================================
# USAGE
# =========================================================================
#
#   # Default: auto-select GPUs (one per job), runs A then B
#   bash run_all_exps.sh
#
#   # Run only one stage
#   STAGES=A bash run_all_exps.sh
#   STAGES=B bash run_all_exps.sh
#
#   # Pin to specific GPUs (round-robin assignment).
#   # If you have 7 idle GPUs, each Stage-A job gets its own:
#   GPUS=0,1,2,3,4,5,6 STAGES=A bash run_all_exps.sh
#
#   # Or 8 GPUs across A (7 jobs) + spillover ignored:
#   GPUS=0,1,2,3,4,5,6,7 bash run_all_exps.sh
#
#   # Run in background:
#   nohup bash run_all_exps.sh > run_all.log 2>&1 &
#
# =========================================================================

set -u
cd "$(dirname "$0")"

STAGES="${STAGES:-AB}"
GPUS="${GPUS:-}"          # comma-separated list, e.g. "0,1,2,3,4,5,6"
SLEEP_BETWEEN=20          # seconds between parallel launches (auto-GPU mode
                          # needs this to avoid two jobs grabbing same GPU)

# Parse GPUS env var into an array; empty array → use auto-select
if [[ -n "$GPUS" ]]; then
    IFS=',' read -ra GPU_LIST <<< "$GPUS"
    echo "[gpu] round-robin across: ${GPU_LIST[*]}"
else
    GPU_LIST=()
    echo "[gpu] auto-select mode (one idle GPU per job, sleep ${SLEEP_BETWEEN}s between launches)"
fi

mkdir -p results
GPU_IDX=0

_run() {
    local tag="$1"; shift
    # Extract --sparsity for stdout filename
    local sp=""
    local args=("$@")
    for ((i=0; i<${#args[@]}; i++)); do
        if [[ "${args[i]}" == "--sparsity" ]]; then
            sp=$(printf "_sp%02d" "$(python -c "print(int(${args[i+1]}*100))")")
            break
        fi
    done
    local logname="${tag}${sp}"

    if [[ ${#GPU_LIST[@]} -gt 0 ]]; then
        # Round-robin pin
        local gpu="${GPU_LIST[$((GPU_IDX % ${#GPU_LIST[@]}))]}"
        GPU_IDX=$((GPU_IDX + 1))
        echo "=== launching ${logname} on GPU ${gpu} ($(date '+%F %T')) ==="
        CUDA_VISIBLE_DEVICES="$gpu" nohup python run_experiment.py "$@" --exp_tag "$tag" \
            > "results/${logname}.stdout.log" 2>&1 &
    else
        # Auto-select
        echo "=== launching ${logname} (auto GPU) ($(date '+%F %T')) ==="
        nohup python run_experiment.py "$@" --exp_tag "$tag" \
            > "results/${logname}.stdout.log" 2>&1 &
        sleep "$SLEEP_BETWEEN"
    fi
}

# =========================================================================
# STAGE A — diagnostic verification (sp10, 3 epoch, 10% data)
# =========================================================================
if [[ "$STAGES" == *A* ]]; then
    echo ">>> STAGE A: 7 diagnostic runs (sp10, 3 epoch, 10% data)"

    # A0 — control: same recipe as phase 5 (small dataset version)
    # Expect: F1 still drops ep1 → ep2 (proportionally, not absolute 85→70)
    _run A0_control \
        --sparsity 0.1 --epochs 3 --eval_last_n 3 \
        --train_subset_frac 0.1 --val_subset_n 1000

    # A1 — disable weight bit reduction (audit's primary verifier)
    # Expect: F1 monotonically rising or flat across all 3 epochs
    _run A1_no_bitreduce \
        --sparsity 0.1 --epochs 3 --eval_last_n 3 \
        --train_subset_frac 0.1 --val_subset_n 1000 \
        --min_bit_wt 16 --max_bit_wt 16

    # A2 — slow projection (delay pruning to ep 2.5)
    _run A2_slowproj \
        --sparsity 0.1 --epochs 3 --eval_last_n 3 \
        --train_subset_frac 0.1 --val_subset_n 1000 \
        --start_pruning_epoch 2.5

    # A3 — LR scheduler only (linear warmup + decay)
    _run A3_lrsched \
        --sparsity 0.1 --epochs 3 --eval_last_n 3 \
        --train_subset_frac 0.1 --val_subset_n 1000 \
        --lr_scheduler linear --warmup_ratio 0.1

    # A4 — bit_reduction=1 (smaller bit step per period)
    _run A4_br1 \
        --sparsity 0.1 --epochs 3 --eval_last_n 3 \
        --train_subset_frac 0.1 --val_subset_n 1000 \
        --bit_reduction 1

    # A5 — calibrate at 4-bit (audit says shouldn't matter)
    _run A5_calib4 \
        --sparsity 0.1 --epochs 3 --eval_last_n 3 \
        --train_subset_frac 0.1 --val_subset_n 1000 \
        --calib_num_bits 4

    # A6 — 16x more calibration samples
    _run A6_bigcalib \
        --sparsity 0.1 --epochs 3 --eval_last_n 3 \
        --train_subset_frac 0.1 --val_subset_n 1000 \
        --calib_batches 64 --calib_batch_size 8

    wait
    echo ">>> STAGE A complete ($(date '+%F %T'))"
fi

# =========================================================================
# STAGE B — combined recovery (sp10, 10 epoch, 10% data)
# =========================================================================
if [[ "$STAGES" == *B* ]]; then
    echo ">>> STAGE B: 3 recovery runs (sp10, 10 epoch, 10% data)"

    # B1 — slow projection + LR scheduler
    _run B1_slowproj_lrsched \
        --sparsity 0.1 --epochs 10 \
        --train_subset_frac 0.1 --val_subset_n 1000 \
        --start_pruning_epoch 5 \
        --lr_scheduler linear --warmup_ratio 0.1

    # B2 — slow projection only
    _run B2_slowproj_only \
        --sparsity 0.1 --epochs 10 \
        --train_subset_frac 0.1 --val_subset_n 1000 \
        --start_pruning_epoch 5

    # B3 — slow projection + LR scheduler + bit_reduction=1
    _run B3_full \
        --sparsity 0.1 --epochs 10 \
        --train_subset_frac 0.1 --val_subset_n 1000 \
        --start_pruning_epoch 5 \
        --lr_scheduler linear --warmup_ratio 0.1 \
        --bit_reduction 1

    wait
    echo ">>> STAGE B complete ($(date '+%F %T'))"
fi

echo "ALL STAGES DONE ($(date '+%F %T'))"
