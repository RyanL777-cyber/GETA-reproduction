#!/usr/bin/env bash
# =========================================================================
# Phase 4.9 experiment matrix
# =========================================================================
# Source of truth for what to run. Each `_run` call is one experiment; comment
# out any block you don't want this round. Each call writes to its own
# results/<exp_tag>_sp<NN>/ directory so multiple experiments coexist safely.
#
# Cost reference (phase 5 baseline ~5.5h / epoch on full SQuAD, one GPU):
#   Stage A: 10% data + 1000 val, 3 epoch x 7 expts
#            -> ~33min/epoch x 3 = ~1.7h/expt; 4 GPUs in parallel ~3h wall
#   Stage B: 10% data + 1000 val, 10 epoch x 3 expts
#            -> ~33min/epoch x 10 = ~5.5h/expt; 3 GPUs in parallel ~5.5h wall
#   Stage C: FULL data, 10 epoch x 4 sparsities
#            -> ~55h/expt; 4 GPUs in parallel ~55h wall (the real reproduction)
#
# Total wall time (default STAGES=AB then C separately): ~3h + ~5.5h = ~8.5h
# for diagnostic; ~55h for final reproduction. Diagnostic numbers are NOT
# directly comparable to paper Table 3 (subset data) — they are for trend
# detection only (does F1 crash? does it recover?).
#
# Run patterns:
#   bash run_all_exps.sh                                 # foreground
#   nohup bash run_all_exps.sh > run_all.log 2>&1 &      # background
#
# To run only one stage:
#   STAGES=A bash run_all_exps.sh
#   STAGES=B bash run_all_exps.sh
#   STAGES=AB bash run_all_exps.sh
# =========================================================================

set -u
cd "$(dirname "$0")"

STAGES="${STAGES:-AB}"   # default: A then B; add C only after picking winner
SLEEP_BETWEEN=15         # seconds between parallel launches (so auto-GPU
                         # selection has time to see the previous job's
                         # memory usage and pick a different GPU)

_run() {
    local tag="$1"; shift
    # Extract --sparsity value (first one if multiple) for stdout filename
    local sp=""
    local args=("$@")
    for ((i=0; i<${#args[@]}; i++)); do
        if [[ "${args[i]}" == "--sparsity" ]]; then
            sp=$(printf "_sp%02d" "$(python -c "print(int(${args[i+1]}*100))")")
            break
        fi
    done
    local logname="${tag}${sp}"
    echo "=== launching ${logname} ($(date '+%F %T')) ==="
    nohup python run_experiment.py "$@" --exp_tag "$tag" \
        > "results/${logname}.stdout.log" 2>&1 &
    sleep "$SLEEP_BETWEEN"
}

mkdir -p results

# =========================================================================
# STAGE A — diagnostic verification (sp10, 3 epochs, 7 experiments)
# Goal: identify which intervention fixes the epoch-2 F1 crash.
# Launch all in parallel; auto-GPU picks idle GPUs round-robin.
# =========================================================================
if [[ "$STAGES" == *A* ]]; then
    echo ">>> STAGE A: 7 diagnostic runs (sp10, 3 epochs each)"

    # A0 — control: reproduce phase 5 result exactly (sanity check).
    # Expect: F1 crashes from ~85 (ep1) to ~70 (ep2).
    _run A0_control \
        --sparsity 0.1 --epochs 3 --eval_last_n 3 \
        --train_subset_frac 0.1 --val_subset_n 1000

    # A1 — disable weight bit reduction (audit's primary verifier).
    # min=max=16 → GETA never lowers bit width.
    # Expect: F1 stays ~85 across all 3 epochs.
    _run A1_no_bitreduce \
        --sparsity 0.1 --epochs 3 --eval_last_n 3 \
        --train_subset_frac 0.1 --val_subset_n 1000 \
        --min_bit_wt 16 --max_bit_wt 16

    # A2 — slow projection: pruning starts at ep 2.5 (vs ~1.67 baseline).
    # Projection has more room for 16→4 walk before pruning kicks in.
    # Expect: gentler F1 dip + better recovery.
    _run A2_slowproj \
        --sparsity 0.1 --epochs 3 --eval_last_n 3 \
        --train_subset_frac 0.1 --val_subset_n 1000 \
        --start_pruning_epoch 2.5

    # A3 — LR scheduler only (linear warmup + decay), no schedule changes.
    # Expect: helps recovery, not a full fix.
    _run A3_lrsched \
        --sparsity 0.1 --epochs 3 --eval_last_n 3 \
        --train_subset_frac 0.1 --val_subset_n 1000 \
        --lr_scheduler linear --warmup_ratio 0.1

    # A4 — bit_reduction=1 (smaller bit step per period).
    # 16→15→14→… vs 16→14→12→…. Same total, smoother curve.
    _run A4_br1 \
        --sparsity 0.1 --epochs 3 --eval_last_n 3 \
        --train_subset_frac 0.1 --val_subset_n 1000 \
        --bit_reduction 1

    # A5 — calibrate at 4-bit instead of 16-bit.
    # Audit says activation projection is OFF in geta.py, so this should
    # have no effect; cheap sanity check that the audit is right.
    _run A5_calib4 \
        --sparsity 0.1 --epochs 3 --eval_last_n 3 \
        --train_subset_frac 0.1 --val_subset_n 1000 \
        --calib_num_bits 4

    # A6 — 16x more calibration samples (32 → 512).
    # Tests whether calibration sample size matters.
    _run A6_bigcalib \
        --sparsity 0.1 --epochs 3 --eval_last_n 3 \
        --train_subset_frac 0.1 --val_subset_n 1000 \
        --calib_batches 64 --calib_batch_size 8

    wait
    echo ">>> STAGE A complete ($(date '+%F %T'))"
fi

# =========================================================================
# STAGE B — combined recovery (sp10, 10 epochs, 3 combos)
# Goal: pick the winning recipe before committing to a 4-sparsity sweep.
# Each ~55h on one GPU; running all 3 in parallel → ~55h wall.
# =========================================================================
if [[ "$STAGES" == *B* ]]; then
    echo ">>> STAGE B: 3 recovery runs (sp10, 10 epochs each)"

    # B1 — slow projection + LR scheduler (default best guess)
    _run B1_slowproj_lrsched \
        --sparsity 0.1 --epochs 10 \
        --train_subset_frac 0.1 --val_subset_n 1000 \
        --start_pruning_epoch 5 \
        --lr_scheduler linear --warmup_ratio 0.1

    # B2 — slow projection only (isolate scheduler contribution)
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

# =========================================================================
# STAGE C — final 4-sparsity sweep (10 epochs, 4 GPUs in parallel)
# Edit flags below to match the winning B* combo before running.
# =========================================================================
if [[ "$STAGES" == *C* ]]; then
    echo ">>> STAGE C: 4-sparsity sweep with winning combo (10 epochs each)"

    for SP in 0.1 0.3 0.5 0.7; do
        _run C1_winner \
            --sparsity $SP --epochs 10 \
            --start_pruning_epoch 5 \
            --lr_scheduler linear --warmup_ratio 0.1
    done

    wait
    echo ">>> STAGE C complete ($(date '+%F %T'))"
fi

# =========================================================================
# Post-processing (run after all stages)
# =========================================================================
# python build_table.py --results_dir ./results --out ./results/table3.md
echo "ALL STAGES DONE ($(date '+%F %T'))"
