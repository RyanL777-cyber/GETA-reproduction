"""
Phase 4.5 — Exp F: Exp E with bit reduction DISABLED.

Same setup as Exp E (quant wrap + fix + GETA sp=0), but pin bit width at 16
by setting `min_bit_wt=16` (upstream default is 4). This isolates whether
the 14-F1 gap between Exp D (88.26) and Exp E (74.27) is caused by the
projection-phase bit reduction 16→4, or by something else inside GETA.

Expected outcomes:
  - F1 ≥ 85  →  14-F1 gap was ENTIRELY bit reduction. quant_fix has no
                residual issue. Proceed to Phase 5 full runs.
  - F1 < 85  →  GETA optimizer still has something breaking fine-tuning
                beyond the quant-wrap path. Need further bisection.

Usage:
    python3 exp_f_geta_sp0_16bit.py
"""
import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime

_GETA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "geta")
if os.path.isdir(_GETA_ROOT) and _GETA_ROOT not in sys.path:
    sys.path.insert(0, _GETA_ROOT)

from _common import select_idle_gpu
select_idle_gpu()

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["DATASETS_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import numpy as np

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

import transformers
import datasets as hf_datasets
transformers.logging.set_verbosity_error()
hf_datasets.logging.set_verbosity_error()

from _common import MODEL_NAME, MAX_LENGTH, setup_logger, load_squad, evaluate, wrap_quant
from quant_fix import apply_ste_fix, calibrate_quant_layers

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--lr_quant", type=float, default=1e-4)
    p.add_argument("--bit_reduction", type=int, default=2)
    p.add_argument("--pruning_periods", type=int, default=6)
    p.add_argument("--projection_periods", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_bits", type=int, default=16)
    p.add_argument("--calib_batches", type=int, default=8)
    p.add_argument("--calib_batch_size", type=int, default=4)
    p.add_argument("--out_dir", type=str, default="./results_exp_f")
    p.add_argument("--log_every", type=int, default=200)
    return p.parse_args()


def run(args):
    os.makedirs(args.out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log = setup_logger(os.path.join(args.out_dir, f"exp_f_{ts}.log"), "exp_f")

    log.info(f"[env] torch={torch.__version__}  device={DEVICE}")
    log.info(f"[cfg] epochs={args.epochs}  bs={args.batch_size}  lr={args.lr}  "
             f"lr_quant={args.lr_quant}  br={args.bit_reduction}  "
             f"P={args.pruning_periods}  Kp={args.projection_periods}  seed={args.seed}")
    log.info(f"[cfg] optimizer=GETA (sp=0, min_bit=max_bit=16)  quant_wrap=ON+FIX")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from transformers import AutoTokenizer, AutoModelForQuestionAnswering
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    raw_val, train_ds, val_ds = load_squad(tokenizer, log)

    log.info("[M1] loading BERT QA")
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME).to(DEVICE)

    log.info("[M2] quantize wrap + STE fix")
    apply_ste_fix()
    model = wrap_quant(model, DEVICE)

    log.info(f"[CALIB] running {args.calib_batches} batches of bs={args.calib_batch_size}")
    calib_batches = []
    for i in range(args.calib_batches):
        start = i * args.calib_batch_size
        if start >= len(train_ds):
            break
        idx = list(range(start, start + args.calib_batch_size))
        batch = train_ds[idx]
        calib_batches.append({
            k: torch.as_tensor(batch[k]).to(DEVICE)
            for k in ("input_ids", "attention_mask", "token_type_ids")
            if k in batch
        })
    calibrate_quant_layers(model, calib_batches, num_bits=args.num_bits, log=log)

    log.info("[M3] building OTO graph")
    enc = tokenizer("What?", "Paris.", max_length=MAX_LENGTH, truncation="only_second",
                    padding="max_length", return_tensors="pt")
    dummy = tuple(v.to(DEVICE) for v in [enc["input_ids"], enc["attention_mask"], enc["token_type_ids"]])
    model.eval()
    from only_train_once import OTO
    oto = OTO(model=model, dummy_input=dummy)
    model.train()
    oto.mark_unprunable_by_param_names(["bert.embeddings.word_embeddings.weight"])

    steps_per_epoch = (len(train_ds) + args.batch_size - 1) // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    period_len = total_steps // args.pruning_periods if args.pruning_periods > 0 else total_steps
    start_pruning_step = max(1, period_len)
    pruning_steps = max(1, period_len * (args.pruning_periods - 1))
    start_projection_step = 0
    projection_steps = max(args.projection_periods, start_pruning_step - start_projection_step)

    log.info(f"[SCHED] steps/epoch={steps_per_epoch}  total={total_steps}  "
             f"start_prune={start_pruning_step}  prune_steps={pruning_steps}  "
             f"proj_steps={projection_steps}")

    log.info("[M4] building GETA optimizer (sp=0, min_bit=max_bit=16)")
    optimizer = oto.geta(
        variant="adamw",
        lr=args.lr,
        lr_quant=args.lr_quant,
        target_group_sparsity=0.0,
        start_projection_step=start_projection_step,
        projection_steps=projection_steps,
        projection_periods=args.projection_periods,
        start_pruning_step=start_pruning_step,
        pruning_steps=pruning_steps,
        pruning_periods=args.pruning_periods,
        bit_reduction=args.bit_reduction,
        min_bit_wt=16,   # <-- KEY DIFFERENCE vs Exp E: pin bit width at 16
        max_bit_wt=16,
    )
    log.info(f"[M4] GETA optimizer built, {len(optimizer.param_groups)} param groups")

    log.info("[TRAIN] starting")
    best_f1, best_epoch = -1.0, 0
    global_step = 0
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        indices = torch.randperm(len(train_ds)).tolist()

        for i in range(0, len(train_ds), args.batch_size):
            batch_idx = indices[i:i + args.batch_size]
            batch = train_ds[batch_idx]
            inputs = {k: v.to(DEVICE) for k, v in batch.items()}

            optimizer.zero_grad()
            out = model(**inputs)
            loss = out.loss

            if torch.isnan(loss) or torch.isinf(loss):
                log.error(f"[!] NaN/Inf at step {global_step}")
                return

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

            if global_step % args.log_every == 0:
                avg = epoch_loss / n_batches
                metrics = optimizer.compute_metrics()
                log.info(f"    step {global_step}  loss={avg:.4f}  "
                         f"grp_sp={metrics.group_sparsity:.3f}  "
                         f"n_imp={metrics.num_important_groups}  n_red={metrics.num_redundant_groups}")

        avg_loss = epoch_loss / max(n_batches, 1)
        metrics = optimizer.compute_metrics()
        elapsed = (time.time() - t_start) / 60
        em, f1, _ = evaluate(model, raw_val, val_ds, args.batch_size * 4, DEVICE)
        log.info(f"[EPOCH {epoch}/{args.epochs}] loss={avg_loss:.4f}  "
                 f"grp_sp={metrics.group_sparsity:.3f}  "
                 f"EM={em:.2f}  F1={f1:.2f}  elapsed={elapsed:.1f}min")
        if f1 > best_f1:
            best_f1, best_epoch = f1, epoch

    log.info(f"[TRAIN] done. best F1={best_f1:.2f} at epoch {best_epoch}")
    log.info(f"[VERDICT] {'PASS' if best_f1 >= 85 else 'FAIL'} (threshold F1>=85)")

    result = {
        "exp": "F_geta_sp0_16bit_pinned",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "lr_quant": args.lr_quant,
        "bit_reduction": args.bit_reduction,
        "pruning_periods": args.pruning_periods,
        "projection_periods": args.projection_periods,
        "seed": args.seed,
        "num_bits": args.num_bits,
        "min_bit_wt": 16,
        "max_bit_wt": 16,
        "best_f1": best_f1,
        "best_epoch": best_epoch,
        "final_group_sparsity": float(metrics.group_sparsity),
        "pass": best_f1 >= 85,
    }
    with open(os.path.join(args.out_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    log.info(f"[SAVE] {args.out_dir}/result.json")


def main():
    args = parse_args()
    try:
        run(args)
    except Exception:
        print(traceback.format_exc(), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
