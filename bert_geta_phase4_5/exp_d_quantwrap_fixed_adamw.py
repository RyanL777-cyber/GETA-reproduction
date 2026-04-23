"""
Phase 4.5 — Exp D: Exp A + quant_fix.

Quant wrap ON (with calibration + STE fix) + vanilla AdamW + linear warmup.
Target: F1 >= 80 (Exp A baseline: 27.33).

If this PASSES, the fix works in pure-AdamW setting. Move to Exp E.
If this FAILS, the fix is insufficient — inspect `exp_c2` numbers.

Usage:
    python3 exp_d_quantwrap_fixed_adamw.py
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

from _common import MODEL_NAME, setup_logger, load_squad, evaluate, wrap_quant
from quant_fix import apply_ste_fix, calibrate_quant_layers

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=12)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_bits", type=int, default=16)
    p.add_argument("--calib_batches", type=int, default=8)
    p.add_argument("--calib_batch_size", type=int, default=4)
    p.add_argument("--out_dir", type=str, default="./results_exp_d")
    p.add_argument("--log_every", type=int, default=200)
    return p.parse_args()


def run(args):
    os.makedirs(args.out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log = setup_logger(os.path.join(args.out_dir, f"exp_d_{ts}.log"), "exp_d")

    log.info(f"[env] torch={torch.__version__}  device={DEVICE}")
    log.info(f"[cfg] epochs={args.epochs}  bs={args.batch_size}  lr={args.lr}  "
             f"wd={args.weight_decay}  warmup={args.warmup_ratio}  seed={args.seed}")
    log.info(f"[cfg] optimizer=AdamW (no GETA)  quant_wrap=ON+FIX  num_bits={args.num_bits}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from transformers import (
        AutoModelForQuestionAnswering, AutoTokenizer, get_linear_schedule_with_warmup,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    raw_val, train_ds, val_ds = load_squad(tokenizer, log)

    # --- Model ---
    log.info("[M1] loading BERT QA")
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME).to(DEVICE)

    log.info("[M2] quantize wrap + STE fix")
    apply_ste_fix()
    model = wrap_quant(model, DEVICE)

    # --- Calibrate activation quant on first few batches of train_ds ---
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

    # --- Optimizer + schedule ---
    steps_per_epoch = (len(train_ds) + args.batch_size - 1) // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    log.info(f"[SCHED] steps/epoch={steps_per_epoch}  total={total_steps}  warmup={warmup_steps}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # --- Training ---
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
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

            if global_step % args.log_every == 0:
                avg = epoch_loss / n_batches
                lr_now = scheduler.get_last_lr()[0]
                log.info(f"    step {global_step}  loss={avg:.4f}  lr={lr_now:.2e}")

        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed = (time.time() - t_start) / 60
        em, f1, _ = evaluate(model, raw_val, val_ds, args.batch_size * 2, DEVICE)
        log.info(f"[EPOCH {epoch}/{args.epochs}] loss={avg_loss:.4f}  "
                 f"EM={em:.2f}  F1={f1:.2f}  elapsed={elapsed:.1f}min")
        if f1 > best_f1:
            best_f1, best_epoch = f1, epoch

    log.info(f"[TRAIN] done. best F1={best_f1:.2f} at epoch {best_epoch}")
    log.info(f"[VERDICT] {'PASS' if best_f1 >= 80 else 'FAIL'} (threshold F1>=80)")

    result = {
        "exp": "D_quantwrap_fixed_adamw",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "seed": args.seed,
        "num_bits": args.num_bits,
        "best_f1": best_f1,
        "best_epoch": best_epoch,
        "pass": best_f1 >= 80,
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
