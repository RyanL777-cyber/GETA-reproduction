"""
Phase 5 — BERT x GETA full SQuAD experiment (Table 3 reproduction)

Paper settings:
  AdamW, lr=3e-5, lr_quant=1e-4, epochs=10, bit_range=[4,16],
  B=4 (batch_size), P=6 (pruning_periods), Kp=6 (projection_periods),
  Kb=1 (start pruning after 1 period), br=2 (bit_reduction)

Usage:
  # Run single sparsity level
  python run_experiment.py --sparsity 0.5

  # Run all 4 levels sequentially
  python run_experiment.py --sparsity 0.1 0.3 0.5 0.7

  # Override settings
  python run_experiment.py --sparsity 0.5 --epochs 10 --seed 42
"""
import argparse
import collections
import json
import logging
import os
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime

# --- GETA source path ---
_GETA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "geta")
if os.path.isdir(_GETA_ROOT) and _GETA_ROOT not in sys.path:
    sys.path.insert(0, _GETA_ROOT)


# --- Auto GPU selection (must precede import torch) ---
def select_idle_gpu(max_used_mem_mb=2000, max_util=10):
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        print(f"[gpu] CUDA_VISIBLE_DEVICES already set: {os.environ['CUDA_VISIBLE_DEVICES']}")
        return
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
    except Exception:
        return
    cands = []
    for line in proc.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3:
            try:
                cands.append((int(parts[0]), int(parts[1]), int(parts[2])))
            except ValueError:
                pass
    if cands:
        cands.sort(key=lambda x: (x[2], x[1]))
        best = cands[0]
        print(f"[gpu] selected GPU {best[0]} (util={best[2]}%, mem={best[1]}MiB)")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(best[0])


select_idle_gpu()

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["DATASETS_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import numpy as np

# --- Performance: TF32 + cudnn autotuner (safe on Ampere+ GPUs, ~1.3-1.5x) ---
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

import transformers
import datasets as hf_datasets
transformers.logging.set_verbosity_error()
hf_datasets.logging.set_verbosity_error()

# =========================================================================
# Constants
# =========================================================================
MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 384
DOC_STRIDE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    p = argparse.ArgumentParser(description="Phase 5: BERT x GETA SQuAD experiment")
    p.add_argument("--sparsity", type=float, nargs="+", required=True,
                    help="target group sparsity (e.g. 0.1 0.3 0.5 0.7)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=4, help="paper B=4")
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--lr_quant", type=float, default=1e-4)
    p.add_argument("--bit_reduction", type=int, default=2, help="paper br=2")
    p.add_argument("--pruning_periods", type=int, default=6, help="paper P=6")
    p.add_argument("--projection_periods", type=int, default=6, help="paper Kp=6")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_root", type=str, default="./results")
    p.add_argument("--eval_last_n", type=int, default=None,
                    help="only evaluate during last N epochs to save time (default: every epoch)")
    return p.parse_args()


# =========================================================================
# Logging
# =========================================================================
def setup_logger(log_path):
    log = logging.getLogger("phase5")
    log.handlers.clear()
    log.setLevel(logging.INFO)
    log.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)
    return log


# =========================================================================
# Data (same as Phase 4 but full SQuAD)
# =========================================================================
def prepare_train_features(examples, tokenizer):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions, examples["context"],
        max_length=MAX_LENGTH, truncation="only_second",
        stride=DOC_STRIDE, return_overflowing_tokens=True,
        return_offsets_mapping=True, padding="max_length",
    )
    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions, end_positions = [], []
    for i, offsets in enumerate(offset_mapping):
        a = answers[sample_map[i]]
        if len(a["answer_start"]) == 0:
            start_positions.append(0); end_positions.append(0); continue
        sc = a["answer_start"][0]
        ec = sc + len(a["text"][0])
        seq_ids = inputs.sequence_ids(i)
        idx = 0
        while seq_ids[idx] != 1: idx += 1
        cs = idx
        idx = len(seq_ids) - 1
        while seq_ids[idx] != 1: idx -= 1
        ce = idx
        if offsets[cs][0] > ec or offsets[ce][1] < sc:
            start_positions.append(0); end_positions.append(0)
        else:
            idx = cs
            while idx <= ce and offsets[idx][0] <= sc: idx += 1
            start_positions.append(idx - 1)
            idx = ce
            while idx >= cs and offsets[idx][1] >= ec: idx -= 1
            end_positions.append(idx + 1)
    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


def prepare_validation_features(examples, tokenizer):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions, examples["context"],
        max_length=MAX_LENGTH, truncation="only_second",
        stride=DOC_STRIDE, return_overflowing_tokens=True,
        return_offsets_mapping=True, padding="max_length",
    )
    sample_map = inputs.pop("overflow_to_sample_mapping")
    inputs["example_id"] = [examples["id"][s] for s in sample_map]
    new_offset = []
    for i, om in enumerate(inputs["offset_mapping"]):
        seq_ids = inputs.sequence_ids(i)
        new_offset.append([(o if seq_ids[j] == 1 else None) for j, o in enumerate(om)])
    inputs["offset_mapping"] = new_offset
    return inputs


def load_squad(tokenizer, log):
    raw_train = hf_datasets.load_dataset("squad", split="train")
    raw_val = hf_datasets.load_dataset("squad", split="validation")
    log.info(f"[DATA] raw train={len(raw_train)}  raw val={len(raw_val)}")

    train_ds = raw_train.map(
        lambda ex: prepare_train_features(ex, tokenizer),
        batched=True, remove_columns=raw_train.column_names,
    )
    train_ds.set_format(type="torch")

    val_ds = raw_val.map(
        lambda ex: prepare_validation_features(ex, tokenizer),
        batched=True, remove_columns=raw_val.column_names,
    )
    log.info(f"[DATA] train features={len(train_ds)}  val features={len(val_ds)}")
    return raw_val, train_ds, val_ds


# =========================================================================
# Evaluation
# =========================================================================
def normalize_answer(s):
    import string
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    s = ' '.join(s.split())
    return s


def compute_f1(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    common = collections.Counter(pred_tokens) & collections.Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0: return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_exact(prediction, ground_truth):
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def postprocess_predictions(raw_val, val_ds, all_start_logits, all_end_logits,
                            n_best=20, max_answer_length=30):
    example_to_features = collections.defaultdict(list)
    for idx, eid in enumerate(val_ds["example_id"]):
        example_to_features[eid].append(idx)

    predictions = {}
    for example in raw_val:
        eid = example["id"]
        context = example["context"]
        best_score, best_answer = -1e9, ""
        for feat_idx in example_to_features[eid]:
            start_logit = all_start_logits[feat_idx]
            end_logit = all_end_logits[feat_idx]
            offsets = val_ds[feat_idx]["offset_mapping"]
            for si in np.argsort(start_logit)[-n_best:].tolist():
                for ei in np.argsort(end_logit)[-n_best:].tolist():
                    if si > ei or ei - si + 1 > max_answer_length: continue
                    if si >= len(offsets) or ei >= len(offsets): continue
                    if offsets[si] is None or offsets[ei] is None: continue
                    score = start_logit[si] + end_logit[ei]
                    if score > best_score:
                        best_score = score
                        best_answer = context[offsets[si][0]:offsets[ei][1]]
        predictions[eid] = best_answer
    return predictions


@torch.no_grad()
def evaluate(model, raw_val, val_ds, batch_size):
    model.eval()
    fwd_columns = ["input_ids", "attention_mask", "token_type_ids"]
    all_start, all_end = [], []
    for i in range(0, len(val_ds), batch_size):
        batch = val_ds[i:i + batch_size]
        inputs = {k: torch.tensor(batch[k]).to(DEVICE) for k in fwd_columns if k in batch}
        out = model(**inputs)
        all_start.append(out.start_logits.cpu().numpy())
        all_end.append(out.end_logits.cpu().numpy())

    all_start = np.concatenate(all_start, axis=0)
    all_end = np.concatenate(all_end, axis=0)
    predictions = postprocess_predictions(raw_val, val_ds, all_start, all_end)

    em_total, f1_total, count = 0.0, 0.0, 0
    for example in raw_val:
        eid = example["id"]
        gold = example["answers"]["text"]
        pred = predictions.get(eid, "")
        em_total += max(compute_exact(pred, ga) for ga in gold)
        f1_total += max(compute_f1(pred, ga) for ga in gold)
        count += 1
    return 100.0 * em_total / count, 100.0 * f1_total / count, predictions


# =========================================================================
# Single experiment
# =========================================================================
def run_single(sparsity, args, tokenizer, raw_val, train_ds, val_ds, log):
    tag = f"sp{int(sparsity*100):02d}"
    out_dir = os.path.join(args.out_root, tag)
    os.makedirs(out_dir, exist_ok=True)

    log.info(f"\n{'='*60}")
    log.info(f"  EXPERIMENT: sparsity={sparsity}  tag={tag}")
    log.info(f"{'='*60}")

    # --- seed ---
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --- M1: load ---
    log.info("[M1] loading BERT QA")
    from transformers import AutoModelForQuestionAnswering
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME).to(DEVICE)

    # --- M2: quantize ---
    log.info("[M2] quantize wrap")
    from only_train_once.quantization.quant_model import model_to_quantize_model
    from only_train_once.quantization.quant_layers import QuantizationMode
    model = model_to_quantize_model(model, quant_mode=QuantizationMode.WEIGHT_AND_ACTIVATION)
    model = model.to(DEVICE)

    # --- M3: OTO ---
    log.info("[M3] building OTO graph")
    enc = tokenizer("What?", "Paris.", max_length=MAX_LENGTH, truncation="only_second",
                    padding="max_length", return_tensors="pt")
    dummy = tuple(v.to(DEVICE) for v in [enc["input_ids"], enc["attention_mask"], enc["token_type_ids"]])
    model.eval()
    from only_train_once import OTO
    oto = OTO(model=model, dummy_input=dummy)
    model.train()
    oto.mark_unprunable_by_param_names(["bert.embeddings.word_embeddings.weight"])

    # --- Schedule ---
    steps_per_epoch = (len(train_ds) + args.batch_size - 1) // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    # Paper: Kb=1 means start pruning after 1 pruning_period worth of steps
    period_len = total_steps // args.pruning_periods if args.pruning_periods > 0 else total_steps
    start_pruning_step = max(1, period_len)  # after 1 period warmup
    pruning_steps = max(1, period_len * (args.pruning_periods - 1))  # remaining periods
    # Projection phase runs in [start_projection_step, start_pruning_step],
    # reducing bit-width br per projection_period. projection_steps must be
    # >= projection_periods or period_duration = projection_steps // projection_periods = 0.
    start_projection_step = 0
    projection_steps = max(args.projection_periods, start_pruning_step - start_projection_step)

    log.info(f"[SCHED] steps/epoch={steps_per_epoch}  total={total_steps}  "
             f"start_prune={start_pruning_step}  prune_steps={pruning_steps}  "
             f"prune_periods={args.pruning_periods}  proj_periods={args.projection_periods}  "
             f"proj_steps={projection_steps}")

    # --- M4: GETA optimizer ---
    log.info("[M4] building GETA optimizer")
    optimizer = oto.geta(
        variant="adamw",
        lr=args.lr,
        lr_quant=args.lr_quant,
        target_group_sparsity=sparsity,
        start_projection_step=start_projection_step,
        projection_steps=projection_steps,
        projection_periods=args.projection_periods,
        start_pruning_step=start_pruning_step,
        pruning_steps=pruning_steps,
        pruning_periods=args.pruning_periods,
        bit_reduction=args.bit_reduction,
        min_bit_wt=4,
        max_bit_wt=16,
    )
    log.info(f"[M4] GETA optimizer built, {len(optimizer.param_groups)} param groups")

    # --- Training ---
    log.info("[TRAIN] starting")
    best_f1, best_epoch = -1, 0
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
                break

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

            if global_step % 200 == 0:
                avg = epoch_loss / n_batches
                metrics = optimizer.compute_metrics()
                log.info(f"    step {global_step}  loss={avg:.4f}  "
                         f"grp_sp={metrics.group_sparsity:.3f}  "
                         f"n_imp={metrics.num_important_groups}  n_red={metrics.num_redundant_groups}")
        else:
            avg_loss = epoch_loss / max(n_batches, 1)
            metrics = optimizer.compute_metrics()
            elapsed = (time.time() - t_start) / 60

            eval_last_n = args.eval_last_n if args.eval_last_n is not None else args.epochs
            do_eval = (epoch > args.epochs - eval_last_n)
            if do_eval:
                em, f1, _ = evaluate(model, raw_val, val_ds, args.batch_size * 4)
                log.info(f"[EPOCH {epoch}/{args.epochs}] loss={avg_loss:.4f}  "
                         f"grp_sp={metrics.group_sparsity:.3f}  "
                         f"EM={em:.2f}  F1={f1:.2f}  elapsed={elapsed:.1f}min")
                if f1 > best_f1:
                    best_f1, best_epoch = f1, epoch
            else:
                log.info(f"[EPOCH {epoch}/{args.epochs}] loss={avg_loss:.4f}  "
                         f"grp_sp={metrics.group_sparsity:.3f}  "
                         f"(eval skipped)  elapsed={elapsed:.1f}min")
            continue
        break

    log.info(f"[TRAIN] done. best F1={best_f1:.2f} at epoch {best_epoch}")

    # --- Capture actual group sparsity (lost after construct_subnet) ---
    final_metrics = optimizer.compute_metrics()
    actual_group_sparsity = float(final_metrics.group_sparsity)
    log.info(f"[FINAL] target sparsity={sparsity}  actual={actual_group_sparsity:.4f}")

    # --- construct_subnet ---
    log.info("[SUBNET] constructing compressed model")
    try:
        oto.construct_subnet(
            export_huggingface_format=False,
            export_float16=False,
            out_dir=out_dir,
        )
        log.info(f"[SUBNET] full  : {oto.full_group_sparse_model_path}")
        log.info(f"[SUBNET] comp  : {oto.compressed_model_path}")

        full_bops = oto.compute_bops(in_million=True)
        full_params = oto.compute_num_params(in_million=True)

        compressed_model = torch.load(oto.compressed_model_path, map_location=DEVICE)
        if hasattr(compressed_model, 'eval'):
            em_c, f1_c, preds = evaluate(compressed_model, raw_val, val_ds, args.batch_size * 4)
        else:
            em_c, f1_c, preds = -1, -1, {}

        log.info(f"[METRICS] BOPs={full_bops['total']:.1f}M  params={full_params:.2f}M")
        log.info(f"[EVAL compressed] EM={em_c:.2f}  F1={f1_c:.2f}")
    except Exception as e:
        log.error(f"[SUBNET] failed: {e}")
        log.error(traceback.format_exc()[-500:])
        em_c, f1_c, preds = -1, -1, {}
        full_bops, full_params = {"total": -1}, -1

    # --- Save results ---
    result = {
        "sparsity": sparsity,
        "actual_group_sparsity": actual_group_sparsity,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "lr_quant": args.lr_quant,
        "bit_reduction": args.bit_reduction,
        "pruning_periods": args.pruning_periods,
        "projection_periods": args.projection_periods,
        "best_f1": best_f1,
        "best_epoch": best_epoch,
        "compressed_em": em_c,
        "compressed_f1": f1_c,
        "bops_million": full_bops["total"],
        "params_million": full_params,
    }
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    if preds:
        with open(os.path.join(out_dir, "predictions.json"), "w", encoding="utf-8") as f:
            json.dump(preds, f, ensure_ascii=False, indent=2)

    log.info(f"[SAVE] results -> {out_dir}/")
    return result


# =========================================================================
# Main
# =========================================================================
def main():
    args = parse_args()

    # --- Logger ---
    os.makedirs(args.out_root, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.out_root, f"phase5_{ts}.log")
    log = setup_logger(log_path)

    log.info(f"[log] {log_path}")
    log.info(f"[env] torch={torch.__version__}  device={DEVICE}")
    log.info(f"[cfg] sparsity={args.sparsity}  epochs={args.epochs}  bs={args.batch_size}  "
             f"lr={args.lr}  lr_quant={args.lr_quant}  br={args.bit_reduction}  "
             f"P={args.pruning_periods}  Kp={args.projection_periods}  seed={args.seed}")

    # --- Tokenizer + data (load once, reuse across sparsity levels) ---
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    raw_val, train_ds, val_ds = load_squad(tokenizer, log)

    # --- Run experiments ---
    all_results = []
    for sp in args.sparsity:
        result = run_single(sp, args, tokenizer, raw_val, train_ds, val_ds, log)
        all_results.append(result)

    # --- Summary table ---
    log.info(f"\n{'='*60}")
    log.info("  SUMMARY")
    log.info(f"{'='*60}")
    log.info(f"  {'Target':>8}  {'Actual':>8}  {'EM':>8}  {'F1':>8}  {'BOPs(M)':>10}  {'Params(M)':>10}")
    for r in all_results:
        log.info(f"  {r['sparsity']:>8.0%}  {r['actual_group_sparsity']:>8.4f}  "
                 f"{r['compressed_em']:>8.2f}  {r['compressed_f1']:>8.2f}  "
                 f"{r['bops_million']:>10.1f}  {r['params_million']:>10.2f}")

    # Save summary
    with open(os.path.join(args.out_root, "summary.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    log.info(f"\n[OK] Phase 5 complete. Results in {args.out_root}/")


if __name__ == "__main__":
    main()
