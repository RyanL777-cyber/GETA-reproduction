"""
Phase 4.9 — BERT x GETA SQuAD experiment, diagnostic + recovery sweeps.

Forked from phase 5 after diagnosis showed:
  - F1 crashes at epoch 2 (85.93 -> 70.76) while grp_sp = 0
  - Audit confirms cause: weight bit projection finishes in ~1.67 epoch
    (16 -> 4) while there's no time to recover
  - Activation projection is commented out in geta.py optimizer.step(),
    so calibration mismatch is NOT the issue (activation stays at 16-bit)

Phase 4.9 adds CLI flags so all the verification & recovery experiments
can be driven from one script:

  --min_bit_wt / --max_bit_wt    override 4 / 16 (e.g. set both = 16 to
                                  fully disable weight bit reduction)
  --calib_num_bits                bit width for calibrate_quant_layers
                                  (default 16; only matters if you want
                                  to align activation calib with min_bit_wt)
  --start_pruning_epoch           when (in epochs) pruning starts; lets you
                                  delay pruning to lengthen the projection
                                  window (default = use paper's 1 period
                                  = total/pruning_periods)
  --pruning_end_epoch             when (in epochs) pruning ends; default =
                                  end of training
  --lr_scheduler                  none | linear  (linear = warmup+decay,
                                  applied to BOTH lr and lr_quant)
  --warmup_ratio                  warmup steps as fraction of total
  --exp_tag                       suffix for output dir
                                  (results/<tag>_sp<NN>/)

See run_all_exps.sh for the planned experiment matrix.
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
    p = argparse.ArgumentParser(description="Phase 4.9: BERT x GETA SQuAD experiment")
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
    p.add_argument("--calib_batches", type=int, default=8,
                    help="number of calibration batches for activation quant init")
    p.add_argument("--calib_batch_size", type=int, default=4,
                    help="calibration batch size")
    # --- phase 4.9 additions ---
    p.add_argument("--min_bit_wt", type=int, default=4,
                    help="GETA min weight bit width (paper=4; set =max_bit_wt to disable bit reduction)")
    p.add_argument("--max_bit_wt", type=int, default=16,
                    help="GETA max weight bit width (paper=16)")
    p.add_argument("--calib_num_bits", type=int, default=16,
                    help="bit width passed to calibrate_quant_layers (default 16; activation projection is OFF in current geta.py)")
    p.add_argument("--start_pruning_epoch", type=float, default=None,
                    help="epoch at which pruning starts; default = total/pruning_periods (paper Kb=1)")
    p.add_argument("--pruning_end_epoch", type=float, default=None,
                    help="epoch at which pruning finishes; default = epochs (paper Kb=1)")
    p.add_argument("--lr_scheduler", type=str, default="none", choices=["none", "linear"],
                    help="LR schedule for both lr and lr_quant (linear = warmup+decay)")
    p.add_argument("--warmup_ratio", type=float, default=0.1,
                    help="warmup fraction of total steps when --lr_scheduler=linear")
    p.add_argument("--exp_tag", type=str, default="",
                    help="prefix for output dir; results/<tag>_sp<NN>/")
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
    sp_tag = f"sp{int(sparsity*100):02d}"
    tag = f"{args.exp_tag}_{sp_tag}" if args.exp_tag else sp_tag
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

    # --- M2: quantize + STE fix (phase 4.5 fix) ---
    log.info("[M2] quantize wrap + STE fix")
    from only_train_once.quantization.quant_model import model_to_quantize_model
    from only_train_once.quantization.quant_layers import QuantizationMode
    from quant_fix import apply_ste_fix, calibrate_quant_layers
    apply_ste_fix()
    model = model_to_quantize_model(model, quant_mode=QuantizationMode.WEIGHT_AND_ACTIVATION)
    model = model.to(DEVICE)

    # --- M2.5: calibrate activation quant on real SQuAD batches ---
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
    calibrate_quant_layers(model, calib_batches, num_bits=args.calib_num_bits, log=log)

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
    period_len = total_steps // args.pruning_periods if args.pruning_periods > 0 else total_steps

    # Paper Kb=1: start pruning after 1 pruning_period (= total/P).
    # Phase 4.9: --start_pruning_epoch lets us delay it (so projection has
    # more room before pruning kicks in).
    if args.start_pruning_epoch is not None:
        start_pruning_step = max(1, int(args.start_pruning_epoch * steps_per_epoch))
    else:
        start_pruning_step = max(1, period_len)

    if args.pruning_end_epoch is not None:
        pruning_end_step = int(args.pruning_end_epoch * steps_per_epoch)
        pruning_steps = max(1, pruning_end_step - start_pruning_step)
    else:
        pruning_steps = max(1, period_len * (args.pruning_periods - 1))

    # Projection phase runs in [start_projection_step, start_pruning_step].
    # By extending start_pruning_step, projection_steps grows automatically.
    start_projection_step = 0
    projection_steps = max(args.projection_periods, start_pruning_step - start_projection_step)

    log.info(f"[SCHED] steps/epoch={steps_per_epoch}  total={total_steps}  "
             f"start_prune={start_pruning_step} ({start_pruning_step/steps_per_epoch:.2f} ep)  "
             f"prune_steps={pruning_steps} ({pruning_steps/steps_per_epoch:.2f} ep)  "
             f"prune_periods={args.pruning_periods}  proj_periods={args.projection_periods}  "
             f"proj_steps={projection_steps} ({projection_steps/steps_per_epoch:.2f} ep)")
    log.info(f"[BIT]   min_bit_wt={args.min_bit_wt}  max_bit_wt={args.max_bit_wt}  "
             f"bit_reduction={args.bit_reduction}  calib_num_bits={args.calib_num_bits}")
    log.info(f"[LR]    scheduler={args.lr_scheduler}  warmup_ratio={args.warmup_ratio}  "
             f"lr={args.lr}  lr_quant={args.lr_quant}")

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
        min_bit_wt=args.min_bit_wt,
        max_bit_wt=args.max_bit_wt,
    )
    log.info(f"[M4] GETA optimizer built, {len(optimizer.param_groups)} param groups")

    # --- LR scheduler (phase 4.9): scales BOTH lr and lr_quant per step ---
    # GETA stores lr_quant per param_group (see geta.py:130), so we snapshot
    # both and rescale every step. self.lr_quant attr on the optimizer is a
    # default and not read on hot path, but we keep it consistent too.
    base_lrs = [g.get("lr", args.lr) for g in optimizer.param_groups]
    base_lr_quants = [g.get("lr_quant", args.lr_quant) for g in optimizer.param_groups]
    base_lr_quant_default = getattr(optimizer, "lr_quant", args.lr_quant)
    if args.lr_scheduler == "linear":
        warmup_steps = max(1, int(args.warmup_ratio * total_steps))
        log.info(f"[LR] linear schedule warmup={warmup_steps} decay over {total_steps - warmup_steps}")

        def lr_multiplier(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            remain = max(0, total_steps - step)
            return remain / max(1, total_steps - warmup_steps)
    else:
        def lr_multiplier(step):
            return 1.0

    # --- Training ---
    log.info("[TRAIN] starting")
    best_f1, best_epoch = -1.0, 0
    final_f1, final_em = -1.0, -1.0
    epoch_log = []  # list of dicts: {epoch, loss, grp_sp, em, f1}
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

            # LR scheduler: scale BOTH lr and lr_quant for every group
            if args.lr_scheduler != "none":
                mult = lr_multiplier(global_step)
                for g, base_lr, base_lq in zip(optimizer.param_groups, base_lrs, base_lr_quants):
                    g["lr"] = base_lr * mult
                    g["lr_quant"] = base_lq * mult
                optimizer.lr_quant = base_lr_quant_default * mult

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
                final_f1, final_em = f1, em
                epoch_log.append({"epoch": epoch, "loss": avg_loss,
                                  "grp_sp": float(metrics.group_sparsity),
                                  "em": em, "f1": f1})
            else:
                log.info(f"[EPOCH {epoch}/{args.epochs}] loss={avg_loss:.4f}  "
                         f"grp_sp={metrics.group_sparsity:.3f}  "
                         f"(eval skipped)  elapsed={elapsed:.1f}min")
                epoch_log.append({"epoch": epoch, "loss": avg_loss,
                                  "grp_sp": float(metrics.group_sparsity),
                                  "em": None, "f1": None})
            continue
        break

    # NB: best_f1 = best across ALL epochs (often epoch 1, not the compressed model)
    # final_f1 = last epoch (closest to compressed model state)
    # compressed_f1 (computed below) = post-construct_subnet, the real number
    log.info(f"[TRAIN] done. best F1={best_f1:.2f} @ ep{best_epoch} (pre-compression)  "
             f"final F1={final_f1:.2f}")

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
        "exp_tag": args.exp_tag,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "lr_quant": args.lr_quant,
        "bit_reduction": args.bit_reduction,
        "pruning_periods": args.pruning_periods,
        "projection_periods": args.projection_periods,
        "min_bit_wt": args.min_bit_wt,
        "max_bit_wt": args.max_bit_wt,
        "calib_num_bits": args.calib_num_bits,
        "start_pruning_epoch": args.start_pruning_epoch,
        "pruning_end_epoch": args.pruning_end_epoch,
        "lr_scheduler": args.lr_scheduler,
        "warmup_ratio": args.warmup_ratio,
        "best_f1": best_f1,
        "best_epoch": best_epoch,
        "final_f1": final_f1,
        "final_em": final_em,
        "compressed_em": em_c,
        "compressed_f1": f1_c,
        "bops_million": full_bops["total"],
        "params_million": full_params,
        "epoch_log": epoch_log,
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
    log.info(f"  {'Target':>8}  {'Actual':>8}  {'fEM':>7}  {'fF1':>7}  {'cEM':>7}  {'cF1':>7}  {'BOPs(M)':>10}  {'Params(M)':>10}")
    for r in all_results:
        log.info(f"  {r['sparsity']:>8.0%}  {r['actual_group_sparsity']:>8.4f}  "
                 f"{r['final_em']:>7.2f}  {r['final_f1']:>7.2f}  "
                 f"{r['compressed_em']:>7.2f}  {r['compressed_f1']:>7.2f}  "
                 f"{r['bops_million']:>10.1f}  {r['params_million']:>10.2f}")
    log.info("  (fEM/fF1 = last-epoch eval before construct_subnet; cEM/cF1 = compressed model)")

    # Save summary
    with open(os.path.join(args.out_root, "summary.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    log.info(f"\n[OK] Phase 5 complete. Results in {args.out_root}/")


if __name__ == "__main__":
    main()
