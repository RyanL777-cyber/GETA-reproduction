"""
Phase 4 — BERT x GETA small-scale SQuAD training
Goal: confirm joint pruning+quantization training runs stably, get preliminary EM/F1.
NOT chasing paper numbers yet -- that's Phase 5.

Usage:
    python train_phase4.py                    # default: 2000 train, 500 val, 3 epochs
    python train_phase4.py --full             # full SQuAD, 10 epochs (Phase 5 preview)
"""
import argparse
import collections
import json
import logging
import os
import subprocess
import sys
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
    except Exception as exc:
        print(f"[gpu] nvidia-smi not available: {exc}")
        return
    cands = []
    for line in proc.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            cands.append((int(parts[0]), int(parts[1]), int(parts[2])))
        except ValueError:
            continue
    if not cands:
        return
    cands.sort(key=lambda x: (x[2], x[1]))
    best_idx, best_mem, best_util = cands[0]
    status = "idle" if (best_util <= max_util and best_mem <= max_used_mem_mb) else "busy-but-best"
    print(f"[gpu] selected GPU {best_idx} ({status}): util={best_util}%, mem_used={best_mem}MiB")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(best_idx)


select_idle_gpu()

# --- Quiet HF/datasets ---
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["DATASETS_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch  # noqa: E402
import numpy as np  # noqa: E402

# =========================================================================
# Logger
# =========================================================================
_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(_LOG_DIR, exist_ok=True)
_LOG_PATH = os.path.join(_LOG_DIR, f"phase4_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

log = logging.getLogger("phase4")
log.setLevel(logging.INFO)
log.propagate = False
_fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
_fh = logging.FileHandler(_LOG_PATH, mode="w", encoding="utf-8")
_fh.setFormatter(_fmt)
_sh = logging.StreamHandler(sys.stdout)
_sh.setFormatter(_fmt)
log.addHandler(_fh)
log.addHandler(_sh)

import transformers  # noqa: E402
import datasets as hf_datasets  # noqa: E402
transformers.logging.set_verbosity_error()
hf_datasets.logging.set_verbosity_error()

# =========================================================================
# Config
# =========================================================================
MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 384
DOC_STRIDE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--full", action="store_true", help="use full SQuAD (Phase 5 preview)")
    p.add_argument("--train_samples", type=int, default=2000, help="train subset size")
    p.add_argument("--val_samples", type=int, default=500, help="val subset size")
    p.add_argument("--epochs", type=int, default=3, help="training epochs")
    p.add_argument("--batch_size", type=int, default=8, help="batch size")
    p.add_argument("--lr", type=float, default=3e-5, help="learning rate")
    p.add_argument("--lr_quant", type=float, default=1e-4, help="quant scalar lr")
    p.add_argument("--target_sparsity", type=float, default=0.5, help="target group sparsity")
    p.add_argument("--bit_reduction", type=int, default=2, help="bit reduction")
    p.add_argument("--out_dir", type=str, default="./phase4_out", help="output directory")
    args = p.parse_args()
    if args.full:
        args.train_samples = 0  # 0 = all
        args.val_samples = 0
        args.epochs = 10
    return args


# =========================================================================
# Data
# =========================================================================
def prepare_train_features(examples, tokenizer):
    """Tokenize training examples with answer span mapping."""
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions, examples["context"],
        max_length=MAX_LENGTH, truncation="only_second",
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions, end_positions = [], []

    for i, offsets in enumerate(offset_mapping):
        a = answers[sample_map[i]]
        if len(a["answer_start"]) == 0:
            start_positions.append(0)
            end_positions.append(0)
            continue
        sc = a["answer_start"][0]
        ec = sc + len(a["text"][0])
        seq_ids = inputs.sequence_ids(i)
        # find context span
        idx = 0
        while seq_ids[idx] != 1:
            idx += 1
        cs = idx
        idx = len(seq_ids) - 1
        while seq_ids[idx] != 1:
            idx -= 1
        ce = idx
        if offsets[cs][0] > ec or offsets[ce][1] < sc:
            start_positions.append(0)
            end_positions.append(0)
        else:
            idx = cs
            while idx <= ce and offsets[idx][0] <= sc:
                idx += 1
            start_positions.append(idx - 1)
            idx = ce
            while idx >= cs and offsets[idx][1] >= ec:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


def prepare_validation_features(examples, tokenizer):
    """Tokenize validation examples, keeping offset_mapping and example_id for postprocessing."""
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions, examples["context"],
        max_length=MAX_LENGTH, truncation="only_second",
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    sample_map = inputs.pop("overflow_to_sample_mapping")
    inputs["example_id"] = [examples["id"][s] for s in sample_map]
    # set offsets of non-context tokens to None (for postprocessing)
    new_offset = []
    for i, om in enumerate(inputs["offset_mapping"]):
        seq_ids = inputs.sequence_ids(i)
        new_offset.append([(o if seq_ids[j] == 1 else None) for j, o in enumerate(om)])
    inputs["offset_mapping"] = new_offset
    return inputs


def load_squad(args, tokenizer):
    """Load SQuAD train/val and tokenize."""
    train_split = "train" if args.train_samples == 0 else f"train[:{args.train_samples}]"
    val_split = "validation" if args.val_samples == 0 else f"validation[:{args.val_samples}]"

    raw_train = hf_datasets.load_dataset("squad", split=train_split)
    raw_val = hf_datasets.load_dataset("squad", split=val_split)

    train_ds = raw_train.map(
        lambda ex: prepare_train_features(ex, tokenizer),
        batched=True, remove_columns=raw_train.column_names,
    )
    train_ds.set_format(type="torch")

    val_ds = raw_val.map(
        lambda ex: prepare_validation_features(ex, tokenizer),
        batched=True, remove_columns=raw_val.column_names,
    )

    return raw_val, train_ds, val_ds


# =========================================================================
# Evaluation (SQuAD EM / F1)
# =========================================================================
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    import re
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
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_exact(prediction, ground_truth):
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def postprocess_predictions(raw_val, val_ds, all_start_logits, all_end_logits, n_best=20, max_answer_length=30):
    """Convert model logits to text predictions, grouped by example_id."""
    example_to_features = collections.defaultdict(list)
    for idx, eid in enumerate(val_ds["example_id"]):
        example_to_features[eid].append(idx)

    predictions = {}
    for example in raw_val:
        eid = example["id"]
        context = example["context"]
        best_score = -1e9
        best_answer = ""

        for feat_idx in example_to_features[eid]:
            start_logit = all_start_logits[feat_idx]
            end_logit = all_end_logits[feat_idx]
            offsets = val_ds[feat_idx]["offset_mapping"]

            start_idxs = np.argsort(start_logit)[-n_best:].tolist()
            end_idxs = np.argsort(end_logit)[-n_best:].tolist()

            for si in start_idxs:
                for ei in end_idxs:
                    if si > ei or ei - si + 1 > max_answer_length:
                        continue
                    if si >= len(offsets) or ei >= len(offsets):
                        continue
                    if offsets[si] is None or offsets[ei] is None:
                        continue
                    score = start_logit[si] + end_logit[ei]
                    if score > best_score:
                        best_score = score
                        best_answer = context[offsets[si][0]:offsets[ei][1]]

        predictions[eid] = best_answer

    return predictions


@torch.no_grad()
def evaluate(model, raw_val, val_ds, batch_size):
    """Run inference on val set and compute EM/F1."""
    model.eval()
    # columns needed for model forward
    fwd_columns = ["input_ids", "attention_mask", "token_type_ids"]
    all_start_logits, all_end_logits = [], []

    n = len(val_ds)
    for i in range(0, n, batch_size):
        batch = val_ds[i:i + batch_size]
        inputs = {k: torch.tensor(batch[k]).to(DEVICE) for k in fwd_columns if k in batch}
        out = model(**inputs)
        all_start_logits.append(out.start_logits.cpu().numpy())
        all_end_logits.append(out.end_logits.cpu().numpy())

    all_start_logits = np.concatenate(all_start_logits, axis=0)
    all_end_logits = np.concatenate(all_end_logits, axis=0)

    predictions = postprocess_predictions(raw_val, val_ds, all_start_logits, all_end_logits)

    # compute metrics
    em_total, f1_total, count = 0.0, 0.0, 0
    for example in raw_val:
        eid = example["id"]
        gold_answers = example["answers"]["text"]
        pred = predictions.get(eid, "")
        em_total += max(compute_exact(pred, ga) for ga in gold_answers)
        f1_total += max(compute_f1(pred, ga) for ga in gold_answers)
        count += 1

    em = 100.0 * em_total / count
    f1 = 100.0 * f1_total / count
    model.train()
    return em, f1, predictions


# =========================================================================
# Main
# =========================================================================
def main():
    args = parse_args()
    log.info(f"[log] writing to {_LOG_PATH}")
    log.info(f"[env] torch={torch.__version__}  device={DEVICE}")
    log.info(f"[cfg] train_samples={args.train_samples or 'ALL'}  val_samples={args.val_samples or 'ALL'}  "
             f"epochs={args.epochs}  batch_size={args.batch_size}  lr={args.lr}  lr_quant={args.lr_quant}  "
             f"target_sparsity={args.target_sparsity}  bit_reduction={args.bit_reduction}")

    os.makedirs(args.out_dir, exist_ok=True)

    # --- M1: load model ---
    log.info("[M1] loading BERT QA + tokenizer")
    from transformers import AutoTokenizer, AutoModelForQuestionAnswering
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME).to(DEVICE)
    log.info("[M1] done")

    # --- M2: quantize wrap ---
    log.info("[M2] model_to_quantize_model")
    from only_train_once.quantization.quant_model import model_to_quantize_model
    from only_train_once.quantization.quant_layers import QuantizationMode
    model = model_to_quantize_model(model, quant_mode=QuantizationMode.WEIGHT_AND_ACTIVATION)
    model = model.to(DEVICE)  # quant wrap adds CPU scalars, move back
    log.info("[M2] done")

    # --- M3: build OTO ---
    log.info("[M3] building OTO graph")
    sample_q, sample_c = "What is the capital of France?", "Paris is the capital of France."
    enc = tokenizer(sample_q, sample_c, max_length=MAX_LENGTH, truncation="only_second",
                    padding="max_length", return_tensors="pt")
    dummy_input = tuple(v.to(DEVICE) for v in [enc["input_ids"], enc["attention_mask"], enc["token_type_ids"]])
    model.eval()
    from only_train_once import OTO
    oto = OTO(model=model, dummy_input=dummy_input)
    model.train()
    log.info("[M3] done")

    # --- mark embeddings unprunable (following test_qbert.py) ---
    oto.mark_unprunable_by_param_names(["bert.embeddings.word_embeddings.weight"])
    log.info("[M3+] marked word_embeddings unprunable")

    # --- Load data ---
    log.info("[DATA] loading SQuAD")
    raw_val, train_ds, val_ds = load_squad(args, tokenizer)
    log.info(f"[DATA] train features={len(train_ds)}  val features={len(val_ds)}  raw val examples={len(raw_val)}")

    # --- Compute pruning schedule ---
    steps_per_epoch = (len(train_ds) + args.batch_size - 1) // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    # GETA schedule: start pruning at 10% of training, finish at 25%
    start_pruning_step = max(1, total_steps // 10)
    pruning_steps = max(1, total_steps // 4)
    log.info(f"[SCHED] steps_per_epoch={steps_per_epoch}  total_steps={total_steps}  "
             f"start_pruning={start_pruning_step}  pruning_steps={pruning_steps}")

    # --- M4: build GETA optimizer ---
    log.info("[M4] building GETA optimizer")
    optimizer = oto.geta(
        variant="adamw",
        lr=args.lr,
        lr_quant=args.lr_quant,
        target_group_sparsity=args.target_sparsity,
        start_pruning_step=start_pruning_step,
        pruning_steps=pruning_steps,
        pruning_periods=10,
        bit_reduction=args.bit_reduction,
        min_bit_wt=4,
        max_bit_wt=16,
    )
    log.info(f"[M4] GETA optimizer built, {len(optimizer.param_groups)} param groups")

    # --- Baseline eval (before training) ---
    log.info("[EVAL] baseline (before training)")
    em0, f10, _ = evaluate(model, raw_val, val_ds, args.batch_size)
    log.info(f"[EVAL] baseline EM={em0:.2f}  F1={f10:.2f}")

    # --- Training loop ---
    log.info("[TRAIN] starting")
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        # shuffle train indices
        indices = torch.randperm(len(train_ds)).tolist()

        for i in range(0, len(train_ds), args.batch_size):
            batch_idx = indices[i:i + args.batch_size]
            batch = train_ds[batch_idx]
            inputs = {k: v.to(DEVICE) for k, v in batch.items()}

            optimizer.zero_grad()
            out = model(**inputs)
            loss = out.loss
            if loss is None:
                raise RuntimeError(f"step {global_step}: model returned no loss")

            # NaN / Inf check
            if torch.isnan(loss) or torch.isinf(loss):
                log.error(f"[!] NaN/Inf loss at step {global_step}, stopping")
                break

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

            if global_step % 50 == 0:
                avg = epoch_loss / n_batches
                metrics = optimizer.compute_metrics()
                log.info(f"    step {global_step}  loss={avg:.4f}  "
                         f"grp_sparsity={metrics.group_sparsity:.3f}  "
                         f"norm_import={metrics.norm_important_groups:.2f}  "
                         f"norm_redund={metrics.norm_redundant_groups:.2f}")
        else:
            # no break = epoch completed normally
            avg_loss = epoch_loss / max(n_batches, 1)
            metrics = optimizer.compute_metrics()
            log.info(f"[EPOCH {epoch}/{args.epochs}] avg_loss={avg_loss:.4f}  "
                     f"grp_sparsity={metrics.group_sparsity:.3f}  "
                     f"n_import={metrics.num_important_groups}  n_redund={metrics.num_redundant_groups}")

            # eval after each epoch
            em, f1, _ = evaluate(model, raw_val, val_ds, args.batch_size)
            log.info(f"[EVAL epoch {epoch}] EM={em:.2f}  F1={f1:.2f}")
            continue
        # break was hit (NaN)
        break

    # --- construct_subnet ---
    log.info("[SUBNET] constructing compressed model")
    try:
        oto.construct_subnet(
            export_huggingface_format=False,
            export_float16=False,
            out_dir=args.out_dir,
        )
        log.info(f"[SUBNET] full model  : {oto.full_group_sparse_model_path}")
        log.info(f"[SUBNET] compressed  : {oto.compressed_model_path}")

        # compute compression metrics
        full_bops = oto.compute_bops(in_million=True)
        full_params = oto.compute_num_params(in_million=True)
        log.info(f"[METRICS] full BOPs={full_bops['total']:.1f}M  params={full_params:.2f}M")

        # eval compressed model
        compressed_model = torch.load(oto.compressed_model_path, map_location=DEVICE)
        if hasattr(compressed_model, 'eval'):
            em_c, f1_c, preds = evaluate(compressed_model, raw_val, val_ds, args.batch_size)
            log.info(f"[EVAL compressed] EM={em_c:.2f}  F1={f1_c:.2f}")

            # save predictions
            pred_path = os.path.join(args.out_dir, "predictions.json")
            with open(pred_path, "w", encoding="utf-8") as f:
                json.dump(preds, f, ensure_ascii=False, indent=2)
            log.info(f"[SAVE] predictions -> {pred_path}")
    except Exception as e:
        log.error(f"[SUBNET] failed: {type(e).__name__}: {e}")
        log.error(traceback.format_exc()[-500:])

    # --- Save config ---
    cfg_path = os.path.join(args.out_dir, "config.json")
    with open(cfg_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    log.info(f"[SAVE] config -> {cfg_path}")

    log.info(f"[OK] Phase 4 complete. Log: {_LOG_PATH}")


if __name__ == "__main__":
    main()
