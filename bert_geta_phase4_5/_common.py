"""
Shared utilities for Phase 4.5 verification experiments.

Copied from bert_geta_phase5/run_experiment.py so that exp_a / exp_b use the
exact same data pipeline and eval code as the failing Phase 5 run. The ONLY
thing each exp script varies is the optimizer choice.
"""
import collections
import logging
import os
import re
import subprocess
import sys

# =========================================================================
# GPU auto-select  (must run BEFORE `import torch`)
# =========================================================================
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


# =========================================================================
# Constants (match phase5)
# =========================================================================
MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 384
DOC_STRIDE = 128


# =========================================================================
# Logger
# =========================================================================
def setup_logger(log_path, name):
    log = logging.getLogger(name)
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
# Data (identical to phase5)
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
    import datasets as hf_datasets
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
# Eval (identical to phase5)
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
    import numpy as np
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


def evaluate(model, raw_val, val_ds, batch_size, device):
    import numpy as np
    import torch
    model.eval()
    fwd_columns = ["input_ids", "attention_mask", "token_type_ids"]
    all_start, all_end = [], []
    with torch.no_grad():
        for i in range(0, len(val_ds), batch_size):
            batch = val_ds[i:i + batch_size]
            inputs = {k: torch.tensor(batch[k]).to(device) for k in fwd_columns if k in batch}
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
# Quant wrap (same call as phase5)
# =========================================================================
def wrap_quant(model, device):
    """Apply GETA's weight+activation quantization wrap. Same as phase5 M2."""
    from only_train_once.quantization.quant_model import model_to_quantize_model
    from only_train_once.quantization.quant_layers import QuantizationMode
    model = model_to_quantize_model(model, quant_mode=QuantizationMode.WEIGHT_AND_ACTIVATION)
    return model.to(device)
