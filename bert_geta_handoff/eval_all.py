"""Evaluate every checkpoint in checkpoints/ on SQuAD v1.1 dev.

Prints a table: file | EM | F1 | params (M).

Run from this folder:
    python eval_all.py
"""
import collections
import glob
import json
import os
import re
import string
import sys

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

_HERE = os.path.dirname(os.path.abspath(__file__))
_GETA_ROOT = os.path.join(os.path.dirname(_HERE), "geta")
if not os.path.isdir(os.path.join(_GETA_ROOT, "only_train_once")):
    raise RuntimeError(f"only_train_once not found under {_GETA_ROOT}")
sys.path.insert(0, _GETA_ROOT)
import only_train_once  # noqa: F401

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 384
DOC_STRIDE = 128
N_BEST = 20
MAX_ANSWER_LEN = 30


def normalize(s):
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    s = " ".join(s.split())
    return s


def f1(pred, golds):
    if not golds:
        return 0.0
    best = 0.0
    for g in golds:
        p_t, g_t = normalize(pred).split(), normalize(g).split()
        if not p_t or not g_t:
            best = max(best, float(p_t == g_t))
            continue
        common = collections.Counter(p_t) & collections.Counter(g_t)
        n = sum(common.values())
        if n == 0:
            continue
        prec = n / len(p_t)
        rec = n / len(g_t)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


def em(pred, golds):
    return float(any(normalize(pred) == normalize(g) for g in golds))


def prepare_val_features(examples, tokenizer):
    inputs = tokenizer(
        [q.strip() for q in examples["question"]], examples["context"],
        max_length=MAX_LENGTH, truncation="only_second",
        stride=DOC_STRIDE, return_overflowing_tokens=True,
        return_offsets_mapping=True, padding="max_length",
    )
    sample_map = inputs.pop("overflow_to_sample_mapping")
    inputs["example_id"] = [examples["id"][s] for s in sample_map]
    for i in range(len(inputs["offset_mapping"])):
        seq = inputs.sequence_ids(i)
        inputs["offset_mapping"][i] = [
            o if seq[k] == 1 else None for k, o in enumerate(inputs["offset_mapping"][i])
        ]
    return inputs


def evaluate(model, raw_val, val_ds, tokenizer, batch_size=32):
    model.eval()
    all_starts, all_ends = [], []
    keep_cols = ["input_ids", "attention_mask", "token_type_ids"]
    val_for_model = val_ds.remove_columns([c for c in val_ds.column_names if c not in keep_cols])
    val_for_model.set_format("torch")
    with torch.no_grad():
        for i in range(0, len(val_for_model), batch_size):
            batch = val_for_model[i:i + batch_size]
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            out = model(**batch)
            all_starts.append(out.start_logits.cpu().numpy())
            all_ends.append(out.end_logits.cpu().numpy())
    import numpy as np
    starts = np.concatenate(all_starts, 0)
    ends = np.concatenate(all_ends, 0)

    # Map example_id -> list of feature indices
    feat_per_ex = collections.defaultdict(list)
    for i, ex_id in enumerate(val_ds["example_id"]):
        feat_per_ex[ex_id].append(i)

    preds = {}
    for ex in raw_val:
        ex_id = ex["id"]
        context = ex["context"]
        candidates = []
        for fi in feat_per_ex[ex_id]:
            offsets = val_ds[fi]["offset_mapping"]
            s_logits, e_logits = starts[fi], ends[fi]
            top_s = np.argsort(s_logits)[-N_BEST:][::-1]
            top_e = np.argsort(e_logits)[-N_BEST:][::-1]
            for si in top_s:
                for ei in top_e:
                    if si >= len(offsets) or ei >= len(offsets):
                        continue
                    if offsets[si] is None or offsets[ei] is None:
                        continue
                    if ei < si or ei - si + 1 > MAX_ANSWER_LEN:
                        continue
                    candidates.append({
                        "score": float(s_logits[si] + e_logits[ei]),
                        "text": context[offsets[si][0]:offsets[ei][1]],
                    })
        if candidates:
            preds[ex_id] = max(candidates, key=lambda x: x["score"])["text"]
        else:
            preds[ex_id] = ""

    em_total, f1_total, n = 0.0, 0.0, 0
    for ex in raw_val:
        golds = ex["answers"]["text"] or [""]
        em_total += em(preds[ex["id"]], golds)
        f1_total += f1(preds[ex["id"]], golds)
        n += 1
    return 100 * em_total / n, 100 * f1_total / n


def main():
    print(f"[env] device={DEVICE}")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    raw_val = load_dataset("squad", split="validation")
    val_ds = raw_val.map(
        lambda ex: prepare_val_features(ex, tokenizer),
        batched=True, remove_columns=raw_val.column_names,
    )

    rows = []
    for ckpt in sorted(glob.glob(os.path.join(_HERE, "checkpoints", "*.pt"))):
        name = os.path.basename(ckpt)
        print(f"\n[eval] {name}")
        model = torch.load(ckpt, map_location=DEVICE, weights_only=False)
        model.to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        em_score, f1_score = evaluate(model, raw_val, val_ds, tokenizer)
        rows.append((name, em_score, f1_score, n_params))
        print(f"  EM={em_score:.2f}  F1={f1_score:.2f}  params={n_params:.2f}M")
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    print("\n=== summary ===")
    print(f"{'file':<40} {'EM':>7} {'F1':>7} {'params(M)':>10}")
    for name, e, f, p in rows:
        print(f"{name:<40} {e:>7.2f} {f:>7.2f} {p:>10.2f}")

    with open(os.path.join(_HERE, "eval_summary.json"), "w") as fout:
        json.dump([{"file": n, "em": e, "f1": f, "params_million": p} for n, e, f, p in rows],
                  fout, indent=2)
    print(f"\nWrote {os.path.join(_HERE, 'eval_summary.json')}")


if __name__ == "__main__":
    main()
