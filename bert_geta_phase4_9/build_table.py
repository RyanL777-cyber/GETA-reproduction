"""
Build paper Table 3 from Phase 5 results.

Loads each sp*/result.json, computes a baseline BOPs from a fresh
uncompressed BERT-base QA (with the same model_to_quantize_model + OTO
pipeline used in phase 5), then prints a formatted table with columns:

    Target | Actual Sparsity | EM(%) | F1(%) | BOPs(G) | Rel.BOPs(%)

Run AFTER phase 5 finishes. Does NOT affect training.

Usage:
    python3 build_table.py
    python3 build_table.py --results_dir ./results
    python3 build_table.py --baseline_bops_m 12345.6   # override baseline
"""
import argparse
import glob
import json
import os
import subprocess
import sys


_GETA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "geta")
if os.path.isdir(_GETA_ROOT) and _GETA_ROOT not in sys.path:
    sys.path.insert(0, _GETA_ROOT)


def select_idle_gpu(max_used_mem_mb=2000, max_util=10):
    if "CUDA_VISIBLE_DEVICES" in os.environ:
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
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cands[0][0])


select_idle_gpu()

import torch

MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 384
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def compute_baseline_bops_m():
    """
    Baseline = uncompressed BERT-base QA through the same pipeline
    (model_to_quantize_model + OTO) but with NO training, NO pruning,
    NO bit reduction. BOPs reflects the initial bit-width (max_bit_wt).
    """
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer
    from only_train_once.quantization.quant_model import model_to_quantize_model
    from only_train_once.quantization.quant_layers import QuantizationMode
    from only_train_once import OTO

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME).to(DEVICE)
    model = model_to_quantize_model(model, quant_mode=QuantizationMode.WEIGHT_AND_ACTIVATION)
    model = model.to(DEVICE).eval()

    enc = tokenizer("What?", "Paris.", max_length=MAX_LENGTH,
                    truncation="only_second", padding="max_length", return_tensors="pt")
    dummy = tuple(v.to(DEVICE) for v in [enc["input_ids"], enc["attention_mask"], enc["token_type_ids"]])

    oto = OTO(model=model, dummy_input=dummy)
    bops = oto.compute_bops(in_million=True)
    return float(bops["total"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    parser.add_argument("--baseline_bops_m", type=float, default=None,
                        help="skip baseline computation, use this value (in M)")
    parser.add_argument("--out", type=str, default=None,
                        help="also write table to this file (markdown)")
    args = parser.parse_args()

    if args.baseline_bops_m is not None:
        baseline_m = args.baseline_bops_m
        print(f"[baseline] using user-provided value: {baseline_m:.1f} M")
    else:
        print("[baseline] computing uncompressed BERT-base QA BOPs...")
        baseline_m = compute_baseline_bops_m()
        print(f"[baseline] BOPs = {baseline_m:.1f} M = {baseline_m/1000:.2f} G\n")

    rows = []
    for path in sorted(glob.glob(os.path.join(args.results_dir, "sp*", "result.json"))):
        with open(path) as f:
            r = json.load(f)
        target = r.get("sparsity", -1)
        actual = r.get("actual_group_sparsity", target)
        em = r.get("compressed_em", -1)
        f1 = r.get("compressed_f1", -1)
        bops_m = r.get("bops_million", -1)
        bops_g = bops_m / 1000 if bops_m >= 0 else -1
        rel_pct = 100.0 * bops_m / baseline_m if (bops_m >= 0 and baseline_m > 0) else -1
        rows.append((target, actual, em, f1, bops_g, rel_pct))

    if not rows:
        print(f"[!] no result.json found under {args.results_dir}/sp*/")
        return

    header = f"{'Target':>8}  {'Actual':>8}  {'EM(%)':>7}  {'F1(%)':>7}  {'BOPs(G)':>9}  {'Rel.BOPs(%)':>12}"
    sep = "-" * len(header)
    lines = [header, sep]
    for tgt, act, em, f1, bops_g, rel in rows:
        lines.append(f"{tgt*100:>7.0f}%  {act:>8.4f}  {em:>7.2f}  {f1:>7.2f}  {bops_g:>9.2f}  {rel:>11.2f}%")

    print("\n".join(lines))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(f"Baseline BOPs: {baseline_m:.1f} M ({baseline_m/1000:.2f} G)\n\n")
            f.write("| Target | Actual | EM(%) | F1(%) | BOPs(G) | Rel.BOPs(%) |\n")
            f.write("|-------:|-------:|------:|------:|--------:|------------:|\n")
            for tgt, act, em, f1, bops_g, rel in rows:
                f.write(f"| {tgt*100:.0f}% | {act:.4f} | {em:.2f} | {f1:.2f} | {bops_g:.2f} | {rel:.2f}% |\n")
        print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
