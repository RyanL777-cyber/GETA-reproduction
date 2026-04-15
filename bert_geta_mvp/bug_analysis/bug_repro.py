#!/usr/bin/env python3
"""Minimal reproduction script for the BERT + GETA param-group bug."""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GETA_ROOT = os.path.join(SCRIPT_DIR, "geta")
if GETA_ROOT not in sys.path:
    sys.path.insert(0, GETA_ROOT)

import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from only_train_once import OTO
from only_train_once.quantization.quant_model import model_to_quantize_model
from only_train_once.quantization.quant_layers import QuantizationMode


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "bert-base-uncased"
    print(f"device={device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    model = model.to(device)

    model = model_to_quantize_model(model, quant_mode=QuantizationMode.WEIGHT_AND_ACTIVATION)
    model = model.to(device)

    sample_q = "What is the capital of France?"
    sample_c = "Paris is the capital and most populous city of France."
    enc = tokenizer(
        sample_q,
        sample_c,
        max_length=384,
        truncation="only_second",
        padding="max_length",
        return_tensors="pt",
    )
    dummy_input = (
        enc["input_ids"].to(device),
        enc["attention_mask"].to(device),
        enc["token_type_ids"].to(device),
    )

    oto = OTO(model=model, dummy_input=dummy_input)
    groups = list(oto._graph.get_param_groups())
    print(f"Graph produced {len(groups)} param_groups")

    dup_map = {}
    for group_idx, group in enumerate(groups):
        for name, param in zip(group["p_names"], group["params"]):
            dup_map.setdefault(id(param), []).append((group_idx, name))

    duplicates = {pid: entries for pid, entries in dup_map.items() if len(entries) > 1}
    if duplicates:
        print(f"Found {len(duplicates)} duplicated parameters across param_groups")
        for pid, entries in list(duplicates.items())[:10]:
            print(f"  param_id={pid}: {entries}")
    else:
        print("No duplicated parameters found in graph param_groups")

    try:
        optimizer = oto.geta(
            lr=3e-5,
            lr_quant=3e-5,
            target_group_sparsity=0.5,
            bit_reduction=2,
            min_bit_wt=4,
            max_bit_wt=16,
        )
        print("OTO.geta() created optimizer:", type(optimizer).__name__)
    except Exception as exc:
        print("OTO.geta() failed:", type(exc).__name__, exc)
        raise


if __name__ == "__main__":
    main()
