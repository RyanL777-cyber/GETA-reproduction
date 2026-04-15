"""
一次性診斷：列出 GETA 在 BERT 上把哪 32 個 param 歸到多個 param_group。

做法：本地重現 Graph.get_param_groups() 的前三個 pass（沒 dedup 那段），
然後列印跨 group 重複的 (param_id -> [(group_id, p_name), ...])。

不動 smoke.py、不動 graph.py。跑完直接看 stdout。
"""
import os
import subprocess
import sys
from collections import defaultdict

# --- 加入 GETA 源代碼路徑 ---
_GETA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "geta")
if os.path.isdir(_GETA_ROOT) and _GETA_ROOT not in sys.path:
    sys.path.insert(0, _GETA_ROOT)


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
    best_idx = cands[0][0]
    print(f"[gpu] selected GPU {best_idx}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(best_idx)


select_idle_gpu()

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 384


def main():
    print(f"[env] torch={torch.__version__} device={DEVICE}")

    # --- M1: load BERT ---
    from transformers import AutoTokenizer, AutoModelForQuestionAnswering
    import transformers
    transformers.logging.set_verbosity_error()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME).to(DEVICE)
    print("[M1] loaded BERT QA")

    # --- M2: wrap with quantized layers ---
    from only_train_once.quantization.quant_model import model_to_quantize_model
    from only_train_once.quantization.quant_layers import QuantizationMode
    model = model_to_quantize_model(model, quant_mode=QuantizationMode.WEIGHT_AND_ACTIVATION)
    model = model.to(DEVICE)  # 量化包裝會新增 CPU scalar，必須搬回 GPU
    print("[M2] wrapped with QuantizedLinear")

    # --- M3: dummy input + OTO ---
    enc = tokenizer(
        "What is the capital of France?",
        "Paris is the capital and most populous city of France.",
        max_length=MAX_LENGTH, truncation="only_second",
        padding="max_length", return_tensors="pt",
    )
    dummy_input = (
        enc["input_ids"].to(DEVICE),
        enc["attention_mask"].to(DEVICE),
        enc["token_type_ids"].to(DEVICE),
    )
    model.eval()
    from only_train_once import OTO
    oto = OTO(model=model, dummy_input=dummy_input)
    model.train()
    print("[M3] OTO graph built")

    # --- 本地重現 Graph.get_param_groups() 的前三 pass（不做 dedup）---
    graph = oto._graph
    raw_param_groups = dict()

    # Pass 1
    for node_group in graph.node_groups.values():
        if node_group.is_trainable and not node_group.is_auxiliary:
            ng_pg = node_group.get_param_groups()
            if len(ng_pg["params"]) > 0:
                raw_param_groups[node_group.id] = ng_pg

    # Pass 2: auxiliary
    for node_group in graph.node_groups.values():
        if hasattr(node_group, "auxilary_node_groups"):
            if node_group.id not in raw_param_groups:
                continue
            depend_pg = raw_param_groups[node_group.id]
            for aux_ng, offset in node_group.auxilary_node_groups:
                if aux_ng.is_auxiliary and aux_ng.is_trainable:
                    depend_pg["auxiliary_ngs"].append((aux_ng.id, offset))

    # Pass 3: drop untrainable
    untrainable = set()
    for pg in raw_param_groups.values():
        if len(pg["auxiliary_ngs"]) > 0:
            continue
        all_no_grad = True
        for p in pg["params"]:
            if p.requires_grad:
                all_no_grad = False
        if all_no_grad:
            untrainable.add(pg["id"])
    for rid in untrainable:
        del raw_param_groups[rid]

    raw_param_groups = dict(sorted(raw_param_groups.items(), key=lambda kv: kv[0]))

    # --- 統計重複 ---
    owners = defaultdict(list)  # param_id -> [(group_id, p_name), ...]
    total_slots = 0
    for group_id, pg in raw_param_groups.items():
        for name, p in zip(pg["p_names"], pg["params"]):
            owners[id(p)].append((group_id, name))
            total_slots += 1

    unique_params = len(owners)
    dup_params = {pid: info for pid, info in owners.items() if len(info) > 1}
    dup_slot_count = sum(len(info) - 1 for info in dup_params.values())

    print()
    print("=" * 78)
    print(f"total groups       : {len(raw_param_groups)}")
    print(f"total param slots  : {total_slots}")
    print(f"unique param ids   : {unique_params}")
    print(f"duplicated params  : {len(dup_params)}  (占掉 {dup_slot_count} 個額外 slot)")
    print("=" * 78)

    # --- 分類：量化 scalar vs 主 weight ---
    QUANT_MARKERS = ("d_quant", "t_quant", "q_m", "clip_val", "quant_")
    quant_dups = []
    weight_dups = []
    for pid, info in dup_params.items():
        name_sample = info[0][1]
        if any(m in name_sample for m in QUANT_MARKERS):
            quant_dups.append((pid, info))
        else:
            weight_dups.append((pid, info))

    print()
    print(f"[分類] 量化 scalar 重複 : {len(quant_dups)}")
    print(f"[分類] 主 weight 重複   : {len(weight_dups)}")
    print()

    print("--- 前 15 個重複參數明細 ---")
    for i, (pid, info) in enumerate(list(dup_params.items())[:15]):
        print(f"[{i}] id={pid}  出現次數={len(info)}")
        for gid, pname in info:
            print(f"     group={gid}  p_name={pname}")

    if weight_dups:
        print()
        print("!!! 主 weight 被重複（需要重新設計 dedup 策略）!!!")
        for pid, info in weight_dups[:10]:
            print(f"  {info}")
    else:
        print()
        print(">>> 所有重複都是量化 scalar → workaround 完全合理，Phase 4 可直接走 <<<")


if __name__ == "__main__":
    main()
