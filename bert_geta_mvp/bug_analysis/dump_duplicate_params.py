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

    # --- 用 model.named_parameters() 建 canonical name 對照表（單一真相）---
    canonical = {}  # id(p) -> canonical name from model.named_parameters()
    for name, p in model.named_parameters():
        canonical[id(p)] = name

    # --- 統計重複 ---
    owners = defaultdict(list)  # param_id -> [(group_id, p_name_in_that_group), ...]
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
    print(f"model.named_params : {len(canonical)}")
    print(f"duplicated params  : {len(dup_params)}  (占掉 {dup_slot_count} 個額外 slot)")
    print("=" * 78)

    # --- 用 canonical name 分類 ---
    QUANT_MARKERS = ("d_quant", "t_quant", "q_m", "clip_val", "quant_", "weight_scale", "act_scale")
    buckets = defaultdict(list)  # category -> [canonical_name, ...]
    unknown_ids = []  # 不在 named_parameters 裡的（理論上不該出現）

    for pid, info in dup_params.items():
        cname = canonical.get(pid)
        if cname is None:
            unknown_ids.append((pid, info[0][1]))
            continue
        is_quant = any(m in cname for m in QUANT_MARKERS)
        # 再依後綴粗分類
        if is_quant:
            cat = "quant_scalar"
        elif cname.endswith(".weight") and "LayerNorm" in cname:
            cat = "layernorm_weight"
        elif cname.endswith(".bias") and "LayerNorm" in cname:
            cat = "layernorm_bias"
        elif cname.endswith(".weight"):
            cat = "linear_weight"
        elif cname.endswith(".bias"):
            cat = "linear_bias"
        elif "embedding" in cname.lower():
            cat = "embedding"
        else:
            cat = "other"
        buckets[cat].append((cname, len(info)))

    print()
    print("--- 依 canonical name 分類 ---")
    for cat in sorted(buckets.keys()):
        print(f"  {cat:20s} : {len(buckets[cat])}")
    if unknown_ids:
        print(f"  NOT_IN_NAMED_PARAMS  : {len(unknown_ids)}  <-- 這些沒出現在 model.named_parameters() 裡")

    # --- 每類印前 5 個 canonical name ---
    print()
    print("--- 每類前 5 個樣本 ---")
    for cat in sorted(buckets.keys()):
        print(f"[{cat}]")
        for cname, ncopies in buckets[cat][:5]:
            print(f"    x{ncopies}  {cname}")

    if unknown_ids:
        print()
        print("--- 不在 named_parameters 的樣本（前 10）---")
        for pid, pname in unknown_ids[:10]:
            print(f"    id={pid}  p_name_in_group={pname}")

    # --- 步驟 1：看 Q/K/V.weight 的兩個 group id 誰大誰小 ---
    # dedup 是 sorted by group id 後先到先得，所以 id 較小的 group 會留下 param。
    print()
    print("--- Q/K/V weight 的 group ownership（決定 dedup 後留在哪）---")
    qkv_targets = []
    for pid, info in dup_params.items():
        cname = canonical.get(pid, "")
        if cname.endswith(".weight") and any(x in cname for x in (".query.", ".key.", ".value.")):
            qkv_targets.append((cname, info))
    # 排序讓同層 Q/K/V 靠在一起
    qkv_targets.sort(key=lambda x: x[0])
    for cname, info in qkv_targets[:9]:  # 前 3 層 × 3
        gids = sorted([gid for gid, _ in info])
        winner = min(gids)  # dedup 後留在 id 最小的 group
        print(f"    {cname}")
        print(f"        groups={gids}  -> kept in group {winner}")

    # 看看各 group 的大小，判斷 winner 是 head 群還是 trunk 群
    print()
    print("--- 各 group 的大小（params 數量，和 smoke log 對照）---")
    for gid, pg in raw_param_groups.items():
        print(f"    group[{gid}]: {len(pg['params'])} params")

    # --- 最後 verdict ---
    print()
    weight_like = sum(len(v) for k, v in buckets.items() if k != "quant_scalar")
    quant_like = len(buckets.get("quant_scalar", []))
    print("=" * 78)
    print(f"VERDICT: quant_scalar={quant_like}  non_quant={weight_like}  unknown={len(unknown_ids)}")
    if weight_like == 0 and len(unknown_ids) == 0:
        print(">>> 全部是量化 scalar → workaround 合理，Phase 4 可直接走 <<<")
    else:
        print("!!! 有非量化 param 被重複 → 需要檢視 dedup 是否丟了 prune mask 投票權 !!!")
    print("=" * 78)


if __name__ == "__main__":
    main()
