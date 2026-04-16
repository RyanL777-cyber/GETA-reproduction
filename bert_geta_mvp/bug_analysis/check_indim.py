"""Diagnose why output.dense in-dim is not pruned by construct_subnet.
Traces the second pass logic for output.dense nodes to find the failure point.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "geta"))
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import subprocess
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    try:
        proc = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                               "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
        cands = []
        for line in proc.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 3:
                cands.append((int(parts[0]), int(parts[1]), int(parts[2])))
        if cands:
            cands.sort(key=lambda x: (x[2], x[1]))
            os.environ["CUDA_VISIBLE_DEVICES"] = str(cands[0][0])
            print(f"[gpu] selected GPU {cands[0][0]}")
    except Exception:
        pass

import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from only_train_once.quantization.quant_model import model_to_quantize_model
from only_train_once.quantization.quant_layers import QuantizationMode
from only_train_once import OTO

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Build OTO (same as smoke.py) ---
print("=== Building model + OTO ===")
tok = AutoTokenizer.from_pretrained("bert-base-uncased")
m = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased").to(DEVICE)
m = model_to_quantize_model(m, quant_mode=QuantizationMode.WEIGHT_AND_ACTIVATION).to(DEVICE)
enc = tok("What?", "Paris.", max_length=384, truncation="only_second", padding="max_length", return_tensors="pt")
dummy = tuple(v.to(DEVICE) for v in [enc["input_ids"], enc["attention_mask"], enc["token_type_ids"]])
m.eval()
oto = OTO(model=m, dummy_input=dummy)
graph = oto._graph

print(f"node_groups: {len(graph.node_groups)}")

# --- 1. Find output.dense nodes in the graph ---
print("\n=== 1. Searching for output.dense nodes ===")
output_dense_nodes = []
for ng_id, ng in graph.node_groups.items():
    for node_id, node in ng.nodes.items():
        # check param names for output.dense
        for pn in node.param_names:
            if "output.dense.weight" in pn and "attention" not in pn:
                output_dense_nodes.append((ng_id, node_id, node, ng))
                break

print(f"Found {len(output_dense_nodes)} output.dense nodes")
for ng_id, node_id, node, ng in output_dense_nodes[:3]:
    print(f"  node={node_id}  ng={ng_id[:60]}...")
    print(f"    param_names={node.param_names}")
    print(f"    node_group_ids={node.node_group_ids}")
    print(f"    is_stem={node.is_stem()}")
    print(f"    has prune_in_dim={hasattr(node.op, 'prune_in_dim')}")
    print(f"    op type={type(node.op).__name__}")
    print(f"    module type={type(node.op.module).__name__}")

# --- 2. Find intermediate.dense nodes ---
print("\n=== 2. Searching for intermediate.dense nodes ===")
inter_dense_nodes = []
for ng_id, ng in graph.node_groups.items():
    for node_id, node in ng.nodes.items():
        for pn in node.param_names:
            if "intermediate.dense.weight" in pn:
                inter_dense_nodes.append((ng_id, node_id, node, ng))
                break

print(f"Found {len(inter_dense_nodes)} intermediate.dense nodes")
for ng_id, node_id, node, ng in inter_dense_nodes[:3]:
    print(f"  node={node_id}  ng={ng_id[:60]}...")
    print(f"    node_group_ids={node.node_group_ids}")
    print(f"    is_stem={node.is_stem()}")
    n_params = len(ng.param_names)
    print(f"    ng has {n_params} param_names, is_prunable={ng.is_prunable}")

# --- 3. Trace backward from first output.dense node ---
print("\n=== 3. Backward trace from output.dense (layer 0) ===")
if output_dense_nodes:
    ng_id, node_id, node, ng = output_dense_nodes[0]
    print(f"Starting from node={node_id}")

    # walk backward up to 5 hops
    current = node
    for hop in range(5):
        incoming = graph.incoming(current)
        print(f"  hop {hop}: {current.id} (op={current.op_name}, groups={current.node_group_ids})")
        if not incoming:
            print(f"    -> no incoming nodes")
            break
        for inc in incoming:
            is_stem = inc.is_stem() if hasattr(inc, 'is_stem') else '?'
            print(f"    <- {inc.id} (op={inc.op_name}, is_stem={is_stem}, groups={inc.node_group_ids})")
        current = incoming[0]

# --- 4. Check if output.dense and intermediate.dense share a node_group ---
print("\n=== 4. Node group overlap check ===")
if output_dense_nodes and inter_dense_nodes:
    out_ng_ids = set()
    for ng_id, _, _, _ in output_dense_nodes:
        out_ng_ids.add(ng_id)
    inter_ng_ids = set()
    for ng_id, _, _, _ in inter_dense_nodes:
        inter_ng_ids.add(ng_id)
    overlap = out_ng_ids & inter_ng_ids
    print(f"  output.dense node_groups: {len(out_ng_ids)}")
    print(f"  intermediate.dense node_groups: {len(inter_ng_ids)}")
    print(f"  overlap: {len(overlap)}")
    if overlap:
        print(f"  SHARED group IDs: {[g[:50]+'...' for g in overlap]}")
        print("  -> output.dense and intermediate.dense are in the SAME group!")
        print("     This means the backward trace won't cross a group boundary.")

# --- 5. Check pruning_redundant_idxes for intermediate groups ---
print("\n=== 5. pruning_redundant_idxes for relevant groups ===")
# simulate set_pruning_redundant_idxes
graph.set_pruning_redundant_idxes()
for ng_id, _, _, ng in inter_dense_nodes[:3]:
    ri = ng.pruning_redundant_idxes
    n_ri = len(ri) if ri is not None else None
    print(f"  intermediate ng={ng_id[:50]}...  redundant_idxes={n_ri}  is_prunable={ng.is_prunable}")

for ng_id in out_ng_ids:
    ng = graph.node_groups[ng_id]
    ri = ng.pruning_redundant_idxes
    n_ri = len(ri) if ri is not None else None
    n_params = len(ng.param_names)
    print(f"  output.dense ng={ng_id[:50]}...  redundant_idxes={n_ri}  is_prunable={ng.is_prunable}  params={n_params}")
