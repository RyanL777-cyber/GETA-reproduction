"""Diagnose construct_subnet shape mismatch in compressed model.
Error was: mat1 and mat2 shapes cannot be multiplied (3072x943 and 3072x768)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "geta"))
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "phase4_out")

# --- 1. List .pt files ---
print("=" * 70)
print("1. Files in phase4_out")
print("=" * 70)
for f in sorted(os.listdir(OUT_DIR)):
    path = os.path.join(OUT_DIR, f)
    size_mb = os.path.getsize(path) / (1024 ** 2)
    print(f"  {f:50s}  {size_mb:.1f} MB")

# --- 2. Load both models ---
print("\n" + "=" * 70)
print("2. Loading models")
print("=" * 70)
full_path = os.path.join(OUT_DIR, "BertForQuestionAnswering_full_group_sparse.pt")
comp_path = os.path.join(OUT_DIR, "BertForQuestionAnswering_compressed.pt")

full_model = torch.load(full_path, map_location="cpu")
comp_model = torch.load(comp_path, map_location="cpu")
print(f"  full model type: {type(full_model).__name__}")
print(f"  comp model type: {type(comp_model).__name__}")

# --- 3. Compare parameter shapes ---
print("\n" + "=" * 70)
print("3. Shape comparison (only showing mismatches and BERT encoder layers)")
print("=" * 70)

full_params = dict(full_model.named_parameters()) if hasattr(full_model, 'named_parameters') else {}
comp_params = dict(comp_model.named_parameters()) if hasattr(comp_model, 'named_parameters') else {}

# If it's a state_dict instead of a model
if not full_params and isinstance(full_model, dict):
    full_params = full_model
    print("  (full model is state_dict)")
if not comp_params and isinstance(comp_model, dict):
    comp_params = comp_model
    print("  (comp model is state_dict)")

print(f"\n  full params: {len(full_params)}")
print(f"  comp params: {len(comp_params)}")

# find params only in full, only in comp, and shape mismatches
only_full = set(full_params.keys()) - set(comp_params.keys())
only_comp = set(comp_params.keys()) - set(full_params.keys())
common = set(full_params.keys()) & set(comp_params.keys())

if only_full:
    print(f"\n  --- params ONLY in full model ({len(only_full)}) ---")
    for n in sorted(only_full)[:20]:
        p = full_params[n]
        s = p.shape if hasattr(p, 'shape') else '?'
        print(f"    {n}: {s}")
    if len(only_full) > 20:
        print(f"    ... and {len(only_full) - 20} more")

if only_comp:
    print(f"\n  --- params ONLY in comp model ({len(only_comp)}) ---")
    for n in sorted(only_comp)[:20]:
        p = comp_params[n]
        s = p.shape if hasattr(p, 'shape') else '?'
        print(f"    {n}: {s}")
    if len(only_comp) > 20:
        print(f"    ... and {len(only_comp) - 20} more")

print(f"\n  --- shape MISMATCHES ({len(common)} common params) ---")
mismatch_count = 0
for n in sorted(common):
    fp = full_params[n]
    cp = comp_params[n]
    fs = fp.shape if hasattr(fp, 'shape') else None
    cs = cp.shape if hasattr(cp, 'shape') else None
    if fs != cs:
        mismatch_count += 1
        print(f"    {n}")
        print(f"      full: {fs}  comp: {cs}")
if mismatch_count == 0:
    print("    (none)")
else:
    print(f"\n  total mismatches: {mismatch_count}")

# --- 4. Focus on FFN layers (error site: 3072x943 vs 3072x768) ---
print("\n" + "=" * 70)
print("4. FFN layer shapes in compressed model (intermediate.dense + output.dense)")
print("=" * 70)
for layer_i in range(12):
    prefix = f"bert.encoder.layer.{layer_i}"
    inter_w = f"{prefix}.intermediate.dense.weight"
    out_w = f"{prefix}.output.dense.weight"
    if inter_w in comp_params and out_w in comp_params:
        iw = comp_params[inter_w]
        ow = comp_params[out_w]
        is_ = iw.shape if hasattr(iw, 'shape') else '?'
        os_ = ow.shape if hasattr(ow, 'shape') else '?'
        # check consistency: intermediate output dim should == output input dim
        ok = "OK" if (hasattr(iw, 'shape') and hasattr(ow, 'shape') and iw.shape[0] == ow.shape[1]) else "MISMATCH!"
        print(f"  layer {layer_i:2d}: intermediate.w={is_}  output.w={os_}  {ok}")

# --- 5. Try forward pass on compressed model ---
print("\n" + "=" * 70)
print("5. Forward pass test on compressed model")
print("=" * 70)
if hasattr(comp_model, 'eval'):
    comp_model.eval()
    try:
        dummy = torch.zeros(1, 16, dtype=torch.long)  # short sequence
        with torch.no_grad():
            out = comp_model(input_ids=dummy)
        print("  PASSED")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        # try to find which layer
        import traceback
        tb = traceback.format_exc()
        # extract the last few relevant lines
        for line in tb.splitlines():
            if 'layer' in line.lower() or 'dense' in line.lower() or 'linear' in line.lower():
                print(f"    {line.strip()}")
else:
    print("  (comp_model has no eval(), skipping)")
