"""Quick diagnostic: print post-dedup param group sizes and first param names."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "geta"))
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from only_train_once.quantization.quant_model import model_to_quantize_model
from only_train_once.quantization.quant_layers import QuantizationMode
from only_train_once import OTO

tok = AutoTokenizer.from_pretrained("bert-base-uncased")
m = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased").cuda()
m = model_to_quantize_model(m, quant_mode=QuantizationMode.WEIGHT_AND_ACTIVATION).cuda()
enc = tok("What?", "Paris.", max_length=384, truncation="only_second", padding="max_length", return_tensors="pt")
dummy = tuple(v.cuda() for v in [enc["input_ids"], enc["attention_mask"], enc["token_type_ids"]])
m.eval()
oto = OTO(model=m, dummy_input=dummy)

pg = oto._graph.get_param_groups()

# figure out what pg actually is
print(f"type={type(pg).__name__}  len={len(pg)}")
groups = list(pg)  # works for list, dict_values, or dict (gives keys)
print(f"element type={type(groups[0]).__name__}")
print()

for i, g in enumerate(sorted(groups, key=lambda x: len(x.get("params", [])))):
    n_params = len(g.get("params", []))
    names = g.get("p_names", [])[:3]
    print(f"  group {i:2d}: {n_params:4d} params  first_names={names}")
