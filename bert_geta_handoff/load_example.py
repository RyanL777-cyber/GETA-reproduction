"""Load a GETA-compressed BERT QA checkpoint and run one inference.

Usage:
    python load_example.py checkpoints/compressed_sp50.pt
    python load_example.py checkpoints/compressed_sp70.pt
"""
import argparse
import os
import sys

import torch
from transformers import AutoTokenizer

# GETA / only_train_once must be importable so torch.load can resolve the
# pickled quant-layer classes. The geta/ folder lives at the project root,
# one level above this folder.
_HERE = os.path.dirname(os.path.abspath(__file__))
_GETA_ROOT = os.path.join(os.path.dirname(_HERE), "geta")
if not os.path.isdir(os.path.join(_GETA_ROOT, "only_train_once")):
    raise RuntimeError(f"only_train_once not found under {_GETA_ROOT}")
sys.path.insert(0, _GETA_ROOT)
import only_train_once  # noqa: F401  (registers classes for unpickling)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", help="path to *_compressed.pt")
    p.add_argument("--question", default="Where is the Eiffel Tower?")
    p.add_argument("--context",
                   default="The Eiffel Tower is a wrought-iron lattice tower in Paris, France.")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # OTO saves the entire nn.Module via torch.save, not just a state_dict.
    # weights_only=False is required (PyTorch >= 2.6 changed the default).
    model = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.eval().to(device)

    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    enc = tok(args.question, args.context,
              max_length=384, truncation="only_second",
              padding="max_length", return_tensors="pt").to(device)

    with torch.no_grad():
        out = model(**enc)
    start = out.start_logits.argmax().item()
    end = out.end_logits.argmax().item()
    if end < start:
        end = start
    answer = tok.decode(enc.input_ids[0, start:end + 1], skip_special_tokens=True)

    print(f"Q: {args.question}")
    print(f"A: {answer}")


if __name__ == "__main__":
    main()
