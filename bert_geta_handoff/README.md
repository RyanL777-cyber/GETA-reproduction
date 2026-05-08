# bert_geta_handoff

GETA-compressed BERT-base SQuAD checkpoints, packaged for handoff.

## Contents

| File | Sparsity | Size | F1 | EM | Params |
|---|---|---|---|---|---|
| `checkpoints/compressed_sp50.pt` | 50% | 308M | 84.02 | 74.45 | 80.5M |
| `checkpoints/compressed_sp70.pt` | 70% | 264M | 83.61 | 73.72 | 69.1M |

Numbers are SQuAD v1.1 dev set, evaluated on the compressed model after
`construct_subnet`. Source run: `bert_geta_phase5/results/combined16_sp{50,70}/`
(16 epochs, bs=16, lr=6e-5, br=1, 6 pruning periods, paper-aligned schedule).
Full per-epoch log lives in `summary.json`.

## How to load

```bash
python load_example.py checkpoints/compressed_sp50.pt
```

See `load_example.py` for the minimal load + inference flow.

Two things to know:

1. OTO saves the **entire `nn.Module`** with `torch.save`, not a state dict.
   So `torch.load(...)` returns a ready-to-use model — no
   `BertForQuestionAnswering.from_pretrained()` needed.
2. Unpickling requires `only_train_once` on `sys.path` so the quant-layer
   classes can be resolved. `load_example.py` adds the sibling `geta/`
   folder automatically.
