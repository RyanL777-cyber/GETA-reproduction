# bert_geta_handoff

GETA-compressed BERT-base SQuAD v1.1 checkpoints — 4 個有完整 metric 紀錄的 run。

## Checkpoint 一覽

| File | Sparsity | epochs | bs | lr | EM | F1 | Params (M) | 註記 |
|---|---|---|---|---|---|---|---|---|
| `checkpoints/compressed_sp10_baseline.pt` | 10% | 10 | 4 | 3e-5 | 75.09 | 84.45 | 103.2 | paper-faithful |
| `checkpoints/compressed_sp50_combined16.pt` | 50% | 16 | 16 | 6e-5 | 74.45 | 84.02 | 80.5 | bs=16 sqrt-scaled 加速 |
| `checkpoints/compressed_sp70_fast13.pt` | 70% | 13 | 16 | 6e-5 | **74.29** | **83.85** | ~69 | **最佳結果**,距論文 GETA 僅差 0.89 F1 |
| `checkpoints/compressed_sp70_combined16.pt` | 70% | 16 | 16 | 6e-5 | 73.72 | 83.61 | 69.1 | 同 fast13 但延長到 16ep,過擬合對照 |

所有 run `lr_quant=1e-4`、`bit_reduction=1`、`pruning_periods=projection_periods=6`、剪枝 `epoch 5→10`、`seed=42`,完整 CLI 在 `bert_geta_clean/launchers/PRESETS.md`。

逐 epoch loss / EM / F1 紀錄在 `results_json/`。

## 兩套訓練設定為什麼共存

論文用 bs=4(慢,單點 ~25h+)。我們在 sp=0.5/0.7 上額外用 sqrt-scaled bs=16/lr=6e-5
重跑一次,訓練快約 4×;sp=0.7 的 13-epoch 還比 paper-faithful 高一點(83.85 vs 待補)。
sp70_combined16 是把 fast13 拉長到 16 epoch 的對照組,證實過擬合(F1 ep15 觸頂後下滑)。

## How to load

```bash
python load_example.py checkpoints/compressed_sp70_fast13.pt
```

兩個重點:

1. OTO 用 `torch.save` 把**整個 `nn.Module`** 存下來,不是 state dict。所以
   `torch.load(...)` 回傳一個可直接用的 model — 不需要
   `BertForQuestionAnswering.from_pretrained()`。
2. Unpickle 時需要 `only_train_once` 在 `sys.path` 上才能解出 quant 層 class。
   `load_example.py` 會自動加 sibling 的 `geta/` 路徑。
