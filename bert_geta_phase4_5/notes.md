# Phase 4.5 — 補驗證

Phase 4 的驗收標準只測「不當機」，沒測訓練本身能不能收斂。Phase 5 全面崩潰後，需要補兩個對照組，定位問題到底在哪一層。

## 為什麼要補

| 階段 | 做了什麼 | 結果 |
|---|---|---|
| `bert_baseline` (Phase 1) | 純 BERT + HF Trainer, full SQuAD, 2 epochs, bs=12 | ✅ EM=81.14 / F1=88.50 |
| `bert_geta_mvp` Phase 4 | 2000 samples, 1 epoch, bs=8, target_sp=0.5 | ❌ loss 5.93→5.93, F1=4.35 |
| `bert_geta_phase5` | full SQuAD, 10 epochs, bs=4, sp=0.1/0.3/0.5/0.7 | ❌ 全部 F1 < 5 |

關鍵觀察：Phase 5 的 epoch 1-3（`grp_sparsity=0`，剪枝還沒啟動）F1 就只到 48.96 且一路下跌。代表問題不在 pruning，在 **dense training 階段**。

純 BERT baseline 跟 Phase 5 差三件事：
1. Quant wrap（`model_to_quantize_model`）
2. GETA optimizer（取代 HF Trainer 的 AdamW + linear warmup）
3. batch size（baseline=12, phase5=4）

## 兩個對照組

| Exp | Quant wrap | GETA optimizer | 預期 F1 | 若失敗則 |
|---|---|---|---|---|
| **A** | ✅ | ❌（純 AdamW + warmup） | ≥ 80 | 問題在 quant wrap |
| **B** | ✅ | ✅（但 `target_sparsity=0`） | ≥ 80 | 問題在 GETA optimizer dense 部分（非 pruning） |

如果 A、B 都過，Phase 5 失敗就確認在 **pruning path**（`target_sparsity > 0` 觸發的 group zero-out 動態），再回頭查 pruning schedule / 瀕死神經元現象。

## 驗收門檻

F1 ≥ 80（bert_baseline 是 88.50，量化理論上略降，留 8.5 分空間）。

## 檔案

- `_common.py` — data pipeline + eval，直接從 `bert_geta_phase5/run_experiment.py` 抽出，確保兩個 exp 跟 phase5 走同一條資料路徑。
- `exp_a_quantwrap_adamw.py` — Quant wrap + `torch.optim.AdamW` + `get_linear_schedule_with_warmup`。預設 bs=12 對齊 bert_baseline。
- `exp_b_geta_sp0.py` — Quant wrap + `oto.geta(..., target_group_sparsity=0.0)`。預設 bs=4 對齊 phase5。

兩個 exp 都預設 **2 epochs full SQuAD**，為了快速拿到訊號（約 2h/run）。拿到結果再決定要不要延長。

## 指令

在 server 上（已 clone 好 repo）：

```bash
cd ~/GETA\ reprocution/GETA-reproduction/bert_geta_phase4_5

# Exp A — 先跑，因為最關鍵（決定 quant wrap 是否乾淨）
python3 exp_a_quantwrap_adamw.py

# Exp B — A 過了再跑
python3 exp_b_geta_sp0.py
```

兩個跑完後看 `results_exp_a/result.json`、`results_exp_b/result.json` 的 `best_f1` 跟 `pass` 欄位。

## 解讀表

| Exp A | Exp B | 結論 | 下一步 |
|---|---|---|---|
| ❌ | — | Quant wrap 壞 | 查 `only_train_once/quantization/quant_layers.py`，比對 quant-wrap 前後 forward 數值 |
| ✅ | ❌ | GETA optimizer dense 部分壞 | 查 `oto.geta(...)` 內部：AdamW 實作 / projection schedule / QADG backward |
| ✅ | ✅ | 兩層都乾淨，問題在 pruning path | 回去看 Phase 5 epoch 4-6（grp_sp 開始長的瞬間），查 group zero-out 後的梯度流動 |
| ❌ | ✅ | 不合理（A 是 B 的子集） | 檢查兩個 script 是否真的在跑同一條路徑 |
