# Project Status

> **更新規則**：每次對話結束、或完成一個可驗收的步驟，更新這份。不要把狀態留在對話歷史裡。

**Last updated**: 2026-04-24

---

## 目前在哪

**Phase 4.5 完成**：兩個對照組都跑完了，**問題定位在 Quant wrap**。下一步是診斷 `only_train_once/quantization/` 內部。

Phase 3（介面打通）跟 Phase 4（規模測試）都還算過關，但 **Phase 4 的 F1=4.35 當時被當作「不當機=pass」放過去**，這個訊號現在回頭看就是 Phase 5 崩潰的早期徵兆。

---

## 最近一次執行：Phase 5（全面失敗）

| Sparsity | Best F1 | Best Epoch | Compressed F1 | 備註 |
|---|---|---|---|---|
| 0.1 | 6.91 | 8 | 4.48 | |
| 0.3 | 6.34 | 9 | 4.63 | |
| 0.5 | 48.96 | 1 | 4.22 | best 是 epoch 1（未剪枝）|
| 0.7 | 48.96 | 1 | 1.87 | best 是 epoch 1（未剪枝）|

Paper 目標：F1 ≈ 84-87。**全部差 >40 分**。

### 關鍵症狀
- 訓練 log 裡 loss 鎖在 5.9507 幾萬步不動（典型 dying-network）
- **Epoch 1-3 的 `grp_sparsity=0`（剪枝還沒啟動）F1 就只到 48.96 且一路往下**，代表問題不在 pruning，在 dense 階段本身
- `bert_baseline` 2 epochs 能到 F1=88.50，同樣資料、同樣 preprocess，所以問題不在資料流

### 可疑層（Phase 5 vs baseline 的差異）
1. Quant wrap（`model_to_quantize_model`）
2. GETA optimizer（取代 HF Trainer 的 AdamW + linear warmup）
3. batch size（baseline=12, phase5=4）

Log: `bert_geta_phase5/results/phase5_20260418_175634.log` 等 4 份

---

## Phase 4.5 結果（2026-04-24）

| Exp | 設定 | Epoch 2 F1 | 判定 |
|---|---|---|---|
| A | Quant wrap + 純 AdamW（無 GETA）| 27.33 | ❌ FAIL |
| B | Quant wrap + GETA(sp=0) | 28.70 | ❌ FAIL |
| bert_baseline 參考 | 無 quant wrap + AdamW | 88.50 | ✅ |

Log:
- `bert_geta_phase4_5/results_exp_a/exp_a_20260423_204616.log`
- `bert_geta_phase4_5/results_exp_b/exp_b_20260423_204646.log`

**結論**：
1. A 跟 baseline 的唯一差異是 quant wrap → **quant wrap 是兇手**（F1 88→27）
2. A ≈ B（F1 差 1.4 分）→ GETA optimizer dense 部分是乾淨的，行為跟純 AdamW 幾乎一樣
3. Phase 5 的崩潰 = 「量化後本來就殘的模型」再被剪枝擊垮

---

## 下一步

診斷 `geta/only_train_once/quantization/`：

1. 查 `quant_model.py` 的 `model_to_quantize_model`——哪些層被包？embedding 有沒有被誤包？
2. 查 `quant_layers.py` 的 `QuantLinear` forward/backward（STE 有沒有接對）
3. 確認 `QuantizationMode.WEIGHT_AND_ACTIVATION` 的初始 bit 寬度（如果起始就低於 16-bit，forward 會直接失真）
4. 寫 `exp_c_quantwrap_diagnostic.py` 做即時對比：wrap 前/後同一 input 的 logit 差異、所有 quant layer 的 scale/zero_point、1-step backward 後的 grad norm

Loss 鎖在 3.2 而不是 0.7 的量級差異，指向量化 range 或 STE 梯度問題。

---

## 已過關 / 未過關

- ✅ **bert_baseline (Phase 1)**：純 BERT 2 epochs full SQuAD，**EM=81.14 / F1=88.50**，對齊 HF 官方 baseline
- ⚠️ **Phase 3 smoke**：`bert_geta_mvp/logs/smoke_20260416_171244.log`，只驗 M1-M5 2 batch 不當機，**沒測訓練收斂**
- ⚠️ **Phase 4 小規模**：`bert_geta_mvp/logs/phase4_20260417_011802.log`，1 epoch 跑完不當機，但 **loss 5.93→5.93、F1=4.35，實際上沒學到東西**。驗收標準過低，這個訊號當時被放過
- ❌ **Phase 5**：全面崩潰（見上表）
- ✅ **Phase 4.5**：已完成，定位問題在 Quant wrap（見上節）

---

## Open questions（非阻塞，但要追蹤）

見 `geta_knowledge/open_questions.md`。

與目前工作相關的：
- `attention.output.LayerNorm` 為何沒被雙重歸類（Bug #1 研究時留下的疑問）
- 三個 bug 是否要回報 GETA upstream
- Bug #2 的 workaround 只 cover FFN pattern，若未來換模型需擴展
- Phase 4 驗收標準為何只測不當機——未來每個 phase 應該都要有「訓練能收斂」的硬門檻
