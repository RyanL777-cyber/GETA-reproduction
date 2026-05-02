# Phase 5 GETA × BERT 重現結果診斷

**結論：沒有重現成功。** Compressed F1 全部落在 76–77，比論文 GETA 的 84–86 低約 8–9 分；EM 落在 65–67，比論文的 75–78 低約 10–13 分。即使對比論文裡 GETA 應該勝過的 OTO+8-bit PTQ baseline，目前的重現也輸 6 分以上。

實驗時間：每個 sparsity 跑了 ~3300 分鐘（55 小時）= 一個 sparsity 一台 GPU 兩天多。四個 sparsity 平行在不同 GPU 上跑。

---

## 1. 結果對照（論文 vs 我跑出來的）

| Sparsity | 論文 EM | 論文 F1 | 我的 EM | 我的 F1 | 差距 (F1) | best_f1 (epoch 1) |
|----------|---------|---------|---------|---------|-----------|---------------------|
| 10%      | 78.26   | 86.06   | 66.72   | 77.17   | **−8.89** | 85.93 |
| 30%      | 77.28   | 85.70   | 66.05   | 77.24   | **−8.46** | 85.51 |
| 50%      | 76.74   | 85.87   | 65.13   | 76.26   | **−9.61** | 85.51 |
| 70%      | 75.88   | 84.74   | 65.09   | 76.25   | **−8.49** | 85.78 |

對比論文的 OTO + 8-bit PTQ（GETA 應該贏這一行）：

| Sparsity | OTO+PTQ EM | OTO+PTQ F1 | 我的 EM | 我的 F1 |
|----------|-----------|-----------|---------|---------|
| 10%      | 73.87     | 83.43     | 66.72   | 77.17   |
| 30%      | 72.95     | 83.31     | 66.05   | 77.24   |
| 50%      | 72.71     | 83.30     | 65.13   | 76.26   |
| 70%      | 71.24     | 82.57     | 65.09   | 76.25   |

→ 連 OTO baseline 都沒打到，更別說 GETA。

BOPs 的部分數值無法直接對照論文（論文寫 GB，我輸出的是 `compute_bops(in_million=True)`），但相對下降比例也明顯不對：
- 論文 sp10→sp70：2.63 → 1.62 GB BOPs（ratio 0.62）
- 我的 sp10→sp70：1,696,683 → 912,102 M BOPs（ratio 0.54）
比例接近，但 absolute 數值要等弄清楚單位後才能下結論。

---

## 2. 最關鍵的訊號：F1 在 epoch 1 達到頂點後崩盤

從 sp10 的訓練紀錄（`phase5_20260424_174757.log`）：

| Epoch | grp_sp | EM | F1 | 備註 |
|-------|--------|----|----|------|
| 1 | 0.000 | 78.15 | **85.93** | bit width 還沒掉到底；幾乎等於 paper GETA 目標 |
| 2 | 0.000 | 59.26 | 70.76 | bit reduction 已跑完到 4-bit；F1 暴跌 15 分 |
| 3 | 0.000 | 65.54 | 76.20 | 開始恢復 |
| 4 | 0.017 | 67.18 | 77.75 | pruning 啟動 |
| 5 | 0.033 | 67.22 | 77.27 | |
| 6 | 0.050 | 68.37 | 77.84 | |
| 7 | 0.050 | 68.46 | 78.25 | |
| 8 | 0.067 | 65.83 | 77.92 | |
| 9 | 0.083 | 68.26 | 78.26 | |
| 10 | 0.100 | 66.72 | 77.18 | 最後 compressed F1 |

關鍵觀察：
1. **Epoch 1 的 F1 (85.93) 已經非常接近論文 GETA 在 sp10 的目標 (86.06)**。代表「OTO graph + GETA optimizer + STE fix + calibration」這條 pipeline 在「bit 還高 + 還沒 prune」的狀態下是對的。
2. **Epoch 2 F1 暴跌 15 分（85.93 → 70.76），但此時 grp_sp 還是 0.000**。也就是說，這次崩潰 **不是 pruning 造成的**，是 **quantization 把 bit 從 16 降到 4** 造成的。
3. **後續 8 個 epoch 都拉不回 epoch 1 的水準**，最高只到 78.26。10 epochs 不足以讓模型在 4-bit 下從 70 → 86。
4. **四種 sparsity 的 compressed_f1 幾乎一樣（76–77）**：sp10 跟 sp70 都是 77 / 76。如果 pruning 是主因，sparsity 越大應該越差；現在沒差，代表 **是量化精度而不是 pruning 在卡上限**。

---

## 3. 最可能的原因（按可能性排序）

### A. Calibration 跟最終 bit width 不匹配 ★★★★★
`calibrate_quant_layers(num_bits=16)` 在 epoch 0 之前對 `q_m_act` / `d_quant_act` 做了 16-bit 的初始化：
```
d_quant_act = q_m_act / (2^15 - 1) = q_m_act / 32767
```
但 paper 設定 `bit_range=[4,16], br=2, projection_periods=6`，最終 bit 會降到 **4-bit**（16, 14, 12, 10, 8, 6, 4），這時 `d_quant_act` 應該是 `q_m_act / 7`，差了 **~4680 倍**。

如果 GETA 的 `d_quant_act` 是 learnable 參數（用 `lr_quant=1e-4` 學），它要在剩下 7~8 個 epoch 內爬 4 個量級，幾乎不可能。  
→ 建議：`calibrate_quant_layers(num_bits=4)` 或在 bit 降到底之後再做一次校準。

### B. Bit-reduction schedule 太快、跟 pruning 撞期 ★★★★
schedule（log 第 16 行）：
- `total=221310`, `period_len=36885`
- bit projection 從 step 0 到 step 36885，**6 個 period 把 bit 從 16 降到 4**
- pruning 從 step 36885 開始 → epoch 2 結尾

→ epoch 1 結束時 bit 大概在 10-bit，F1 仍然是 85.93。  
→ epoch 2 結束時 bit 已經跌到 4-bit + pruning 同時啟動，F1 直接掉 15 分。

論文 Kp=6（projection_periods）但搭的是 P=6 完整 6 個 period 拉開，可能訓練 budget 該配更長 epoch（paper 的具體 schedule 待查），現在 10 epoch 中 bit-降低與 pruning 的 warm-up 各只佔 1 epoch，模型沒時間適應。

### C. 沒有 LR scheduler ★★★
`run_experiment.py` 用固定 `lr=3e-5, lr_quant=1e-4`。BERT QA 通常會用 linear warmup + linear decay。沒有 scheduler 讓 epoch 2 之後的訓練偏慢、又容易抖。從 log 看 loss 在 epoch 2~3 完全沒下降甚至上升（1.27 → 1.53）也佐證這點。

### D. "best_f1" 指標誤導 ★★
`run_experiment.py` 把 `best_f1` 當「整個 run 最佳 F1」，但 epoch 1 的 best 是 16-bit-ish 模型的 F1，不是最終壓縮模型的 F1。報告應該只看 `compressed_f1`/`compressed_em`。這不影響真實重現品質，但會誤判「best_f1=85.9 看起來重現了」。

### E. Calibration batch 數太少 ★
只用 8 batch × bs 4 = 32 個樣本來估每層 max|x|。對一個 73 層的模型來說可能不夠穩。`max|x|` 從 2.66 到 137.5 跨度大，少量樣本可能低估實際 max。

### F. BOPs 數值單位不確定 ★
論文寫 GB，我輸出 `bops_million`。先把單位對清楚再判斷 BOPs 比例是否符合預期。

---

## 4. 建議的下一步（依 cost/benefit 排序）

1. **確認 GETA 內部 `d_quant_act` 是否會跟著 bit 改變自動 rescale**（讀 `geta/only_train_once/quantization/quant_layers.py` 跟 `optimizer.geta`）。如果不是，calibration 要改成 `num_bits=4`（最終值），或在 bit 降完後二次校準 → **這是 1 行改動但可能直接解掉 8 分差距**。
2. 確認 BOPs 單位：算一個 baseline `compute_bops` 對應 GFLOPs 是多少，跟論文 13.57 GB 對齊。
3. 實驗：把 `projection_periods` 拉長到佔總訓練量 1/3（不是 1/6），讓 bit 降到 4 之後還剩 6+ epoch 給模型 recover。
4. 實驗：加 LR scheduler（linear warmup 1 epoch + linear decay）。
5. 在 epoch 1 跟 epoch 2 各存一份 model checkpoint，比較「bit=10 + sp=0」與「bit=4 + sp=0」的 F1 差距，把 quantization-only loss 跟 pruning loss 拆開。

**先驗證 hypothesis A（calibration bit mismatch）成本最低，回報可能最大。** 不需要重跑全 10 epoch，跑 sp10 / 3 epoch 就能確認 epoch 2 的暴跌是否消失。

---

## 5. 已正確的部分（不要動）

- OTO graph build 沒問題（epoch 1 F1 = 85.93）
- STE fix + calibration（at 16-bit）對於 epoch 1 的精度是夠的
- Group sparsity 的 ramp 接近目標（target 0.7 → actual 0.6999，誤差 < 1e-4）
- BOPs / params 隨 sparsity 單調下降（趨勢對，絕對值待校）
- GPU auto-select 跟 4 個 sparsity 平行跑的工程架構 OK

---

## 6. 一句話結論

**不是 pipeline 壞掉，是 quantization schedule（bit 降太快 + calibration 只校了 16-bit）讓模型在 4-bit 下沒時間 recover。先試把 calibration 改成最終 bit width，再決定要不要加長 projection schedule。**
