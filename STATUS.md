# Project Status

> **更新規則**：每次對話結束、或完成一個可驗收的步驟，更新這份。不要把狀態留在對話歷史裡。

**Last updated**: 2026-04-23

---

## 目前在哪

**Phase 4.5（新增）**：補對照驗證。Phase 5 四個 sparsity 全崩，回頭發現 Phase 4 驗收標準只測「不當機」、沒測訓練能否收斂，需要補兩個對照組定位失敗層。

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

## 下一步

**跑 Phase 4.5 兩個對照組**（在 server 上）：

```bash
cd ~/GETA\ reprocution/GETA-reproduction/bert_geta_phase4_5

# Exp A — Quant wrap ✅, GETA ❌（純 AdamW + linear warmup）
python3 exp_a_quantwrap_adamw.py

# Exp B — Quant wrap ✅, GETA ✅ 但 target_sparsity=0
python3 exp_b_geta_sp0.py
```

兩個都是 2 epochs full SQuAD，約 2h/run。建議 tmux 或 nohup。

**解讀表**見 `bert_geta_phase4_5/notes.md`。摘要：

| Exp A | Exp B | 結論 |
|---|---|---|
| ❌ | — | 問題在 quant wrap |
| ✅ | ❌ | 問題在 GETA optimizer dense 部分（非 pruning） |
| ✅ | ✅ | 問題在 pruning path，回 Phase 5 查 group zero-out 動態 |

---

## 已過關 / 未過關

- ✅ **bert_baseline (Phase 1)**：純 BERT 2 epochs full SQuAD，**EM=81.14 / F1=88.50**，對齊 HF 官方 baseline
- ⚠️ **Phase 3 smoke**：`bert_geta_mvp/logs/smoke_20260416_171244.log`，只驗 M1-M5 2 batch 不當機，**沒測訓練收斂**
- ⚠️ **Phase 4 小規模**：`bert_geta_mvp/logs/phase4_20260417_011802.log`，1 epoch 跑完不當機，但 **loss 5.93→5.93、F1=4.35，實際上沒學到東西**。驗收標準過低，這個訊號當時被放過
- ❌ **Phase 5**：全面崩潰（見上表）
- 🔄 **Phase 4.5**：已寫好 script，待 server 執行

---

## Open questions（非阻塞，但要追蹤）

見 `geta_knowledge/open_questions.md`。

與目前工作相關的：
- `attention.output.LayerNorm` 為何沒被雙重歸類（Bug #1 研究時留下的疑問）
- 三個 bug 是否要回報 GETA upstream
- Bug #2 的 workaround 只 cover FFN pattern，若未來換模型需擴展
- Phase 4 驗收標準為何只測不當機——未來每個 phase 應該都要有「訓練能收斂」的硬門檻
