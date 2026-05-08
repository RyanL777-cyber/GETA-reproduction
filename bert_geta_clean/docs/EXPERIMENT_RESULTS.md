# 復現結果 — BERT-base SQuAD

所有 run 都已套用本資料夾的補丁(修法 #1 + #2 + `quant_fix.py` + `projection_steps`)。
硬體:單卡 GPU(Linux server)。所有 run seed=42。

## 主要數字 vs 論文

| Sparsity | Run config | EM | F1 | 論文 F1 | 差距 |
|---|---|---|---|---|---|
| 0.0(baseline,無壓縮) | 2ep, bs=4 | 81.14 | **88.50** | 88.50 | 0.00 ✅ |
| 0.1 | 10ep, bs=4 | 75.09 | 84.45 | 86.06 | −1.61 |
| 0.5 | 16ep, bs=16 | 74.45 | 84.02 | 85.87 | −1.85 |
| **0.7**(最佳) | **13ep, bs=16** | **74.29** | **83.85** | **84.74** | **−0.89** |
| 0.7 | 16ep, bs=16 | 73.72 | 83.61 | 84.74 | −1.13(過擬合) |

對照論文 Table 3 的 OTO + 8-bit PTQ baseline:

| Sparsity | OTO+PTQ F1 | 我們的 F1 | Δ |
|---|---|---|---|
| 0.1 | 83.43 | 84.45 | +1.02 |
| 0.5 | 83.30 | 84.02 | +0.72 |
| 0.7 | 82.57 | 83.85 | **+1.28** |

→ 全部 run 都贏過 OTO+PTQ。但都還沒追上論文 GETA(差 ~1 F1)。

## 診斷:差距來自 pre-prune 階段(剪枝前就已經輸了)

所有 run 在**剪枝開始之前**的 best F1 = 86.31–87.06。論文 baseline(無壓縮)= 88.50。**還沒做任何壓縮就已經比論文低 1.5–2 F1**,剪枝再吃掉 2–3 F1。

這代表差距不可能靠調 prune / fine-tune 階段補回來。
最可疑的根因:**所有 run 都用 `bit_reduction=1`,論文 default = 2**。尚未驗證。

## 沒有幫助的東西(不要重複嘗試)

- **超過最佳點還繼續延長 epoch** — sp=0.7 從 13ep 拉到 16ep,F1 反而從 83.85 退到 83.61。
  `train_loss` 已經掉到 0.038,`val_F1` 在 ep15 觸頂、ep16 開始掉。Linear LR schedule 末段衰減到 0 → 後面想救也救不回來。
  教訓:不要因為 train loss 還在降就自動加 epoch。

## 確實有效的加速

- **bs=4 → 16, lr 3e-5 → 6e-5**(sqrt-scaled):每 epoch 速度 2.2×
  (150min → 67min,同一張 GPU)。24GB 卡沒 OOM,訓練穩定。

## 下一步建議實驗

1. **`sp=0.7, bit_reduction=2, bs=16, ~12ep`** — 直接驗證是不是 br=1 害的
2. 如果 #1 還不夠,**退回論文 config: `bs=4, br=2`**(代價 ~25h/sparsity 點)


/home/h24116081/GETA reprocution/GETA-reproduction/bert_geta_clean
