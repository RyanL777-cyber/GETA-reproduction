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

## Exp C 診斷結果（2026-04-24）

Script: `bert_geta_phase4_5/exp_c_quantwrap_diagnostic.py`（<1 min 跑完）

**確認三個問題**：

1. **Activation quant 參數被用 weight 統計初始化（H1，主要元兇）**
   - `initialize_quant_layer`（`quant_layers.py:436-440`）用 `max|weight|` 設 `q_m_act`
   - 結果：`qm_act/max|x|` 中位數 0.33，最糟 0.004（量化範圍 = activation 範圍的 1/200）
   - 平均 32% activation 被 saturate，qa_outputs 輸入 saturate 76%
   - Wrapped logits std=0.015 vs dense std=0.222，輸出被壓扁 15×

2. **STE 前後向不對應**
   - `SymQuantizerNonLinear.backward` 只在 `|x|>=clip_val(2.0)` 歸零梯度
   - 但 forward 在 `|x|>=q_m_act`（常 << 2）就已經 saturate 成常數
   - 中間那段區域 forward 常數、backward 當 identity 傳 → 梯度 4 數量級不平衡

3. **梯度雪崩不平衡**
   - L0.output.dense: `|grad|/|W|=13.5`（爆炸）
   - L11.att.query: `|grad|/|W|=8.8e-4`（等同不訓練）
   - 差 ~15,000 倍，配合 lr=3e-5，上層根本沒動

**被否決**：H2（`clip_val=(-2,2)` 砍梯度）——gradZero% 平均只 0.3%，不是主因。

---

## 修法實作完成（2026-04-24）

Fix 實作：`bert_geta_phase4_5/quant_fix.py`
- `apply_ste_fix()` — monkey-patch `SymQuantizerNonLinear.backward` 跟 `SymQuantizerLinear.backward`，讓 `|x| >= q_m` 的區域梯度歸零（跟 forward 的 saturate 對齊）
- `calibrate_quant_layers(model, batches)` — 暫時把 activation quant 關掉，hook 每個 `QuantizeLinear` 的 input 收集 `max|x|`，用 `max|x| * 1.05` 設 `q_m_act`
- `wrap_quant_fixed(model, device, batches)` — 三件事的 convenience wrapper

**不動 upstream GETA 原始檔**，全靠 monkey-patch。想回到原版只要不呼叫 `apply_ste_fix()` 即可。

## Exp D / E 結果（2026-04-24）

| Exp | 設定 | F1 | 判定 | 備註 |
|---|---|---|---|---|
| D | Quant wrap+fix + 純 AdamW + 16-bit | **88.26** | ✅ PASS | 貼近 baseline 88.50 |
| E | Quant wrap+fix + GETA(sp=0) + 16→4-bit | **74.27** | ❌ FAIL | 但比 Exp B 的 28.70 進步 +45 |

**Fix 完全有效**。Exp D 證明量化那層徹底乾淨。

**Exp E 缺的 14 F1 來自 bit reduction**（不是 fix 有問題）：
- step 0-7377（projection 階段前）loss 曲線跟 Exp D 完全一致（等資料量下 1.63 vs 1.63）
- step 7377（projection 結束、bit 固定 4-bit）瞬間 loss 跳升 1.6→1.9 並鎖住
- 剩下 83% 訓練都在 4-bit 下跑，2 epoch 不夠恢復

Log: `results_exp_d/exp_d_20260424_042635.log`、`results_exp_e/exp_e_20260424_040115.log`

---

## 下一步（兩條路並行）

**路線 A — 回頭跑 Phase 5（最終驗證）** ← 優先

`bert_geta_phase5/run_experiment.py` 已改好：在 M2 加 `apply_ste_fix()`、在 M2 跟 M3 之間加 calibration pass（預設 8 batches × bs=4）。`quant_fix.py` 已複製到 phase5 資料夾內，self-contained。

```bash
cd ~/GETA\ reprocution/GETA-reproduction/bert_geta_phase5
python3 run_experiment.py --sparsity 0.1 0.3 0.5 0.7
# 可改 --calib_batches / --calib_batch_size 調整 calibration
```

Paper 跑 10 epoch，模型有 5× 時間在 4-bit 下 fine-tune 恢復。paper Table 3 F1 ≈ 84-87 就是 quant+prune 聯合結果，這才是最終對照。預期跑很久，建議 tmux。

**路線 B — 隔離 bit reduction（可選，~2h）**

`bert_geta_phase4_5/exp_f_geta_sp0_16bit.py` 已寫好：同 Exp E 但設 `min_bit_wt=max_bit_wt=16`（關閉 bit reduction）。
- 若 F1 ≥ 85 → 14 分缺口全是 bit reduction 的代價，fix 無殘留問題
- 若 F1 < 85 → GETA optimizer 另有問題需繼續隔離

```bash
cd ~/GETA\ reprocution/GETA-reproduction/bert_geta_phase4_5
python3 exp_f_geta_sp0_16bit.py
```

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
