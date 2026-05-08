# GETA × BERT-base SQuAD 復現 — 完整實驗報告

**目標**: 復現 GETA paper (CVPR 2025) Table 3 的 BERT/SQuAD 結果(joint pruning + quantization,F1 ≈ 84-87)。

---

## Phase 1 — Baseline (✅ 過關)

純 BERT,2 epochs full SQuAD,**EM=81.14 / F1=88.50**,對齊 HF 官方數字。確認資料流、preprocess 沒問題。

---

## Phase 3 — 介面 smoke test (🔧 改了 GETA 原始碼第一次)

### 撞到的問題

M4 呼叫 `oto.geta(...)` 建 optimizer 直接報:

```
ValueError: some parameters appear in more than one parameter group
```

### 診斷

- model 本身 637 個 unique param,沒重複 → 問題在 GETA。
- 寫 `dump_duplicate_params.py` 重現 `Graph.get_param_groups()` 的 raw 輸出 → **314 個 param 各重複 2 次**(Q/K/V 的 weight/bias、output.LayerNorm、量化 scalar)。
- 為什麼 Q/K/V 重複? 因為 QADG 把它們同時塞進兩個依賴群:
  - **head 群** — Q/K/V 的 768 輸出維被切成 12 head × 64,要剪 head 必須三個一起剪
  - **trunk 群** — Q/K/V 的 768 輸入維是 residual trunk

  同一份 weight,兩個群用不同 `p_transform` 算 mask,**QADG 層語意是對的**。但 PyTorch optimizer 不允許 param 出現在兩個 param_group(會 state 衝突 + GETA 的 step() in-place `p.data.add_` 會減兩次 lr×grad)。

### 修法 #1 — `geta/only_train_once/graph/graph.py:1371`

在 `get_param_groups()` 末端加 dedup,但**順序按「群大小遞增」**而不是 id 字典序:

```python
dedup_order = sorted(
    param_groups.items(),
    key=lambda kv: (len(kv[1].get("params", [])), kv[0]),
)
```

**為什麼一定要這樣排**: 初版用 id 字典序時 layer.0 的 head 群(`node-765_...`)字典序大於 trunk 群(`node-663_...`),導致 layer.0 的 head 群被整組清空(其他 11 層正常)。改成「小群優先」後:per-layer head 群 24 param < trunk 群 245 param → head 群永遠贏,語意上也對(具體群 > 廣域群)。

驗收: smoke.py M1→M5 全通,27 groups 全保留。

---

## Phase 4 — 小規模訓練 (🔧 改了 GETA 原始碼第二次)

### 撞到的問題

訓完呼叫 `oto.construct_subnet()` 產壓縮模型 → forward 爆 shape error:

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (3072x943 and 3072x768)
```

12 層 encoder 的 `output.dense` 都中。

### 診斷

`construct_subnet` 兩遍剪:

1. prune_out_dim — `intermediate.dense` out 從 3072 剪到 ~1008
2. prune_in_dim — `output.dense` in 應該同步縮

但 `output.dense` 在 graph 裡 **沒有 incoming edges**(`graph.incoming(node)` 空 list)→ 跳過 in-dim pruning → in_features 留 3072 → 跟前面 1008 對不起來。

**Root cause**: `model_to_quantize_model` 把 Linear 換成 QuantizeLinear 後,`torch.jit.trace` 在 GELU → output.dense 之間插了量化中間節點,graph builder 後處理沒保留這段邊。純 prune 路徑(`oto.hesso()`)沒中,所以 CV tutorials 不會撞到,**joint prune+quant 才會觸發**。

### 修法 #2 — `geta/only_train_once/subnet_construction/pruning_compression.py`

在第二遍 in-dim pruning 之後,加第三遍 scan:對每個 `*.layer.N.output.dense`,找對應的 `*.layer.N.intermediate.dense` 的 `pruning_redundant_idxes`,手動同步剪 `output.dense.in_features`。

驗收: subnet 產出成功,params 從 ~110M → 80.45M。**但 F1=4.35**(loss 5.93→5.93,沒學到東西)— **這個訊號當時被誤判成「不當機=pass」放過去**,後來證明是 Phase 5 崩潰的早期徵兆。

---

## Phase 5 第一次 — 全面崩潰

| sp | Best F1 | Compressed F1 | 備註 |
|---|---|---|---|
| 0.1 | 6.91 | 4.48 | |
| 0.3 | 6.34 | 4.63 | |
| 0.5 | 48.96 | 4.22 | best 在 ep1(未剪枝) |
| 0.7 | 48.96 | 1.87 | best 在 ep1(未剪枝) |

差 paper >40 分。額外發現 **Bug #3**: `projection_period_duration = projection_steps // projection_periods`,paper 用 Kp=6 但沒傳 `projection_steps` → `1//6 = 0` → step() 除零。在 caller 層(`run_experiment.py`)算對 `projection_steps` 補上,**沒動 GETA 原始碼**。

關鍵觀察: **ep1-3 grp_sparsity=0(剪枝還沒啟動)F1 就只到 48.96 並一路往下** → 問題不在剪枝,在 dense 階段本身。

---

## Phase 4.5 — 隔離 quant wrap (定位真兇)

設了三組對照:

| Exp | 設定 | F1 | 結論 |
|---|---|---|---|
| A | Quant wrap + 純 AdamW | 27.33 | ❌ |
| B | Quant wrap + GETA(sp=0) | 28.70 | ❌ |
| baseline | 無 quant wrap + AdamW | 88.50 | ✅ |

**A 跟 baseline 唯一差異 = quant wrap → quant wrap 是兇手(F1 88→27)**。GETA optimizer 在 dense 階段是乾淨的(A≈B)。

### Exp C 診斷出三個問題

1. **Activation quant 參數被用 weight 統計初始化(主因)** — `initialize_quant_layer` 用 `max|weight|` 設 `q_m_act`,實際 `qm_act/max|x|` 中位數 0.33,最糟 0.004 → 量化範圍只有 activation 的 1/200 → 平均 32% activation 被 saturate,qa_outputs 的輸入 76% saturate
2. **STE 前後向不對應** — `SymQuantizerNonLinear.backward` 在 `|x|>=clip_val(2.0)` 才歸零,但 forward 在 `|x|>=q_m_act`(常 << 2)就 saturate → 中間區段 forward 常數 / backward identity → 梯度差 4 個數量級
3. **梯度雪崩不平衡** — L0 dense `|grad|/|W|=13.5`,L11 query `|grad|/|W|=8.8e-4`,差 1.5 萬倍

### 修法 — `bert_geta_phase4_5/quant_fix.py` (monkey-patch,不動 upstream)

- `apply_ste_fix()`: patch backward,讓 `|x|>=q_m` 區域梯度歸零(對齊 forward 的 saturate)
- `calibrate_quant_layers()`: hook 收集 `max|x|`,用 `max|x|*1.05` 重設 `q_m_act`
- 不改 GETA 原始檔。回到原版只要不呼叫 `apply_ste_fix()`。

### 驗證 Exp D / E

| Exp | 設定 | F1 |
|---|---|---|
| D | Quant wrap+fix + AdamW + 16-bit | **88.26** ✅ |
| E | Quant wrap+fix + GETA(sp=0) + 16→4-bit | **74.27** (從 28.70 +45) |

E 缺的 14 F1 來自 bit reduction(step 7377 projection 結束 bit 固定 4-bit 的瞬間 loss 1.6→1.9 跳升)。fix 本身完全有效。

---

## Phase 5 重跑(套用 quant_fix)

`run_experiment.py` 在 M2 加 `apply_ste_fix()`,M2/M3 之間加 calibration pass。

| Run | EM | F1 | Paper F1 | Gap |
|---|---|---|---|---|
| sp=0.1 (10ep, bs=4) | 75.09 | 84.45 | 86.06 | −1.61 |
| sp=0.5 (16ep, bs=16) | 74.45 | 84.02 | 85.87 | −1.85 |
| **sp=0.7 (13ep, bs=16)** | 74.29 | **83.85** | 84.74 | **−0.89(最佳)** |
| sp=0.7 (16ep, bs=16) | 73.72 | 83.61 | 84.74 | −1.13(過擬合) |

全部贏過 OTO+8bit PTQ baseline,但都還沒到 paper。

### 中間調整

- bs 4→16(2.2× 加速,67min/epoch),lr 用 sqrt scaling 3e-5→6e-5
- 發現 16ep 過擬合(train loss 已降到 0.038)

### 關鍵診斷

**全部 run 的 best_f1(剪枝前)= 86.31–87.06,paper baseline = 88.50** → 還沒剪枝就已經低 1.5–2 F1,剪枝再吃 2–3 F1。問題出在 pre-prune 階段。

懷疑根因: **所有 run 都用 `bit_reduction=1`,paper default = 2**。當初打 baseline command 時打成 1 然後一路沿用。

---

## 三個改動總整理

| # | 檔案 | 為什麼改 |
|---|---|---|
| **修法 #1** | `geta/only_train_once/graph/graph.py:1371` | QADG 的 Q/K/V/output.LayerNorm 同時掛在 head 群和 trunk 群,PyTorch optimizer 不接受 → dedup 並按**群大小遞增**排序(否則 layer.0 的 head 群會被誤殺) |
| **修法 #2** | `geta/only_train_once/subnet_construction/pruning_compression.py` | QuantizeLinear 把 graph 的 GELU→output.dense 邊弄丟 → output.dense 變孤立節點 → in-dim 沒同步剪 → 加第三遍 scan,根據對應 intermediate.dense 補剪 |
| **修法 #3 (monkey-patch,不算改 upstream)** | `bert_geta_phase4_5/quant_fix.py` | activation quant 用 weight 統計初始化導致 75% saturation + STE 前後向不對應;monkey-patch backward + calibration 重設 `q_m_act` |

(另外還有 Bug #3 `projection_steps // projection_periods` 除零,但只在 `run_experiment.py` 補,沒動 GETA 內部。)

---

## 目前狀況 / 下一步

**待跑**: `sp=0.7, br=2, bs=16, ~12ep` — 驗證 br=2 是不是補上 ~1 F1 缺口的關鍵。若 br=2 還不夠,再退回 paper 的 bs=4(代價 ~25h/點)。
