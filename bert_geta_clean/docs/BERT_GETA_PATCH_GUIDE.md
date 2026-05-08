# BERT × GETA 補丁指南

GETA 的 joint pruning + quantization pipeline 在 BERT-base SQuAD 上會踩到 3 個 bug。原因是這套 codebase 主要在 CV model 上驗證過,attention-heavy + 量化的 use case 沒人測過。本文是復現論文所需的最小補丁集。

**TL;DR**:2 個原始碼 patch(`git apply` 一鍵套)、1 個 runtime monkey-patch(`quant_fix.py`)。

---

## 修法 #1 — Param-group dedup 必須由「小群」優先迭代

**檔案**: `only_train_once/graph/graph.py`(約 1370 行,`Graph.get_param_groups()` 內)

**症狀**

```
ValueError: some parameters appear in more than one parameter group
```

…在 `oto.geta(...)` 階段就炸,還沒進到訓練。

**根因**

QADG 把 BERT 的 Q/K/V weight 同時放進**兩個** dependency group:

- **head group** — Q/K/V 的 768-dim 輸出被切成 12 head × 64,要剪 head 必須三個一起剪。
- **trunk group** — Q/K/V 的 768-dim 輸入掛在 residual trunk 上。

這在語意上是對的(兩群有不同的 `p_transform`)。但 PyTorch optimizer 不接受同一個 param 出現在兩個 `param_group`。GETA 本來就有 dedup 邏輯 — 但它是按 **group_id 字典序**迭代。BERT 的 layer.0 剛好 head group id 排在 trunk group id 之後 → trunk 先走、把 head 的 param 都標成 seen → head group 被整組清空。其他 11 層剛好排序方向對才沒事。

**修法**

把 dedup 迭代順序改成按 `(group_size, group_id)` 排 — **小群優先**。這樣 per-layer 的具體群(head, ~24 param)永遠搶在廣域 trunk 群(~245 param)之前認領自己的 param。具體 > 廣域。

```diff
+        # Iterate smaller groups first so per-layer (e.g. head) groups keep their
+        # params instead of losing them to the large trunk group. Falling back to
+        # group_id for deterministic tie-breaking.
         seen_param_ids = set()
-        for group_id, param_group in list(param_groups.items()):
+        dedup_order = sorted(
+            param_groups.items(),
+            key=lambda kv: (len(kv[1].get("params", [])), kv[0]),
+        )
+        for group_id, param_group in dedup_order:
```

→ 套用 [patches/01-graph-dedup.patch](patches/01-graph-dedup.patch)

**驗證**

```python
oto = OTO(model, dummy_input)
oto.geta(...)                # 不應該 raise
print(len(oto.optimizer.param_groups))  # BERT-base 應該是 27 個,沒有缺
```

---

## 修法 #2 — 剪枝後同步 FFN `output.dense.in_features`

**檔案**: `only_train_once/subnet_construction/pruning_compression.py`
(`automated_pruning_compression` 內,放在原本兩遍剪枝邏輯之後)

**症狀**

訓練完呼叫 `oto.construct_subnet()` 產壓縮模型,forward 立刻爆:

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (3072x943 and 3072x768)
```

12 層 encoder 的 `output.dense` 全部對不齊。

**根因**

BERT 的 FFN 是兩個成對的 Linear:

```
intermediate.dense:  768 → 3072
output.dense:        3072 → 768
```

剪掉 3072-dim channel 必須**兩個一起縮** — `intermediate.dense.out_features` 跟 `output.dense.in_features` 要同步。GETA 的 `construct_subnet` 跑兩遍:

1. `prune_out_dim` 把 `intermediate.dense` out 從 3072 → ~1008 ✓
2. `prune_in_dim` 應該循 graph 邊回追,把 `output.dense` in 同步縮

但 `model_to_quantize_model` 把每個 Linear 包成 `QuantizeLinear` 之後,`torch.jit.trace` 在 GELU 跟 `output.dense` 之間插了 quant/dequant 中間節點。graph builder 的後處理把這段邊弄丟了 → `output.dense` 變**孤兒節點**(沒有 incoming edges)。第二遍 prune_in_dim 看不到上游模組就跳過,`in_features` 留在 3072 → shape mismatch。

純剪枝路徑(`oto.hesso()`)不會包 QuantizeLinear → 不會中。**只有 joint prune + quant 才會觸發**,所以 CV tutorials 沒人撞到。

**修法**

在第二遍剪枝後加第三遍 scan,改用**模組命名規則**配對(不依賴 graph 邊)。對每個 `*.layer.N.output.dense`,找對應的 `*.layer.N.intermediate.dense`,讀它的 `pruning_redundant_idxes`,手動把 `output.dense.in_features` 跟 `weight` 跟著縮。

→ 套用 [patches/02-pruning-compression.patch](patches/02-pruning-compression.patch)
(在原本 in-dim 剪枝迴圈後加 ~50 行)

**驗證**

```python
ckpt_full, ckpt_compressed = oto.construct_subnet(out_dir="./out")
m = torch.load(ckpt_compressed)
sum(p.numel() for p in m.parameters())  # sp=0.5 約 80M(原 110M)
m(input_ids=..., attention_mask=...)    # forward 通過,不再 shape error
```

⚠️ **警告 — 這個檢查通過 ≠ 模型健康**。修了 #1 + #2 之後 subnet 能組、能 forward,但 F1 大概只有 4(隨機水準)。還要加修法 #3 才會真的開始學。

---

## 修法 #3 — 量化 runtime patch(`quant_fix.py`)

**性質**: runtime monkey-patch,**完全不動原始碼**。在訓練腳本裡 import,在量化包裝之前呼叫。

**症狀**

訓練 loss 卡在 ~5.93 不動,val F1 從 epoch 1 就 < 30,**剪枝還沒開始**就已經這樣。沒當機,就是不會學。

**根因**(兩個問題,要一起修)

### 3a. `q_m_act` 用 weight 統計初始化 — **upstream bug**

`initialize_quant_layer` 用 `max|weight|` 設 activation quantizer 的範圍 `q_m_act`。但 BERT:

| | 典型量級 |
|---|---|
| weight     | ~0.05 |
| activation(LayerNorm 後) | ~5–50 |

結果 `q_m_act / max|x|` 中位數 0.33,最糟 0.004。**75% 的 activation 在量化之前就被 clip 到 ±q_m_act** → 大量資訊直接損失。

CV model 的 weight 跟 activation 量級剛好接近 → 蒙混過去。BERT 把 bug 暴露出來。

### 3b. STE 前後向用不同的 clip 邊界 — **設計脆弱點**

`SymQuantizerNonLinear.backward` 在 `|x| >= clip_val`(寫死 2.0)才把梯度歸零。但 forward 在 `|x| >= q_m_act` 就 saturate。3a 把 `q_m_act` 壓到 << 2 之後,`q_m_act < |x| < 2` 這段區間:

- forward: 飽和(輸出常數)
- backward: identity(梯度直接通過)

梯度量級差 4 個數量級。Layer-0 dense `|grad|/|W|` = 13.5,layer-11 query `|grad|/|W|` = 8.8e-4。optimizer 處理不來。

這個寫法是 PACT-style 的常見簡化,很多量化框架都這樣做 — 在 `q_m > clip_val` 時是對的。bug 3a 把這個前提打破了。

**修法** — `quant_fix.py` 提供兩個 function:

```python
from quant_fix import apply_ste_fix, calibrate_quant_layers

# 在 model_to_quantize_model 之後、訓練之前呼叫一次:
apply_ste_fix()                      # 修 3b: backward 在 |x|>=q_m 歸零
calibrate_quant_layers(model, calib_loader, num_batches=8)  # 修 3a: 用真實 activation 重設 q_m_act
```

`apply_ste_fix()` monkey-patch `SymQuantizerNonLinear.backward`(以及 linear 版本),讓 backward 跟著 `q_m` 走,不再用寫死的 2.0。

`calibrate_quant_layers()` 暫時關掉 activation quant 跑幾個 forward batch,hook 每層收集 `max|x|`,然後設 `q_m_act = max|x| * 1.05`,再從 `num_bits` 推 `d_quant_act`。

**不動原始碼**。要還原只要不呼叫這兩個 function。

**驗證**(剪枝開始前的 sanity baseline)

| 設定 | 預期 F1 |
|---|---|
| Quant wrap, 不修, 純 AdamW(sp=0)             | ~27 ❌ |
| Quant wrap **+ fix**, 純 AdamW, 16-bit       | ~88 ✅ |
| Quant wrap **+ fix**, GETA, sp=0, 16→4-bit   | ~74 |

如果 dense 階段 F1(epoch 2-3,剪枝還沒啟動)能到 86-87 區間,就是修對了。

---

## 為什麼這些只在 BERT + joint pipeline 上會中

| 修法 | CV model 為什麼蓋住了 bug |
|---|---|
| #1 | 沒有 QADG head/trunk 重疊 — dependency group 都是單一模式 |
| #2 | 沒有 `QuantizeLinear` 包裝 → graph trace 乾淨 → in-dim sync 自然能用 |
| #3a | weight 跟 activation 量級接近 → 用 weight 設 `q_m_act` 剛好沒事 |
| #3b | bug #3a 沒爆 → forward 都待在 `clip_val=2.0` 內 → STE 不一致永遠不暴露 |

每個 fix 對應的都是「在 CV 上成立、在 attention-heavy 量化模型上不成立」的隱性假設。這些 bug 不是 BERT 專屬 — 任何類似架構(其他 transformer、joint quant + prune)都很可能踩到。

---

## 套用順序

1. 先套 patch #1(`graph.py`)— 不修這個訓練根本起不來
2. 再套 patch #2(`pruning_compression.py`)— 不修這個訓完 `construct_subnet()` 會炸
3. 在訓練腳本加 `quant_fix.py` 的兩個呼叫 — 不修這個能訓但學不到東西

#1 + #2 一次 `git apply` 套完就忘了它。#3 住在你的訓練腳本裡(本 repo 的 `src/run_experiment.py` 已經寫好了)。
