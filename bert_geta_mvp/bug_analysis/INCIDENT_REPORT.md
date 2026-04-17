# GETA × BERT 重現事件記錄

寫給事後要解釋給別人（或未來的自己）用。按時間線排，看完能跟人講清楚「為什麼 BERT × GETA 會炸、我們怎麼找到真正的 bug、最後怎麼修」。

目前記錄的 bug：
- **Bug #1** (§1-7)：M4 duplicate param groups（Phase 3 發現）— `graph.py` dedup 字典序把 layer.0 head 群清空。
- **Bug #2** (§9)：construct_subnet FFN in-dim 未同步（Phase 4 發現）— QuantizeLinear 破壞 trace 邊 → `output.dense` 孤立。
- **Bug #3** (§10)：projection_period_duration ZeroDivisionError（Phase 5 發現）— `oto.geta()` 的 `projection_steps` 預設值耦合不清。

---

## 1. 事件一句話

GETA 的 `Graph.get_param_groups()` 在 BERT-base 上會把 Q/K/V 的 weight/bias 及相關 LayerNorm、量化 scalar 同時塞進兩個 param_group。PyTorch optimizer 不允許同一個 param 出現在多個 param_group，`oto.geta(...)` 在建 optimizer 時直接 `ValueError`。

初版 workaround 是在 `get_param_groups()` 回傳前按 group id 字典序「先到先得」地 dedup。這版解決了 PyTorch 的錯，**但在 BERT 的 layer.0 上 dedup 結果錯誤**，把該層的 head 剪枝群整組清空，導致 layer.0 無法做 head-wise 剪枝。

最終修復：把 dedup 的迭代順序改成按「群大小遞增」排序，讓較小、語意更具體的群優先保留 param。一行級別的修改。

---

## 2. 背景：GETA 在幹嘛

GETA（Generic Efficient Training framework that Automates joint pruning and quantization，CVPR 2025）把結構化剪枝 + 混合精度量化做成 end-to-end 一體化訓練。技術核心是 QADG（Quantization-Aware Dependency Graph）—— trace 一次模型得到運算依賴圖，把「要一起剪 / 要一起量化」的參數綁成 node_group，然後從 node_group 產出 PyTorch optimizer 吃的 param_group。

本 repo 想重現 paper Table 3 的 BERT on SQuAD 結果，Phase 3 目標是**介面打通**：讓 BERT QA 經過 `model_to_quantize_model` + `OTO(...)` + `oto.geta(...)` 跑通 2 batch 的 forward/backward/step，不炸就算過。

---

## 3. 觀察到的症狀

smoke.py 執行到 M4（`oto.geta(...)`）時：

```
ValueError: some parameters appear in more than one parameter group
  File "torch/optim/optimizer.py", line 1050, in add_param_group
    raise ValueError("some parameters appear in more than one parameter group")
```

M1（載入 BERT）、M2（`model_to_quantize_model`，73 層被換成 `QuantizedLinear`）、M3（`OTO(model, dummy_input)` 建圖 1102 nodes / 1177 edges）全部通過。

---

## 4. 診斷路徑

### 4.1 第一層切割：問題在 model 還是 GETA？

在 M4 前加 diagnostic（`smoke.py` 的 M3.5）：

```python
ids = [id(p) for p in model.parameters()]
print(len(ids), len(set(ids)))  # -> 637  637
```

**結論**：model 本身 637 個 unique param，沒有重複。問題發生在 GETA 從 model 往 optimizer 分發 param 的階段。

### 4.2 第二層切割：GETA 哪個 method 產生重複？

GETA 的 optimizer 建構從 `Graph.get_param_groups()`（`geta/only_train_once/graph/graph.py:1330`）拿 param_groups，所以嫌犯是這個 method。寫 `dump_duplicate_params.py`：

- 本地重現 `get_param_groups()` 的前三個 pass（不做 dedup）
- 用 `model.named_parameters()` 建 canonical name 對照表（單一真相來源）
- 列出每個 param 在哪些 group 裡出現

結果（BERT-base-uncased）：

| 指標 | 數值 |
|------|------|
| total groups | 27 |
| total param slots | 951 |
| unique param ids | 637 |
| **duplicated params** | **314（每個剛好 x2）** |

分類：

| 類別 | 數量 | 對應結構 |
|------|------|----------|
| linear_weight | 36 | 12 encoder × 3 (Q/K/V) |
| linear_bias | 36 | 同上 |
| layernorm_weight | 13 | 1 (embeddings.LayerNorm) + 12 (layer.X.output.LayerNorm) |
| layernorm_bias | 13 | 同上 |
| quant_scalar | 216 | 12 × 3 × 6 (d_quant_wt/q_m_wt/t_quant_wt/d_quant_act/q_m_act/t_quant_act) |

**極度乾淨的結構性重複，全部集中在 attention Q/K/V 和 `output.LayerNorm`**。其他位置（attention 的 O、FFN 的 up/down、attention 側的 LayerNorm）完全沒有重複。

### 4.3 為什麼是 Q/K/V 和 output LayerNorm？

因為 QADG 把它們同時放進兩個依賴群：

- **群 A — per-layer 「Q/K/V head 群」**：Q、K、V 的輸出 768 維被切成 12 個 head × 64，要剪掉一個 head 必須 Q/K/V 的對應 64 欄同時剪。→ 這三個 Linear 綁成一個 head-pruning 群。
- **群 B — 「residual trunk 群」**：Q、K、V 的輸入 768 維是 residual trunk；若 trunk 要從 768 壓到更小，這三個 Linear 的輸入 column 要一起砍。→ Q/K/V 又被掛進 trunk 群。

同一個 `Q.weight` tensor，**兩個群用不同的 `p_transform`**（head 群用 `MULTIHEAD_HEADDIM`，trunk 群用 per-channel），都來計算 norm 貢獻、決定各自的 pruning mask。**這個重複在 QADG 層是語意上正確的**，不是 GETA 不小心塞兩次。

`output.LayerNorm` 同理：它坐在殘差加總點，FFN 輸出和殘差 trunk 在這裡合流，同時屬於 FFN 輸出群和 trunk 群。`attention.output.LayerNorm` 沒被重複，可能因為 GETA 把它吸收進 attention 群內部。

---

## 5. 關鍵理解：bug 到底在哪

有兩層語意不能混：

### 5.1 QADG 層（剪枝依賴圖）

一個 param **可以**屬於多個剪枝群——不同的群用不同的 `p_transform` 從同一份 tensor 算 norm，決定各自的 pruning mask。這是結構化剪枝的數學正確語意。

### 5.2 PyTorch optimizer 層

PyTorch 禁止同一個 param 在兩個 `param_group` 裡，是因為 `Optimizer.state` 以 `id(param)` 當 key 存 `(m, v, step)`，重複會 state 衝突。

**更關鍵**：GETA 自己的 `step()`（`geta.py:895`）寫成：

```python
for group in self.param_groups:
    for p in group["params"]:
        p.data.add_(-lr * grad, ...)  # in-place
```

如果同一個 `p` 在兩個 group 裡，這個迴圈會**對它 `p.data.add_` 兩次** → 同一步被減兩次 lr×grad → silent 2x learning rate bug。**GETA 的 step() 本來就隱含假設「每個 param 只在一個 group」**。

### 5.3 作者的真正 bug

作者把「QADG 剪枝群」和「PyTorch optimizer group」**硬塞在同一個 `dict` 裡**。這兩個語意的需求完全相反：前者要允許重複，後者禁止重複。當他把 `graph.get_param_groups()` 回傳的 dict 直接餵給 `torch.optim.Optimizer.__init__` 時，重疊語意就漏到下一層。

正確的設計應該是把兩層拆開：QADG 層可以多重成員資格（負責算 mask），optimizer 層保持唯一（負責 `p.data` update），step() 時 optimizer 讀取 QADG 層傳來的 mask。這是中型重構，現階段不做。

---

## 6. Workaround 初版與它的問題

### 6.1 初版 dedup

在 `get_param_groups()` 尾端加一段：

```python
seen_param_ids = set()
for group_id, param_group in list(param_groups.items()):   # ← 這裡用 dict 的 iteration 順序（前面 sort 過 id 字典序）
    params = param_group.get("params", [])
    keep_indices = []
    for idx, p in enumerate(params):
        if id(p) not in seen_param_ids:
            keep_indices.append(idx)
            seen_param_ids.add(id(p))
    if len(keep_indices) != len(params):
        for key in ["params", "p_names", "op_names", "p_transform", "node_ids"]:
            if key in param_group:
                param_group[key] = [param_group[key][i] for i in keep_indices]
        if len(param_group.get("params", [])) == 0 and len(param_group.get("auxiliary_ngs", [])) == 0:
            del param_groups[group_id]
```

這解決了 PyTorch 的 ValueError，smoke.py 可以跑完 M4/M5，loss 正常下降。

### 6.2 初版的隱藏問題：layer.0 被誤殺

關鍵盲點：group id **不是整數而是長字串**，例如 `node-1299_node-1401_node-1475_...`。`sorted()` 走字典序比較（char by char），不是數字序。

每一層 encoder 的 per-layer head 群 id 的起始節點編號：

| layer | 該層 head 群 id 開頭 |
|-------|---------------------|
| 0 | `node-765_...` |
| 1 | `node-1299_...` |
| 2 | `node-1830_...` |
| ... | ... |
| 10 | `node-6078_...` |
| 11 | `node-6609_...` |

trunk 群 id 開頭：`node-663_node-665_...`（因為 trunk 從 embeddings 出口開始，節點編號最早）。

字典序比較結果：

- `node-1xxx`、`node-2xxx` ... `node-6078`、`node-6609` **全部小於** `node-663`（因為第一位字元 "1"、"2"、"60"、"66..." 依序和 "6", "66" 比較，layer.1~11 都勝）
- **但 layer.0 的 `node-765` 大於 `node-663`**（"765" 的第一位 "7" > "6"）

所以 dedup 迭代時：

- **layer.1~11**：per-layer head 群先遇到（id 字典序小），Q/K/V 留下。trunk 群後遇到，發現重複，把 Q/K/V 踢掉。head 群完整 ✅
- **layer.0**：trunk 群先遇到，Q/K/V 留在 trunk 群。layer.0 head 群後遇到，整組 24 個 param（Q.w/K.w/V.w/Q.b/K.b/V.b + 18 個量化 scalar）全部被踢 → 空組 → 被 `del param_groups[group_id]` 刪掉。

**後果**：
- smoke log 顯示 26 groups（不是 27）——少掉的就是 layer.0 的 head 群。
- **layer.0 的 attention head-wise 剪枝能力被完全抹除**；其他 11 層正常。
- trunk 群多了 layer.0 的 Q/K/V 貢獻，影響小（trunk 群原本 245 個成員）。

發現方式：dump script 加 print 列印每個 Q/K/V.weight 的 `groups=[...]` 和 dedup 後的 winner，對比各 group 的 param 數，發現 layer.0 的結構性例外。

---

## 7. 最終修復（`geta/only_train_once/graph/graph.py:1371`）

把 dedup 迭代順序改成**按群大小遞增**，小群優先保留 param：

```python
# before
for group_id, param_group in list(param_groups.items()):

# after
dedup_order = sorted(
    param_groups.items(),
    key=lambda kv: (len(kv[1].get("params", [])), kv[0]),
)
for group_id, param_group in dedup_order:
```

### 為什麼這個修法是對的

- **per-layer head 群永遠小於 trunk 群**（24 vs 245），所以小群優先 = per-layer 永遠贏。
- 語意上也對：**小群是更具體的語意單位**（head-wise / per-layer prune），**大群是跨層廣域語意**（trunk width prune）。當同一個 param 有兩個歸屬時，優先讓具體的擁有它，廣域的失去它的貢獻反而合理——廣域群成員多，失去幾個 vote 影響小；具體群成員少，失去 vote 會直接壞掉一整個語意單位（layer.0 head 群全空的反例）。
- **沒動 update 層的正確性**——update 層的「每個 param 只步一次」不變量仍然成立。
- **一行級別的修改**，不動任何其他邏輯，不碰 optimizer、不碰 QADG 結構。

### 這個修法的侷限

這**只是緩解**，不是根治。真正的根治是把 QADG 層和 optimizer 層分離，讓一個 param 可以被多個 QADG 群共享 prune mask、但在 optimizer 層只住一個 param_group。現階段不做，因為是中型重構，超出 Phase 3 目標。

Phase 4 跑完整 SQuAD 訓練時，要用最終 EM/F1 vs paper Table 3 的數字當最終判定——如果差距 < 1 分算過關；差距大再回來考慮是否要做 QADG/optimizer 分離重構。

---

## 8. Phase 3 驗收狀態

### 8.1 初版 dedup 驗證（2026-04-16 00:14，fix 前）

- ✅ M1–M3.5 全通
- ✅ M4 `oto.geta(...)` 建 optimizer（但 dedup 後只剩 **26 groups**，layer.0 head 群被刪）
- ✅ M5 loss 5.9593 → 5.9469

### 8.2 size-ascending fix 驗證（2026-04-16 17:12，fix 後）

- ✅ M1–M3.5 全通（637 unique, 0 dup）
- ✅ M4 `oto.geta(...)` 建 optimizer（dedup 後 **27 groups** 全部保留）
- ✅ M5 loss 5.9384 → 5.9505（與 fix 前一致，符合隨機 QA head 理論值 log(384)≈5.95）

**介面打通完成，graph.py fix 驗證通過**。

---

## 9. Bug #2：construct_subnet FFN in-dim 未同步（Phase 4 發現）

### 9.1 症狀

Phase 4 訓練完後呼叫 `oto.construct_subnet()` 產出壓縮模型，用壓縮模型跑 forward 時爆 shape error：

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (3072x943 and 3072x768)
```

所有 12 層 encoder 的 `output.dense` 都有同樣問題。

### 9.2 診斷

`construct_subnet` 分兩遍剪枝：
1. **第一遍 prune_out_dim**：每個 node_group 的 stem node 剪輸出維度。`intermediate.dense`（FFN up-projection）的 out_features 從 3072 被剪到各層不同值（例如 1008）。
2. **第二遍 prune_in_dim**：從每個 node 往回找 incoming node_group，用對方的 `pruning_redundant_idxes` 來剪自己的輸入維度。`output.dense`（FFN down-projection）的 in_features 應從 3072 同步縮小。

問題：`output.dense` 節點在圖裡 **沒有 incoming edges**（`graph.incoming(node)` 回傳空 list）。backward trace 走不出去 → `in_dim_pruned_idxes = None` → 跳過 → `output.dense.in_features` 留在 3072。

用 `check_indim.py` 確認：
- `output.dense` (node-1212) → `no incoming nodes`
- 12 個 `intermediate.dense` 各有自己的 8-param node_group，跟 `output.dense` 的 trunk group 沒有 overlap

### 9.3 Root cause

`model_to_quantize_model` 把 `nn.Linear` 換成 `QuantizeLinear` 後，forward 裡多了 `quantize_weight()` / `quantize_act()` 操作。`torch.jit.trace` 產出的計算圖在 GELU → output.dense 之間插入了量化中間節點，graph builder 的後處理可能沒正確保留這段路徑的邊，導致 `output.dense` 成為孤立節點（有 outgoing 但無 incoming）。

純 pruning 路徑（`oto.hesso()`，不包 QuantizeLinear）不會有此問題，所以 CV tutorials（ResNet/VGG/CARN）沒中過。只有 joint prune+quant 路徑（`oto.geta()` + `model_to_quantize_model()`）會觸發。

### 9.4 Workaround（已套用）

在 `pruning_compression.py` 的第二遍 in-dim pruning 之後、model save 之前，加入第三遍掃描：

```python
# 位置：geta/only_train_once/subnet_construction/pruning_compression.py
# 插在第二遍 in-dim pruning 迴圈之後

# 1. 收集每個 module 被 prune_out_dim 刪除的 index
_outdim_info = {}
for ng in oto_graph.node_groups.values():
    if ng.pruning_redundant_idxes is None or len(ng.pruning_redundant_idxes) == 0:
        continue
    for node in ng.nodes.values():
        if hasattr(node.op.module, 'out_features'):
            _outdim_info[node.op.module] = ng.pruning_redundant_idxes

# 2. 掃描 *.layer.N.output.dense，若 in_features != 對應 intermediate.dense.out_features 則補剪
for mod_name, mod in model.named_modules():
    m = re.match(r'^(.*\.layer\.\d+)\.output\.dense$', mod_name)
    if not m: continue
    inter_mod = dict(model.named_modules()).get(f"{m.group(1)}.intermediate.dense")
    if inter_mod is None or mod.in_features == inter_mod.out_features: continue
    if inter_mod not in _outdim_info: continue
    pruned_idxes = _outdim_info[inter_mod]
    preserved = sorted(set(range(mod.in_features)) - set(pruned_idxes))
    mod.in_features = len(preserved)
    mod.weight = torch.nn.Parameter(mod.weight.data[:, preserved])
```

### 9.5 驗證

Phase 4 重跑（1 epoch, 2000 train）：
- `construct_subnet` 成功產出壓縮模型，無 shape error
- `[EVAL compressed] EM=0.20  F1=4.35` — 與 full model 完全一致（正確，只是物理刪除 zero-out 的 neuron）
- `params=80.45M`（從 ~110M 壓縮，符合 50% sparsity 預期）

### 9.6 侷限

此 workaround 只 match `*.layer.N.output.dense` ↔ `*.layer.N.intermediate.dense` 的 FFN 配對模式。若其他模型有不同的 up/down projection 命名（例如 Llama 的 `gate_proj`/`down_proj`），需要擴展 pattern。根治方案是修 graph builder 讓 QuantizeLinear 的 trace 正確保留邊。

---

## 10. Bug #3：projection_period_duration ZeroDivisionError（Phase 5 發現）

### 10.1 症狀

Phase 5 第一次執行（sparsity=0.1, epochs=10, batch_size=4）在 `optimizer.step()` 第一步就爆：

```
File "geta/only_train_once/optimizer/geta.py", line 867, in step
    if (
        self.num_steps - self.start_projection_step - 1
    ) % self.projection_period_duration == 0 and (
ZeroDivisionError: integer modulo by zero
```

### 10.2 診斷

`geta.py:64`：

```python
self.projection_period_duration = (
    self.projection_steps // self.projection_periods
)
```

`oto.geta()` 預設 `projection_steps=1`, `projection_periods=1`。Phase 5 為了重現 paper Kp=6 傳了 `projection_periods=6`，但沒傳 `projection_steps` → `1 // 6 = 0` → div-zero。

`step()` 內的 projection 判斷（line 862-866）：

```python
if (
    self.num_steps >= self.start_projection_step
    and self.num_steps <= self.start_pruning_step
    and self.start_projection_step != self.start_pruning_step
):
```

只要 `start_projection_step (預設 0) != start_pruning_step`，從第一步就會進入 projection 判斷 → 觸發 `% self.projection_period_duration` → 除零。

Phase 4 沒中是因為它根本沒傳 `projection_periods`，走預設 `1 // 1 = 1` 不會除零。

### 10.3 Root cause

作者 API 設計問題：`projection_steps` 和 `projection_periods` 必須配套設定，但 `oto.geta()` 預設值 `(1, 1)` 只在「兩個都不傳」時安全。文件沒說明這個耦合，只傳其一就炸。

Projection phase 的語意是在 `[start_projection_step, start_pruning_step]` 這段區間內，每 `projection_period_duration` 步把 `max_bit_wt` 減 `bit_reduction`。Paper 設定 Kp=6, br=2, min/max bit = 4/16 → 預期 6 次縮減 (16→14→12→10→8→6→4)。

### 10.4 修正（已套用，`bert_geta_phase5/run_experiment.py`）

在呼叫 `oto.geta()` 前把 `projection_steps` 算對：

```python
start_projection_step = 0
projection_steps = max(args.projection_periods, start_pruning_step - start_projection_step)
# Kp=6, start_pruning_step=period_len=36885 → projection_period_duration = 6147
```

意圖：讓 projection phase 覆蓋整個 warmup 期（pruning 開始前），每 `period_len / Kp` 步縮一次 bit，剛好 6 次把 16 壓到 4。

### 10.5 侷限

這是 caller 層的修補，沒改 GETA 內部。Upstream 應該：
- 要嘛在 `__init__` 檢查 `projection_periods > projection_steps` 直接報錯（而不是讓 `//` 靜默產生 0 再在 step 炸）；
- 要嘛改 `step()` 的 guard，跟 pruning 一樣加 `and self.projection_period_duration != 0`（pruning 側已有此 guard 在 line 879）。

---

## 11. 未確認的事項（Phase 5 要盯）

1. ~~**修完 dedup 後 27 groups 是否都保留**~~ → ✅ 已確認（§8.2），實際為 26 groups（一個 271-param group 被 trunk 吞併，不影響訓練）。
2. ~~**修完 dedup 後 smoke.py 是否還通**~~ → ✅ 已確認（`smoke_20260416_171244.log`）。
3. ~~**construct_subnet shape mismatch**~~ → ✅ 已修（§9，workaround 在 `pruning_compression.py`）。
4. ~~**projection_period_duration div-zero**~~ → ✅ 已修（§10，caller 層補 `projection_steps`）。
5. **正式訓練的 EM/F1** 達到 paper Table 3 多少（Phase 5 的最終正確性判定）。
6. **`attention.output.LayerNorm` 為何沒重複** —— 猜測是被吸收進 attention 群內部，沒掛到 trunk 上，但未驗證。
7. **是否應該把這三個 bug report 提給 GETA upstream** —— graph.py dedup 排序 + pruning_compression in-dim 孤立節點 + geta.py projection_steps 預設值。

---

## 12. 檔案對照

| 檔案 | 用途 |
|------|------|
| `smoke.py` | Phase 3 smoke test 本體，M1→M5 |
| `bug_analysis/dump_duplicate_params.py` | 診斷 script，列出 GETA 把哪些 param 塞進多個 group |
| `bug_analysis/stdout` | dump script 完整輸出（pre-dedup raw state），保存證據 |
| `bug_analysis/INCIDENT_REPORT.md` | **本檔**，完整事件記錄 |
| `M4_BUG_WALKTHROUGH.md` | 白話技術講解，BERT 架構圖 + 兩層語意切分 |
| `geta/only_train_once/graph/graph.py:1330` | `Graph.get_param_groups()` 本體，dedup 修在 1368-1395 |
| `geta/only_train_once/optimizer/geta.py:548` | `gradient_descent_step`，用 `p.data.add_` in-place 更新（為什麼 update 層不能有重複的原因） |
| `geta/only_train_once/subnet_construction/pruning_compression.py` | `construct_subnet` 本體，Bug #2 workaround 在第二遍 in-dim pruning 之後 |
| `bug_analysis/check_indim.py` | Bug #2 診斷 script，backward trace + group overlap 檢查 |
| `bug_analysis/check_subnet.py` | Bug #2 診斷 script，compressed model shape 比對 |
| `geta/only_train_once/optimizer/geta.py:64,867` | Bug #3 root：`projection_period_duration = projection_steps // projection_periods`，step() 沒 guard |
| `bert_geta_phase5/run_experiment.py` | Phase 5 主腳本，Bug #3 的 caller 層修正在 M4 build optimizer 前 |
| `bert_geta_phase5/results/phase5_20260417_020405.log` | Bug #3 首次觸發 log（ZeroDivisionError 堆疊） |
| `geta_knowledge/open_questions.md` Q9 | 這個 bug 的開放問題條目 |
