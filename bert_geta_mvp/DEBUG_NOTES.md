# Phase 3 Smoke Test — 問題分析

## 目前進度

| 階段 | 狀態 | 說明 |
|------|------|------|
| M1: 載入 BERT | ✅ **通過** | 成功從 HuggingFace 載入 `bert-base-uncased` 和 tokenizer |
| M2: 量化包裝 | ✅ **通過** | 成功用 `model_to_quantize_model` 包裝成可量化版本（轉換 73 層） |
| M3: 建構 OTO | ✅ **通過** | 成功建構依賴圖（1102 nodes, 1177 edges） |
| M3.5: 檢查 model 參數重複 | ✅ **通過** | 模型本身無參數重複（637 unique parameters） |
| M3.6: 檢查 OTO param_groups 重複 | ✅ **診斷完成** | **找到 314 個參數在多個 group 中** |
| M4: 建構 GETA optimizer | ❌ **失敗** | `ValueError: some parameters appear in more than one parameter group` |
| M5: 訓練循環 | ⏸️ **未測試** | - |

---

## 🎯 根本原因已確認（方向 B）

### M3.6 的診斷結果

```
FOUND 314 parameters in multiple groups!
  param_id=130275592283600: appears in groups 
    [(1, 'bert.encoder.layer.1.attention.self.query.weight'), 
     (23, 'bert.encoder.layer.1.attention.self.query.weight')]
  param_id=130275592283360: appears in groups 
    [(1, 'bert.encoder.layer.1.attention.self.query.bias'), 
     (23, 'bert.encoder.layer.1.attention.self.query.bias')]
  ...（還有 311 個這樣的重複）
```

### 問題的根本原因

**GETA 的計算圖構建邏輯有 bug**：

1. **層級結構**：GETA 為 BERT 的每一層建構了 `OperatorNodeGroup`
   - group[0..22]: 各層的 query/key/value/dense/LayerNorm 等
   - group[23]: `BertForQuestionAnswering.qa_outputs` （最後的分類頭）

2. **參數重複註冊**：某些參數（尤其是 encoder 層的 attention weights）被同時加入：
   - ✗ 它所在層的 node group（group[1], group[3], ...）
   - ✗ 一個 auxiliary / merged group（group[23]）
   
   導致同一個 `nn.Parameter` 物件被多個 `param_group['params']` 列表引用

3. **PyTorch optimizer 拒絕**：
   ```python
   # torch/optim/optimizer.py line 1050
   for param_group in param_groups:
       for param in param_group['params']:
           if id(param) in params_seen:  # ← 檢測到重複！
               raise ValueError("some parameters appear in more than one parameter group")
   ```

---

## 診斷過程

### M3.5 排除了 model 端的問題  ✓
- 637 unique parameters，0 duplicates in `model.parameters()`
- 結論：M2 的量化包裝沒有複製參數 ✓

### M3.6 追蹤到 GETA 內部  ✓
- 27 個 param_groups 總共
- 同一個參數被 1~2 個不同 group 引用
- 314 個參數有重複

---

## 為什麼 GETA 會這樣做

GETA 的計算圖在「追蹤依賴」時可能採用了「為每層建 group」 + 「建 auxiliary group 來處理依賴」的策略，但在合併 param_groups 時沒有去重。

### 推測的 GETA 內部流程

1. 遍歷模型的每個 operator（Linear, LayerNorm, ...）
   - 為每個 operator 建 node → 再聚合成 node_group
   - 每個 node_group 對應一個 param_group

2. 建構依賴圖（QADG）時發現某些層有共同的依賴
   - 建 auxiliary node_group 來管理這些依賴
   - **但在 aux group 當中加入參數時，沒有從原始 param_groups 中移除**

3. 結果：同一個參數同時在原始 group 和 aux group 中

---

## 解決方案

### 方案 A：修改 GETA 的 `get_param_groups()` 去重 ✅ (建議)

在 `only_train_once/graph/graph.py` 或 `node_group.py` 的 `get_param_groups()` 方法中，加入去重邏輯：

```python
def get_param_groups(self):
    param_groups = [...]  # 現有邏輯
    
    # 去重：track all param ids seen so far
    seen_param_ids = set()
    deduplicated_groups = []
    
    for pg in param_groups:
        new_params = []
        new_p_names = []
        for p, name in zip(pg['params'], pg['p_names']):
            pid = id(p)
            if pid not in seen_param_ids:
                new_params.append(p)
                new_p_names.append(name)
                seen_param_ids.add(pid)
        
        if len(new_params) > 0:
            pg['params'] = new_params
            pg['p_names'] = new_p_names
            deduplicated_groups.append(pg)
    
    return deduplicated_groups
```

### 方案 B：在 smoke.py 中繞過 bug

在呼叫 `oto.geta()` 前，手動去重 param_groups：
- 讀取 `oto._graph.get_param_groups()`
- 去除重複參數
- 改寫 `oto._graph.param_groups`（如果可以）
- 再呼叫 `oto.geta()`

但這需要從外部修改 GETA 的內部狀態，比較「hacky」。

### 方案 C：向 GETA 團隊提 issue

提供這份診斷報告給 GETA 開發者，他們應該會優先 fix。

---

## 建議下一步

1. **確認 GETA 版本**：檢查 clone 下來的 GETA repo 是否已有 fix（在 graph.py 或 node_group.py 搜 deduplicate/dedup）
2. **嘗試方案 A 或 B**：如果 GETA 還沒 fix，可以用上面的去重邏輯改 smoke.py
3. **如果方案 A/B 行得通**：smoke test 應該能完整跑完，進入 M5（訓練循環）

自訂 training loop 換掉的是「**如何呼叫 optimizer**」，不是「optimizer 被建立時 param group 衝突」這個問題。換 loop 也建不出 optimizer，無解。只有當 Phase 4 要用 HF Trainer 注入失敗時，才該考慮自寫 loop。現階段放一邊。

### 關於 GETA 官方 tutorial 的事實

Tutorials 全是 CV（ResNet18、CARN、qVGG7），**沒有任何 BERT/transformer 範例**。不用去翻了。sanity_check 裡有 Phi2/Llama 這類 HF transformer 可以參考接法，但它們都是 causal LM 不是 QA，只能看「HF kwargs 怎麼接」這個面向。詳見 `geta_knowledge/repo_map.md`。

### 建議執行順序

1. 跑一次有 M3.5 的 smoke.py，拿到 `dup` 數字。
2. **若 `dup > 0`**（方向 A）：貼前 10 個重複名字給 Claude，我指示怎麼改 quant wrapping 或在 M3 前手動 dedupe。
3. **若 `dup = 0`**（方向 B）：
   - 先試 `oto.hesso(variant="adam", lr=3e-5, target_group_sparsity=0.5)` 看是否同樣失敗。
   - 無論成敗都貼 log，再決定下一步（改 GETA source 或 workaround）。
4. **不要**再往「寫自訂 loop」或「提 GitHub issue」的方向走，先把根因找到。

