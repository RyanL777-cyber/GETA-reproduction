# Phase 3 Smoke Test — 問題分析

## 目前進度

| 階段 | 狀態 | 說明 |
|------|------|------|
| M1: 載入 BERT | ✅ **通過** | 成功從 HuggingFace 載入 `bert-base-uncased` 和 tokenizer |
| M2: 量化包裝 | ✅ **通過** | 成功用 `model_to_quantize_model` 包裝成可量化版本（轉換 73 層） |
| M3: 建構 OTO | ✅ **通過** | 成功建構依賴圖（1102 nodes, 1177 edges） |
| M4: 建構 GETA optimizer | ❌ **失敗** | `ValueError: some parameters appear in more than one parameter group` |
| M5: 訓練循環 | ⏸️ **未測試** | - |

---

## M4 失敗的根本原因

### 錯誤內容
```
ValueError: some parameters appear in more than one parameter group
File "torch/optim/optimizer.py", line 1050, in add_param_group
    raise ValueError("some parameters appear in more than one parameter group")
```

### 技術分析

1. **BERT 經過 `model_to_quantize_model` 後的結構變化**
   - 原始 BERT: 標準 PyTorch 模組，每層只出現一次在 `model.named_parameters()`
   - 量化後: 每個原始層被替換成 `QuantizedLinear` / `QuantizedLayerNorm` 等
   - **潛在問題**: 量化層可能複製了參數，導致同一個參數在多個地方被引用

2. **OTO 的參數群組建構邏輯**
   - `OTO(model, dummy_input)` 遍歷計算圖中的所有節點
   - 在建構 GETA optimizer 時，為「可量化」和「不可量化」的層創建不同的參數群組
   - `oto.geta(**kwargs)` 呼叫時，PyTorch optimizer 在 `add_param_group()` 時檢查：
     ```python
     # PyTorch source: optimizer.py
     for group in self.param_groups:
         for param in group['params']:
             if id(param) in params_set:  # 檢查重複
                 raise ValueError("some parameters appear in more than one parameter group")
     ```

3. **推測的問題根源**
   - 量化層的某些內部參數（如 `weight_scale`, `bias_scale` 等）可能被**重複註冊**到模型中
   - 或者 GETA 的參數群組邏輯與 BERT 的複雜層結構（12 層編碼器 × 多頭注意力 × ...）產生了不相容

---

## 為什麼 dict 輸入失敗（已修復）

**原始問題**: M3 用 dict 裝 dummy_input → OTO 把它傳給 model(dummy_input)
- BERT forward 簽名是 `forward(input_ids, attention_mask=None, token_type_ids=None, ...)`
- 接收到 dict 時，模型試圖呼叫 `input_ids.size()` → dict 沒有 `.size()` 方法

**修復**: 改用 tuple
```python
dummy_input = (
    dummy_dict["input_ids"],
    dummy_dict["attention_mask"],
    dummy_dict["token_type_ids"],
)
```
這樣 BERT 收到的是位置參數，正常解包

---

## 可能的解決方向

### 方向 1: 檢查量化層是否有參數重複
```python
# 在 M4 前加診斷代碼
all_params = list(model.parameters())
param_ids = [id(p) for p in all_params]
if len(param_ids) != len(set(param_ids)):
    print(f"WARNING: {len(param_ids) - len(set(param_ids))} duplicate parameters found")
    # 找出重複的參數
```

### 方向 2: 檢查 GETA 的參數群組
```python
# 在 oto.geta() 前加診斷
print(f"Model has {len(list(model.parameters()))} total parameters")
# 查看 oto 內部的參數計數
```

### 方向 3: 跳過量化並只測試 pruning
- 改用 `oto.hesso()` 而不是 `oto.geta()`（純 pruning，不量化）
- 看是否是量化層的問題

### 方向 4: 向 GETA 團隊提 issue
- 提供：BERT-base 量化後 + OTO initializer 的複現代碼
- 他們可能會指出是否是已知的相容性問題

---

## 架構總結

```
smoke.py 流程:
  
  M1: AutoModel → bert-base-uncased (standard PyTorch)
        ↓
  M2: model_to_quantize_model(model) → QuantizedBERT
        ↓ (層被替換 + 參數可能複製?)
  M3: OTO(model, dummy_input) → 計算圖 (1102 nodes)
        ↓ (圖建構成功，但參數引用可能有問題?)
  M4: oto.geta(...) ❌ (參數群組衝突)
```

---

## 下一步

1. **診斷**: 在 M2 後檢查參數重複情況
2. **測試**: 嘗試 `oto.hesso()` 看是否是量化的問題
3. **回溯**: 檢查 GETA 官方教程 (Tutorial 03) 是否有 BERT 的例子
4. **替代**: 考慮是否要用「自訂 training loop + 手動 optimizer.step()」而不是依賴 `oto.geta()` 的自動包裝

---

## Claude 的意見（2026-04-15 追加）

### 對目前分析的回饋

Copilot 的錯誤定位是對的（`add_param_group` 重複檢查），但**方向 1～4 順序有問題**：方向 3（官方 tutorial）和方向 4（自訂 training loop）都是「繞路」而不是「找原因」。在還不知道是 *model 本身* 還是 *GETA 自身* 出錯前，就跳去寫替代方案等於瞎猜。**應該先做最便宜的診斷，把可能性切一半。**

### 切一半的 3 行診斷（已加進 smoke.py 的 M3.5）

```python
ids = [id(p) for p in model.parameters()]
log.info(f"param_count={len(ids)}  unique={len(set(ids))}  dup={len(ids)-len(set(ids))}")
```

- **`dup > 0`** → **方向 A**：問題在 M2 的 `model_to_quantize_model`。量化包裝時沒有正確 detach 舊 weight，同一個 `nn.Parameter` 被兩個不同的 module attribute 同時指向，導致 `model.parameters()` 迭代時出現兩次。這類問題 GETA 完全不用改，只要修 quant wrapping（或在 wrapper 之後手動去重）。
- **`dup = 0`** → **方向 B**：model 本身沒有 duplicate，問題在 `oto.geta(...)` 自己的 param grouping 邏輯 — 它可能把「要剪枝的 group」和「要量化的 group」當成兩個獨立 group 同時塞進同一個 base optimizer。這才需要去讀 `only_train_once/optimizer/` 底下 GETA 的 param groups 建構程式碼。

M3.5 若 `dup > 0`，會接著印出前 10 個重複的 `(name_a, name_b)` pair，直接指出是哪一層。

### BERT 有沒有 weight tying 的補充

`BertForQuestionAnswering` **不該有 weight tying**（和 `BertForMaskedLM` 不同，QA head 是一個獨立的 `qa_outputs` Linear），所以 baseline 狀態下 `dup` 應該是 0。如果 M3.5 在 M2 *之前* 就有 dup，才要懷疑 HF `tie_weights()` 的 side effect；若 M3.5 在 M2 *之後* 才有 dup，那就鐵定是量化包裝的鍋。**更嚴謹的診斷會在 M2 前後各量一次**，這點之後可以再加。

### 方向 3（hesso fallback）什麼時候有用

只在 **`dup = 0`** 的情況下才做為「再切一半」的診斷工具：
- `hesso` 純剪枝，不量化 → 如果 hesso 也炸同一個錯 → GETA 整個 optimizer family 對 HF transformer 都有 param grouping bug，要讀 source 修。
- 如果 hesso 通 → 問題鎖定在 `geta()` 的 *joint* prune+quant grouping 那段。

### 方向 4（自訂 training loop）不該現在做

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

