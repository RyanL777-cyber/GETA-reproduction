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

