# GETA Param Group 重複問題 — 調查指南

**建立時間**：2026-04-15 19:30  
**確認狀態**：✓ 根本原因已診斷，M4 症狀已重現，M5 新症狀出現

---

## 問題總結

BERT × GETA smoke test 在建構 GETA optimizer 時失敗，根本原因是 **OTO 的 `get_param_groups()` 返回了含有重複參數的參數群組列表**。

### 三個關鍵事實

| 項目 | 狀態 | 意義 |
|------|------|------|
| 模型參數重複検 | ✓ 無重複（637 unique） | 問題不在 M2 的量化包裝 |
| OTO param_groups 重複検 | ✓ **314 個重複** | **根本原因在 GETA** |
| M4 oto.geta() | ✗ ValueError | 因為重複參數 PyTorch optimizer 拒絕 |

---

## 症狀 1：M4 崩潰（已解決）

**錯誤**：
```
ValueError: some parameters appear in more than one parameter group
  File "torch/optim/optimizer.py", line 1050, in add_param_group
```

**根本原因**：同一個 `nn.Parameter` 被塞進了多個 param_group 的 `params` 列表

**具體例子**（從 M3.6 診斷）：
```
param_id=133602790209840: appears in 
  - group[1]  ('bert.encoder.layer.1.attention.self.query.weight')
  - group[23] ('bert.encoder.layer.1.attention.self.query.weight')
```

同一個權重被登記了兩次 → PyTorch optimizer 的 `add_param_group()` 檢查失敗

---

## 症狀 2：M5 新錯誤（待調查）

**錯誤**（在去重後出現）：
```
RuntimeError: shape '[768, -1]' is invalid for input of size 1
  File "geta/only_train_once/transform/tensor_transform.py", line 72
    return tensor.view(num_groups, -1)
```

**發生位置**：
- `magnitude.py:9` 呼叫 `tensor_transformation_param_group(param.data, p_transform, param_group)`
- 最終在 `tensor_transform.py:72` 試著把某個參數 reshape 成 `(768, -1)` 形狀
- 但該參數只有 1 個元素 → 無法 reshape

**推測原因**：
- 去重後某個 param_group 的參數少了（可能從 768 個變成 1 個）
- 但 param_group 的 metadata 沒有更新（`num_groups` 仍是 768）
- GETA 在計算 importance score 時依賴 `num_groups` 而不是實際的參數個數

---

## 調查方向

### 方向 A：確認 param_group 重複的根本成因

**檔案**：`only_train_once/graph/graph.py` 和 `node_group.py`

**要調查的問題**：
1. **`get_param_groups()` 返回值結構**
   - 是否 `param_groups` 涵蓋了所有 node groups？
   - 是否有「auxiliary node groups」會重複指向相同的參數？

2. **層級建構邏輯**
   - BERT 有 12 層 encoder，對應 12 個 node groups（大概）
   - 最後還有 `qa_outputs` 的 node group（group[23]）
   - **為什麼 encoder 層的參數也被 qa_outputs group 引用？**

3. **搜尋策略**
   ```bash
   grep -n "auxiliary\|auxilary" geta/only_train_once/graph/graph.py
   grep -n "auxiliary\|auxilary" geta/only_train_once/graph/node_group.py
   ```
   查看是否有 auxiliary group 被建構，以及如何添加參數。

**期望發現**：可能在 `graph.py` 的某處有這樣的邏輯：
```python
for node_group in node_groups:
    if node_group.has_auxiliary:
        aux_group = create_auxiliary_group(node_group)
        # ← 這裡可能沒有正確去重，或者從 node_group copy 了參數
        param_groups.extend(aux_group.get_param_groups())
```

---

### 方向 B：確認去重應該在哪一層做

**問題**：`get_param_groups()` 返回時是否應該已經確保無重複？

**調查步驟**：
1. 看看有沒有其他地方（如 `build_pruning_dependency_graph`）也會建構 param groups
2. 檢查 GETA 的 tutorial 是否有相同的問題（只是他們沒測過 BERT）
3. 查看是否有針對「單個層內部的量化層」的特殊處理（量化層會增加 `d_quant_wt`、`q_m_wt` 等輔助參數）

---

### 方向 C：確認 M5 錯誤是否是 M3.7 去重的副作用

**關鍵問題**：去重後 param_group 的元數據是否保持一致？

**應檢查的元數據**：
```python
param_group_keys = [
    'id', 'num_groups', 'is_prunable', 'is_auxiliary',
    'p_names', 'params', 'p_transform', 'op_names',
    'node_ids', 'auxiliary_ngs'
]
```

**檢查清單**：
- [ ] `len(params)` vs `num_groups`：去重後是否還匹配？
- [ ] `len(p_names)` vs `len(params)`：是否始終相等？
- [ ] `p_transform` 長度：是否與 `params` 長度相同？
- [ ] `num_groups` 如何計算：是否基於 `len(params)` 還是固定值？

**如何測試**：
```python
# 在 M3.7 去重完成後加這段
for pg_idx, pg in enumerate(deduplicated):
    num_params = len(pg['params'])
    num_groups = pg.get('num_groups', -1)
    num_p_transform = len(pg.get('p_transform', []))
    print(f"group[{pg_idx}]: params={num_params}, num_groups={num_groups}, p_transform={num_p_transform}")
    if num_groups != num_params:
        print(f"  ⚠️ MISMATCH: num_groups ({num_groups}) != actual params ({num_params})")
```

---

### 方向 D：確認是否需要同時修復 `num_groups`

如果 M3.7 的去重導致 `num_groups` 不匹配，解決方案可能是：

```python
# 在去重時同步更新 num_groups
for pg in deduplicated:
    # 重新計算 num_groups 基於向量維度 + 分組策略
    params = pg['params']
    # GETA 的 num_groups 定義是什麼？通常是：
    #   - 對 Linear: num_groups = output_features
    #   - 對 Conv2d: num_groups = out_channels
    #   - 或者某種自動分組？
    pg['num_groups'] = len(params)  # ← 這可能是錯的
```

**需要確認**：`num_groups` 的定義和計算方式

---

## 調查工件和命令

### 1. 快速重現 param_group 重複

```bash
cd /home/h24116081/GETA\ reprocution/GETA-reproduction/bert_geta_mvp
python3 -c "
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from only_train_once import OTO
from only_train_once.quantization.quant_model import model_to_quantize_model
from only_train_once.quantization.quant_layers import QuantizationMode

model = AutoModelForQuestionAnswering.from_pretrained('bert-base-uncased').cuda()
model = model_to_quantize_model(model, quant_mode=QuantizationMode.WEIGHT_AND_ACTIVATION)

tok = AutoTokenizer.from_pretrained('bert-base-uncased')
enc = tok('What?', 'Answer', return_tensors='pt', max_length=384, padding='max_length', truncation='only_second')
dummy = tuple(v.cuda() for v in enc.values())

oto = OTO(model=model, dummy_input=dummy)

# 檢查 param_groups
pgs = list(oto._graph.get_param_groups().values())
print(f'Total groups: {len(pgs)}')

seen = {}
for pg_idx, pg in enumerate(pgs):
    for p in pg['params']:
        pid = id(p)
        if pid in seen:
            print(f'DUPLICATE: param in group[{seen[pid]}] AND group[{pg_idx}]')
        seen[pid] = pg_idx
"
```

### 2. 查看 GETA 源代碼中的 auxiliary group 邏輯

```bash
cd /home/h24116081/GETA\ reprocution/GETA-reproduction/geta

# 搜尋 auxiliary 引用
grep -r "auxiliary" only_train_once/graph/ --include="*.py" | head -20

# 搜尋 param_groups 的構建
grep -r "get_param_groups" only_train_once/graph/ --include="*.py" -A 10
```

### 3. 檢查官方 tutorial 是否有類似問題

```bash
cd /home/h24116081/GETA\ reprocution/GETA-reproduction/geta

# 查看 sanity_check 中是否用過 transformer
ls -la sanity_check/
file sanity_check/*.py | grep -i transformer

# 看看有沒有 BERT 或 Llama 的測試
grep -r "BERT\|Llama\|Phi" sanity_check/ tutorials/
```

---

## 預期發現

根據現有信息，你可能會發現：

1. **方向 A 最可能**：
   - `graph.py` 的 `get_param_groups()` 在處理 auxiliary node groups 時沒有檢查重複
   - Auxiliary groups 直接 copy 了原始 node groups 的參數

2. **解決方案**：
   - 在 `get_param_groups()` 回傳前加入去重邏輯
   - 同時更新相關的 metadata（`num_groups` 等）

3. **為什麼只影響 BERT**：
   - Tutorial 的 QVGG7 / Phi2 / Llama 可能：
     - 沒有產生 auxiliary groups，或者
     - Auxiliary groups 的設計在 BERT 上有特殊行為

---

## 報告建議

當你調查完後，建議向 GETA 官方提交的 issue 包含：

```markdown
## Title
Parameter duplication in `get_param_groups()` for BERT-based models

## Description
When using GETA with quantized BERT (via `model_to_quantize_model` + `OTO`), 
the optimizer fails with "some parameters appear in more than one parameter group".

## Reproduction
[粘貼快速重現命令]

## Expected
OTO param_groups should contain unique parameters without duplication.

## Actual
314 parameters appear in multiple groups (e.g., group[1] AND group[23]).

## Root Cause
[根據你的調查填入]
```

---

祝調查順利！有任何發現可以隨時更新這份文件。
