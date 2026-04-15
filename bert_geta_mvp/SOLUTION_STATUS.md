# 方案 B 測試結果（2026-04-15）

## 進展概況

| 階段 | 狀態 | 備註 |
|------|------|------|
| M1-M3 | ✅ 通過 | BERT 加載、量化、OTO 建成功 |
| M3.5-M3.7 | ✅ 診斷+去重 | 發現 314 個重複參數，去重邏輯已加入 |
| M4 oto.geta() | ✅ 通過 | 去重後 GETA optimizer 成功建立！ |
| M5 訓練迴圈 | ❌ 新錯誤 | `RuntimeError: shape '[768, -1]' is invalid for input of size 1` |

---

## 方案 B 的成功和局限

### ✅ 成功之處
1. **診斷精確**：找到了 314 個在多個 param_group 中的重複參數
2. **去重有效**：去重後 M4 成功建立 GETA optimizer（從 27 個 groups 保留 26 個）
3. **證實根因**：這確實是 GETA 的 bug，不是 BERT 或量化層的問題

### ❌ 局限性
方案 B 只做了「表面去重」（去除 params 列表中的重複），但 GETA 內部的許多計算依賴於 param_group 的元數據（如 `num_groups`）與實際參數的一致性。

**M5 的新錯誤根因**：
```
RuntimeError: shape '[768, -1]' is invalid for input of size 1
```
- 在 `tensor_transformation.py` 的 `basic_transformation(tensor, num_groups)` 中
- 嘗試把形狀為 (1,) 的張量 reshape 成 (768, -1)
- 這表示某個 param_group 的 `num_groups=768`，但實際只有 1 個元素的參數

**原因**：去重後某個 param_group 可能變成了：
- 原本有 768 個元素、被分成 768 個 groups（每 group 1 個元素）
- 去重移除了 767 個重複的參數
- 只剩 1 個參數，但 `num_groups` 仍舊是 768
- GETA 計算 importance score 時試著 reshape 這  1 個元素的參數到 (768, -1) 形狀 → 崩潰

---

## 結論

### 真正的解決方案應該是：
1. **GETA 官方修復**：在 `only_train_once/graph/graph.py:get_param_groups()` 中加入去重邏輯
2. **或者修改更深層的邏輯**：在構建 auxiliary node groups 時，確保不會把相同的參數放入多個 group

### 方案 B 的侷限性：
- 去重本身有效，但 GETA 內部對參數結構的假設被破壞
- 需要同時更新 `num_groups`、重新計算 `p_transform` 的對應關係等
- 這樣的修改實在太複雜，不值得在 smoke.py 中手工實作

### 推薦的下一步
1. **向 GETA 團隊報告**：提供這份診斷（M3.6 和本備忘）
2. **等待官方修復**，或者
3. **改用 `oto.hesso()` 只做 pruning（不量化）**試試看是否是量化特有的 bug

---

## 代碼位置

已修改的方法：
- M3.5：檢查 model 端的參數重複 ✓
- M3.6：檢查 OTO 端的參數重複，診斷根因 ✓
- M3.7：嘗試去重（部分有效，但不完全）⚠️

