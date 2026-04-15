# Open Questions

Phase 2 接 GETA 之前、或實作中會撞到的**還沒驗證**的點。每驗證一項就移到底下的「已驗證」區，並在原條目標 ✅ + 日期。

## 待驗證
1. `OTO.__init__(dummy_input=...)` 是否支援 dict 形式的輸入？HF transformer 預設吃 dict，但 tutorials 全是 tensor。
2. `model_to_quantize_model` 能否正確遞迴替換 HF `BertForQuestionAnswering` 內所有 Linear？LayerNorm / Embedding 的行為？
3. `OTO.geta(...)` 的完整 kwargs list（README 只給 7 個，實際可能更多，例如 `weight_decay` / `pruning_steps` / `warmup_steps`）。
4. `construct_subnet(...)` 對 HF 模型是否支援 `export_huggingface_format=True`？若不支援，匯出格式是什麼？
5. HF `Trainer(optimizers=(oto_opt, None))` 注入 GETA optimizer 能否正常工作？accelerate wrapping 會不會破壞 GETA 的內部狀態？
6. GETA 訓練期間的 loss 曲線是否會明顯高於 baseline？paper 應該有 ablation。
7. Paper Table 3 中 BERT/SQuAD 的實際超參（target_group_sparsity、bit_reduction、epochs、lr）是多少？
8. 量化後的 F1 / EM 與 baseline 差距是多少（paper 的 reference 數字）？
9. **[Phase 3 留下的隱憂 — 已部分修復]** `Graph.get_param_groups()` 在 BERT-base 上會讓 **314 個 param**（不是原本估的 32）跨 group 重複（26 groups 合計 669 個 param slot vs model 637 unique params）。目前 workaround 是在 `geta/only_train_once/graph/graph.py` 的 `get_param_groups()` 末尾加全域 `seen_param_ids` dedup。**已確認（2026-04-16）**：(a) 重複是 QADG 的**設計**——Q/K/V 同時屬於 per-layer head 群和 residual trunk 群，用不同 p_transform 做兩種 prune 決策，語意正確；(b) 初版 dedup 用 id 字典序排序，**layer.0 的 head 群被整組清空**（id 開頭 `node-765` > trunk 群 `node-663`），layer 1-11 正常。**已修復**：dedup 改成按群大小遞增排序，小群優先（`geta/only_train_once/graph/graph.py:1371`），完整事件記錄 `bert_geta_mvp/bug_analysis/INCIDENT_REPORT.md`。**Phase 4 仍要盯**：正式訓練的 EM/F1 vs paper Table 3，若差距 >1 分需考慮 QADG/optimizer 分離重構。

## 已驗證
- ✅ 2026-04-16 Q1：`OTO(dummy_input=...)` **不**支援 dict — BERT forward 收到 dict 會 call `input_ids.size()` 爆掉。解法是用 tuple `(input_ids, attention_mask, token_type_ids)` 按位置傳。見 `bert_geta_mvp/smoke.py` M3。
- ✅ 2026-04-16 Q2：`model_to_quantize_model(..., WEIGHT_AND_ACTIVATION)` 在 `BertForQuestionAnswering` 上成功轉換 **73 層**，M3 OTO graph 建成（1102 nodes / 1177 edges），M5 forward+backward+step 通過。LayerNorm/Embedding 沒爆。
- ✅ 2026-04-16 Q3（部分）：`oto.geta(variant, lr, lr_quant, target_group_sparsity, bit_reduction, min_bit_wt, max_bit_wt)` 這 7 個 kwargs 在 BERT 上足以建構 optimizer（搭配 Q9 的 dedup workaround）。其餘 kwargs 未測。
