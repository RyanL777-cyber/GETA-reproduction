# BERT on SQuAD — Baseline 實驗小日誌

## 目前進度
- [x] 建立最小 baseline script (`baseline_bert_squad.py`)
- [x] 修掉三個實質 bug（見下方）
- [ ] Smoke test 在本機跑通（64 train / 32 val）
- [ ] Server 上用較完整資料重跑，記錄 EM / F1
- [ ] 判斷是否可進入 Phase 2（GETA 整合）

## 當前設定（Smoke test）
| 項目 | 值 | 來源 |
|---|---|---|
| model | `bert-base-uncased` | **已知事實**（phase 1.md 指定）|
| tokenizer | `bert-base-uncased` | **已知事實** |
| head | `AutoModelForQuestionAnswering` | **已知事實** |
| dataset | `squad` (HF datasets) | **已知事實** |
| max_length | 384 | **假設**（HF 官方教學預設）|
| doc_stride | 128 | **假設**（HF 官方教學預設）|
| learning_rate | 3e-5 | **假設** |
| batch size | 8 (train & eval) | **假設** |
| epochs | 1 | **假設**，smoke test 用 |
| weight_decay | 0.01 | **假設** |
| train samples | 64 | **假設**，純線路驗證 |
| val samples | 32 | **假設**，純線路驗證 |
| seed | 未設 | **待決定** |

> 所有標「假設」的項目都不是 paper-confirmed。Phase 2 接 GETA / 對齊 Table 3 之前必須重新確認。

## 已知事實
- Phase 1 只做 baseline，不做 GETA / pruning / quantization。
- Bottom-up：只維持 `baseline_bert_squad.py`、`notes.md`、（未建立）`requirements.txt`。
- HF 標準 QA post-process：training 時 span 對不上要標到 CLS(0)；eval 時需保留 `example_id` 與 `offset_mapping` 供還原 answer。
- `Trainer.predict` 無法處理 string / None 欄位，因此預測前必須 `remove_columns(["example_id","offset_mapping"])`，但 post-process 仍要拿原版 features。

## 暫時假設（需驗證）
- 64/32 的 smoke test 足以驗證「流程通」，但 F1 不具參考價值。
- 完整 baseline 的超參與 HF 官方教學一致即可接上 GETA。論文 Table 3 的實際設定還沒查。
- `eval_strategy="no"` 只影響訓練中評估，不影響最後手動 eval + EM/F1。

## 遇到的錯誤 & 修法

### Bug 1：`compute_metrics` 是壞的死碼
**症狀**：原本定義了 `compute_metrics(eval_pred)`，但：
- 沒有傳進 `Trainer(...)`，根本不會被呼叫。
- 內部又呼叫 `trainer.predict(eval_dataset)`，如果真被呼叫會遞迴/炸掉。
- 但 `TrainingArguments` 又設 `load_best_model_at_end=True` + `metric_for_best_model="f1"`，訓練中評估時拿不到 f1，會在 epoch 結束時報錯。

**修法**：整段移除；改成「訓練期間不評估」，訓練結束後手動跑一次 eval + post-process + squad metric。符合 Phase 1 最低成功標準。

### Bug 2：`Trainer.predict(eval_dataset)` 欄位不相容
**症狀**：validation preprocess 後，`eval_dataset` 保留了 `example_id`（str）與 `offset_mapping`（list，含 None），`DefaultDataCollator` 轉 tensor 時會爆炸。

**修法**：
```python
eval_dataset_for_model = eval_dataset.remove_columns(["example_id", "offset_mapping"])
raw_predictions = trainer.predict(eval_dataset_for_model).predictions
final_predictions = postprocess_qa_predictions(eval_examples, eval_dataset, raw_predictions)
```
關鍵：post-process 仍用**原版** `eval_dataset`，因為需要 `offset_mapping` 還原 answer span。

### Bug 3：`Trainer(tokenizer=...)` 在新版 transformers 被移除
**症狀**：`TypeError: Trainer.__init__() got an unexpected keyword argument 'tokenizer'`。新版（>=4.46）把 `tokenizer=` 改名為 `processing_class=`。

**修法**：直接移除 `tokenizer=tokenizer` 參數。tokenizer 在 preprocess 階段已經用掉，Trainer 這裡只要 `DefaultDataCollator`，不需要 tokenizer。

### 載入模型時的 UNEXPECTED / MISSING 訊息（非 bug）
載 `bert-base-uncased` 進 `AutoModelForQuestionAnswering` 時會列出：
- UNEXPECTED：`cls.predictions.*`（MLM head）、`cls.seq_relationship.*`（NSP head）、`bert.pooler.*` — pre-training 用的 head，QA 任務不需要，丟掉。
- MISSING：`qa_outputs.weight/bias` — QA head 需要新建隨機初始化的 Linear，由 fine-tune 去訓練。

這是預期行為，不是錯誤。

### Bug 4：資料規模對 smoke test 來說過大
**症狀**：原本 2000/500 的 subset 在本機仍要跑幾分鐘，不適合「純驗證線路」。

**修法**：降到 64/32。完整規模留到 server 再跑。

## Pylance 靜態警告（非 bug）
- `trainer.predict(eval_dataset_for_model)` 會噴 `Dataset` vs `Dataset[Unknown]` 型別錯誤，這是 HF `datasets.Dataset` 與 torch `Dataset` 的 stub 不相容，runtime 正常，忽略即可。

## 下一步
1. 在本機跑 smoke test，確認：能 load squad、能 train 1 epoch、能印出 `{'exact_match': ..., 'f1': ...}`。
2. Smoke test 通過後，把 `TRAIN_SAMPLES` / `VAL_SAMPLES` 放大（或整份資料）丟 server。
3. 把 server 結果填回本檔「目前進度」與新建的「Baseline 結果」小節。
4. 評估是否可進 Phase 2。
