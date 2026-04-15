# BERT on SQuAD — Baseline 實驗小日誌

## 目前進度
- [x] 建立最小 baseline script (`baseline_bert_squad.py`)
- [x] 修掉四個實質 bug（見下方）
- [x] Smoke test 在本機跑通（64 train / 32 val，F1 ≈ 9.76，線路驗證用）
- [x] Server 上用 full data 重跑（87599 train / 10570 val，2 epoch）
- [x] **Phase 1 結案：EM = 81.14 / F1 = 88.50，可進 Phase 2（GETA 整合）**

## Baseline 結果（2026-04-15，server full run）
| 項目 | 數值 |
|---|---|
| train samples | 87599（full SQuAD v1.1 train）|
| val samples | 10570（full SQuAD v1.1 validation）|
| epochs | 2 |
| batch size | 12 (train) / 32 (eval) |
| learning_rate | 3e-5 |
| weight_decay | 0.01 |
| total steps | 14754 |
| train_runtime | 6504s ≈ 108 min |
| throughput | 27.2 samples/s, 2.27 steps/s |
| train_loss (avg) | 0.978 |
| loss 起 → 終 | 4.61 → 0.68（健康收斂，無 NaN/爆炸）|
| grad_norm 峰值 | ~23（未爆炸）|
| **Exact Match** | **81.14** |
| **F1** | **88.50** |

**對照**：HuggingFace 官方 `bert-base-uncased` + SQuAD v1.1 標準 baseline 約 EM ~80.8 / F1 ~88.5，本次結果完全貼合。這驗證了：
- preprocess 正確（answer span 對齊無誤）
- post-process 正確（offset_mapping 還原 answer 無誤）
- metric 計算正確（squad metric 輸出合理）
- 沒有資料流 bug

**Phase 1 驗收 checklist**
- [x] 成功載入 bert-base-uncased
- [x] 成功載入 SQuAD
- [x] training preprocess 正常
- [x] validation preprocess 正常
- [x] script 可以正常 train
- [x] script 可以正常 eval
- [x] 可正確輸出 EM / F1
- [x] 沒有阻塞性錯誤
- [x] F1 ≥ 85（自訂硬門檻，實際 88.50）
- [x] baseline 結果已紀錄（本節）

## Phase 2 前的 open items
接 GETA 之前應該回頭確認的事項（目前都是 HF 預設或假設值，非 paper-confirmed）：
- paper 是否用 `bert-base-uncased` 還是 cased？（目前假設 uncased）
- paper 的 batch size / epochs / lr / warmup / weight_decay 是否與本 baseline 一致？
- paper 的 max_length / doc_stride 是否 384 / 128？
- seed 還沒設 → 若 GETA 實驗要報 mean±std 需要固定
- 本次結果是 single run，沒跑多 seed 誤差

## 當前設定（script 預設，smoke / full 只差 SAMPLES）
| 項目 | 值 | 來源 |
|---|---|---|
| model | `bert-base-uncased` | **已知事實**（phase 1.md 指定）|
| tokenizer | `bert-base-uncased` | **已知事實** |
| head | `AutoModelForQuestionAnswering` | **已知事實** |
| dataset | `squad` (HF datasets) | **已知事實** |
| max_length | 384 | **假設**（HF 官方教學預設）|
| doc_stride | 128 | **假設**（HF 官方教學預設）|
| learning_rate | 3e-5 | **假設** |
| train batch size | 12 | **假設** |
| eval batch size | 32 | **假設** |
| epochs | 2 | **假設**（full run 用）|
| weight_decay | 0.01 | **假設** |
| TRAIN_SAMPLES | 64（smoke）/ None（full）| 切換 |
| VAL_SAMPLES | 32（smoke）/ None（full）| 切換 |
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
