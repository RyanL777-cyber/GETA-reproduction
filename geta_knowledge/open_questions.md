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

## 已驗證
*(尚無)*
