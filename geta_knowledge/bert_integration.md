# BERT baseline 接上 GETA — 具體介面與地雷

Phase 2 的實作備忘。**目前還沒真的接過**，這份是根據 README + api_index 推導的，實作時會不斷更新。

## 1. Baseline 端要改的地方
目前 [bert_baseline/baseline_bert_squad.py](../bert_baseline/baseline_bert_squad.py) 用 HuggingFace `Trainer`。`Trainer` 把 optimizer 藏在內部，要接 GETA 必須：

**選項 A — 自己寫 training loop（推薦）**
- 丟掉 `Trainer`，改成純 PyTorch loop：手動建 `DataLoader`、手動 `optimizer.step()`、手動算 loss。
- 好處：optimizer 我們自己掌控，GETA optimizer 可以直接塞進來。
- 壞處：要重寫 train/eval 流程，大概多 80 行 code。

**選項 B — 用 `Trainer(optimizers=(oto_optimizer, None))`**
- HF `Trainer` 支援外部 optimizer 注入：`Trainer(..., optimizers=(opt, scheduler))`。
- 好處：preprocess/post-process/eval 完全重用 Phase 1 的程式。
- 壞處：Trainer 可能會在內部對 optimizer 做一些 accelerate 的包裝（gradient accumulation, DDP, mixed precision），GETA optimizer 是否能被正確處理**未經驗證**，是主要地雷。
- **首選嘗試這條**，若踩到雷再退回選項 A。

## 2. Dummy input 要怎麼組
BERT QA 吃的是 dict：`{"input_ids": ..., "attention_mask": ..., "token_type_ids": ...}`。`OTO(model, dummy_input)` 的 `dummy_input` 傳入 dict 應該可行，但 GETA tutorials 全是 CV 模型吃 tensor，**行為未驗證**。

保險作法：
```python
dummy_input = {
    "input_ids": torch.zeros(1, 384, dtype=torch.long).cuda(),
    "attention_mask": torch.ones(1, 384, dtype=torch.long).cuda(),
    "token_type_ids": torch.zeros(1, 384, dtype=torch.long).cuda(),
}
oto = OTO(model=model.cuda(), dummy_input=dummy_input)
```
若不吃 dict，改成 tuple 或要求使用 `torch.jit.trace` 對 dict 的支援。**先用看看再說。**

## 3. `model_to_quantize_model` 對 BERT 的可行性
README 顯示 Tutorial 3 是 VGG7，`sanity_check/` 有 Phi2/Llama 等 transformer-based 模型被測過。BERT 的 `BertForQuestionAnswering` 內部結構（embedding → encoder × 12 → qa_outputs Linear）應該全部都是 Linear / LayerNorm / Embedding，理論上 `model_to_quantize_model` 應該能包住。

**地雷**：
- LayerNorm 要不要量化？預設應該是 keep in fp。
- Embedding 要不要量化？Phi2/Llama 的 sanity_check 可參考其做法。
- `qa_outputs` 這個最後一層（2 個輸出 channel, start/end logits）的 pruning group 會很小，可能被 target_group_sparsity 忽略，**不影響**。

## 4. Optimizer 超參數初值建議
Paper Table 3 的 BERT/SQuAD 行會有官方超參，**Phase 2 開始前要去 paper 找**。目前沒拿到，先用 README 範例改：
```python
optimizer = oto.geta(
    variant="adam",
    lr=3e-5,                    # 沿用 Phase 1 baseline
    lr_quant=3e-5,              # 假設同步，後續可調
    target_group_sparsity=0.5,  # 假設 50%，paper 可能不同
    bit_reduction=2,
    min_bit_wt=4,
    max_bit_wt=16,
)
```

## 5. Construct subnet 後的 eval
GETA 匯出的子網 weight 要能被 HF tokenizer / post-process 吃回去。兩條路：
- **好的情況**：`construct_subnet(..., export_huggingface_format=True)` 能寫出標準 HF dir，之後 `AutoModelForQuestionAnswering.from_pretrained(export_dir)` 直接載入，接原本 eval pipeline。
- **差的情況**：子網是自訂格式，要手寫轉換。參考 `sanity_check/test_phi2.py` 看 HF 模型他們是怎麼處理的。

## 6. 現階段 TODO（Phase 2 啟動時按順序執行）
1. 把 BERT baseline 改成選項 B 的 Trainer 注入式（不動 preprocess / post-process）。
2. 在小 subset（回到 TRAIN_SAMPLES=64）上先跑通「GETA optimizer 接得上 BERT」這條線，不追求分數。
3. 驗證 `dummy_input` dict 形式可被 OTO 接受。
4. 驗證 `model_to_quantize_model` 能包住整個 BERT 而不崩潰。
5. 驗證 `construct_subnet` 匯出後能被 HF 模型重新載入。
6. 四步都通過才換 full data 跑真正 Phase 2 實驗。
