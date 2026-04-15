# GETA 最小使用流程（白話 10 句版）

目標：讓未來的我三秒看懂「使用者怎麼呼叫 GETA 完成一次壓縮」。不講論文，不講 QADG/QASSO 內部。

1. 準備一個跑得通的 PyTorch 模型，並放到 GPU 上。
2. 準備一個 `dummy_input`，shape 與 dtype 要和真正訓練時一筆輸入一致（transformer 要注意 `input_ids`/`attention_mask` 這種 dict input 的處理）。
3. 用 `model_to_quantize_model(model, quant_mode=QuantizationMode.WEIGHT_AND_ACTIVATION)` 把模型內部的 Linear/Conv 換成量化版本。
4. 把包好的 model 丟進 `OTO(model=..., dummy_input=...)`，拿到 `oto` 物件。GETA 會在這步把 dependency graph 建好。
5. 呼叫 `oto.geta(variant="adam", lr=..., lr_quant=..., target_group_sparsity=..., bit_reduction=..., min_bit_wt=..., max_bit_wt=...)` 拿到一個 optimizer。
6. 之後就是**一般 PyTorch 訓練迴圈**：`optimizer.zero_grad()` → forward → `loss.backward()` → `optimizer.step()`。不要再套第二層 optimizer。
7. 訓練中 optimizer 會自己同時推動 group sparsity（剪枝）與 bit allocation（量化），使用者什麼都不用調。
8. 訓練結束後呼叫 `oto.construct_subnet(out_dir='./compressed/')`，GETA 會把最終剪枝 + 量化後的子網寫到 `out_dir`。
9. 匯出的子網再用原本的 eval pipeline 評測（對我們就是 SQuAD EM/F1），和 baseline 比較掉點幅度。
10. 若要 HF 可讀格式，`construct_subnet` 可能接受 `export_huggingface_format=True`（待驗證）；否則要自己寫轉換。

**關鍵原則**：GETA 取代的是 optimizer，不是 training loop 本身。任何原本能 `loss.backward()` 的流程都可以接上。
