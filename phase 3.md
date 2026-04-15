## Phase 3：BERT 與 GETA 的介面打通

### 目標

讓 BERT QA model 至少可以被 GETA framework 包起來並做最小 forward/backward。

### 任務

- 嘗試將 Hugging Face QA model 包成 quantized model
- 設計適用於 BERT QA 的 dummy input
- 用 dummy input 建立 OTO
- 建立 GETA optimizer
- 跑一到兩個 batch 的 smoke test

### 驗收

- 不因 graph tracing 爆掉
- 不因 input interface 爆掉
- loss 可 backward
- optimizer.step 可執行

### 這一步的主要技術風險

- Hugging Face forward 介面與 GETA 預期不一致
- attention / QA head 的 group dependency 處理
- dummy input 設計不對導致 tracing 失敗
- shape-related op 使 graph 建立失敗