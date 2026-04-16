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

---

## Phase 4：小規模 GETA-BERT smoke test

### 目標

不是追成績，而是確認 joint training 能穩定跑。

### 任務

- 用小 subset
- 少量 epoch 或少量 steps
- 開啟 GETA 的 BERT 參數配置
- 觀察是否有 NaN、爆 gradient、爆 graph、爆 projection/pruning stage

### 驗收

- 訓練可完成
- 沒有明顯 numerical instability
- 可以得到初步 EM / F1 與 sparsity / bit 相關資訊

---

## Phase 5：正式 BERT GETA 實驗

### 目標

按論文已知設定，逐步跑 target sparsity。

### 任務

依序跑：

1. 10%
2. 30%
3. 50%
4. 70%

固定 paper-confirmed 設定：

- AdamW
- lr = 3e-5
- lr_quant = 1e-4
- epochs = 10
- bit range = [4,16]
- B=4, Kb=1, P=6, Kp=6, br=2。

### 驗收

每個 sparsity 都要有：

- EM
- F1
- 若可行則補 BOPs / relative BOPs
- 運行紀錄
- seed 與設定檔