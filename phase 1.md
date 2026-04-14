# Phase 1 任務簡報 — 建立 BERT on SQuAD 的 baseline 流程

## 專案背景
這個專案的最終目標，是復現 GETA 論文中 **Table 3 的 BERT on SQuAD** 實驗。

但目前還 **不進入 GETA、不做 pruning、不做 quantization**。

目前進入的是 **Phase 1：先建立純 baseline 的 BERT QA 流程**，目的是先確認任務本身、資料流程、訓練流程、評估流程都正確。

---

## Phase 1 的目標
建立一條可正常執行的 **BERT question answering baseline**，使用：

- `bert-base-uncased`
- `AutoTokenizer`
- `AutoModelForQuestionAnswering`
- `SQuAD`

這個階段的目標不是追高分，而是先確認：

1. baseline 訓練流程可以正常跑  
2. evaluation 可以正常跑  
3. 可以得到合理的 **EM / F1**  
4. 後續 GETA 接入時，有一條已知正常的 baseline 可比較  

---

## 這個階段包含什麼

### 包含
- 載入 SQuAD 資料集
- 載入 `bert-base-uncased` tokenizer
- 載入 `AutoModelForQuestionAnswering`
- 完成 training preprocessing
- 完成 validation preprocessing
- 完成 QA 的 post-processing
- 計算 EM / F1
- 建立最小可執行 baseline script
- 先用小資料做 smoke test
- 必要時再擴展到較完整資料規模

### 不包含
這些事現在**不要做**：

- GETA 整合
- pruning
- quantization
- QADG / QASSO 細節實作
- matching Table 3
- 改模型
- 改資料集
- 自行引入其他 QA backbone
- 做方法改良

---

## Phase 1 需要產出的東西
這個階段結束時，我希望至少有：

1. 一個 baseline script  
   例如：`baseline_bert_squad.py`

2. 一份可以跑通的 baseline 設定  
   至少包含：
   - model name
   - dataset
   - max length
   - doc stride
   - batch size
   - epochs

3. 一份 baseline 結果紀錄  
   至少包含：
   - train 是否正常
   - eval 是否正常
   - EM
   - F1
   - 使用的資料規模與主要設定

4. 一份簡短說明  
   說明目前 baseline 是否可作為後續 GETA integration 的基準

---

## 驗收標準
只有以下條件都成立，Phase 1 才算完成：

- 可以成功載入 `bert-base-uncased`
- 可以成功載入 SQuAD
- training preprocessing 正常
- validation preprocessing 正常
- script 可以正常 train
- script 可以正常 eval
- 可以正確輸出 EM / F1
- 沒有阻塞性的資料處理或 evaluation 錯誤
- baseline 結果已被紀錄

### Phase 1 的最低成功標準
即使不是完整資料、不是高分，只要：
- 流程通
- 訓練通
- 評估通
- EM / F1 可正確產生

就算達標。

---

## 已知固定事實
以下內容視為目前專案的固定事實，除非之後有更直接證據推翻，否則不要自行更改：

- 最終專案目標是 GETA 論文 Table 3 的 BERT on SQuAD
- 目前還沒有進入 GETA
- 目前只是在做 baseline
- baseline backbone 使用：`bert-base-uncased`
- tokenizer 使用：`bert-base-uncased`
- model head 使用：`AutoModelForQuestionAnswering`
- dataset 使用：`SQuAD`
- 目前專案仍採用 **bottom-up**，不先建立完整最終架構
- 現階段可以只維持簡單檔案，例如：
  - `baseline_bert_squad.py`
  - `notes.md`
  - `requirements.txt`

---

## 你在這個階段的角色
請你扮演 **baseline reproduction engineer**。

你的任務是：
- 幫我建立一條穩定、最小、可執行的 BERT QA baseline
- 保證流程正確
- 儘量減少不必要複雜度
- 協助我確認哪些設定是已知、哪些只是暫時假設

### 重要限制
- 不要跳到 GETA integration
- 不要現在做 pruning 或 quantization
- 不要改模型
- 不要改資料集
- 不要引入不必要的新框架
- 不要把任務擴張成 improvement project
- 優先建立「穩定可跑」的 baseline，而不是過度優化

---

## 希望你協助我的工作順序
請依照這個順序協助我：

1. 建立最小 baseline script
2. 先確認 imports / dataset / model 正常
3. 完成 QA preprocessing
4. 完成 QA post-processing
5. 先用小 subset 做 smoke test
6. 確認 train / eval / EM / F1 都正常
7. 整理 baseline 結果
8. 說明是否可以進入 Phase 2

---

## 我希望你的回答格式
請盡量把回答拆成這三部分：

- **已確認事實**
- **你的假設**
- **建議的下一步**

回答請務實、直接，不要跳到太後面的 phase。

---

# 以下是給我自己看的，不是要你自行補完的設定

## 目前仍未知 / 尚未鎖定的項目
以下內容在 Phase 1 可以暫時使用合理預設，但必須標記為 **assumption**，不能冒充為 paper-confirmed：

- baseline 的精確 batch size
- baseline 的精確 epochs
- baseline 的 max sequence length
- baseline 的 doc stride
- 是否先用小 subset 還是完整資料
- learning rate 是否先採 Hugging Face 常見設定
- weight decay
- warmup
- seed
- gradient accumulation
- checkpoint selection 規則
- evaluation 的實作細節是否完全對齊 paper 最終做法

---

## Phase 1 的真正目的
這一階段的目的只有一個：

**先確認原始 BERT on SQuAD 任務鏈本身是正確的。**

也就是先確保：
- 模型沒問題
- 資料沒問題
- preprocess 沒問題
- postprocess 沒問題
- 評估沒問題

只有 baseline 穩定了，後面接 GETA 才有意義。