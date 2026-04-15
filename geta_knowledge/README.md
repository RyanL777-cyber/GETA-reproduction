# GETA Knowledge Base

這個資料夾是**我（Claude）對 GETA 這個專案與它原始碼的持續理解紀錄**。
目的是避免每次對話都從零重新理解 repo，節省 token。

所有內容都是**我自己累積的**，不是給人類讀的正式文件。
使用者可以看，但結構與用字以我自己能快速 lookup 為優先。

## 資料夾結構

```
geta_knowledge/
├── README.md          <- 你現在看的，這個資料夾的入口
├── overview.md        <- GETA 是什麼 / 論文目標 / 我們用它做什麼
├── repo_map.md        <- only_train_once 等子資料夾的樹狀索引與用途（逐步填）
├── api_index.md       <- 關鍵 symbols（OTO / geta / construct_subnet ...）輸入輸出與呼叫位置
├── usage_flow.md      <- GETA 最小使用流程（白話 10 句內）
├── bert_integration.md<- BERT baseline 如何接上 GETA 的具體介面 / 注意事項
├── open_questions.md  <- 我還不確定的點、暫時假設、待驗證事項
└── file_notes/        <- 每讀完一個檔就存一份極短摘要
    └── <file>.md
```

## 每個 file_note 的固定格式

```markdown
---
file: <相對於 repo 根目錄的路徑>
last_read: <YYYY-MM-DD>
---
- **用途**: 一句話
- **關鍵 symbols / API**: 只列名字，不複製 code
- **主流程位置**: 這個檔在 GETA 使用流程的第幾步
- **Phase 2 必看?**: yes / no / later
- **下一個應該看哪個檔**: 指向下一份 note 或檔名
- **open questions**: 還沒搞懂的地方
```

## 閱讀 / 累積策略（源自 GPT_suggues.md）

分三層，禁止一次吃整個 repo：

1. **Layer 1 — 建地圖**：先看 README 和各層資料夾的 tree，填 `repo_map.md`，只標「用途 + 必看/略過」，不讀 code。
2. **Layer 2 — 只吃入口檔**：`__init__.py`、`OTO` 定義、README 的 example、tutorials 裡最短的一個、sanity_check 對應 model 的入口 script。每看一個就新增 `file_notes/<xxx>.md`，並同步更新 `api_index.md`。
3. **Layer 3 — 整理流程**：等地圖和 API 都有了才寫 `usage_flow.md`，限 10 句，只講使用者怎麼呼叫，不重講論文。

## 硬規則

- **不要一次貼整包 `only_train_once/`**，會爆 token。
- 每讀完一份新檔必須更新對應 note + 必要時更新 `api_index.md` / `open_questions.md`。
- 不重寫論文背景。只寫「使用者怎麼呼叫」。
- 發現假設錯了要在原 note 標「SUPERSEDED」而不是默默改寫。
- 禁止直接讀 `.ipynb`，要先 `jupyter nbconvert --to script` 轉成 `.py` 再讀。
- **每份新的 GPU training/smoke script 都要內建 `select_idle_gpu()`**（模板在 `bert_baseline/baseline_bert_squad.py:19-69`），且必須在 `import torch` 之前呼叫，否則 torch 會預先綁定 GPU。用 `nvidia-smi` 挑 util 最低、記憶體占用最少的那張，設 `CUDA_VISIBLE_DEVICES`。Server 是共享環境，不加這段會撞到別人的 job。
