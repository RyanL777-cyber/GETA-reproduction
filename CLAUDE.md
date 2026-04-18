# GETA Reproduction — Project Orientation

給 Claude 新 session onboard 用。讀完這份 + `STATUS.md` + `bert_geta_mvp/bug_analysis/INCIDENT_REPORT.md` 就能上手。

---

## 目標

重現 GETA 論文 (CVPR 2025, *Generic Efficient Training framework that Automates joint pruning and quantization*) Table 3 的 **BERT-base on SQuAD v1.1** 結果。四個 target sparsity：10% / 30% / 50% / 70%，報 EM / F1 / BOPs。

**不**重現 CV / GLUE / 其他 backbone，**不**調 paper 以外的 hyperparameter。

---

## 環境

### Local（Windows, VSCode）
- 路徑：`c:\Users\iwant\Downloads\GETA reproduction\`
- 只用來編輯程式碼、看 log、寫文件。**不在本機跑訓練**。

### Server（Linux, GPU）
- SSH 帳號：`h24116081`
- 專案路徑：`~/GETA\ reprocution/GETA-reproduction/`（注意 typo `reprocution`，是 server 端資料夾名，不要改）
- 所有訓練在 server 上跑。
- GPU 多卡環境 → 新訓練/smoke 腳本都要在 `import torch` 之前呼叫 `select_idle_gpu()`（template 在 `bert_baseline/baseline_bert_squad.py`）。

---

## 目錄結構

```
GETA reproduction/
├── CLAUDE.md                  # 本檔，新 session 先讀
├── STATUS.md                  # 目前進度、下一步（每次結束更新）
├── phase_.md                  # Phase 3/4/5 的目標與驗收標準
├── phase 1.md, phase 3.md     # 早期 phase 筆記
│
├── geta/                      # GETA 原始碼（upstream fork，已有 local patch）
│   └── only_train_once/
│       ├── graph/graph.py:1371               # Bug #1 fix
│       ├── subnet_construction/pruning_compression.py  # Bug #2 workaround
│       └── optimizer/geta.py:64,867          # Bug #3 root（caller 層修正）
│
├── bert_baseline/             # Baseline BERT QA（不含 GETA），建立 EM/F1 參考線
│   └── baseline_bert_squad.py
│
├── bert_geta_mvp/             # Phase 3/4：smoke test + 小規模訓練
│   ├── smoke.py               # Phase 3 smoke（M1-M5）
│   ├── train_phase4.py        # Phase 4 小規模訓練 + construct_subnet
│   ├── PHASE3_NOTE.md
│   ├── logs/                  # 所有 phase 3/4 training log
│   └── bug_analysis/
│       ├── INCIDENT_REPORT.md # 三個 bug 的完整記錄
│       ├── check_indim.py, check_subnet.py, check_groups.py, dump_duplicate_params.py
│       └── ...
│
├── bert_geta_phase5/          # Phase 5：正式 SQuAD 實驗
│   ├── run_experiment.py      # 支援 --sparsity 0.1 0.3 0.5 0.7
│   └── results/               # log + result.json + predictions.json per sparsity
│
└── geta_knowledge/            # Claude 事前整理的 GETA repo 筆記
    ├── overview.md, repo_map.md, api_index.md
    ├── usage_flow.md, bert_integration.md
    └── open_questions.md      # 跨 bug 的未解問題清單
```

---

## Paper 設定（SQuAD, Table 3）

**注意**：以下數值是從 `phase_.md` 第 68-74 行抄來，沒有直接查 paper PDF。若要精準對比應自行確認。

| 項目 | 值 |
|------|---|
| optimizer | AdamW |
| lr | 3e-5 |
| lr_quant | 1e-4 |
| epochs | 10 |
| batch_size (B) | 4 |
| warmup pruning periods (Kb) | 1 |
| pruning_periods (P) | 6 |
| projection_periods (Kp) | 6 |
| bit_reduction (br) | 2 |
| bit_range | [4, 16] |

---

## 怎麼跑各 Phase

### Baseline
```bash
cd ~/GETA\ reprocution/GETA-reproduction/bert_baseline
python3 baseline_bert_squad.py
```

### Phase 3 smoke
```bash
cd ~/GETA\ reprocution/GETA-reproduction/bert_geta_mvp
python3 smoke.py
```

### Phase 4（小規模，default 2000 train / 500 val / 3 epochs）
```bash
python3 train_phase4.py               # 小規模
python3 train_phase4.py --full        # 完整 SQuAD
```

### Phase 5（正式）
```bash
cd ~/GETA\ reprocution/GETA-reproduction/bert_geta_phase5
python3 run_experiment.py --sparsity 0.1 0.3 0.5 0.7
# 或單獨：
python3 run_experiment.py --sparsity 0.5
```

Phase 5 每個 sparsity 產出 `results/sp{10,30,50,70}/{result.json, predictions.json}`，結束印 summary table。

---

## 已知 bug（細節見 `bert_geta_mvp/bug_analysis/INCIDENT_REPORT.md`）

| # | 位置 | 症狀 | 狀態 |
|---|------|------|------|
| 1 | `graph/graph.py:1371` | M4 dedup 字典序清空 layer.0 head 群 | ✅ 已 fix（size-ascending 排序） |
| 2 | `subnet_construction/pruning_compression.py` | `construct_subnet` 後 FFN `output.dense` 沒同步剪 in-dim，shape error | ✅ 已 workaround（第三 pass 補剪） |
| 3 | `optimizer/geta.py:64,867` | `projection_steps` 預設 1 + 傳 `projection_periods=6` → div-zero | ✅ 已在 caller 層修（`run_experiment.py`） |

---

## 工作慣例

- **一個 phase 結束更新一次 `STATUS.md`**（不要把狀態留在對話裡）
- **新 bug 寫進 `INCIDENT_REPORT.md`**，不要另開檔案
- **新的跨 phase open question 寫進 `geta_knowledge/open_questions.md`**
- **不 amend commit**；每次修復新開 commit
- **plot / comment 一律英文**（matplotlib 不支援中文）
- 對話內可以用中文溝通，檔案內 commit/comment 用英文
