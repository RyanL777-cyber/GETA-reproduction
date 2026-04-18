# Project Status

> **更新規則**：每次對話結束、或完成一個可驗收的步驟，更新這份。不要把狀態留在對話歷史裡。

**Last updated**: 2026-04-17

---

## 目前在哪

**Phase 5**：正式 SQuAD 實驗，等 server 重跑 sparsity=0.1/0.3/0.5/0.7。

Phase 3（介面打通）與 Phase 4（小規模穩定性）已過關。

---

## 最近一次執行

| 項目 | 內容 |
|------|------|
| Log | `bert_geta_phase5/results/phase5_20260417_020405.log` |
| 結果 | **失敗** — 首次執行就在 `optimizer.step()` 第一步 div-zero |
| 原因 | Bug #3：`projection_steps` 預設值 1，但 `projection_periods=6` → `1 // 6 = 0` |

### 已完成的修復
- `bert_geta_phase5/run_experiment.py` 在 M4 build optimizer 前補上 `start_projection_step=0` + `projection_steps = max(projection_periods, start_pruning_step)`
- `INCIDENT_REPORT.md` §10 記錄 Bug #3

---

## 下一步

1. **Server 上重跑 Phase 5**：
   ```bash
   cd ~/GETA\ reprocution/GETA-reproduction/bert_geta_phase5
   python3 run_experiment.py --sparsity 0.1 0.3 0.5 0.7
   ```
   （跑很久，建議 tmux / nohup）

2. 等 4 個 sparsity 跑完後，開新對話分析結果，對比 paper Table 3：
   - 給 Claude 讀：`CLAUDE.md`, `STATUS.md`, `INCIDENT_REPORT.md`, `results/sp*/result.json`
   - 如果 EM/F1 差 > 1 分，回頭考慮做 QADG/optimizer 分離的根治重構（見 Incident §7 最後一段）

---

## 已過關

- **Phase 3 smoke**：`bert_geta_mvp/logs/smoke_20260416_171244.log`，M1-M5 全通，27 groups 全保留。
- **Phase 4 小規模**：`bert_geta_mvp/logs/phase4_20260417_011802.log`，1 epoch 訓練穩定，`construct_subnet` 成功產出壓縮模型，EM=0.20 / F1=4.35（與 full model 一致，只是物理刪除 zero-out neuron）。

---

## Open questions（非阻塞，但要追蹤）

見 `geta_knowledge/open_questions.md`。

與目前工作相關的：
- `attention.output.LayerNorm` 為何沒被雙重歸類（Bug #1 研究時留下的疑問）
- 三個 bug 是否要回報 GETA upstream
- Bug #2 的 workaround 只 cover FFN pattern，若未來換模型需擴展
