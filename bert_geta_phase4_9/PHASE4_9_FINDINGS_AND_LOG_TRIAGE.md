# Phase 4.9 Findings and Log Triage

本筆記整理 `bert_geta_phase4_9` 目前留下的部分實驗結果、可支持的推論，以及 log 的保留價值。這次整理採用「整合成一份索引」的方式；原始 log 沒有刪除。

## 一句話結論

Phase 4.9 **還不能證明已成功復刻論文結果**，但已經足夠支持一個關鍵判斷：

**分數下降的主因不是 activation calibration 16-bit 對不上 activation 4-bit，而是 GETA 的 weight bit projection / bit reduction schedule 太激進，且 pruning/projection 時程安排讓模型沒有足夠時間 recover。**

目前最有希望的方向是 `B3_full` 類型的 recipe：

```text
start_pruning_epoch = 5
bit_reduction = 1
lr_scheduler = linear
warmup_ratio = 0.1
calib_num_bits = 16
min_bit_wt = 4
max_bit_wt = 16
```

但這個方向目前只有 partial log，沒有完整 10 epoch、沒有 final compressed result，所以不能宣稱復刻成功。

## 可以確認的事

### 1. Phase 4.9 是診斷實驗，不是正式復刻

`run_all_exps.sh` 開頭明確寫：

```text
Phase 4.9 EXPERIMENT matrix (NOT the final reproduction)
Diagnostic + recovery experiments only.
```

Stage A / Stage B 的設計目標是找出 intervention，不是產出正式論文表格。

### 2. activation calibration mismatch 不是主因

原先假說是：

```text
activation calibration 用 16-bit，但最後 activation 變成 4-bit，所以 mismatch 導致崩盤
```

這個說法目前不成立，原因有兩個：

1. GETA optimizer 裡 activation projection 的呼叫是註解掉的，實際 schedule 主要作用在 weight bit。
2. `A5_calib4` 嘗試把 calibration 改成 4-bit，早期表現反而很差；這不支持「改 calibration bits 就能解決」。

### 3. weight bit projection / schedule 是最強嫌疑

Phase 5 的原始崩盤點是 epoch 2：F1 從高分掉下去，但 `grp_sp=0.000`，代表還不是 pruning 已經剪壞。

Phase 4.9 的部分實驗也支持這點：

- control / aggressive schedule 容易低分或不穩。
- delayed pruning / slower projection 可以讓前期 F1 維持高分。
- `B3_full` 在前 3 epoch 維持健康 F1，代表只要 schedule 放慢，模型不是天生不能訓練。

## 目前完成的正式 result.json

目前只有兩個實驗資料夾有 `result.json`：

| Result | 狀態 | 重點 |
|---|---|---|
| `results/A0_control_sp10/result.json` | 完成 | 3 epoch、10% train、1000 val；compressed F1 = 57.30 |
| `results/A2_slowproj_sp10/result.json` | 完成 | 3 epoch、10% train、1000 val；epoch 1-2 尚可，但 epoch 3/pruning 後掉到 compressed F1 = 46.39 |

### A0_control_sp10

設定摘要：

```text
epochs = 3
train_subset_frac = 0.1
val_subset_n = 1000
bit_reduction = 2
start_pruning_epoch = default
lr_scheduler = none
```

結果：

| Epoch | grp_sp | F1 |
|---:|---:|---:|
| 1 | 0.017 | 39.10 |
| 2 | 0.050 | 56.72 |
| 3 | 0.100 | 57.36 |

compressed F1 = `57.30`。

這個結果不能跟論文比，因為它用 subset data；但它可以當 aggressive/control 參考。

### A2_slowproj_sp10

設定摘要：

```text
epochs = 3
train_subset_frac = 0.1
val_subset_n = 1000
start_pruning_epoch = 2.5
bit_reduction = 2
lr_scheduler = none
```

結果：

| Epoch | grp_sp | F1 |
|---:|---:|---:|
| 1 | 0.000 | 78.94 |
| 2 | 0.000 | 78.52 |
| 3 | 0.017 | 46.49 |

compressed F1 = `46.39`。

解讀：

- 延後 pruning 前，F1 可維持在約 78-79。
- 一進入 pruning/低 bit 壓力後，epoch 3 明顯崩掉。
- 這支持「時程和壓縮壓力」是問題，不支持「activation calibration bits 是唯一主因」。

## 最重要的 partial evidence：B3_full

`results/phase5_20260427_184836.log` / `results/B3_full_sp10.stdout.log` 是目前最有價值的 partial recovery log。

設定：

```text
epochs = 10
sparsity = 0.1
start_prune = 5.00 epoch
proj_steps = 5.00 epoch
bit_reduction = 1
lr_scheduler = linear
warmup_ratio = 0.1
```

前 3 epoch：

| Epoch | grp_sp | EM | F1 |
|---:|---:|---:|---:|
| 1 | 0.000 | 75.20 | 83.84 |
| 2 | 0.000 | 79.00 | 86.73 |
| 3 | 0.000 | 79.30 | 87.06 |

解讀：

- 這是目前最強的「schedule 修正有效」訊號。
- `grp_sp=0.000` 時 F1 已經到 86-87，表示 quantized BERT 本身可以學到合理分數。
- 但這個 run 沒有完整跑完 10 epoch，也沒有 `result.json` / compressed F1，所以不能當成成功復刻。

## C1_winner 嘗試

C1 看起來是在嘗試把 winner recipe 套到多個 sparsity：

| Log | 狀態 | 看點 |
|---|---|---|
| `C1_winner_sp10.stdout.log` | partial | 只看到 early training / epoch 1 左右，未完成 |
| `C1_winner_sp30.stdout.log` | failed | CUDA OOM |
| `C1_winner_sp50.stdout.log` | partial | 有 epoch 1 訊號，未完成 |
| `C1_winner_sp70.stdout.log` | failed | CUDA OOM |

這組不能證明正式復刻成功。sp30/sp70 的 OOM 也表示並行資源安排需要修正。

## Log 品質注意

這批 log 不能全部直接當乾淨證據，原因：

1. 多個 job 平行啟動，stdout 和 `phase5_*.log` 有交叉污染或片段不完整。
2. 很多 log 只有 step 或 epoch 1，沒有 `[TRAIN] done`、`[SAVE]`、`result.json`。
3. `summary.json` 會被最後完成的 job 覆蓋，目前不代表完整 matrix。

因此判讀優先順序應該是：

```text
result.json > 完整 stdout with [SAVE] > partial phase5_*.log > summary.json
```

## 建議保留的檔案

這些最有證據價值：

```text
results/A0_control_sp10/result.json
results/A0_control_sp10/predictions.json
results/A2_slowproj_sp10/result.json
results/A2_slowproj_sp10/predictions.json
results/B3_full_sp10.stdout.log
results/phase5_20260427_184836.log
results/C1_winner_sp30.stdout.log
results/C1_winner_sp70.stdout.log
run_experiment.py
run_all_exps.sh
quant_fix.py
build_table.py
```

理由：

- A0/A2 是目前僅有的完整 `result.json`。
- B3 是最有希望的 schedule 修正 partial evidence。
- C1 sp30/sp70 保留 OOM 證據，方便之後修 GPU 排程。
- 程式檔保留用來對照參數與 log。

## 低價值 log / 可清理候選

以下不是說內容完全無用，而是它們目前沒有形成完整結果，且大多可由本筆記取代：

```text
results/phase5_20260427_184618.log
results/phase5_20260427_184628.log
results/phase5_20260427_184643.log
results/phase5_20260427_184659.log
results/phase5_20260427_184714.log
results/phase5_20260427_184729.log
results/phase5_20260427_184755.log
results/phase5_20260427_184807.log
results/phase5_20260427_184809.log
results/phase5_20260427_184820.log
results/phase5_20260427_184823.log
results/phase5_20260427_184837.log
results/phase5_20260427_184841.log
results/phase5_20260427_184852.log
results/phase5_20260427_184854.log
results/phase5_20260427_184908.log
results/phase5_20260427_184909.log
results/phase5_20260427_184923.log
results/phase5_20260427_184924.log
results/phase5_20260427_190024.log
```

stdout 低價值候選：

```text
results/A1_no_bitreduce_sp10.stdout.log
results/A3_lrsched_sp10.stdout.log
results/A4_br1_sp10.stdout.log
results/A5_calib4_sp10.stdout.log
results/A6_bigcalib_sp10.stdout.log
results/B1_slowproj_lrsched_sp10.stdout.log
results/B2_slowproj_only_sp10.stdout.log
results/C1_winner_sp10.stdout.log
results/C1_winner_sp50.stdout.log
```

保留或刪除取決於是否還要追溯中途 step。若只保留結論，本筆記已經整合了主要看點。

## 下一步建議

不要直接宣稱 Phase 4.9 已復刻成功。比較穩的下一步是：

1. 用 B3 類設定只跑 `sp10` full data 到完整 10 epoch。
2. 必須拿到 `result.json`，尤其是 `compressed_f1` / `compressed_em`。
3. 如果 sp10 compressed F1 接近論文，再跑 sp30/sp50/sp70。
4. 多 sparsity 不要同時塞滿同一張 GPU；C1 的 OOM 顯示 GPU 排程需要修正。

成功復刻的判準應該是：

```text
full data
10 epoch 完整結束
construct_subnet 成功
compressed_f1 接近論文
四個 sparsity 都有 result.json
```

目前 Phase 4.9 達成的是「問題定位與候選 recipe」，不是「正式復刻完成」。

