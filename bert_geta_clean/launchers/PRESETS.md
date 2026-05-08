# 預設配置與我們跑過的實驗

`run.sh` 是給你自己改參數用的範本。如果你想直接重現特定結果,在這裡找對應的 CLI 命令貼上即可。

所有命令假設你站在 `bert_geta_clean/` 根目錄執行。

---

## Preset B — 最佳結果(我們跑出最高 F1 的配置)

對應 EXPERIMENT_RESULTS.md 那張表的 **sp=0.7 (13ep, bs=16)** 那一列,F1=83.85,**比論文 GETA 只差 0.89**。

```bash
python src/run_experiment.py \
  --sparsity 0.7 \
  --epochs 13 \
  --batch_size 16 \
  --lr 6e-5 \
  --lr_quant 1e-4 \
  --bit_reduction 1 \
  --pruning_periods 6 \
  --projection_periods 6 \
  --start_pruning_epoch 5 \
  --pruning_end_epoch 10 \
  --lr_scheduler linear \
  --warmup_ratio 0.1 \
  --seed 42 \
  --out_root ./results \
  --exp_tag preset_B
```

**為什麼這組是最好**: bs=16 訓練快(~67min/epoch),lr=6e-5 是 sqrt-scaled 上去夠穩,13 epoch 在 overfitting 之前就停(我們驗證過 16ep 反而退步)。

---

## Preset C — 論文還原(嚴格 reproduction)

最接近論文 Table 3 的設定。**注意**: 我們**沒跑過這組**,所以不確定能不能補上 ~1 F1 的 gap。如果你有時間驗證請告訴我們結果。

跟 Preset B 主要差異: `bs=4`(paper B=4)、`bit_reduction=2`(paper br=2)、`lr=3e-5`(paper)。

```bash
python src/run_experiment.py \
  --sparsity 0.7 \
  --epochs 10 \
  --batch_size 4 \
  --lr 3e-5 \
  --lr_quant 1e-4 \
  --bit_reduction 2 \
  --pruning_periods 6 \
  --projection_periods 6 \
  --start_pruning_epoch 5 \
  --pruning_end_epoch 10 \
  --lr_scheduler linear \
  --warmup_ratio 0.1 \
  --seed 42 \
  --out_root ./results \
  --exp_tag preset_C
```

**代價**: bs=4 比 bs=16 慢約 4×,單點預估 ~25h+。

---

## 我們實際跑過的 4 個 run

| 標記 | sp | epochs | bs | lr | br | EM | F1 | params(M) | BOPs(GB) | Rel.(%) | 備註 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sp10_baseline | 0.1 | 10 | 4 | 3e-5 | 1 | 75.09 | 84.45 | 103.21 | 3.27 | 24.12 | 第一次驗證 pipeline 跑得起來 |
| sp50_combined16 | 0.5 | 16 | 16 | 6e-5 | 1 | 74.45 | 84.02 | 80.45 | 2.32 | 17.07 | |
| **sp70_fast13** | **0.7** | **13** | **16** | **6e-5** | **1** | **74.29** | **83.85** | **69.08** | **1.93** | **14.20** | **最佳,= Preset B** |
| sp70_combined16 | 0.7 | 16 | 16 | 6e-5 | 1 | 73.72 | 83.61 | 69.08 | 1.86 | 13.68 | 過擬合,F1 ep15 觸頂 |

> BOPs(GB) = `bops_million / 1000`(OTO `compute_bops(in_million=True)` 換算)。Rel.(%) 用論文 Table 3 baseline 13.57 GB 為分母。我們 br=1,論文 GETA br=2,故我們 BOPs 比論文同 sparsity 高 ~18–24%。

### sp10_baseline(2026-05-02 第一個成功 run)

```bash
python src/run_experiment.py \
  --sparsity 0.1 \
  --epochs 10 \
  --batch_size 4 \
  --lr 3e-5 \
  --lr_quant 1e-4 \
  --bit_reduction 1 \
  --pruning_periods 6 \
  --projection_periods 6 \
  --start_pruning_epoch 5 \
  --pruning_end_epoch 10 \
  --lr_scheduler linear \
  --warmup_ratio 0.1 \
  --seed 42 \
  --out_root ./results \
  --exp_tag baseline
```

### sp50_combined16

```bash
python src/run_experiment.py \
  --sparsity 0.5 \
  --epochs 16 \
  --batch_size 16 \
  --lr 6e-5 \
  --lr_quant 1e-4 \
  --bit_reduction 1 \
  --pruning_periods 6 \
  --projection_periods 6 \
  --start_pruning_epoch 5 \
  --pruning_end_epoch 10 \
  --lr_scheduler linear \
  --warmup_ratio 0.1 \
  --seed 42 \
  --out_root ./results \
  --exp_tag combined16
```

### sp70_fast13 — 最佳結果(同 Preset B,參考用)

見上方 Preset B。

### sp70_combined16(過擬合對照)

跟 fast13 同設定,只把 epochs 延長到 16。F1 反而從 83.85 退到 83.61,訓練 loss 掉到 0.038 → val F1 ep15 觸頂後下滑。教訓:不要因為 train loss 還在降就加 epoch。

```bash
python src/run_experiment.py \
  --sparsity 0.7 \
  --epochs 16 \
  --batch_size 16 \
  --lr 6e-5 \
  --lr_quant 1e-4 \
  --bit_reduction 1 \
  --pruning_periods 6 \
  --projection_periods 6 \
  --start_pruning_epoch 5 \
  --pruning_end_epoch 10 \
  --lr_scheduler linear \
  --warmup_ratio 0.1 \
  --seed 42 \
  --out_root ./results \
  --exp_tag combined16
```

---

## 同時跑多個 sparsity(sweep)

`--sparsity` 接多個值,會依序跑完並寫到 `results/<exp_tag>_sp10/`、`results/<exp_tag>_sp30/` 等等。例如 sp50+sp70 sweep:

```bash
python src/run_experiment.py \
  --sparsity 0.5 0.7 \
  --epochs 16 \
  --batch_size 16 \
  --lr 6e-5 \
  --lr_quant 1e-4 \
  --bit_reduction 1 \
  --pruning_periods 6 \
  --projection_periods 6 \
  --start_pruning_epoch 5 \
  --pruning_end_epoch 10 \
  --lr_scheduler linear \
  --warmup_ratio 0.1 \
  --seed 42 \
  --out_root ./results \
  --exp_tag sweep
```

時間 = 兩個點各自時間相加。
