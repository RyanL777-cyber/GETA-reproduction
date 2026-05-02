# Phase 5 GETA x BERT 診斷推論複核

本筆記只根據程式碼與既有 log/result 做唯讀複核；沒有修改原始程式、log、result 或原本的 `REPRODUCTION_DIAGNOSIS.md`。

## 結論

原診斷的大方向是對的：**Phase 5 的崩盤主要發生在 quantization，而不是 pruning。**

但原診斷中這句推論：

> activation calibration 16-bit 對不上 activation 4-bit

照字面看不成立，至少沒有被目前程式碼支持。程式確實用 16-bit 做 activation calibration，但 GETA optimizer 實際訓練時沒有啟用 activation bit projection。也就是說，目前看不到「activation 被 schedule 降到 4-bit，但 calibration 還停在 16-bit」這件事。

更有證據的主因是：

**GETA 的 weight bit projection 太早、太快，把 weight quantization 很快壓到低 bit；F1 在 pruning 尚未真正造成稀疏前就崩掉。**

## 程式碼證據

### 1. Activation calibration 確實是 16-bit

`bert_geta_phase5/run_experiment.py`：

```python
calibrate_quant_layers(model, calib_batches, num_bits=16, log=log)
```

`bert_geta_phase5/quant_fix.py` 裡 `calibrate_quant_layers()` 會用 `num_bits` 算 activation quantization step：

```python
qmax_divisor = (2 ** (num_bits - 1)) - 1
new_d = new_qm / qmax_divisor
mod.q_m_act.fill_(new_qm)
mod.d_quant_act.fill_(new_d)
```

所以「activation calibration 使用 16-bit」這一點是事實。

### 2. 但 activation projection 沒有真的啟用

`geta/only_train_once/optimizer/geta.py` 裡雖然有 activation projection 函式：

```python
def partial_projected_gradient_descent_step_range_act(self, param_group):
```

但在 optimizer 的 `step()` 中，實際呼叫被註解掉：

```python
self.partial_projected_gradient_descent_step_range_wt(group)
# self.partial_projected_gradient_descent_step_range_act(group) # Uncomment this line if apply activation quantization
```

pruning 分支裡 activation projection 也同樣是註解：

```python
# self.partial_projected_gradient_descent_step_range_act(group)
```

因此目前的訓練路徑主要在做 **weight bit projection**，不是 activation bit projection。

這表示原本「activation 最終應該變 4-bit，所以 `d_quant_act` 應該從 `q_m/32767` 變成 `q_m/7`」的說法，缺少實際程式碼支撐。

### 3. Weight bit projection 確實會快速下降

`bert_geta_phase5/run_experiment.py` 建立 GETA optimizer 時設定：

```python
bit_reduction=args.bit_reduction,
min_bit_wt=4,
max_bit_wt=16,
```

預設參數是：

```python
bit_reduction = 2
projection_periods = 6
```

`geta.py` 的 `step()` 中會在 projection period 更新：

```python
self.max_bit_wt = self.max_bit_wt - self.bit_reduction
```

所以 weight bit 大致會走：

```text
16 -> 14 -> 12 -> 10 -> 8 -> 6 -> 4
```

### 4. Schedule 很早就把 projection 做完

sp10 log `phase5_20260424_174757.log` 顯示：

```text
steps/epoch=22131
total=221310
start_prune=36885
proj_steps=36885
proj_periods=6
```

`36885 / 22131 ~= 1.67 epoch`。

也就是說，weight bit projection 在第 2 個 epoch 還沒結束前就走完主要下降流程。這和 F1 崩盤時間點吻合。

## Log 證據

sp10 的 epoch 結果：

| Epoch | grp_sp | EM | F1 | 解讀 |
|---|---:|---:|---:|---|
| 1 | 0.000 | 78.15 | 85.93 | 還沒有 pruning，F1 幾乎達到論文目標 |
| 2 | 0.000 | 59.26 | 70.76 | F1 暴跌，但 group sparsity 仍是 0 |
| 3 | 0.000 | 65.54 | 76.20 | 有恢復，但回不到 epoch 1 |
| 10 | 0.100 | 66.72 | 77.18 | 最終 compressed F1 約 77 |

關鍵點是 epoch 2：

```text
grp_sp=0.000
F1=70.76
```

當時 group sparsity 還是 0，代表 F1 暴跌不是 pruning 已經剪掉大量 group 造成的。更合理的解釋是 projection 階段的 quantization，尤其是 weight bit reduction。

## 對原診斷各推論的判斷

| 原推論 | 判斷 | 原因 |
|---|---|---|
| 沒有重現成功 | 成立 | compressed F1 約 76-77，低於論文 84-86 |
| F1 在 epoch 1 之後崩盤 | 成立 | sp10 log 顯示 85.93 -> 70.76 |
| 不是 pruning 主因 | 成立 | epoch 2 崩盤時 `grp_sp=0.000` |
| activation calibration 16-bit 對不上 activation 4-bit | 不成立或未證明 | activation projection 在 `step()` 裡被註解，沒有證據顯示 activation 被 schedule 降到 4-bit |
| bit-reduction schedule 太快 | 成立，而且是最強主因 | weight bit projection 約 1.67 epoch 內完成，和 F1 暴跌吻合 |
| `best_f1` 指標誤導 | 成立 | `best_f1` 是訓練中最佳 epoch；最終 compressed model 要看 `compressed_f1` |
| LR scheduler 可能有影響 | 可能成立但未證明 | 程式使用固定 lr；但目前證據較弱，需實驗驗證 |

## 更精準的一句話

不是「activation 4-bit calibration mismatch」已被證實；真正有程式碼與 log 支撐的是：

**GETA 的 weight bit projection 在很短時間內把權重量化壓到低 bit，F1 在 pruning 尚未造成稀疏前就崩掉，後續 epoch 也無法完全恢復。**

## 建議的下一個驗證實驗

如果要用最小成本驗證主因，建議先不要動 pruning，先做 sp10 的短跑實驗：

1. 固定 `target_group_sparsity=0` 或只跑 sp10。
2. 把 `min_bit_wt=16, max_bit_wt=16`，等於關閉 weight bit reduction。
3. 跑 2-3 epochs，看 epoch 2 是否還會從 85+ 掉到 70 左右。

預期：

- 如果不再崩盤，主因就是 weight bit reduction。
- 如果仍然崩盤，才需要繼續查 GETA optimizer 的其他訓練行為。

另一路可以測：

1. 保留 `min_bit_wt=4`。
2. 拉長 `projection_steps` 或延後 projection。
3. 看 F1 是否能在低 bit 後恢復更多。

