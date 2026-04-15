# GETA — Overview

**Paper**: *Automatic Joint Structured Pruning and Mixed Precision Quantization for Efficient Neural Network Training and Compression*, CVPR 2025
**Repo**: https://github.com/microsoft/geta
**Python package name**: `only_train_once`（沿用前身 OTO 的名字，不要被騙以為是另一個 repo）

## 它是什麼
GETA 是一個「**訓練即壓縮**」框架：不用先訓練完整模型再事後剪枝/量化，而是把 **structured pruning + mixed precision quantization** 直接編進 optimizer 裡，跟一般訓練一起跑。跑完就得到一個已經剪枝 + 已經量化的子網路。

## 兩個核心模組
| 模組 | 全名 | 做什麼 |
|---|---|---|
| QADG | Quantization-Aware Dependency Graph | 在計算圖層級分析權重/activation 的量化依賴，建立「可剪哪些 group」的搜尋空間 |
| QASSO | Quantization-Aware Structured Sparse Optimizer | 帶結構稀疏化的 optimizer，在訓練過程同時考慮剪枝目標稀疏度與量化位元限制 |

使用者基本上看不到 QADG/QASSO，它們被包在 `OTO` 這個類別後面。

## 使用者實際會碰到的 API
`OTO`、`oto.geta(...)`、`oto.construct_subnet(...)`、`model_to_quantize_model(...)`、`QuantizationMode`。這五個就夠完成一次完整的 GETA 壓縮流程，詳見 `api_index.md` 與 `usage_flow.md`。

## 對我們專案的定位
GETA 是 **Phase 2** 才要接的東西。Phase 1 已經把 BERT/SQuAD baseline 穩定起來（EM 81.14 / F1 88.50，見 `bert_baseline/notes.md`）。Phase 2 的工作是：把 baseline 的 `model` 丟給 GETA 的入口，跑出剪枝 + 量化後的 BERT，再用同一條 eval 流程量 EM/F1，和 baseline 對比掉點幅度，復現論文 Table 3。
