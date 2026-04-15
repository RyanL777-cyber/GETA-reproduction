# GETA repo 地圖（microsoft/geta @ main, 2026-04-15 fetched）

只記需要知道的東西。**禁止一次把 `only_train_once/` 全部讀進 context。**

## 頂層
```
microsoft/geta/
├── only_train_once/     <- 主 package，使用者 import 的就是這個
├── tutorials/           <- 三個 Jupyter notebook 範例（全是 CV 任務，沒 NLP）
├── sanity_check/        <- 各模型的最小整合測試（有 transformer 類但無 BERT）
├── test_scripts/        <- 量化相關測試腳本（全是 CV）
├── img_folder/          <- 論文配圖，忽略
├── README.md            <- 有 QuickStart 範例，**必看**
├── LICENSE.txt
└── SECURITY.md
```

## `only_train_once/` 一層樹
```
only_train_once/
├── __init__.py              <- 暴露 OTO class，public API 入口 ★
├── quantization/            <- model_to_quantize_model, QuantizationMode ★
├── optimizer/               <- GETA / HESSO optimizer 實作（OTO.geta 背後）
├── subnet_construction/     <- construct_subnet 背後的 export 邏輯
├── dependency_graph/        <- QADG（量化感知依賴圖分析）
├── graph/                   <- 計算圖抽取與追蹤
├── operation/               <- 各類 op 在 pruning/quant 下的行為定義
├── transform/               <- graph transform passes
├── tests/                   <- 內部 unit tests，不用看
└── assets/                  <- 資源檔，忽略
```

★ = Phase 2 實作時會直接 import 的東西。其它只在 debug 掉進底層時才需要看。

## `tutorials/`
| 檔 | 任務 | 對我有用? |
|---|---|---|
| `01.resnet18_cifar10.ipynb` | ResNet18 + CIFAR-10 的 pruning+quant 示範 | 看一次學 API call sequence，但不必深究 |
| `02.carnx2_super_resolution.ipynb` | 超解析度 | 跳過 |
| `03.qvgg7_cifar10.ipynb` | QuantizationMode 完整流程示範 | 對量化部分有參考價值 |
| `utils/` | 範例共用工具 | 看了再說 |

> **注意**：全部是 CV 任務，沒有 BERT/SQuAD 的範本。Phase 2 接 BERT 要自己類推。
> **硬規則**：這些是 `.ipynb`，要看必須先 `jupyter nbconvert --to script xxx.ipynb` 轉 py 再讀。

## `sanity_check/`
和 Phase 2 可能相關的 transformer/NLP 類：
- `test_llamav1.py`、`test_llamav2.py`、`test_llama_embed_lora.py`、`test_llamav1_lora.py`
- `test_phi2.py`、`test_phi3.py`
- `test_tnlg.py`、`test_tnlg_lora.py`
- `test_mamba.py`、`test_mamba_onnx.py`
- `test_whisper.py`
- `backends/` 放各模型的 model definition

**沒有 `test_bert.py`**。接 BERT 時沒有官方 sanity check 可抄，要用 `test_phi2.py` / `test_llamav1.py` 這類 HuggingFace transformer 的入口當作最接近的參考模板。

## `test_scripts/`
全是 CV 量化測試（`Qtest_VIT_imagenet.py`、`Qtest_resnet56_ablation.py` 等）。
**對 BERT 整合基本沒有參考價值**，除非要看他們怎麼組 CLI 跑完整 train+eval。

## 結論：Phase 2 接 BERT 時應該看哪些檔
按優先序：
1. `README.md` 的 QuickStart 範例（已擷取於 `file_notes/README.md.md`）
2. `only_train_once/__init__.py`（OTO class 的完整簽名）
3. `only_train_once/quantization/quant_model.py`（`model_to_quantize_model` 實作，確認 transformer block 能不能被 handle）
4. `sanity_check/test_phi2.py` 或 `test_llamav1.py`（找 HF transformer 的接法模板）
5. `only_train_once/optimizer/` 底下 `GETA` 的類別簽名（全部 hyperparam）
