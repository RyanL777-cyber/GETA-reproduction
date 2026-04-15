---
file: README.md
last_read: 2026-04-15
source: https://github.com/microsoft/geta
---

- **用途**: GETA 專案的說明書兼 QuickStart。是我目前對 GETA API 的主要資訊來源。
- **關鍵 symbols / API**:
  - `OTO`（from `only_train_once`）
  - `model_to_quantize_model`（from `only_train_once.quantization.quant_model`）
  - `QuantizationMode`（from `only_train_once.quantization.quant_layers`）
  - `oto.geta(...)`、`oto.construct_subnet(...)`
- **主流程位置**: 覆蓋完整五步流程（建模型 → 包量化 → 建 OTO → 拿 optimizer → 匯出子網）。是唯一官方寫下的 end-to-end 範例。
- **Phase 2 必看?**: **yes**（已完全擷取到 api_index.md 與 usage_flow.md）
- **下一個應該看哪個檔**:
  1. `only_train_once/__init__.py` — 取得 `OTO` 類別完整簽名與其他公開 symbols
  2. `only_train_once/quantization/quant_model.py` — 驗證 `model_to_quantize_model` 能否吃 HF BERT
  3. `sanity_check/test_phi2.py`（或 test_llamav1.py）— HF transformer 接 GETA 的最接近模板
- **open questions**:
  - README 的 QuickStart 範例是 VGG7 + CIFAR-10，沒有任何 HF transformer / dict input 的示範。BERT 接法需要自己推斷。
  - `OTO.geta` 的 kwargs 不全，需要讀原始碼補完。
  - 沒有提到 `construct_subnet` 能否匯出成 HF 格式。

## 擷取到的完整 QuickStart code
```python
from only_train_once.quantization.quant_model import model_to_quantize_model
from only_train_once.quantization.quant_layers import QuantizationMode
from sanity_check.backends.vgg7 import vgg7_bn
from only_train_once import OTO
import torch

model = vgg7_bn()
model = model_to_quantize_model(model, quant_mode=QuantizationMode.WEIGHT_AND_ACTIVATION)
dummy_input = torch.rand(1, 3, 32, 32)
oto = OTO(model=model.cuda(), dummy_input=dummy_input.cuda())

optimizer = oto.geta(
    variant="adam",
    lr=1e-3,
    lr_quant=1e-3,
    target_group_sparsity=0.5,
    bit_reduction=2,
    min_bit_wt=4,
    max_bit_wt=16,
)

oto.construct_subnet(out_dir='./')
```
