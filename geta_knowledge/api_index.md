# GETA 關鍵 API 索引

只列使用者端真的會碰到的 symbols。內部 QADG/QASSO 不寫，除非 debug 需要。

## 1. `OTO` — 主入口類別
```python
from only_train_once import OTO
```
**檔案位置**: `only_train_once/__init__.py`
**作用**: 吃一個 PyTorch `nn.Module` + 一個 `dummy_input`，幫你建立內部 dependency graph，之後透過它呼叫 optimizer 工廠方法與子網匯出。

**初始化簽名（推斷自 README QuickStart）**
```python
oto = OTO(model=model.cuda(), dummy_input=dummy_input.cuda())
```
- `model`: 已經放到 device 上的 `nn.Module`（若要量化，要先經 `model_to_quantize_model` 包一層）
- `dummy_input`: 用來追蹤計算圖的範例輸入張量

### 主要方法
| 方法 | 回傳 | 用途 |
|---|---|---|
| `.geta(...)` | 一個 PyTorch-compatible optimizer 物件 | 建 QASSO 最佳化器，訓練時 `optimizer.step()` 即可 |
| `.hesso(...)` | optimizer | 純 pruning（不含 quant）時用；GETA 不需要 |
| `.construct_subnet(out_dir=...)` | None（寫檔到 out_dir）| 訓練完成後匯出剪枝+量化後的子網 |

---

## 2. `OTO.geta(...)` — GETA optimizer 工廠
```python
optimizer = oto.geta(
    variant="adam",                # 底層 optimizer 型態: "adam" / "sgd"
    lr=1e-3,                       # 一般參數的 learning rate
    lr_quant=1e-3,                 # quant-related 參數（如 step size, clip）的 lr
    target_group_sparsity=0.5,     # 要剪掉多少 group（0~1）
    bit_reduction=2,               # 每層 weight 平均要比 full precision 少幾個 bit（mixed precision 目標）
    min_bit_wt=4,                  # weight 最低位元（不會壓到比這低）
    max_bit_wt=16,                 # weight 最高位元
    # 其它可能的 kwargs: weight_decay, pruning_steps 等
)
```
**使用方式**: 跟一般 PyTorch optimizer 一樣，`optimizer.zero_grad()` / `optimizer.step()`。
**注意**: 這個 optimizer 同時推 pruning 與 quantization，不要自己再包另一層。

---

## 3. `model_to_quantize_model(...)` — 先把模型改成可量化版本
```python
from only_train_once.quantization.quant_model import model_to_quantize_model
from only_train_once.quantization.quant_layers import QuantizationMode

model = model_to_quantize_model(model, quant_mode=QuantizationMode.WEIGHT_AND_ACTIVATION)
```
**作用**: 把原本的 `nn.Linear` / `nn.Conv2d` 等替換成量化版本。在 `OTO(...)` 之前呼叫。
**Mode 推測**: `QuantizationMode.WEIGHT_ONLY` / `.WEIGHT_AND_ACTIVATION`（activation-only 應該也存在，debug 時再查）。
**Open question**: 對 HuggingFace `BertForQuestionAnswering` 這種 subclass 是否能吃到所有 Linear？要讀 quant_model.py 原始碼確認。

---

## 4. `QuantizationMode` — 量化目標設定
```python
from only_train_once.quantization.quant_layers import QuantizationMode
```
**作用**: enum / class，指定要量化 weight、activation 或兩者。在 `model_to_quantize_model` 的 `quant_mode` 參數用。

---

## 5. `OTO.construct_subnet(out_dir='./')`
**作用**: 訓練完成後呼叫，會在 `out_dir` 下匯出壓縮後的子網（權重 + 結構）。README 沒提供完整參數，但從 OTO `__init__.py` 推斷支援：
- `merge_lora_to_base`: 若用 LoRA fine-tune，合併回 base weight
- `export_huggingface_format`: 是否匯出成 HF 可讀格式 ← **對 BERT 特別重要**

---

## 標準呼叫順序（五行版）
```python
model = BertForQuestionAnswering.from_pretrained(...)            # 1. 建 HF 模型
model = model_to_quantize_model(model, quant_mode=...)           # 2. 包量化層
oto = OTO(model=model.cuda(), dummy_input=dummy_input.cuda())    # 3. 建 OTO 入口
optimizer = oto.geta(variant="adam", lr=..., target_group_sparsity=..., ...)  # 4. 拿 QASSO optimizer
# ...normal PyTorch training loop: zero_grad / backward / step...
oto.construct_subnet(out_dir='./compressed_bert/')               # 5. 匯出子網
```
