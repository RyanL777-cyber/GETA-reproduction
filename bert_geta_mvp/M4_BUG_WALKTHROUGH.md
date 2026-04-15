# Phase 3 smoke test 全流程 + M4 bug 白話講解

寫給你自己看的，不是給 AI 同事看的。看完你應該能對任何人解釋「BERT × GETA 到底接在哪、為什麼 M4 會炸、workaround 為什麼不算完美」。

---

## 1. smoke.py 在做什麼（M1 → M5）

目的：**只驗證「介面打通」**，不訓練、不 eval、不看 sparsity 是否真的達到 0.5。所以只要 loss 能 backward + optimizer.step() 不丟錯，就算過。

```
  HF 原始 BERT (PyTorch nn.Module)
        │
        ▼  M1  AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")
  BERT + QA head （隨機初始化的 qa_outputs Linear）
        │
        ▼  M2  model_to_quantize_model(model, WEIGHT_AND_ACTIVATION)
  BERT（73 個 Linear / LayerNorm 被換成 QuantizedLinear / QuantizedLayerNorm）
        │        └─ 新增 weight_clip_val、act_clip_val 等「量化用」的可學 scalar
        │
        ▼  M3  OTO(model, dummy_input=(input_ids, attention_mask, token_type_ids))
  OTO 物件：
    - _model：量化後的 BERT
    - _graph：跑一次 torch.jit.trace 得到的依賴圖（1102 nodes, 1177 edges）
    - 每個「算子 / 參數」被歸入一個或多個 node_group
        │
        ▼  M3.5  診斷：list(model.parameters()) 有沒有重複 id
  結果：637 個 param、0 個重複 —— 代表 model 層沒問題
        │
        ▼  M4  oto.geta(variant="adam", lr=3e-5, lr_quant=3e-5,
        │                target_group_sparsity=0.5, bit_reduction=2, ...)
  內部：
    1. graph.get_param_groups()  ← 這裡就是 bug 的點
    2. 把 param_groups 丟給 torch.optim.Adam(param_groups, ...)
    3. Adam 在 add_param_group 時檢查「同一個 param 有沒有重複出現」
  產出：GETA optimizer 物件（包住 Adam）
        │
        ▼  M5  2 個真 SQuAD batch
  for batch in batches:
      optimizer.zero_grad()
      out = model(**batch)       # 走 eager PyTorch，不是 traced graph
      loss.backward()            # QA head 的 start/end loss
      optimizer.step()           # GETA 在這裡同時做 prune step + quant step
```

M5 log 顯示兩個 batch 分別 5.9593 / 5.9469。參考值：`log(384) ≈ 5.95`。這是「QA head 隨機初始化、完全沒學過」的理論 loss，也就是在 384 個 token 裡均勻猜 start 位置的交叉熵。loss 能下降 + 兩個 batch 的差（0.0124）合理，代表 forward/backward/step 的整條梯度路徑是通的。

---

## 2. M4 的 bug — 到底是什麼狀況

### 2.1 錯誤原文

```
ValueError: some parameters appear in more than one parameter group
    File "torch/optim/optimizer.py", line 1050, in add_param_group
```

PyTorch 的 `torch.optim.Optimizer` 有一條硬規則：**每個 `nn.Parameter` 只能出現在一個 param_group 裡**。原因是 optimizer state（momentum、variance、step count）是以 `id(param)` 當 key 存的，如果同個 param 出現兩次，state 會互相覆蓋，拿哪個 group 的 lr / weight_decay 去更新也歧義。

### 2.2 錯誤從哪來

不是在 M2（量化包裝），也不是在 model 本體。M3.5 已經證明：

```
param_count=637  unique=637  dup=0
```

model 層面沒有重複 param。但進到 GETA optimizer 建構時卻炸了——代表重複發生在 **GETA 自己把 model 的 param 重新分組**的那一步，也就是 `graph.get_param_groups()`。

### 2.3 什麼叫「重新分組」

PyTorch 原生 optimizer 只要吃 `model.parameters()` 的一個 flat list。但 GETA 要做「**結構化剪枝 + 混合精度量化**」，必須知道哪些 param 屬於同一個「可剪群」（例如 attention head 的 Q/K/V/O 要一起剪、FFN 的 up_proj/down_proj 要綁在一起），以及哪些 param 是「量化專屬」的（weight_clip_val 這類）。

所以 GETA 用 `node_group`（依賴圖上的強連通分量）來組織 param，然後從 node_group 產出 param_group，丟給底層 Adam。**26 個 node_groups 對應 26 個 param_groups**，log 裡就是這樣。

### 2.4 為什麼會有重複

log 裡的 group 分佈（按順序）：

| group idx | params | prunable |
|-----------|--------|----------|
| 0,2,4,6,...,18（10 個） | 8   | True |
| 1,3,5,7,...,19（10 個） | 24  | True |
| **20**    | **245** | True |
| 21        | 8   | True |
| 22        | 24  | True |
| 23        | 24  | True |
| 24        | 8   | True |
| 25        | 8   | False |

**加總：10×8 + 10×24 + 245 + 8 + 24 + 24 + 8 + 8 = 669 個 slot**
**但 model 只有 637 個 unique param → 32 個 slot 是重複的**。

重複從哪來？QADG 把同一個 param 映射到多個 node_group 的原因有兩類：

1. **跨層共享的 tensor** 被多個 group 的依賴路徑都指到——像 LayerNorm 的 weight/bias、residual 路徑上的 tensor，會同時落在「前一層的 FFN group」和「後一層的 attention group」的依賴集合裡。
2. **量化新增的 clip_val scalar** 可能被歸進「該 Linear 的 prune group」和「它自己的 quant group」各一次。

哪一種是主因，目前**沒驗證**——這就是隱憂的起點。

---

## 3. 原代碼 vs 新代碼（`geta/only_train_once/graph/graph.py` 的 `Graph.get_param_groups()`）

位置：[graph.py:1330](geta/only_train_once/graph/graph.py#L1330)

### 3.1 原代碼（沒 patch 時）

```python
def get_param_groups(self):
    param_groups = dict()

    # Pass 1：每個 trainable、非 auxiliary 的 node_group 自己回一包 param
    for node_group in self.node_groups.values():
        if node_group.is_trainable and not node_group.is_auxiliary:
            ng_param_group = node_group.get_param_groups()
            if len(ng_param_group["params"]) > 0:
                param_groups[node_group.id] = ng_param_group

    # Pass 2：處理 auxiliary node group 的偏移量（剪枝用）
    for node_group in self.node_groups.values():
        if hasattr(node_group, "auxilary_node_groups"):
            ...

    # Pass 3：丟掉整組都不 requires_grad 的 group
    ...

    return param_groups.values()
```

關鍵：**從頭到尾沒有任何「同一個 param 有沒有被兩個 group 拿走」的檢查**。每個 `node_group.get_param_groups()` 只看自己負責的節點，拉出那些節點對應的 `nn.Parameter`；如果兩個 node_group 在依賴圖上「輻射到」同一個 param，就各自抓一份。最後丟給 Adam → 重複 → 炸。

### 3.2 新代碼（patch 後，[graph.py:1368-1388](geta/only_train_once/graph/graph.py#L1368-L1388)）

```python
# Deduplicate parameters across groups to avoid invalid PyTorch optimizer
# initialization when the same parameter appears in multiple groups.
seen_param_ids = set()
for group_id, param_group in list(param_groups.items()):
    params = param_group.get("params", [])
    if len(params) == 0:
        continue

    keep_indices = []
    for idx, param in enumerate(params):
        if id(param) not in seen_param_ids:
            keep_indices.append(idx)
            seen_param_ids.add(id(param))

    # 只有這個 group 裡有某個 param 之前被別人拿過，才做截斷
    if len(keep_indices) != len(params):
        for key in ["params", "p_names", "op_names", "p_transform", "node_ids"]:
            if key in param_group:
                param_group[key] = [param_group[key][i] for i in keep_indices]

        # 截完後如果變空組，就整組刪掉
        if len(param_group.get("params", [])) == 0 and len(param_group.get("auxiliary_ngs", [])) == 0:
            del param_groups[group_id]

return param_groups.values()
```

邏輯白話：

1. 開一個全域 `seen_param_ids` set。
2. **按 `param_groups.items()` 的順序**（前面剛好 sort 過 id，所以順序是確定的）一個一個 group 走。
3. 對每個 param：如果 `id(param)` 還沒被任何前面的 group 拿過 → 這個 group 可以留。如果已經被拿過 → 從這個 group 裡刪掉（連 `p_names / op_names / p_transform / node_ids` 一起同步刪，不然 index 會錯位）。
4. 如果某個 group 被刪空了，整組抽掉。

**效果：PyTorch optimizer 不再報錯，因為每個 param 保證只在一個 group 裡。**

---

## 4. 為什麼這叫「workaround」不是「修好」

剪枝 / 量化的語意是：**某個 param 屬於某個 group = 這個 group 的 sparsity / bit-width 規則會管它**。

現在的 dedup 做的是：「**誰先拿到就誰的**」——完全是 iteration order 決定，不是依據哪個 group 對這個 param 有「正確的」所有權。

具體假設情境（還沒驗證，但這就是隱憂）：

- 假設 BERT 第 3 層 attention 的 `Q.weight` 同時落在：
  - node_group A：「第 3 層 attention 的 Q/K/V/O 剪枝群」（要按 head 整塊剪）
  - node_group B：「跨層的 residual 寬度剪枝群」（要和第 2、4 層的某些 param 一起壓）
- dedup 後 A 先拿到 → B 的 param list 裡 `Q.weight` 被刪掉 → **B group 在 optimizer step 時根本不會更新 `Q.weight` 的 prune mask**。
- 結果：paper 說「target_group_sparsity=0.5」是整個 B group 一起達成 50% 稀疏；實際上 B 少了一個成員，剩餘成員要達成 50% 可能更難或更鬆，**最終量到的 sparsity 和你設的不是同一回事**。

更糟的是**哪個 group 先拿到是由 dict 的 key 排序決定**（patch 前面有 `sorted(param_groups.items(), key=lambda kv: kv[0])`），這個順序完全和「剪枝語意誰該擁有它」無關，純 ID 大小。

所以：
- ✅ 對 smoke test（M5 就跑 2 個 batch，只看 loss 有沒有 backward）：完全沒差。
- ✅ 對 Phase 3 驗收：達成，介面打通。
- ❌ 對 Phase 4 正式訓練 + 達成 paper Table 3 的 sparsity / bit-width 目標：**不可信**，必須先釐清下面三件事：

  1. GETA 的設計裡，同一個 param 本來就該出現在多個 node_group 嗎？（去讀 `node_group.get_param_groups()` 和 `construct_node_groups` 的源碼）
  2. 如果是設計，**正確修法**是在 optimizer 層讓多個 group 對同一個 param「共享 mask / 共享 grad 但各自記 state」，不是砍 slot。
  3. 如果是 bug，**正確修法**是去修 node_group 建構邏輯，讓 param 歸屬唯一。

這三個問題被記在 [open_questions.md](geta_knowledge/open_questions.md#L9) 的 Q9，Phase 4 開工前要回來解。

---

## 5. 被操作的是 BERT 的哪塊

### 5.1 bert-base-uncased 架構圖

```
Input: (input_ids, attention_mask, token_type_ids)     shape: [B, 384]
   │
   ▼
┌──────────────────────────────────────────────────────┐
│ BertEmbeddings                                       │
│   ├─ word_embeddings    [30522, 768]   ← 大表        │
│   ├─ position_embeddings [512, 768]                  │
│   ├─ token_type_embeddings [2, 768]                  │
│   └─ LayerNorm + Dropout                             │
└──────────────────────────────────────────────────────┘
   │  hidden: [B, 384, 768]
   ▼
┌──────────────────────────────────────────────────────┐
│ BertEncoder                                          │
│   └─ 12 × BertLayer                                  │
│        ┌──────────────────────────────────┐          │
│        │ BertAttention                    │          │
│        │   ├─ self.query   Linear 768→768 │ ← Q      │
│        │   ├─ self.key     Linear 768→768 │ ← K      │ ← 這四顆
│        │   ├─ self.value   Linear 768→768 │ ← V      │   是同一
│        │   ├─ (softmax・matmul)           │          │   個 prune
│        │   └─ output.dense Linear 768→768 │ ← O      │   group
│        │       + LayerNorm + residual     │          │
│        ├──────────────────────────────────┤          │
│        │ BertIntermediate                 │          │
│        │   └─ dense Linear 768→3072       │ ← FFN↑   │ ← 這兩顆
│        ├──────────────────────────────────┤          │   是同一
│        │ BertOutput                       │          │   個 prune
│        │   └─ dense Linear 3072→768       │ ← FFN↓   │   group
│        │       + LayerNorm + residual     │          │
│        └──────────────────────────────────┘          │
└──────────────────────────────────────────────────────┘
   │  hidden: [B, 384, 768]
   ▼
┌──────────────────────────────────────────────────────┐
│ qa_outputs  Linear 768 → 2                           │
│   output: start_logits[B,384] + end_logits[B,384]    │
└──────────────────────────────────────────────────────┘
```

### 5.2 M2 量化替換的 73 層（對應 log 裡的 `Converted 73 layers`）

- 每個 BertLayer 6 個 Linear：Q / K / V / attn_out / ffn_up / ffn_down → 12 層 × 6 = **72** 個 Linear
- 加上最後的 `qa_outputs`：**+1**
- = **73 個 Linear** 被換成 `QuantizedLinear`
- LayerNorm 在 `WEIGHT_AND_ACTIVATION` 模式下也會被 wrap 但計入方式不同，這裡 log 只算了 Linear family。

### 5.3 M3 建出的 1102 nodes / 1177 edges

nodes 遠多於 param 數（637），因為 jit.trace 會把每個 op（matmul / add / softmax / layer_norm / gelu / reshape / transpose）都記成 node，**每個參數會是它所屬 Linear op 節點的一個「輸入」**。QADG 從這張圖裡抽出「寬度耦合」的 node_group——就是上面架構圖裡我用框標起來的 Q/K/V/O 四顆 + FFN 兩顆。

### 5.4 26 個 param_groups 的對應猜測（未完全驗證）

回去對照 log 的 group 分佈：

| 區塊 | 推測對應 |
|------|---------|
| 10 個 (8 params) + 10 個 (24 params) 交錯 | 10 層 encoder 的「attention 群 + FFN 群」？（但 BERT 是 12 層，差 2 層，可能前 2 層或後 2 層被併到 group 20 了） |
| group 20（245 params，最大） | **Embedding 表 + 殘差幹道**——跨層共享寬度最大的那條，也最可能是 32 個重複 slot 的主要貢獻者 |
| 21–24（8/24/24/8）| 剩下 2 層 encoder 的群？ |
| 25（8 params，`prunable=False`）| `qa_outputs` Linear（不剪，因為是 task head） |

**這張表是猜的**，要驗證得去印每個 group 的 `p_names`（smoke.py 原本有印但我剛剛砍掉了，之後有需要可以再加一個 diagnostic）。

---

## 6. 2026-04-16 追加：讀完 source 後的反轉

上面第 4 節說 dedup 是「workaround、有隱憂」。讀完 `node_group.py` / `geta.py` / `tutorials/01.resnet18_cifar10.py` 後，**部分結論要修正**。

### 發現 A：GETA 的 `step()` 本來就假設「每個 param 只被步一次」

[geta.py:895](geta/only_train_once/optimizer/geta.py#L895) 的 `step()` 主迴圈是：

```python
for group in self.param_groups:
    if not group["is_prunable"] or len(group["active_redundant_idxes"]) == 0:
        ...
        self.gradient_descent_step(group)   # ← 這裡
    elif group["is_prunable"] and len(group["active_redundant_idxes"]) > 0:
        ...
```

而 `gradient_descent_step` ([geta.py:548](geta/only_train_once/optimizer/geta.py#L548))：

```python
def gradient_descent_step(self, param_group):
    for p_name, p in zip(param_group["p_names"], param_group["params"]):
        ...
        p.data.add_(param_group["grad_variant"][p_name], alpha=-param_group["lr"])
```

**關鍵**：`p.data.add_(...)` 是 **in-place 直接改 `p.data`**。如果同一個 `p` 在兩個 group 裡，`for group in self.param_groups` 會對它呼叫兩次 `p.data.add_(...)` → **同一步被減去 2×lr×grad**，silent 2x update。

所以 PyTorch 的「不許重複 param」檢查其實**和 GETA 自己 step() 的語意是一致的**：GETA 根本就沒預期一個 param 被多個 group 步到。這不是 PyTorch 的潔癖，這是 GETA 內建的不變量。

**結論修正**：dedup workaround 在 **update 層**不是 hack，是**恢復 GETA 自己一直假設的不變量**。原本第 4 節擔心的「optimizer state 歧義」其實不存在，因為 GETA 根本沒用 PyTorch 原生的 Adam state——它自己算 `grad_variant` 然後直接 in-place update，沒有 `m/v/step` 共享問題。

### 發現 B：但「哪個 group 的 prune mask 管這個 param」的問題還在

雖然 update 層穩了，sparsity 層還是有疑慮。看 [node_group.py:178](geta/only_train_once/graph/node_group.py#L178) 的 `set_pruning_redundant_idxes`：它從該 node_group 的 `params` 算 `norm_group`，決定哪些「欄/列」是 redundant，然後剪掉。

如果一個 param 本來應該被兩個 group 看著「兩份 norm 貢獻」投票決定 prune 與否，dedup 後它只貢獻給先到的那一組。第一節點組少了一票、第二節點組完全看不到它。**這個疑慮第 4 節講的沒變**。

但——實際測試 smoke log 顯示 M4 成功、M5 loss 正常下降、patch 後 26 個 group 總和剛好 637 個 unique param（=model 全部參數）。這代表 dedup 後**每個 param 還是被某個 group 管到**，沒有 param 變孤兒。最差情況只是「被管得比原作者意圖少一份」，不是「完全沒人管」。

### 發現 C：為什麼 CV tutorials 沒撞到這個 bug

Tutorial 01（ResNet18/CIFAR10）跑的是 `oto.hesso(...)`，**不是 `oto.geta(...)`**：
- 純剪枝，沒呼叫 `model_to_quantize_model`，沒有 `QuantizedLinear` wrapper。
- ResNet 的 op 結構是 Conv → BN → ReLU → (add/skip)，每個 Conv / BN 的 weight 各自屬於一個節點，殘差幹道透過 **auxiliary node groups** 處理（第一節點組之外單獨一層），不會產生 param 級的重複。
- Tutorials 2、3 同理（CARN super-res、VGG7 quant only）都是 CNN 家族。

**沒有任何 tutorial 同時跑「transformer + GETA（joint prune+quant）」**。sanity_check 裡雖然有 Phi2/Llama/Mamba/Whisper，但看檔名都是 `*_prune.py` 或 `*_quant.py`，不是 joint。

### 發現 D：bug 的真正來源（強化猜測）

結合 A+B+C，32 個重複 slot 最可能的來源是：

1. `model_to_quantize_model` 在 BERT 的 73 個 Linear 上都插了 `QuantizedLinear`。每個 `QuantizedLinear` 新增 `weight_clip_val`、`act_clip_val`、`d_quant_wt`、`t_quant_wt`、`q_m_wt` 等可學 scalar（從 [geta.py:556](geta/only_train_once/optimizer/geta.py#L556) 的 `if "d_quant" in p_name or "t_quant" in p_name or "q_m" in p_name` 可看出這些 name 是 GETA 認得的量化參數）。
2. 這些 scalar attach 在原 Linear 節點上，當該 Linear 節點同時屬於「這一層自己的 prune group」和「殘差幹道的 width group」時，scalar 會被兩個 group 都 extend 進 `params` list。
3. 12 層 × (某幾個 scalar) ≈ 幾十個重複，量級吻合 32。

這個猜測如果對，**重複的都是量化 scalar 而不是主 weight**，那 dedup 的影響更小——量化 scalar 在 step() 裡本來就是單一 learning 更新（不像 prune mask 需要跨 group 投票），哪個 group 先拿到都無所謂。

### 發現 E：是否只有 BERT？

**幾乎肯定不是**。任何走 `model_to_quantize_model + oto.geta` 路徑的 HF transformer 都會中。repo 沒提供 transformer+joint 的 tutorial，代表作者群要嘛沒測過這條路徑，要嘛測過但沒 merge 修復。這是個「**發行版就帶著的 upstream bug，剛好被 CV-only 的測試覆蓋率掩蓋**」。

---

## 7. 結論修正 + 下一步

原本第 4 節說「workaround 有兩個隱憂（optimizer state、sparsity 語意）」。修正後：

| 隱憂 | 狀態 |
|------|------|
| (1) optimizer state 被兩個 group 搶著寫 | ❌ 不存在。GETA 不用 PyTorch 原生 Adam state，自己 in-place add |
| (2) `p.data` 被兩個 group 各 step 一次 → 2x update | ✅ 是真的隱憂，但 dedup **解決了這個**，不是製造問題 |
| (3) sparsity 語意失真（某 param 本該被兩個 prune mask 投票）| ⚠️ 還在，但**只有當重複的是主 weight 時才嚴重**。若重複的都是量化 scalar（發現 D 的猜測），影響可忽略 |

**因此下一步只有一件事**：**驗證發現 D**——印出 32 個重複 param 的具體 name，看它們是 `.weight`/`.bias` 還是 `weight_clip_val`/`d_quant_wt` 這類量化 scalar。

- 如果是後者：workaround 完全合理，Phase 4 可以直接走，甚至可以考慮把修改提 issue 給 GETA upstream。
- 如果是前者（主 weight 被重複）：需要重新評估，可能要改成「optimizer 層 dedup、prune mask 層保留多重成員」的分離設計。

### 執行步驟

寫一個小 script（不動 smoke.py），暫時**註解掉 [graph.py:1368-1388](geta/only_train_once/graph/graph.py#L1368-L1388) 的 dedup 那段**，跑一次印出：

```python
# pseudocode
from collections import defaultdict
owners = defaultdict(list)  # param_id → [(group_id, p_name), ...]
for group_id, pg in original_param_groups.items():
    for name, p in zip(pg['p_names'], pg['params']):
        owners[id(p)].append((group_id, name))
for pid, info in owners.items():
    if len(info) > 1:
        print(info)
```

跑完再還原 dedup。看輸出的 `p_name` 長什麼樣就知道發現 D 是否成立。

---

## 8. 一句話總結（改寫）

> GETA 的 `step()` 本來就假設每個 param 只被步一次（`p.data.add_` 是 in-place），所以 dedup workaround 在 update 層其實是**恢復原設計不變量**，不是 hack。唯一還有風險的是「重複的那 32 個 param 本來該被多個 prune group 投票」——但若它們只是量化 scalar（`weight_clip_val`/`d_quant_wt` 這類，猜測但未證），影響可忽略。**要確定就只需跑一個 script 印出 32 個 name**；根據結果再決定 Phase 4 能否直接走。
