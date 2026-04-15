# BERT+GETA 集成 Bug 分析与修复

## 问题症状

执行 smoke test 时，在 M4（`oto.geta()` 阶段）失败：
```
RuntimeError: some parameters appear in more than one parameter group
```

## 根本原因

GETA 内部的 `Graph.get_param_groups()` 方法存在参数重复问题。

### 具体位置
- 文件：`geta/only_train_once/graph/graph.py`
- 方法：`Graph.get_param_groups()` （约第 1330 行）

### 问题细节

在 `get_param_groups()` 方法中，当收集各个 node_group 的参数时，**没有去重机制**，导致同一个参数（相同的 `id(param)`）出现在多个 param_groups 字典中。

原代码流程：
```python
def get_param_groups(self):
    param_groups = dict()
    for node_group in self.node_groups.values():
        if node_group.is_trainable and not node_group.is_auxiliary:
            ng_param_group = node_group.get_param_groups()
            if len(ng_param_group["params"]) > 0:
                param_groups[node_group.id] = ng_param_group
```

此时，如果多个 node_group 都包含同一个参数，该参数会被多次添加到不同的 param_groups 中。当 PyTorch optimizer 尝试注册这些 param_groups 时，检测到重复参数就会报错。

## 修复方案

在返回 param_groups 之前，添加**跨 group 的参数去重**逻辑：

### 修复步骤

在 `get_param_groups()` 末尾（return 前），插入以下代码：

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

    if len(keep_indices) != len(params):
        for key in ["params", "p_names", "op_names", "p_transform", "node_ids"]:
            if key in param_group:
                param_group[key] = [param_group[key][i] for i in keep_indices]

        if len(param_group.get("params", [])) == 0 and len(param_group.get("auxiliary_ngs", [])) == 0:
            del param_groups[group_id]

return param_groups.values()
```

### 修复原理

1. 维护全局 `seen_param_ids` 集合，记录已处理过的参数对象标识
2. 遍历每个 param_group 中的参数，只保留未见过的参数
3. 根据保留的参数索引，同步更新该 group 的元数据列表（`p_names`、`p_transform` 等）
4. 如果 group 变空且无辅助 groups，删除该 group

## 验证

修复后 smoke test 完整通过：
- M1-M5 全部成功
- Forward/backward/step 正常
- Loss 收敛正常（batch1: 5.9679, batch2: 5.9860）

## 关键教训

- 参数在多个 node_group 间共享时需要显式去重
- PyTorch optimizer 要求 param_groups 中每个参数唯一
- 去重需要同步更新关联的元数据列表，保持对齐
