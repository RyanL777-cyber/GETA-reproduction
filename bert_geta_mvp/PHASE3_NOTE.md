# Phase 3 集成注意事项

## 概览

Phase 3 集成 BERT (with quantization) + GETA 遇到的主要问题与解决方案。

## 核心问题

**参数重复在 param_groups 中**
- 症状：`OTO.geta()` 时 PyTorch optimizer 报错
- 原因：GETA 的 `Graph.get_param_groups()` 方法没有处理参数去重
- 修复：在 `get_param_groups()` 末尾添加跨 group 去重逻辑

详细分析见 → [BUG_ANALYSIS.md](BUG_ANALYSIS.md)

## 当前状态

✅ **已修复，smoke test 通过**
- M1-M5 全部成功执行
- 可正常运行 2 batch 的 forward/backward/step

## 快速参考

- **重现脚本**：`bug_repro.py`
- **完整测试**：`smoke.py`
- **修复位置**：`geta/only_train_once/graph/graph.py` Line ~1330-1395（`get_param_groups()` 末尾）
- **测试日志**：`logs/smoke_*.log`
