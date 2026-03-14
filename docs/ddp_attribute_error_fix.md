# DDP 模式下 AttributeError 修复：自定义方法不可通过包装对象访问

## 问题描述

使用 5 卡 DDP 训练时，所有 rank 在进入训练循环的第一个 batch 即崩溃：

```
[rank3]: File "scripts/train.py", line 302, in train_one_epoch
[rank3]:     losses = model.compute_loss(
[rank3]: AttributeError: 'DistributedDataParallel' object has no attribute 'compute_loss'
```

该错误在全部 5 个 rank 上同时触发，训练无法进行。

---

## 根因分析

### PyTorch DDP 的方法代理机制

`DistributedDataParallel`（DDP）对 `nn.Module` 的包装方式决定了哪些方法可以直接调用：

| 方法类型 | 是否可直接通过 DDP 包装对象调用 | 原因 |
|---|---|---|
| `forward()` / `__call__()` | ✅ 可以 | DDP 重写了 `forward()`，内部调度到 `self.module.forward()` 并插入梯度同步 |
| `nn.Module` 内置方法 | ✅ 可以 | `parameters()`、`state_dict()` 等由 `nn.Module` 基类提供，DDP 继承了它们 |
| **自定义方法** | ❌ 不可以 | `compute_loss()`、`_prepare_target()` 等定义在 PENCI 模型类上，DDP 不知道它们的存在 |

DDP 的 `__getattr__` 只查找 `nn.Module` 注册的子模块/参数/缓冲区，**不会**将未知属性转发到 `self.module`。因此对 DDP 包装对象调用 `model.compute_loss(...)` 直接触发 `AttributeError`。

### 受影响的调用点

排查发现共 4 处裸调用：

| 文件 | 位置 | 调用 |
|---|---|---|
| `scripts/train.py` | `train_one_epoch()` | `model.compute_loss(...)` |
| `scripts/train.py` | `evaluate()` | `model.compute_loss(...)` |
| `scripts/train.py` | `evaluate()` | `model._prepare_target(...)` |
| `scripts/evaluate.py` | 传感器空间评估循环 | `model._prepare_target(...)` |

其中 `train.py` 的 `evaluate()` 中已有一处使用了 `hasattr(model, 'module')` 模式做条件分支，但不够统一且未覆盖 `compute_loss` 调用。

---

## 修复方案

### 引入 `unwrap_model()` 辅助函数

在 `train.py` 和 `evaluate.py` 中分别添加统一的解包函数：

```python
def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model
```

所有自定义方法调用统一改为通过 `unwrap_model()` 访问底层模型：

```python
# 修复前
losses = model.compute_loss(x, pos, sensor_type, ...)
target = model._prepare_target(x, length)

# 修复后
losses = unwrap_model(model).compute_loss(x, pos, sensor_type, ...)
target = unwrap_model(model)._prepare_target(x, length)
```

### 为什么不直接用 `model.module`

- 单卡训练时 `model` 是原始 `PENCI` 对象，没有 `.module` 属性，直接访问会报错。
- `unwrap_model()` 兼容单卡和多卡两种模式，代码无需根据运行环境做分支。

### 为什么 `model(x, ...)` 不需要修改

`model(x, pos, sensor_type, ...)` 调用的是 `__call__` → `forward()`，这是 DDP 唯一重写并代理的入口。DDP 在此处插入梯度 all-reduce 同步，因此前向传播必须经过 DDP 包装，**不能**用 `unwrap_model(model)(x, ...)` 替代。

总结区分规则：
- **前向传播** `model(x, ...)` → **保持**通过 DDP 包装调用（需要梯度同步）
- **自定义方法** `compute_loss`、`_prepare_target` → 通过 `unwrap_model(model)` 调用（纯计算，不涉及梯度同步）

---

## 修改文件清单

| 文件 | 修改内容 |
|---|---|
| `scripts/train.py` | 新增 `unwrap_model()` 辅助函数 |
| `scripts/train.py` | `train_one_epoch()`: `model.compute_loss` → `unwrap_model(model).compute_loss` |
| `scripts/train.py` | `evaluate()`: `model.compute_loss` → `unwrap_model(model).compute_loss` |
| `scripts/train.py` | `evaluate()`: 移除 `hasattr` 条件分支，统一为 `unwrap_model(model)._prepare_target` |
| `scripts/evaluate.py` | 新增 `unwrap_model()` 辅助函数 |
| `scripts/evaluate.py` | 传感器空间评估: `model._prepare_target` → `unwrap_model(model)._prepare_target` |

## 验证

```bash
python -m py_compile scripts/train.py      # ✓ 通过
python -m py_compile scripts/evaluate.py    # ✓ 通过
pytest tests/test_smoke.py -v               # ✓ 11 passed
```

## 日志佐证

修复前的运行日志（`outputs/ddp_diag_5gpu_20260312_235915/full_terminal_20260312_235915.log`）显示：

1. DDP 进程组初始化成功（5 rank 全部 `Init COMPLETE`，第 164-177 行）
2. 导联场预热正常完成（仅 rank 0 执行，14 个指纹全部缓存命中，第 247-270 行）
3. 进入训练循环后第一个 batch 即在 `model.compute_loss` 处崩溃（第 277-326 行）

这证实之前的 NCCL 超时修复和导联场预热修复均已生效，本次 `AttributeError` 是训练循环中的独立问题。
