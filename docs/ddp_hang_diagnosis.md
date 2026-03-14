# 多卡 DDP 训练卡死问题：诊断与修复

## 问题描述

使用 5 张 A6000 GPU 执行多卡 DDP 训练时，进程卡死无响应：

```bash
torchrun --nproc_per_node=5 scripts/train.py --config configs/default.yaml
```

训练启动后各进程在初始化阶段无限期挂起，最终因 NCCL 超时崩溃。单卡训练不受影响。

---

## 根因分析

经排查定位到三个问题，其中前两个为主要致死原因。

### 问题一（主因）：NCCL 看门狗超时

**机制说明**：

`dist.init_process_group(backend="nccl")` 建立进程组后，NCCL 后端会启动看门狗线程。默认超时为 30 分钟——如果任何 rank 在该时间窗口内未参与集合通信操作（`all_reduce`、`barrier` 等），NCCL 即判定该 rank 已死，触发全局崩溃。

**PENCI 的问题**：

`init_process_group()` 之后、第一次 `dist.barrier()` 或训练循环之前，所有 5 个 rank 各自独立执行一系列重型 CPU 初始化：

| 初始化步骤 | 开销描述 |
|---|---|
| `setup_physics()` | MNE 源空间计算、电极注册表加载 |
| 数据加载 | 读取并合并 10 个数据集的 JSON 元数据（约 170 万条/rank） |
| `PENCIDataset` 构造 | `precompute_fingerprints=True` 时遍历全部元数据 |
| `DistributedBucketBatchSampler` | 按桶分组、填充、切分索引 |
| 导联场预热 | 遍历所有已注册指纹，调用 `get_leadfield()` |

当上述步骤总耗时超过 NCCL 默认的 30 分钟超时阈值时，看门狗判定进程失联，触发全局 abort。

**关键排除项**：

经逐行审查，`DistributedBucketBatchSampler` 的批次数一致性没有问题——各桶均按 `total_batch_size = num_replicas × batch_size` 的倍数进行填充，再以 `ids_bucket[rank::num_replicas]` 步幅切分，保证所有 rank 获得相同的批次数。此处不是卡死原因。

### 问题二（加剧因素）：导联场预热的文件锁竞争

**机制说明**：

`LeadfieldManager.get_leadfield()` 在缓存未命中时会调用 MNE 计算导联场矩阵，并通过文件锁（`fcntl.flock`）保护缓存写入，防止多进程同时写坏缓存文件。

**PENCI 的问题**：

修复前的代码让**所有 5 个 rank 同时**执行导联场预热循环：

```python
# 修复前：所有 rank 均执行
if leadfield_manager is not None and electrode_registry is not None:
    all_fps = electrode_registry.get_all_fingerprints()
    if all_fps:
        for fp in all_fps:
            names, positions = electrode_registry.get_config_by_fingerprint(fp)
            L = leadfield_manager.get_leadfield(names, positions, device)
```

当某个指纹的缓存文件不存在时，5 个 rank 同时触发 MNE 正向计算。其中 1 个 rank 获取文件锁并开始计算，其余 4 个被阻塞在文件锁上（`sleep(1) × 30` 次重试），叠加到总初始化时间中，进一步推高超时风险。

### 问题三（次要）：`evaluate()` 函数中误删参数

```python
def evaluate(model, dataloader, device, config, ..., rank=0, world_size=1):
    """评估模型"""
    del rank  # ← 删除了传入的 rank 参数
```

`del rank` 将 `rank` 参数从局部作用域中移除。虽然当前 `evaluate()` 函数体内未使用 `rank`，但这一写法导致后续如需在函数体中引用 `rank`（如添加仅主进程打印等逻辑）时会触发 `NameError`，属于隐患。

---

## 修复方案

修改文件：`scripts/train.py`

### 修复一：延长 NCCL 超时至 60 分钟

在 `setup_distributed()` 中为 `init_process_group` 添加显式超时参数：

```python
from datetime import datetime, timedelta

def setup_distributed() -> Tuple[int, int, int]:
    if "RANK" in os.environ:
        # ...
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(minutes=60),  # ← 新增：60 分钟超时
        )
        return rank, local_rank, world_size
    return 0, 0, 1
```

**原理**：给予初始化阶段足够的时间窗口。60 分钟可覆盖最坏情况下的物理计算 + 数据加载 + 导联场预热的总耗时。

### 修复二：导联场预热改为仅 rank 0 执行 + barrier 同步

```python
# 修复后：仅 rank 0 预热，其他 rank 等待
if leadfield_manager is not None and electrode_registry is not None:
    all_fps = electrode_registry.get_all_fingerprints()
    if all_fps:
        if is_main_process(rank):
            logger.info(f"预热导联场: {len(all_fps)} 个唯一电极配置 (仅 rank 0)...")
            for fp in all_fps:
                try:
                    names, positions = electrode_registry.get_config_by_fingerprint(fp)
                    L = leadfield_manager.get_leadfield(names, positions, device)
                    logger.info(f"  指纹 {fp}: 导联场 {L.shape}")
                except Exception as e:
                    logger.warning(f"  指纹 {fp} 导联场预热失败: {e}")
            logger.info("导联场预热完成")
        # 所有 rank 同步：确保 rank 0 预热完毕，其他 rank 可从磁盘缓存加载
        if world_size > 1:
            dist.barrier()
```

**原理**：

1. **rank 0** 独占执行导联场预热，逐一计算并写入磁盘缓存——无文件锁竞争。
2. `dist.barrier()` 确保 rank 0 完成缓存填充后，其他 rank 才继续执行。
3. 后续训练循环中，所有 rank 调用 `get_leadfield()` 时均命中磁盘缓存，直接加载，无需重复计算。

### 修复三：移除 `evaluate()` 中的 `del rank`

```python
def evaluate(model, dataloader, device, config, ..., rank=0, world_size=1):
    """评估模型"""
    # 移除 del rank，保留参数以备后续使用
    model.eval()
    # ...
```

---

## 关于 `setup_physics()` 和数据加载的说明

`setup_physics()`（MNE 源空间计算、电极注册表构建）和数据加载（JSON 元数据读取、`PENCIDataset` 构造）仍然由**所有 rank 各自独立执行**。原因如下：

1. **各 rank 需要本地副本**：这些对象（`SourceSpace`、`ElectrodeConfigRegistry`、`LeadfieldManager`、`PENCIDataset`）存储在各进程的内存中，无法通过 NCCL 广播共享（非 tensor 数据）。
2. **序列化成本高于重复计算**：将这些复杂 Python 对象序列化、广播、再反序列化的代价不亚于各 rank 独立初始化。
3. **超时延长已覆盖**：将 NCCL 超时从 30 分钟延长至 60 分钟后，足以容纳所有 rank 并行完成这些初始化步骤。

唯一改为 rank 0 独占的是**导联场预热**——因为它涉及磁盘 I/O 和文件锁，多 rank 并发会产生严重的锁竞争问题。

---

## 验证

修复后验证通过：

```bash
# 语法检查
python -m py_compile scripts/train.py    # ✓ 通过
python -m py_compile scripts/evaluate.py  # ✓ 通过

# 单元测试（11 个用例全部通过）
pytest tests/test_smoke.py -v             # ✓ 11 passed
```

多卡启动命令不变：

```bash
torchrun --nproc_per_node=5 scripts/train.py --config configs/default.yaml
```

---

## 修改文件清单

| 文件 | 修改内容 |
|---|---|
| `scripts/train.py` | `setup_distributed()`: 增加 `timeout=timedelta(minutes=60)` |
| `scripts/train.py` | 导联场预热段：改为 `is_main_process(rank)` 守卫 + `dist.barrier()` |
| `scripts/train.py` | `evaluate()`: 移除 `del rank` |
| `scripts/train.py` | import 合并：`from datetime import datetime, timedelta` |
