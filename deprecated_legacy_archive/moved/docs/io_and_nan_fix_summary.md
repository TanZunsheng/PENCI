# 训练 I/O 与 NaN 两轮修复总结

## 背景

本次总结覆盖两轮连续修复：

1. **第一轮**：多卡训练时读取数据占用大量共享存储带宽，导致 NFS 抖动、节点卡顿。  
2. **第二轮**：训练过程中出现 `loss = NaN`，且此前无法快速定位“具体是哪个样本/哪个环节”导致异常。

---

## 第一轮修复：I/O 卡顿与节点抖动

### 问题表现

- 多卡 + 多 worker 场景下，数据读取触发高并发 I/O。
- 即使数据已转为 HDF5，训练仍出现明显卡顿，并影响同节点/同存储其他任务。

### 根因

- 仅“格式改成 HDF5”不足以自动解决随机读放大问题。
- 训练采样顺序与实际读取顺序不一致，导致共享存储侧存在突发读取压力。
- 训练前缺乏针对“本轮即将使用数据”的缓存预热策略。

### 修复方法

1. **文件级调度（file scheduler）**  
   在分布式分桶采样器中引入文件级调度能力：同一 batch 尽量来自同一 HDF5 文件，并支持文件内随机采样。  
   目标是把 I/O 模式从“离散随机”拉向“更可控的文件局部访问”。

2. **训练前 HDF5 预热（page cache warmup）**  
   在每个 epoch（可配置）启动前，按计划顺序把即将使用的 HDF5 文件顺序读取到 OS page cache。  
   训练进程随后的读取命中缓存，降低训练时段对 NFS 的瞬时冲击。

3. **预热过程可观测化**  
   输出预热进度（已读 GiB、百分比、速度、ETA），便于判断是否达到预热目标，以及是否出现读速异常。

4. **保守并发策略**  
   预热逻辑默认按低并发/单线程顺序读取；`training.num_workers` 也下调为低并发配置，避免“预热和训练同时打爆 NFS”。

### 关键配置

- `data.file_scheduler.enabled`
- `data.file_scheduler.shuffle_within_file`
- `data.io_prefetch.enabled`
- `data.io_prefetch.warmup_gb`
- `data.io_prefetch.warmup_each_epoch`
- `data.io_prefetch.prefetch_threads`
- `data.io_prefetch.read_chunk_mb`
- `training.num_workers`

### 结果

- 训练读取链路变得可控，I/O 峰值明显更平滑。
- 节点级卡顿风险下降，训练阶段对共享存储的扰动减轻。

---

## 第二轮修复：`loss = NaN` 诊断与拦截

### 问题表现

- 训练中偶发 `loss` 变为 `NaN`。
- 之前只能看到日志中的 NaN 结果，无法快速回答：
  - 是输入问题、导联场问题、前向输出问题，还是反向梯度爆炸？
  - 具体是哪些样本触发了异常？

### 根因

- 训练循环缺少系统化的 NaN/Inf 守卫与分阶段诊断。
- DDP 场景下若某一 rank 异常，其他 rank 可能继续推进，容易进入更难排查的状态。
- batch 元数据此前未完整透传 `hdf5_path/hdf5_idx`，定位到 HDF5 样本粒度困难。

### 修复方法

1. **新增 NaNGuard 诊断器**  
   在训练循环中新增 `NaNGuard`，按阶段检查非有限值：
   - 输入阶段（`x / pos / leadfield`）
   - 前向阶段（`reconstruction / source_activity / loss`）
   - 反向阶段（梯度范数、首个非有限梯度参数）

2. **DDP 一致化处理**  
   使用跨 rank 同步判断：任一 rank 命中异常，所有 rank 走一致分支，避免“部分 rank 继续、部分 rank 中断”的失步问题。

3. **坏 batch 可追溯记录**  
   发生异常时落盘诊断 JSONL，包含：
   - `epoch / global_step / rank / stage / reason`
   - 当前学习率、AMP scale
   - 关键张量统计（min/max/mean/std、nan_count、inf_count）
   - 样本来源信息（`dataset / path / fingerprint / hdf5_path / hdf5_idx / sample_index`）

4. **可选张量快照**
   支持按配置保存异常 batch 的小样本张量快照，便于离线最小复现。

5. **数据元数据增强**  
   在 dataset 返回的 metadata 中补充 `hdf5_path / hdf5_idx / sample_index`，用于精准定位问题样本。

### 关键配置（`training.nan_guard`）

- `enabled`: 是否启用 NaN/Inf 守卫
- `fail_fast`: 首次异常是否立即终止（默认建议开启）
- `skip_bad_batch`: 非 fail-fast 模式下是否跳过坏 batch
- `max_records`: 每个 rank 最多记录条数
- `max_metadata_items`: 每条记录最多写入多少样本元信息
- `dump_tensors`: 是否写 `.pt` 快照
- `dump_max_samples`: 快照中保留样本数
- `record_file` / `dump_dir`: 诊断文件命名

### 诊断产物位置

- `outputs/<run>/diagnostics/bad_batches.rank*.jsonl`
- `outputs/<run>/diagnostics/bad_batch_tensors/*.pt`（开启 `dump_tensors` 时）

### 结果

- 出现 NaN 时不再“黑盒”，可以快速定位到具体阶段和具体样本来源。
- 为后续根因修复（学习率、AMP、数据清洗、导联场异常等）提供直接证据。

---

## 涉及文件

- `scripts/train.py`
- `penci/data/dataset.py`
- `configs/default.yaml`

---

## 当前建议运行策略

1. 先用默认 `nan_guard.fail_fast=true` 跑一次，优先拿到第一份高质量异常证据。  
2. 如果希望任务不中断，可切到 `fail_fast=false` + `skip_bad_batch=true`，但建议仅在已经明确异常来源后使用。  
3. 结合 `bad_batches.rank*.jsonl` 定位后，再做有针对性的数值稳定修复（如 AMP 策略、学习率、梯度裁剪、特定样本清洗）。
