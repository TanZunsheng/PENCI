# 训练/验证数据加载与 NFS 防卡顿策略

本文档说明 PENCI 当前训练与验证阶段的数据加载路径，以及如何通过采样调度与预热策略降低 NFS I/O 抖动。

## 1. 总体结论

- 训练和验证都按 `batch` 读取，不是一次性整集读入内存。
- 数据主路径优先走 HDF5（`hdf5_path + hdf5_idx`），失败时才回退 `.pt`。
- “内存缓存”主要指 **OS page cache**（页缓存），不是应用层维护的大块内存池。

## 2. 训练阶段如何加载

1. 元数据来源：
- 从各数据集 `train.json` 合并得到训练样本列表。

2. 样本读取：
- `__getitem__` 优先从 HDF5 读取 `x[idx]`；
- 每个 DataLoader worker 持有自己的 HDF5 文件句柄缓存；
- `pos` / `sensor_type` 做 per-file 缓存，避免重复读取。

3. 组 batch：
- 使用 Bucket Sampler 按 `(channels, fingerprint)` 分桶；
- 启用 `file_scheduler` 时，同一 batch 尽量来自同一 HDF5 文件，减少跨文件随机读。

4. 预热（抗 NFS 抖动）：
- 每个 epoch 训练前按计划文件顺序读取，预热到 OS page cache；
- 关键参数：`warmup_gb`、`read_chunk_mb`、`prefetch_threads`、`max_files`。

## 3. 验证阶段如何加载（本次更新后）

1. 元数据来源：
- 从各数据集 `val.json` 合并得到验证样本列表。

2. 组 batch：
- 验证仍按 batch 跑，`shuffle=False`；
- 现在支持独立配置验证阶段文件调度：
  - `data.file_scheduler.val_enabled`
  - `data.file_scheduler.val_shuffle_within_file`
- 推荐：`val_enabled=true`、`val_shuffle_within_file=false`，兼顾 NFS 稳定与评估可复现。

3. 验证前预热（本次新增）：
- 每次 `evaluate()` 前，对验证将访问的 HDF5 文件做顺序预热；
- 默认复用训练同一套预热参数（也可单独覆盖）：
  - `data.io_prefetch.warmup_before_val`
  - `data.io_prefetch.warmup_val_gb`
  - `data.io_prefetch.max_files_val`
  - `read_chunk_mb` / `prefetch_threads` 与训练共用。

## 4. 为什么这样能减少 NFS 卡顿

- 随机跨文件读取会放大 NFS 元数据与小 I/O 请求开销，容易抖动。
- 文件级调度把 batch 聚簇到同一文件，减少跨文件跳转。
- 顺序预热把热点文件提前放入页缓存，正式迭代时更多命中内存。
- `prefetch_threads=1` 可显著抑制并发拉满 NFS 的风险。

## 5. 建议参数（NFS 场景）

- `data.file_scheduler.enabled: true`
- `data.file_scheduler.shuffle_within_file: true`（训练）
- `data.file_scheduler.val_enabled: true`
- `data.file_scheduler.val_shuffle_within_file: false`（验证）
- `data.io_prefetch.enabled: true`
- `data.io_prefetch.warmup_before_val: true`
- `data.io_prefetch.prefetch_threads: 1`
- `data.io_prefetch.read_chunk_mb: 8`
- `training.num_workers: 1`（若仍抖动可降到 0）

## 6. 日志中应观察到的关键信号

- 训练开始前出现 `开始 HDF5 预热` 与 `[预热进度]`；
- 验证开始前同样出现一轮预热日志；
- 数据加载器初始化日志中可看到训练/验证文件级调度是否启用。

