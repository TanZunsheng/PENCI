# Stage1 真实数据的 HDF5 / NFS 防卡顿策略

本文档只适用于当前活动主线：
`scripts/v1/train_stage1.py --config configs/stage1_real.yaml --mode real_finetune`

## 1. 总体结论

- Stage1 真实数据训练与验证都按 `batch` 读取，不会整集载入内存
- 样本读取优先走 HDF5（`hdf5_path + hdf5_idx`），失败时再回退 `.pt`
- “预热”主要利用的是 OS page cache，而不是应用层大内存池

## 2. 训练阶段的数据路径

1. 元数据来源：
- `configs/stage1_real.yaml` 中的 `real_train_metadata`
- 通过 `PENCIDataset` 和 `create_dataloader()` 构建 DataLoader

2. 样本读取：
- `__getitem__` 优先从 HDF5 读取 `x[idx]`
- 每个 worker 维护自己的 HDF5 句柄缓存
- `pos` / `sensor_type` 走 per-file 缓存，减少重复 I/O

3. Batch 组织：
- `BucketBatchSampler` / `DistributedBucketBatchSampler` 按 `(channels, fingerprint)` 分桶
- `file_scheduler` 启用后，同一段 batch 尽量来自同一 HDF5 文件

4. 预热：
- 每个 epoch 开始前可按计划顺序预热热点文件
- 训练中可根据低水位触发异步续热

## 3. 验证阶段的数据路径

- 验证集来自 `real_val_metadata`
- 默认 `shuffle=False`
- 可独立配置验证阶段文件调度：
  - `data.file_scheduler.val_enabled`
  - `data.file_scheduler.val_shuffle_within_file`
- 可在验证前单独做一轮预热：
  - `data.io_prefetch.warmup_before_val`
  - `data.io_prefetch.warmup_val_gb`
  - `data.io_prefetch.max_files_val`

## 4. 为什么这条链路更稳

- HDF5 合并减少了海量小 `.pt` 文件随机打开带来的元数据压力
- file scheduler 把 batch 聚簇到同一文件，降低跨文件随机读
- node-union prefetch 在正式迭代前把热点文件提前拉入页缓存
- `prefetch_threads=1` 能避免多线程把 NFS 拉满

## 5. 推荐配置

建议从 `configs/stage1_real.yaml` 的当前基线起步：

```yaml
data:
  use_bucket_sampler: true
  use_fingerprint: true
  file_scheduler:
    enabled: true
    shuffle_within_file: true
    val_enabled: true
    val_shuffle_within_file: false
  io_prefetch:
    enabled: true
    warmup_before_val: true
    prefetch_threads: 1
    read_chunk_mb: 8

training:
  num_workers: 1
```

## 6. 日志中应该看到什么

- 训练前出现 HDF5 预热日志
- DDP 模式下 rank schedule 被汇总为 node-union plan
- 验证前若开启 `warmup_before_val`，会再触发一轮验证集预热
- DataLoader 初始化日志里能看到 file scheduler / fingerprint / bucket sampler 是否启用
