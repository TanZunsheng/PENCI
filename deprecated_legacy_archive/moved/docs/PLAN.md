# 节点级联合预热与训练期滑动续热方案

## Summary

把当前“`rank 0` 只预热自己本地 file plan、epoch 开始前一次性读一段”的方案，升级为 **单节点所有 rank 联合计划 + 训练期后台续热的 page cache 工作集管理**。

目标行为固定为：

- 预热范围：覆盖 **本节点所有 rank** 在当前 epoch 早期会用到的 HDF5 文件，而不是只覆盖 `rank 0`
- 启动策略：**达到高水位后才开始训练**
- 训练期策略：后台单线程顺序续热，维持 **低/高水位**
- 缓存层：继续使用 **OS page cache**，不做显式 Python/Tensor 内存缓存
- 工作集单位：**HDF5 文件消费窗口**（物理缓存仍然是整文件 page cache）
- 进度单位：**epoch 内 batch index**，不使用 `global_step`

## Implementation Changes

### 1. 采样器导出“每个 rank 的文件消费窗口计划”

在 `DistributedBucketBatchSampler` 上新增一个面向预热的稳定接口，返回 **当前 rank 在当前 epoch 的文件消费窗口计划**，而不是只返回简单文件列表。

接口输出固定为按消费顺序排列的列表，每项至少包含：

- `hdf5_path`
- `n_batches`
- `first_batch_idx`
- `last_batch_idx`

规则固定为：

- 仅在 `file_scheduler=true` 时启用该计划接口
- `first_batch_idx/last_batch_idx` 基于当前 rank 的 epoch 内 batch 序号计算
- 同一个文件如果在计划中是连续消费，只保留一段
- 同一个文件如果在 epoch 内被打散成多个不连续消费窗口，必须保留多段
- 如果 `file_scheduler=false`，训练侧回退到现有一次性本地预热逻辑，不启用新预取器

这样训练侧不需要猜“后面要读什么文件”，而是直接基于 sampler 的确定性计划做预热。

### 2. 训练侧构建“本节点所有 rank 的并集窗口计划”

在 `train.py` 中，`set_epoch(epoch)` 之后、训练开始之前：

- 每个 rank 获取自己的本地文件消费计划
- 使用 `dist.all_gather_object` 收集 **本节点所有 rank** 的计划
- 由 `rank 0` 构建 **节点级窗口计划**

节点级计划的构建规则固定为：

- 先按 `hdf5_path` 分组，再按 `first_batch_idx` 排序
- 仅对 **同一路径且时间上重叠/相邻** 的窗口做合并
- 如果同一路径在 epoch 内存在明显时间间隔，必须保留多个分离窗口
- 每个窗口保留：
  - `first_use_step`
  - `last_use_step`
  - `n_batches`
  - `ranks`
- 节点级计划按 `(first_use_step, hdf5_path, last_use_step)` 排序
- `rank 0` 负责解析绝对路径和文件大小
- 其他 rank 不做预热，只等待 `rank 0` 的预热与后台续热

这一步解决当前“只预热 `rank 0` 局部计划”的缺口。

### 3. 新增节点级后台预取器

在 `train.py` 内新增一个预取器组件，例如 `NodePageCachePrefetcher`，仅由 `rank 0` 创建和驱动。

组件职责固定为：

- 接收节点级窗口计划
- 管理窗口级逻辑工作集，而不是样本级工作集
- 在后台线程中顺序读取文件，填充 OS page cache
- 根据训练进度持续补未来文件

实现固定为：

- 后台线程数：`1`
- 读取方式：顺序读整文件，按现有 `read_chunk_mb`
- 不做随机读，不做多线程并发，不做 mmap，不做用户态 tensor cache

组件内部维护的状态固定为：

- `prefetched_cursor`：已经预热到计划中的哪个窗口
- `current_batch_idx`：训练当前进行到 epoch 内哪个 batch
- `active_prefetched_bytes`：已预热且 `last_use_step >= current_batch_idx` 的逻辑工作集大小
  - 这里需要按物理文件路径去重，不能因为同一文件有多个未来窗口而重复计数
- `high_watermark_bytes`
- `low_watermark_bytes`

### 4. 启动前预热与训练期续热逻辑

行为固定如下：

- 训练开始前：
  - `rank 0` 同步执行预热
  - 直到 `active_prefetched_bytes >= high_watermark_bytes`，或计划文件耗尽
  - 然后 `barrier`，所有 rank 再开始训练

- 训练进行中：
  - 每处理一个 batch，`rank 0` 用 **epoch 内 `batch_idx`** 更新预取器进度
  - 预取器根据新的 `current_batch_idx` 重算 `active_prefetched_bytes`
  - 当 `active_prefetched_bytes < low_watermark_bytes` 时，后台继续顺序读取后续文件
  - 一直补到 `active_prefetched_bytes >= high_watermark_bytes` 或文件耗尽

这里的“淘汰”不做显式删除，固定解释为：

- 当某窗口的 `last_use_step < current_batch_idx`，它从“逻辑工作集”中移出
- OS page cache 由内核自然回收
- 预取器只负责补新的未来文件，不主动 drop cache

### 5. 配置与参数

保留现有 `data.io_prefetch.warmup_gb` 作为 **高水位**，新增：

- `data.io_prefetch.low_watermark_gb`
- `data.io_prefetch.async_refill: true`
- `data.io_prefetch.scope: "node_union"`
- `data.io_prefetch.startup_policy: "high_watermark"`

默认值固定为：

- `high_watermark_gb = warmup_gb`
- `low_watermark_gb = 0.75 * high_watermark_gb`
- 例如 `80 GiB -> 60 GiB`
- `prefetch_threads` 仍强制按 `1` 执行

CLI 覆盖策略固定为：

- 保留现有 `--io_prefetch_warmup_gb` 覆盖高水位
- 本轮不新增低水位 CLI 参数，低水位走 YAML 配置

### 6. 日志与可观测性

日志固定补充以下信息：

- 节点级计划摘要：
  - `all ranks union` 文件数
  - 总文件体积
  - 高/低水位
- 启动前预热进度：
  - 已预热 GiB
  - 达到高水位百分比
  - 速度、ETA
- 训练期续热状态：
  - 当前 `batch_idx`
  - 当前逻辑工作集 GiB
  - 是否触发后台补水
  - 已预热到第几个窗口
- 文件计划来源：
  - 每个 rank 本地窗口条数
  - 节点级窗口条数
  - 节点级唯一文件条数

## Public Interfaces / Config Changes

需要新增或稳定化的接口/配置如下：

- 采样器新接口：
  - `get_prefetch_rank_schedule()`  
  返回当前 rank 的文件消费窗口计划，元素包含 `hdf5_path / n_batches / first_batch_idx / last_batch_idx`

- 训练侧新组件：
  - `NodePageCachePrefetcher`  
  封装启动前同步预热、训练期后台续热、进度更新和线程生命周期

- 新配置项：
  - `data.io_prefetch.low_watermark_gb`
  - `data.io_prefetch.async_refill`
  - `data.io_prefetch.scope`
  - `data.io_prefetch.startup_policy`

兼容策略固定为：

- `file_scheduler=false` 时，不启用节点级联合预热和后台续热
- `io_prefetch.enabled=false` 时，行为保持现状
- 验证集不启用该机制

## Test Plan

### 单元测试

1. 采样器计划导出
- 给定固定 epoch 和 file scheduler，`get_prefetch_rank_schedule()` 返回稳定顺序
- `first_batch_idx/last_batch_idx` 与本 rank batch 数一致
- 同文件连续 batch 被正确聚合
- 同文件的不连续 batch 会被保留为多个窗口

2. 节点级计划合并
- 多个 rank 的本地窗口输入后，能正确合并重叠/相邻窗口
- 同文件的分离窗口不会被错误压成一个大区间
- 排序按 `(first_use_step, path, last_use_step)` 稳定

3. 预取器水位逻辑
- 启动前能补到高水位
- `current_batch_idx` 推进后，旧文件退出逻辑工作集
- 当低于低水位时能继续补到高水位

### 集成/烟测

1. 单机 DDP 3 卡或 5 卡
- 日志中能看到 `all ranks union` 计划摘要
- 训练启动前确实等待到高水位
- 训练开始后能看到后台续热日志，而不是只有 epoch 开头一次预热

2. 回归检查
- `io_prefetch.enabled=false` 时训练流程不变
- `file_scheduler=false` 时回退行为正常
- 验证阶段不触发预取器

### 验收标准

- 训练期 NFS 读不再主要由“未覆盖的 rank 本地文件”触发
- `/proc/self/mountstats`、`io PSI`、`vmstat` 相比当前实现明显下降
- 训练吞吐不因预取线程引入新的卡死或同步阻塞

## Assumptions / Defaults

- 本轮实现范围固定为 **单节点 DDP**
- 缓存层固定为 **OS page cache**，不做显式共享内存 tensor cache
- 启动策略固定为 **达到高水位再启动训练**
- 高水位默认沿用 `warmup_gb`；低水位默认 `0.75 * high_watermark`
- `80 GiB` 场景下默认 `low_watermark = 60 GiB`
- 背景续热线程固定为 `1`
- 文件读取失败只记 warning，不中断训练；后续该文件仍允许按现有路径回退到训练时按需读取
