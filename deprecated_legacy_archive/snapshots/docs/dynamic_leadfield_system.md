# PENCI 动态导联场系统

## 一、概述

PENCI 采用物理约束的深度学习架构，其核心挑战之一是处理不同被试、不同数据集之间各异的
电极配置（通道数量、位置及命名）。动态导联场系统通过集成 MNE-Python 的物理前向建模能力，
为每种电极配置自动计算对应的导联场矩阵，作为 **不可训练的固定物理约束** 注入解码器。

主要设计目标：

| 目标 | 实现方式 |
|------|----------|
| 物理一致性 | 基于 MNE 的 BEM 前向模型，确保源→传感器映射符合电磁物理 |
| 动态适配 | 根据通道配置 hash 自动计算，无需手动预计算 |
| 高效复用 | 两级缓存（内存 LRU + 磁盘 .pt），相同配置只计算一次 |
| 多进程安全 | 文件锁防止分布式训练中重复计算 |
| 固定约束 | 导联场不可训练，使用 `register_buffer` 注册 |

---

## 二、架构设计

系统由三个协同工作的核心模块组成，全部位于 `penci/physics/` 目录：

### 2.1 SourceSpace（源空间定义）

**文件**: `penci/physics/source_space.py`

定义固定的 72 个源点，使用 `fsaverage` 模板头模型作为标准空间：

| 分类 | 数量 | 来源 | 说明 |
|------|------|------|------|
| 皮层脑区 | 68 | DK68 (aparc) 标签 | 每个标签取质心坐标 |
| 左海马体 | 1 | aseg.mgz 体积分割 | Left-Hippocampus |
| 右海马体 | 1 | aseg.mgz 体积分割 | Right-Hippocampus |
| 左杏仁核 | 1 | aseg.mgz 体积分割 | Left-Amygdala |
| 右杏仁核 | 1 | aseg.mgz 体积分割 | Right-Amygdala |

主要属性：
- `positions`: `(72, 3)` MNI 米制坐标
- `names`: 长度 72 的名称列表
- `cortical_positions` / `subcortical_positions`: 分组坐标
- `src`: MNE 混合源空间对象（延迟构建）

### 2.2 ElectrodeUtils（电极工具）

**文件**: `penci/physics/electrode_utils.py`

负责从 BIDS 格式的 `electrodes.tsv` 中提取电极坐标并过滤无效通道：

- `find_electrodes_tsv()`: 根据数据集名/被试 ID/站点自动定位文件
- `read_electrodes_tsv()`: 解析 TSV，返回 `Dict[str, np.ndarray]`（通道名→(3,)坐标）
- `filter_channels_like_postprocess()`: 复制 BrainOmniPostProcess 的过滤逻辑（移除 NaN、全零坐标）
- `get_valid_channels_for_dataset()`: 一键式接口，包含查找+读取+排除非 EEG 通道+过滤

### 2.3 LeadfieldManager（导联场管理器）

**文件**: `penci/physics/leadfield_manager.py`

系统的调度中心。根据通道配置计算唯一 hash，管理两级缓存，调用 MNE 计算前向解：

- `get_leadfield(channel_names, channel_positions)` → `(n_channels, 72)` 张量
- `get_leadfield_for_batch(...)` → `(batch_size, n_channels, 72)` 张量
- 内部方法: `_compute_leadfield()`, `_extract_region_leadfield()`

---

## 三、数据流

```text
ProcessedData/{dataset}/bids/derivatives/.../electrodes.tsv
      │
      │  find_electrodes_tsv() + read_electrodes_tsv()
      ▼
Dict[通道名 → (3,) 米制坐标]
      │
      │  filter_channels_like_postprocess() + 排除 Cz 等非 EEG 通道
      ▼
有效通道列表 (channel_names) + 坐标矩阵 (n_channels, 3)
      │
      │  _compute_channel_hash(): SHA256(排序通道名 + 坐标)[:16]
      ▼
16 位十六进制 hash (例如: 2d5f98901176184b)
      │
      ├── [内存缓存命中] → 直接返回 (< 1ms)
      ├── [磁盘缓存命中] → torch.load → 返回 (< 10ms)
      └── [缓存未命中]
            │
            │  MNE: create_info → make_forward_solution → convert_forward_solution
            ▼
      完整导联场 (n_channels, n_total_sources × 3)  ← 自由方向
            │
            │  _extract_region_leadfield(): 提取 72 个脑区
            ▼
      导联场张量 (n_channels, 72) → 缓存到磁盘 + 内存
            │
            │  PhysicsDecoder.forward(): einsum('bcn,bntd->bctd', L, source)
            ▼
      重建传感器信号 (B, n_channels, T, D)
```

---

## 四、技术细节

### 4.1 混合源空间的导联场提取

MNE 的前向解包含 6 个源空间：

| 索引 | 类型 | 内容 | 活跃源点数 |
|------|------|------|-----------|
| 0 | surface | 左半球皮层 | ~10,242 |
| 1 | surface | 右半球皮层 | ~10,242 |
| 2 | volume | 左海马体 | ~39 |
| 3 | volume | 右海马体 | ~42 |
| 4 | volume | 左杏仁核 | ~14 |
| 5 | volume | 右杏仁核 | ~19 |

**关键技术决策**: 使用 `surf_ori=True, force_fixed=False`

- `force_fixed=True` 会对混合源空间报错（体积源无表面法线）
- `surf_ori=True` 将皮层源旋转到表面法线坐标系，每个源点 3 列：
  - `col[3i+0]` = 切向方向 1
  - `col[3i+1]` = 切向方向 2
  - `col[3i+2]` = **法线方向** ← 这是我们需要的
- 体积源保持 XYZ 三自由度

### 4.2 导联场列提取策略

**68 个皮层脑区**（DK68 aparc 标签）:

对每个标签内所有属于源空间的顶点，取法线方向列（偏移 `+2`）的平均值：

```python
# 每个源点法线方向列 = src_idx * 3 + 2
normal_col_indices = [vert_to_src_idx[v] * 3 + 2 for v in label.vertices if v in vert_to_src_idx]
L_72[:, label_idx] = L_full[:, normal_col_indices].mean(axis=1)
```

依据：MNE `convert_forward_solution` 中 `force_fixed` 分支取 `source_nn[2::3]`
（`mne/forward/forward.py` 第 819 行），证实法线在每组 3 列中的第 3 个位置。

**4 个皮层下结构**（海马体 + 杏仁核）:

体积源无法线方向，取 3 个方向导联场向量的 L2 范数再平均：

```python
magnitudes = np.sqrt(Lx**2 + Ly**2 + Lz**2)  # (n_channels, n_active)
L_72[:, 68 + sub_idx] = magnitudes.mean(axis=1)
```

### 4.3 缓存机制

两级缓存确保相同电极配置只需计算一次：

| 级别 | 存储 | 容量 | 读取速度 | 淘汰策略 |
|------|------|------|---------|---------|
| 内存 | `OrderedDict` | 16 条目（可配） | < 1ms | LRU |
| 磁盘 | `{cache_dir}/{hash}.pt` | 无限 | < 10ms | 不淘汰 |

**Hash 计算**: SHA256（排序后的通道名 + 8 位小数坐标），取前 16 位十六进制。

**文件锁**: 使用 `O_CREAT | O_EXCL` 原子创建 `.lock` 文件，防止多进程同时计算。
超时 30 次重试后强制删除残留锁。

### 4.4 PhysicsDecoder 集成

`PhysicsDecoder`（`penci/models/physics_decoder.py`）在 `use_fixed_leadfield=True` 模式下，
通过 `einsum` 将导联场矩阵应用于源空间活动：

```python
# leadfield: (B, n_sensors, n_sources) 或 (n_sensors, n_sources)
# source_activity 经过线性投影得到 source_signal: (B, n_sources, T)
# 输出: reconstruction = leadfield @ source_signal → (B, n_sensors, T)
```

导联场通过 `PENCI.forward(leadfield=...)` 参数逐层传递至 `PhysicsDecoder`。

---

## 五、使用示例

### 基本用法

```python
import numpy as np
from penci.physics import SourceSpace, LeadfieldManager, get_valid_channels_for_dataset

# 1. 初始化物理环境
ss = SourceSpace(subjects_dir="/path/to/mne_data/MNE-fsaverage-data")
lm = LeadfieldManager(
    source_space=ss,
    subjects_dir="/path/to/mne_data/MNE-fsaverage-data",
    cache_dir="/path/to/leadfield_cache",
)

# 2. 获取特定被试的电极配置
valid_channels, names = get_valid_channels_for_dataset(
    processed_data_dir="/path/to/ProcessedData",
    dataset_name="HBN_EEG",
    subject_id="NDARVA281NVV",
    site="R11",
)
pos = np.array([valid_channels[n] for n in names])  # (128, 3)

# 3. 获取导联场（首次约 75 秒，后续 < 10ms）
leadfield = lm.get_leadfield(names, pos)  # (128, 72), float32

# 4. 传入模型
output = model(x, pos_tensor, sensor_type, leadfield=leadfield)
```

### 批量训练用法

```python
# BucketBatchSampler 确保同一 batch 内通道配置相同
leadfield_batch = lm.get_leadfield_for_batch(
    names, pos, batch_size=32
)  # (32, 128, 72)

losses = model.compute_loss(x, pos, sensor_type, leadfield=leadfield_batch)
```

---

## 六、配置参数

在 `configs/default.yaml` 中的相关配置：

```yaml
# 模型 - 物理解码器设置
model:
  physics:
    use_fixed_leadfield: true      # 使用固定物理导联场（不通过注意力学习）
    leadfield_path: null           # null 表示动态模式，forward 时传入

# 物理约束配置
physics:
  subjects_dir: "/path/to/MNE-fsaverage-data"  # fsaverage 数据目录
  leadfield_cache_dir: "/path/to/leadfield_cache"   # 导联场缓存目录
  processed_data_dir: "/path/to/ProcessedData"       # BIDS 数据目录（含 electrodes.tsv）

# 数据配置（必须开启 BucketBatchSampler）
data:
  use_bucket_sampler: true         # 按通道数分桶，确保同 batch 内配置一致
```

**注意**:
- `use_bucket_sampler: true` 是动态导联场的前提条件，确保同一 batch 内所有样本的通道配置相同
- `subjects_dir` 必须包含已下载的 `fsaverage`（BEM + label + surf + mri/aseg.mgz）
- `fsaverage` 不会自动下载（用户网络无法访问 OSF），需要手动准备
