# 电极指纹系统

## 概述

电极指纹系统解决了 PENCI 在多数据集混合训练时的核心问题：**如何确保每个样本使用正确的导联场矩阵，同时避免冗余计算**。

### 问题背景

训练数据来自多个数据集（HBN_EEG、SEED-DV、ThingsEEG 等），存在以下情况：

1. **不同数据集通道数不同**（如 128ch、129ch、6ch）
2. **通道数相同但电极位置不同**（不同采集设备）
3. **同一数据集内不同被试电极配置不同**（如 HBN_EEG 有多个采集站点）
4. **大量样本共享相同的电极配置**（同一设备采集的数据）

旧方案按通道数分桶、假设同通道数 = 同电极配置，这在多设备混合时不成立。
旧方案还假设每个数据集只有一种电极配置（只读取第一个被试的 electrodes.tsv），在 HBN_EEG 等多站点数据集上会导致 `KeyError`。

### 解决方案：两阶段架构

引入**离线预计算 + 在线秒级加载**的两阶段架构：

**阶段一（离线）**：`scripts/precompute_all_leadfields.py`
- 全局扫描所有样本的 `.pt` 文件，提取电极坐标，计算指纹
- 支持增量计算：已有指纹的样本零 I/O 复用，仅计算新增样本
- 智能去重：相同指纹只计算一次导联场
- 通过 TSV 文件获取通道名和米制坐标，调用 MNE 计算导联场
- 输出：指纹注册表存档 `.pt` 文件

**阶段二（在线）**：训练时 O(1) 加载
- `setup_physics()` 检测到 `physics.fingerprint_registry_path` 配置后
- 调用 `ElectrodeConfigRegistry.load_from_archive()` 直接加载
- 跳过一切运行时扫描和 MNE 计算
- 向后兼容：未配置存档路径时沿用旧的运行时扫描模式

---

## 历史 Bug 修复记录

### Bug #1：采样推断导致少数派电极配置被静默吞并

**影响版本**：初始实现
**影响文件**：`scripts/precompute_all_leadfields.py`、`penci/data/dataset.py`
**修复日期**：2026-03-10

**问题描述**：

旧版本的 `scan_and_fingerprint()` 和 `_precompute_fingerprints()` 采用**采样推断**策略：
按 `(dataset, channels)` 分组后，每组仅均匀采样 5 个代表样本计算指纹。如果 5 个代表的
指纹完全一致，则判定整组为"同质组"，将该指纹直接赋给组内全部样本。

```python
# 旧代码（已删除）
max_representatives = 5
step = len(indices) // max_representatives
rep_indices = [indices[i * step] for i in range(max_representatives)]
# ... 如果 len(valid_fps) <= 1 → 整组共享一个指纹
```

**触发场景**：

HBN_EEG 数据集包含多个采集站点（SI、RU、CBIC 等），不同站点使用不同的 EEG 设备，
导致同一 `(HBN_EEG, 129)` 组内存在多种电极配置。当某站点的样本数量远少于其他站点时
（例如 500 个 vs 20000 个），5 个均匀采样的代表极有可能全部来自多数派站点，
导致少数派站点的全部样本被静默分配到错误的指纹 → 使用错误的导联场矩阵。

**后果**：
- 少数派电极配置的样本在物理解码阶段使用错误的导联场矩阵
- 不产生任何错误或警告（静默错误）
- 训练损失可能略微偏高但不会崩溃，难以被发现

**修复方案**：

彻底废弃采样推断策略，改为**精确计算每个样本的指纹**：

- 离线脚本（`precompute_all_leadfields.py`）：使用 `ProcessPoolExecutor` 多进程并行计算
  所有样本的指纹，通过 `--max_workers` 参数控制并行度（默认 16 进程）
- 在线回退（`dataset.py`）：逐一串行计算每个样本的指纹，并附带警告建议先运行离线脚本

---

### Bug #2：`all()` 门控导致增量场景下全量重算

**影响版本**：Bug #1 修复后的版本
**影响文件**：`scripts/precompute_all_leadfields.py`、`penci/data/dataset.py`
**修复日期**：2026-03-10

**问题描述**：

Bug #1 修复后，代码使用 `all()` 函数判断是否所有样本都已有指纹：

```python
# 旧代码（已删除）
has_fp_field = all(
    "fingerprint" in m and m["fingerprint"] not in (None, "", "unknown")
    for m in all_metadata
)
if has_fp_field:
    # 策略 1: 全部零 I/O 复用
else:
    # 策略 2: 全部重新计算（ProcessPoolExecutor 并行）
```

**触发场景**：

假设 PENCIData 中已有 170 万个样本，且均已通过历史运行回写了 `fingerprint` 字段。
此时用户新增了一个包含 1 万个样本的新数据集（其 metadata 尚未计算指纹）。

由于 `all()` 检查 171 万条记录中只要有 1 条缺失指纹就返回 `False`，
导致已有指纹的 170 万老样本全部被丢进 `ProcessPoolExecutor` 重新计算。

**后果**：
- 170 万次无谓的磁盘 I/O 读取 `.pt` 文件
- 原本只需几分钟的增量计算被拖慢为数小时的全量重算
- 在线 `dataset.py` 启动时间从秒级退化为十几分钟

**修复方案**：

将 `all()` 二元门控替换为**逐条增量检测**：

```python
# 新代码
for idx, meta in enumerate(all_metadata):
    fp = meta.get("fingerprint")
    if fp and fp not in (None, "", "unknown"):
        # 已有有效指纹 → 零 I/O 复用
        path_to_fingerprint[pt_path] = fp
    else:
        # 缺失指纹 → 加入待计算列表
        need_compute.append((idx, pt_path))
```

- 170 万老样本：零 I/O，直接从 metadata 读取指纹字符串
- 1 万新样本：并行计算（离线脚本）或串行计算（在线回退）
- 两个文件同步修复，保持逻辑一致

---

### Bug #3：`next(iter(set))` 随机选取 dataset_name 导致 TSV 查找路径错误

**影响版本**：Bug #2 修复后的版本
**影响文件**：`scripts/precompute_all_leadfields.py`
**修复日期**：2026-03-10

**问题描述**：

`find_tsv_for_fingerprint()` 使用 `next(iter(datasets))` 从 `set` 中取一个 dataset_name，
但 Python 的 `set` 是无序的，选取结果不确定。同时 `example_pt` 指向的样本可能来自另一个 dataset：

```python
# 旧代码（已删除）
pt_path = fp_info["example_pt"]
datasets = fp_info["datasets"]
dataset_name = next(iter(datasets))  # 从无序 set 中随机选取！
```

**触发场景**：

Broderick2018 数据集包含多个子数据集（Broderick2018_CocktailParty、Broderick2018_NaturalSpeech 等），
它们共享相同的电极配置（相同 fingerprint）。`unique_fingerprints[fp]["datasets"]` 为
`{"Broderick2018_CocktailParty", "Broderick2018_NaturalSpeech"}`。

假设 `example_pt` 指向 `Broderick2018_CocktailParty/sub-22` 的 `.pt` 文件，
但 `next(iter(datasets))` 随机选到了 `Broderick2018_NaturalSpeech`。
此时 `find_electrodes_tsv()` 会在 `ProcessedData/.../Broderick2018_NaturalSpeech/sub-22/` 下查找
electrodes.tsv，但 sub-22 只存在于 CocktailParty 子数据集中，导致 `FileNotFoundError`。

**后果**：
- 该 fingerprint 组的导联场计算被跳过（`find_tsv_for_fingerprint` 返回 `None`）
- 所有共享该 fingerprint 的样本在训练时无法获取正确的导联场
- 随机性：每次运行结果可能不同（取决于 `set` 的迭代顺序）

---

### Bug #4：单一 `example_pt` 故障导致整个 fingerprint 组失败

**影响版本**：Bug #2 修复后的版本
**影响文件**：`scripts/precompute_all_leadfields.py`
**修复日期**：2026-03-10

**问题描述**：

每个 fingerprint 只存储一个 `example_pt` 路径。如果该样本的 `electrodes.tsv` 缺失或损坏，
整个 fingerprint 组（可能包含 1000+ subjects）全部失败，即使组内其他 999 个 subject 的 TSV 都正常：

```python
# 旧代码（已删除）
unique_fingerprints[fp] = {
    "datasets": {dataset_name},
    "channels": n_channels,
    "example_pt": pt_path,       # 单一路径，无 fallback
}
```

**触发场景**：

某个 fingerprint 组覆盖 3 个数据集的 1200 个 subject。`example_pt` 恰好指向其中一个
subject 的 `.pt` 文件。如果该 subject 的 `electrodes.tsv` 丢失（数据预处理问题），
`find_tsv_for_fingerprint` 直接返回 `None`，整个 fingerprint 组被跳过。

**后果**：
- 1200 个 subject 的导联场全部无法计算
- 错误率放大：1 个文件缺失 → 1200 个样本受影响

**Bug #3 + Bug #4 联合修复方案**：

将 `example_pt` 单一路径替换为 `example_candidates` 候选列表，每个候选为
`(pt_path, dataset_name)` 元组，严格配对。`find_tsv_for_fingerprint()` 依次尝试
每个候选，第一个成功即返回：

```python
# 新代码
MAX_EXAMPLE_CANDIDATES = 5

# scan_and_fingerprint() 中构建候选列表
unique_fingerprints[fp] = {
    "datasets": {dataset_name},
    "channels": n_channels,
    "example_candidates": [(pt_path, dataset_name)],  # 配对存储
}
# 后续遇到同一 fingerprint 的不同 dataset 时追加候选
candidates = unique_fingerprints[fp]["example_candidates"]
existing_datasets = {ds for _, ds in candidates}
if dataset_name not in existing_datasets and len(candidates) < MAX_EXAMPLE_CANDIDATES:
    candidates.append((pt_path, dataset_name))

# find_tsv_for_fingerprint() 中带 fallback 的查找
for attempt_idx, (pt_path, dataset_name) in enumerate(candidates):
    subject_id, session, site = extract_subject_info_from_pt_path(pt_path)
    if subject_id is None:
        continue
    try:
        tsv_path = find_electrodes_tsv(...)
    except FileNotFoundError:
        continue
    # 读取 + 过滤 → 成功则返回
    return channel_names, channel_positions

# 所有候选均失败时才返回 None
```

**设计要点**：
- Bug #4 方案是 Bug #3 的超集：每个候选自带配对的 `dataset_name`，天然消除错配
- 候选收集策略优先跨 dataset 多样化（`dataset_name not in existing_datasets`），
  最大化 fallback 覆盖面
- `MAX_EXAMPLE_CANDIDATES = 5` 限制内存开销
- 候选尝试失败时使用 `logger.debug()` 记录，最终全部失败时 `logger.warning()`

---

## 坐标空间与 Scheme B 设计

### 坐标空间差异

系统中存在两个坐标空间：

| 来源 | 坐标范围 | 精度 | 用途 |
|------|---------|------|------|
| `.pt` 文件 `pos[:,:3]` | ~[-0.65, 0.65]（归一化） | bfloat16 | 运行时指纹计算 |
| `electrodes.tsv` | ~[-0.1, 0.1]（米制 CapTrak） | float64 | MNE 导联场计算 |

`.pt` 文件中的坐标经过 BrainOmniPostProcess 的 `normalize_pos()` 处理（去中心 + 缩放），
并以 bfloat16 精度存储。这导致 `.pt` 坐标和 TSV 坐标计算出的指纹**永远不同**。

### Scheme B：双源指纹策略

离线预计算脚本采用 Scheme B 解决坐标空间不匹配问题：

1. **指纹来源**：从 `.pt` 文件的 `pos[:,:3]` 计算 `pos_fingerprint`
   - 保证与运行时 `PENCIDataset` 计算的指纹完全一致
2. **导联场来源**：从 `electrodes.tsv` 读取米制坐标 + 通道名
   - MNE 前向模型需要真实物理坐标
3. **双向注册**：将 `.pt` 指纹和 TSV 指纹都指向同一个通道配置
   - 运行时通过 `.pt` 指纹查找 → 得到正确的通道名和米制坐标 → 计算导联场

这样做的好处是**完全不需要在 PENCI 中复刻 `normalize_pos()` 的逆变换**。

---

## 双指纹设计

系统维护两种指纹：

| 指纹类型 | 输入 | 用途 | 计算函数 |
|---------|------|------|---------|
| **full_fingerprint** | 通道名 + 坐标 | LeadfieldManager 缓存键 | `_compute_channel_hash()` |
| **pos_fingerprint** | 仅坐标 (x,y,z) | Dataset/Sampler 分桶键 | `compute_fingerprint_from_pos()` |

### 为什么需要两种指纹？

- `.pt` 样本文件只有 `pos: (C, 6)` 张量，**不含通道名**
- 导联场计算需要通道名（用于匹配 electrodes.tsv 中的通道）
- 解决方案：
  1. Dataset 从 `.pt` 文件提取 `pos[:, :3]` 计算 `pos_fingerprint`
  2. Registry 注册时同时计算两种指纹，建立 `pos_fp → full_fp` 映射
  3. 训练时通过 `pos_fingerprint` 查找完整配置（通道名 + 坐标）

---

## 核心组件

### 1. `compute_fingerprint_from_pos(positions)`

位于 `penci/physics/leadfield_manager.py`。

```python
from penci.physics.leadfield_manager import compute_fingerprint_from_pos

pos = np.array([[0.02, 0.08, 0.01], [-0.02, 0.08, 0.01]])
fp = compute_fingerprint_from_pos(pos)  # → "a1b2c3d4e5f6a7b8"
```

**特性**：
- 输入：`(n_channels, 3)` 坐标数组（米制或归一化均可，但同一系统内必须一致）
- 输出：16 字符十六进制字符串
- 行顺序无关（内部按 `np.lexsort` 排序）
- float32/float64 一致（量化到 0.1mm 整数后哈希）
- 形状和 dtype 编入哈希，不同通道数必然产生不同指纹

### 2. `_compute_channel_hash(names, positions)`

位于 `penci/physics/leadfield_manager.py`。

```python
from penci.physics.leadfield_manager import _compute_channel_hash

full_fp = _compute_channel_hash(["Fp1", "Fp2"], positions)
```

与 `compute_fingerprint_from_pos` 类似，但额外包含通道名。用于 `LeadfieldManager` 的磁盘缓存键。

### 3. `ElectrodeConfigRegistry`

位于 `penci/physics/electrode_utils.py`。

```python
from penci.physics import ElectrodeConfigRegistry

# === 方式 A：从离线存档加载（推荐） ===
registry = ElectrodeConfigRegistry.load_from_archive("leadfield_registry.pt")

# === 方式 B：运行时扫描（向后兼容） ===
registry = ElectrodeConfigRegistry(processed_data_dir)
registry.register_dataset("HBN_EEG")

# === 方式 C：直接注册（无需文件 I/O） ===
pos_fp = registry.register_config(channel_names, channel_positions)

# 查询
names, positions = registry.get_config_by_fingerprint(pos_fp)
has = registry.has_fingerprint(pos_fp)
all_fps = registry.get_all_fingerprints()

# 保存为存档（用于离线脚本）
registry.save_to_archive("leadfield_registry.pt")
```

**内部索引**：
- `_configs`: `(dataset, n_channels) → (names, positions)` — 旧接口兼容
- `_fingerprint_configs`: `pos_fp → (names, positions)` — 指纹查询
- `_pos_to_full_fingerprint`: `pos_fp → full_fp` — 指纹桥接

### 4. `PENCIDataset` 指纹预计算

位于 `penci/data/dataset.py`。

```python
dataset = PENCIDataset(
    metadata_path="train.json",
    precompute_fingerprints=True,  # 启用指纹预计算
)

# 获取单个样本的指纹
fp = dataset.get_fingerprint(0)
```

**增量计算策略**：

对所有样本逐条检查 metadata 中是否已有 `"fingerprint"` 字段：
- 已有有效指纹 → 零 I/O 复用（直接读取字符串）
- 缺失指纹 → 逐一加载 `.pt` 文件计算（附带进度日志和警告）

这样在新增数据集时，老样本零开销，仅计算新增样本的指纹。

### 5. `BucketBatchSampler` 指纹分桶

```python
sampler = BucketBatchSampler(
    dataset=dataset,
    batch_size=32,
    use_fingerprint=True,  # 按 (ch, fp) 分桶
)
```

- `use_fingerprint=True`：桶键为 `(n_channels, pos_fingerprint)` 元组
- `use_fingerprint=False`（默认）：桶键为 `n_channels`，向后兼容

### 6. `PENCICollator` 指纹传递

Collator 自动从第一个样本的 metadata 中提取 `fingerprint` 字段，附加到 batch 输出：

```python
batch = collator(samples)
batch["fingerprint"]  # → "a1b2c3d4e5f6a7b8"
```

---

## 离线预计算脚本

### 用法

```bash
# 基本用法：扫描所有数据集，计算导联场，保存注册表存档
python scripts/precompute_all_leadfields.py \
    --penci_data /work/2024/tanzunsheng/PENCIData \
    --processed_data /work/2024/tanzunsheng/ProcessedData \
    --subjects_dir /work/2024/tanzunsheng/mne_data/MNE-fsaverage-data \
    --cache_dir /work/2024/tanzunsheng/leadfield_cache \
    --archive_out /work/2024/tanzunsheng/leadfield_cache/fingerprint_registry.pt

# 干跑模式：只扫描和报告，不计算导联场
python scripts/precompute_all_leadfields.py --dry_run --verbose

# 同时将指纹回写到 metadata JSON（加速训练初始化）
python scripts/precompute_all_leadfields.py --update_metadata

# 指定并行进程数（默认 16）
python scripts/precompute_all_leadfields.py --max_workers 32
```

### 四步流水线

1. **扫描与指纹化**（增量）：发现所有 `*-metadata/` 目录中的 JSON，逐条检查已有指纹（零 I/O 复用），仅对缺失指纹的样本并行计算
2. **查找 TSV**：根据 fingerprint 的多个候选 `(pt_path, dataset_name)` 依次尝试逆向定位对应的 `electrodes.tsv`，第一个成功即使用；应用 BrainOmniPostProcess 过滤规则获取通道名和米制坐标
3. **计算导联场**：对每个唯一指纹调用 `LeadfieldManager` 进行 MNE 前向计算
4. **保存存档**：将注册表序列化为 `.pt` 存档；可选地将指纹回写到 metadata JSON

### 存档格式

```python
{
    "__version__": 1,
    "__created_at__": "2025-01-15T10:30:00",
    "configs": {
        "a1b2c3d4e5f6a7b8": {       # pos_fingerprint
            "channel_names": ["Fp1", "Fp2", ...],
            "channel_positions_m": np.ndarray,  # (N, 3) 米制坐标
            "full_fingerprint": "x9y8z7w6v5u4t3s2",
            "leadfield_cache_path": "/path/to/cache/abc123.pt",  # 可选
        },
        ...
    }
}
```

---

## 数据流

### 推荐流程（两阶段）

```
阶段一：离线预计算
==================

scripts/precompute_all_leadfields.py
  │
  ├─ 扫描 PENCIData/*-metadata/*.json
  │   ├─ 已有 fingerprint 字段 → 零 I/O 复用
  │   └─ 缺失 fingerprint → 并行加载 .pt 计算
  │       └─ pos[:,:3] → compute_fingerprint_from_pos() → pos_fingerprint
  │
  ├─ 逆向定位 ProcessedData/.../electrodes.tsv
  │   └─ filter_channels_like_postprocess() → (channel_names, positions_m)
  │
  ├─ LeadfieldManager.get_leadfield() → MNE 前向计算（或命中缓存）
  │
  └─ registry.save_to_archive("fingerprint_registry.pt")
      └─ 可选: --update_metadata 将 fingerprint 回写到 JSON


阶段二：在线训练
================

1. setup_physics() 检测 physics.fingerprint_registry_path
   └─ ElectrodeConfigRegistry.load_from_archive() → O(1) 初始化

2. PENCIDataset(precompute_fingerprints=True)
   ├─ 有 metadata["fingerprint"]? → 零 I/O 直接使用
   └─ 无? → 逐一加载 .pt 计算指纹（增量：仅计算缺失的样本）

3. BucketBatchSampler 按 (ch, pos_fp) 分桶

4. 训练循环:
   PENCICollator → batch["fingerprint"]
     → resolve_leadfield_for_batch()
       → registry.get_config_by_fingerprint()
       → leadfield_manager.get_leadfield() (内存/磁盘缓存)
       → leadfield: (n_channels, 72) → PhysicsDecoder
```

### 旧流程（运行时扫描，向后兼容）

```
1. setup_physics() 未检测到 fingerprint_registry_path
   └─ ElectrodeConfigRegistry(processed_data_dir) + register_dataset()

2. PENCIDataset(precompute_fingerprints=True)
   └─ 逐一计算所有样本指纹（无缓存可复用）

3-4. 同上
```

---

## 配置

### YAML 配置项

```yaml
data:
  use_bucket_sampler: true    # 必须启用（动态导联场模式自动启用）
  datasets:                    # 会自动注册电极配置
    - HBN_EEG
    - SEED-DV
    - ThingsEEG

physics:
  subjects_dir: "/path/to/mne_data/MNE-fsaverage-data"
  leadfield_cache_dir: "/path/to/leadfield_cache"
  processed_data_dir: "/path/to/ProcessedData"
  # 离线预计算存档路径（推荐，设为 null 则使用运行时扫描模式）
  fingerprint_registry_path: "/path/to/leadfield_cache/fingerprint_registry.pt"
```

### 自动行为

当 `physics` 配置启用动态导联场时，训练脚本自动：

1. 启用 `use_bucket_sampler = True`
2. 启用 `use_fingerprint = True`
3. 预计算所有样本的电极指纹
4. 在训练开始前预热所有唯一指纹的导联场

---

## API 参考

### `compute_fingerprint_from_pos(positions, precision_mm=0.1)`

| 参数 | 类型 | 说明 |
|------|------|------|
| `positions` | `np.ndarray (N, 3)` | 电极坐标 |
| `precision_mm` | `float` | 量化精度（毫米） |
| **返回** | `str` | 16 字符十六进制指纹 |

### `ElectrodeConfigRegistry`

| 方法 | 说明 |
|------|------|
| `load_from_archive(path)` | **（类方法）** 从离线存档加载注册表 |
| `save_to_archive(path)` | 将注册表序列化到存档文件 |
| `register_dataset(name)` | 从 ProcessedData 注册数据集 |
| `register_config(names, pos, dataset_name=None)` | 直接注册配置 |
| `get_config_by_fingerprint(pos_fp)` | 按指纹查询 `(names, positions)` |
| `has_fingerprint(pos_fp)` | 检查指纹是否已注册 |
| `get_all_fingerprints()` | 列出所有已注册指纹 |
| `get_config(dataset, n_ch)` | 旧接口（向后兼容） |

### `load_registry_from_archive(archive_path)`

| 参数 | 类型 | 说明 |
|------|------|------|
| `archive_path` | `str` | 存档文件路径 (.pt) |
| **返回** | `ElectrodeConfigRegistry` | 已填充的注册表实例 |

便捷函数，等价于 `ElectrodeConfigRegistry.load_from_archive(archive_path)`。

### `resolve_leadfield_for_batch(fingerprint, manager, registry, device)`

| 参数 | 类型 | 说明 |
|------|------|------|
| `fingerprint` | `str` | batch 的电极位置指纹 |
| `manager` | `LeadfieldManager` | 导联场管理器 |
| `registry` | `ElectrodeConfigRegistry` | 电极配置注册表 |
| `device` | `torch.device` | 计算设备 |
| **返回** | `torch.Tensor (C, 72)` | 导联场矩阵 |

---

## 常见问题

### Q: 如果 `.pt` 文件中电极坐标有微小浮点差异怎么办？

量化精度为 0.1mm（`precision_mm=0.1`），即坐标差异小于 0.05mm 的被视为相同。这足以吸收 float32↔float64 转换和 bfloat16 存储带来的舍入误差。

### Q: 为什么 `.pt` 指纹和 TSV 指纹不同？

`.pt` 文件中的坐标经过 `normalize_pos()` 处理（去中心 + 缩放到 [-1,1] 范围）并以 bfloat16 精度存储，而 TSV 坐标是米制 CapTrak 原始值。`compute_fingerprint_from_pos()` 使用 `round(coord * 10000)` 量化，在这两个坐标空间中会产生完全不同的整数，因此指纹永远不匹配。

Scheme B 设计通过在离线脚本中**同时注册两种指纹指向同一配置**来解决这个问题。

### Q: 如果某个 pos_fingerprint 在 registry 中找不到怎么办？

`get_config_by_fingerprint()` 会抛出 `KeyError`，训练会中断。解决方法：
1. 确保运行了 `precompute_all_leadfields.py` 且覆盖了所有数据集
2. 如果使用运行时扫描模式，确保所有数据集都已通过 `register_dataset()` 注册

### Q: HBN_EEG 没有 electrodes.tsv 怎么办？

HBN_EEG 的 ProcessedData 目录中没有 electrodes.tsv 文件。离线脚本会记录警告并跳过该数据集的导联场计算。需要手动提供 electrodes.tsv 或使用其他方式获取电极坐标。

### Q: 向后兼容性如何？

- 不配置 `physics.fingerprint_registry_path`（或设为 `null`）→ 完全沿用旧的运行时扫描模式
- `use_fingerprint=False`（默认）时，`BucketBatchSampler` 退化为按通道数分桶
- `get_config(dataset, n_channels)` 旧接口保留
- 不启用物理约束时，指纹系统不被激活，对现有流程零影响

### Q: 新增数据集后需要全量重算吗？

不需要。指纹系统采用增量策略：

- **离线脚本**：逐条检查 metadata，已有 `fingerprint` 字段的样本零 I/O 跳过，仅对缺失指纹的新样本并行计算。新增 1 万样本到 170 万样本的库中，只需计算 1 万次。
- **在线回退**：同样逐条检查，已有指纹的直接复用，缺失的逐一计算。

建议新增数据集后重新运行 `precompute_all_leadfields.py --update_metadata`，
将新样本的指纹回写到 metadata JSON，确保后续训练启动时全部走零 I/O 路径。

### Q: 预计算指纹的性能如何？

| 场景 | 离线脚本 | 在线回退 |
|------|---------|---------|
| 全部已有指纹（170 万样本） | < 10 秒（纯 JSON 读取） | < 5 秒 |
| 全量首次计算（170 万样本） | ~9 分钟（16 进程并行） | ~2 小时（串行） |
| 增量新增（1 万 / 170 万） | ~30 秒 | ~3 分钟 |

离线脚本使用 `ProcessPoolExecutor` 多进程并行，可通过 `--max_workers` 调整进程数。
