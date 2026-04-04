# V1 动态导联场系统

本文档描述当前 V1 主线如何在真实数据 Stage1 训练中解析、缓存并注入动态导联场。

## 1. 作用范围

动态导联场主要服务于：
- `scripts/v1/train_stage1.py --mode real_finetune`
- `scripts/v1/evaluate_stage1.py --dataset_mode real`

仿真数据路径通常直接在样本中携带 `leadfield`，不需要运行时计算。

## 2. 相关模块

### 2.1 物理基础
- `penci/physics/source_space.py`：72 脑区源空间定义
- `penci/physics/electrode_utils.py`：电极坐标、指纹注册表、TSV 解析
- `penci/physics/leadfield_manager.py`：MNE 导联场计算与缓存

### 2.2 训练接线
- `penci/training/physics.py`
  - `setup_physics(config)`
  - `resolve_leadfield_for_batch(...)`
- `penci/shared/models/physics_decoder.py`
- `penci/v1/models/stage1_model.py`

## 3. 数据流

```text
metadata / electrodes.tsv / fingerprint archive
        │
        ▼
ElectrodeConfigRegistry
        │
        ▼
LeadfieldManager
        │
        ▼
resolve_leadfield_for_batch()
        │
        ▼
Stage1Model.forward(..., leadfield=L)
        │
        ▼
PhysicsDecoder.project_source_state(..., leadfield=L)
```

## 4. 工作机制

### 4.1 静态与动态两种来源

1. 若配置了固定导联场文件：
- `load_default_leadfield()` 直接加载默认矩阵
- 整个 batch 复用同一份 `L`

2. 若没有固定导联场：
- `setup_physics()` 初始化 `LeadfieldManager` 与 `ElectrodeConfigRegistry`
- `resolve_leadfield_for_batch()` 依据 batch 中的指纹或电极坐标获取 `L`
- 同配置样本优先命中注册表和缓存

### 4.2 缓存层次

- 电极指纹注册表：避免重复查找同一电极配置
- 导联场磁盘缓存：相同配置只需计算一次
- 导联场内存缓存：训练过程中重复 batch 可直接复用

### 4.3 DDP 注意点

- 多卡下仍然是每个进程独立持有自己的物理对象
- rank 0 负责关键准备与预热时，要配合 barrier 同步
- 真实数据路径中，leadfield 解析和 page-cache prefetch 是两条并行的“稳态基础设施”

## 5. 与 Stage1 的接口

`Stage1Model` 并不自己决定导联场来源，它只接受外部传入的 `leadfield`：

```python
output = model(
    x,
    pos,
    sensor_type,
    leadfield=leadfield,
    target_length=x.shape[-1],
)
```

随后 `PhysicsDecoder` 将 `S_t` 投影回传感器空间，用于重建损失与物理闭环。

## 6. 配置入口

当前主要配置位于 `configs/stage1_real.yaml`：

```yaml
model:
  physics:
    use_fixed_leadfield: true
    leadfield_path: null

physics:
  subjects_dir: "/work/2024/tanzunsheng/mne_data/MNE-fsaverage-data"
  leadfield_cache_dir: "/work/2024/tanzunsheng/leadfield_cache"
  processed_data_dir: "/work/2024/tanzunsheng/ProcessedData"
  fingerprint_registry_path: "/work/2024/tanzunsheng/leadfield_cache/fingerprint_registry.pt"
```

说明：
- `leadfield_path` 非空时，可视作固定导联场模式
- `fingerprint_registry_path` 非空时，优先走离线指纹注册表
- `processed_data_dir` 用于回退到 TSV 查找链路

## 7. 推荐实践

- 先运行 `scripts/precompute_all_leadfields.py` 建好指纹注册表与导联场缓存
- 真实数据训练优先使用 `configs/stage1_real.yaml`
- 若 NFS 较慢，和 `data.io_prefetch` 配合使用，而不是只调大 `num_workers`
- 修改 `penci/physics/` 逻辑后，最好重新验证 `tests/test_smoke.py` 中的导联场相关用例
