# PENCI: V1 两阶段神经连接建模

PENCI 当前的活动主线是 V1 两阶段架构，用于从 EEG/MEG 信号恢复显式脑区状态并进一步估计静态有效连接。
旧版单阶段主线及其历史文档已迁移到 `deprecated_legacy_archive/`，当前根目录只保留新主线相关内容。

## 当前主线

### Stage1: 状态恢复 + 物理闭环
- 入口模型：`penci.v1.models.Stage1Model`
- 目标：从 `x (B, C, T)` 恢复显式脑区状态 `S_t (B, N, T')`
- 约束：通过导联场将 `S_t` 投影回传感器空间，形成重建闭环
- 训练脚本：`scripts/v1/train_stage1.py`

### Stage2: 静态连接学习
- 入口模型：`penci.v1.models.StaticConnectivityModel`
- 目标：在冻结的 `S_t` 上学习 `A_base`
- 训练脚本：`scripts/v1/train_stage2.py`

### 共享基础设施
- 共享模型组件：`penci/shared/`
- 真实数据加载与分桶：`penci/data/dataset.py`
- DDP / prefetch / leadfield 训练基础设施：`penci/training/`
- 编码器与基础模块：`penci/encoders/`、`penci/modules/`

## 安装

```bash
/work/2024/tanzunsheng/anaconda3/envs/EEG/bin/pip install -e .
/work/2024/tanzunsheng/anaconda3/envs/EEG/bin/pip install -e ".[dev]"
```

## 项目结构

```text
PENCI/
├── penci/
│   ├── v1/                  # V1 两阶段主线
│   │   ├── models/          # Stage1Model, StaticConnectivityModel, StateHead
│   │   └── data/            # 仿真数据集
│   ├── shared/              # DynamicsCore, PhysicsDecoder 等共享模型组件
│   ├── training/            # DDP / prefetch / leadfield 训练基础设施
│   ├── data/                # 真实数据加载、HDF5、file scheduler
│   ├── physics/             # SourceSpace、LeadfieldManager、电极工具
│   ├── encoders/            # BrainOmni 移植编码器
│   └── modules/             # BrainOmni 移植基础模块
├── scripts/
│   ├── v1/                  # Stage1/Stage2 训练、评估、仿真数据生成
│   ├── convert_to_hdf5.py
│   ├── convert_hbn_to_hdf5.py
│   ├── convert_to_hdf5_by_fingerprint.py
│   └── precompute_all_leadfields.py
├── configs/
│   ├── stage1_sim.yaml
│   ├── stage1_real.yaml
│   ├── stage1_eval.yaml
│   ├── stage2_connectivity.yaml
│   └── stage2_eval.yaml
├── docs/                    # 当前新主线文档
├── deprecated_legacy_archive/
│   ├── moved/               # 已归档的旧代码/旧文档
│   └── snapshots/           # 重写前快照
└── tests/
    └── test_smoke.py
```

## 训练与评估

### Stage1 仿真预训练

```bash
/work/2024/tanzunsheng/anaconda3/envs/EEG/bin/python \
  scripts/v1/train_stage1.py \
  --config configs/stage1_sim.yaml \
  --mode sim_pretrain
```

### Stage1 真实数据微调

```bash
/work/2024/tanzunsheng/anaconda3/envs/EEG/bin/python \
  scripts/v1/train_stage1.py \
  --config configs/stage1_real.yaml \
  --mode real_finetune
```

### Stage1 多卡 DDP

```bash
torchrun --nproc_per_node=4 \
  scripts/v1/train_stage1.py \
  --config configs/stage1_real.yaml \
  --mode real_finetune
```

### Stage1 评估

```bash
/work/2024/tanzunsheng/anaconda3/envs/EEG/bin/python \
  scripts/v1/evaluate_stage1.py \
  --config configs/stage1_eval.yaml \
  --checkpoint outputs/stage1_real/best_model.pt \
  --dataset_mode real
```

### Stage2 训练与评估

```bash
/work/2024/tanzunsheng/anaconda3/envs/EEG/bin/python \
  scripts/v1/train_stage2.py \
  --config configs/stage2_connectivity.yaml \
  --stage1_checkpoint outputs/stage1_real/best_model.pt

/work/2024/tanzunsheng/anaconda3/envs/EEG/bin/python \
  scripts/v1/evaluate_stage2.py \
  --config configs/stage2_eval.yaml \
  --stage1_checkpoint outputs/stage1_real/best_model.pt \
  --stage2_checkpoint outputs/stage2/best_model.pt
```

## I/O 优化链路

V1 的 `real_finetune` 训练已经接入旧主线打磨过的 I/O 优化底座：

- HDF5 优先、`.pt` 回退的数据读取路径
- `BucketBatchSampler` + `DistributedBucketBatchSampler`
- `file_scheduler` 降低跨文件随机读
- `node-union page-cache prefetch` 预热
- 指纹驱动的动态导联场解析与缓存

核心入口分别在：
- `penci/data/dataset.py`
- `penci/training/prefetch.py`
- `penci/training/distributed.py`
- `penci/training/physics.py`

## 测试

```bash
/work/2024/tanzunsheng/anaconda3/envs/EEG/bin/python -m pytest tests/test_smoke.py
```

当前 smoke 已覆盖：
- 共享模块与编码器
- Stage1 / Stage2 主路径
- 动态导联场链路
- prefetch / file scheduler 关键逻辑
- 新主线命名空间导出

## 当前文档

- `docs/training_guide.md`：V1 训练与评估指南
- `docs/dynamic_leadfield_system.md`：动态导联场系统说明
- `docs/electrode_fingerprint_system.md`：电极指纹与注册表说明
- `docs/train_val_hdf5_nfs_strategy.md`：HDF5 / NFS / prefetch 策略

历史文档已移动到 `deprecated_legacy_archive/moved/docs/`。
