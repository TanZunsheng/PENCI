# PENCI V1 训练指南

本文档只描述当前活动主线：V1 两阶段建模。
旧版单阶段训练入口和对应文档已经迁移到 `deprecated_legacy_archive/`。

---

## 1. 环境准备

### 1.1 Python 环境
- Conda 环境：`/work/2024/tanzunsheng/anaconda3/envs/EEG`
- Python：`/work/2024/tanzunsheng/anaconda3/envs/EEG/bin/python`

### 1.2 安装

```bash
/work/2024/tanzunsheng/anaconda3/envs/EEG/bin/pip install -e .
```

### 1.3 依赖路径
- 数据根目录：`/work/2024/tanzunsheng/PENCIData`
- fsaverage：`/work/2024/tanzunsheng/mne_data/MNE-fsaverage-data`
- 导联场缓存：`/work/2024/tanzunsheng/leadfield_cache`
- ProcessedData：`/work/2024/tanzunsheng/ProcessedData`

---

## 2. 主线路径概览

### 2.1 Stage1
`Stage1Model` 负责从 EEG 恢复显式脑区状态 `S_t`，并通过导联场将 `S_t` 投影回传感器空间。

### 2.2 Stage2
`StaticConnectivityModel` 在冻结的 `S_t` 上学习静态有效连接 `A_base`。

### 2.3 共享底座
当前主线共用以下训练基础设施：
- `penci/data/dataset.py`：真实数据/HDF5/file scheduler
- `penci/training/distributed.py`：DDP 初始化、同步、unwrap helpers
- `penci/training/prefetch.py`：node-union page-cache prefetch
- `penci/training/physics.py`：动态导联场准备与 batch 解析

---

## 3. Stage1 训练

### 3.1 仿真预训练

```bash
/work/2024/tanzunsheng/anaconda3/envs/EEG/bin/python \
  scripts/v1/train_stage1.py \
  --config configs/stage1_sim.yaml \
  --mode sim_pretrain
```

### 3.2 真实数据微调

```bash
/work/2024/tanzunsheng/anaconda3/envs/EEG/bin/python \
  scripts/v1/train_stage1.py \
  --config configs/stage1_real.yaml \
  --mode real_finetune
```

### 3.3 多卡 DDP

```bash
torchrun --nproc_per_node=4 \
  scripts/v1/train_stage1.py \
  --config configs/stage1_real.yaml \
  --mode real_finetune
```

### 3.4 恢复训练

```bash
/work/2024/tanzunsheng/anaconda3/envs/EEG/bin/python \
  scripts/v1/train_stage1.py \
  --config configs/stage1_real.yaml \
  --mode real_finetune \
  --resume outputs/stage1_real/latest.pt
```

### 3.5 关键说明
- `training.batch_size` 在 DDP 下是每卡 batch size
- 当前脚本不会自动做学习率线性缩放，`training.learning_rate` 按配置原值使用
- `real_finetune` 是当前 I/O 优化最完整的一条路径

---

## 4. Stage1 评估

### 4.1 仿真评估

```bash
/work/2024/tanzunsheng/anaconda3/envs/EEG/bin/python \
  scripts/v1/evaluate_stage1.py \
  --config configs/stage1_eval.yaml \
  --checkpoint outputs/stage1_sim/best_model.pt \
  --dataset_mode sim
```

### 4.2 真实数据评估

```bash
/work/2024/tanzunsheng/anaconda3/envs/EEG/bin/python \
  scripts/v1/evaluate_stage1.py \
  --config configs/stage1_eval.yaml \
  --checkpoint outputs/stage1_real/best_model.pt \
  --dataset_mode real
```

评估输出默认写入 `paths.stage1_eval_output` 或 `--output` 指定路径。

---

## 5. Stage2 训练与评估

### 5.1 训练

```bash
/work/2024/tanzunsheng/anaconda3/envs/EEG/bin/python \
  scripts/v1/train_stage2.py \
  --config configs/stage2_connectivity.yaml \
  --stage1_checkpoint outputs/stage1_real/best_model.pt
```

### 5.2 评估

```bash
/work/2024/tanzunsheng/anaconda3/envs/EEG/bin/python \
  scripts/v1/evaluate_stage2.py \
  --config configs/stage2_eval.yaml \
  --stage1_checkpoint outputs/stage1_real/best_model.pt \
  --stage2_checkpoint outputs/stage2/best_model.pt
```

Stage2 当前主要面向仿真连接恢复，不走 HDF5 / file scheduler 真实数据链路。

---

## 6. I/O 与 NFS 优化

当前只在 `Stage1 real_finetune` 路径上完整启用：
- HDF5 优先读取
- `BucketBatchSampler` / `DistributedBucketBatchSampler`
- `file_scheduler`
- `node-union page-cache prefetch`
- 电极指纹驱动的动态导联场查找

对应配置主要位于 `configs/stage1_real.yaml`：
- `data.file_scheduler.*`
- `data.io_prefetch.*`
- `physics.*`
- `training.num_workers`

推荐起点：
- `training.num_workers: 1`
- `data.file_scheduler.enabled: true`
- `data.io_prefetch.enabled: true`
- `data.io_prefetch.prefetch_threads: 1`

---

## 7. 常见配置文件

- `configs/stage1_sim.yaml`：Stage1 仿真预训练
- `configs/stage1_real.yaml`：Stage1 真实数据微调
- `configs/stage1_eval.yaml`：Stage1 独立评估
- `configs/stage2_connectivity.yaml`：Stage2 训练
- `configs/stage2_eval.yaml`：Stage2 评估

---

## 8. 常见问题

### 8.1 DDP 卡住
优先检查：
- `torch.cuda.set_device(local_rank)` 是否在初始化进程组前执行
- rank 0 的 leadfield / prefetch 预热后是否有 barrier 同步
- 数据路径和 HDF5 文件是否对所有 rank 可见

### 8.2 NFS 抖动导致吞吐不稳
优先检查：
- `training.num_workers` 是否过高
- `file_scheduler` 是否开启
- `node-union page-cache prefetch` 是否开启
- HDF5 转换是否完整

### 8.3 Stage2 表现差
优先检查：
- Stage1 checkpoint 是否足够稳定
- `use_ground_truth_state` 是否需要作为对照开启
- `lag_order`、`max_radius`、`l1_sparsity` 是否需要重调

---

## 9. 验证

```bash
/work/2024/tanzunsheng/anaconda3/envs/EEG/bin/python -m pytest tests/test_smoke.py
```

当前 smoke 已覆盖 V1 主线的训练组件、动态导联场链路和 prefetch 关键逻辑。
