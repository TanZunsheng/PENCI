# PENCI 训练指南

本文档提供 PENCI (Physics-constrained End-to-end Neural Connectivity Inference) 项目的训练指导，涵盖项目现状、环境准备、多卡训练流程、监控指标、评估体系及配置说明。

---

## 一、项目现状总结

PENCI 项目已进入生产就绪状态，核心架构与训练流水线已完全打通。

- **架构实现**：完整实现"三明治"架构：Encoder → Dynamics Core → Physics Decoder。
- **分布式支持**：原生支持多 GPU DDP 训练，具备学习率线性缩放和梯度同步机制。
- **评估系统**：构建了独立的三层评估体系，涵盖传感器空间、源空间及仿真验证。
- **验证情况**：所有 11 个核心测试用例全部通过，系统在 10 个 EEG 数据集上完成了端到端验证。
- **训练数据集**：支持 HBN_EEG, SEED-DV, THINGS-EEG 等 10 个主流数据集，样本量超百万。

---

## 二、环境准备

### 2.1 Conda 环境
请使用预配置的专用环境：
- **环境路径**：`/work/2024/tanzunsheng/anaconda3/envs/EEG`
- **Python 路径**：`/work/2024/tanzunsheng/anaconda3/envs/EEG/bin/python`

### 2.2 安装方式
在项目根目录下执行可编辑模式安装：
```bash
/work/2024/tanzunsheng/anaconda3/envs/EEG/bin/pip install -e .
```

### 2.3 必要数据路径
确保以下路径在服务器上可用：
- **fsaverage 头模型**：`/work/2024/tanzunsheng/mne_data/MNE-fsaverage-data/`
- **导联场缓存**：`/work/2024/tanzunsheng/leadfield_cache/`
- **BIDS 元数据**：`/work/2024/tanzunsheng/ProcessedData/`（含 `electrodes.tsv`）

---

## 三、训练方法

### 3.1 运行前验证
建议在正式训练前运行 smoke test 确保组件正常：
```bash
/work/2024/tanzunsheng/anaconda3/envs/EEG/bin/pytest tests/test_smoke.py -v
```

### 3.2 启动单卡训练
```bash
python scripts/train.py --config configs/default.yaml
```

### 3.3 多卡分布式训练 (DDP)
系统使用 `torchrun` 启动多卡并行训练，通过 `DistributedBucketBatchSampler` 确保各卡负载均衡。

**启动命令 (以 4 卡为例):**
```bash
torchrun --nproc_per_node=4 scripts/train.py --config configs/default.yaml
```

**关键机制说明:**
- **Batch Size**: 配置文件中的 `batch_size` 为每张显卡的局部批大小。全局 Batch Size = `batch_size × world_size`。
- **学习率缩放**: 系统自动执行线性缩放。`effective_lr = base_lr × world_size`。
- **步数定义**: `warmup_steps` 和 `max_steps` 均为全局迭代步数，不随 GPU 数量变化。
- **示例**: 若单卡 LR=1e-4, Batch=32，使用 8 卡训练时，实际 LR 变为 8e-4，全局 Batch 变为 256。

### 3.4 恢复训练
若训练中断，可通过 `--resume` 参数指定检查点路径继续训练：
```bash
python scripts/train.py --config configs/default.yaml --resume checkpoints/latest.pt
```

---

## 四、训练监控指标

训练过程中，系统会自动计算传感器空间的重建质量指标并记录至 TensorBoard。

| 指标名称 | 物理含义 | 取值范围 | 监控目标 |
|---------|---------|---------|---------|
| **Pearson** | 信号形状相似度 (相关系数) | [-1, 1] | 越高越好 (越接近 1) |
| **SNR (dB)** | 重建信噪比 | (-∞, +∞) | 越高越好 |
| **NRMSE** | 归一化均方根误差 | [0, +∞) | 越低越好 |

**TensorBoard 标量视图:**
- `val/pearson`, `val/snr_db`, `val/nrmse`: 验证集重建性能。
- `train/loss`, `train/lr`: 训练损失与实时学习率。

---

## 五、独立评估体系

通过 `scripts/evaluate.py` 进行多维度性能评估，分为三个层级：

1. **层级 1 (传感器空间)**: 在测试集上计算 Pearson, SNR, NRMSE。
2. **层级 2 (源空间对比)**: 以 sLORETA 为基准，对比推断出的 72 脑区活动与传统物理逆解的相关性。
3. **层级 3 (仿真测试)**: 使用模拟偶极子信号，计算定位误差 (DLE)。

**评估命令:**
```bash
# 执行完整三层评估
python scripts/evaluate.py --config configs/default.yaml --checkpoint path/to/model.pt --eval_mode all

# 仅执行传感器空间重建评估
python scripts/evaluate.py --config configs/default.yaml --checkpoint path/to/model.pt --eval_mode sensor
```

---

## 六、配置说明 (`default.yaml`)

### 6.1 核心训练参数
- `training.batch_size`: 每张 GPU 的批大小。
- `training.learning_rate`: 单卡基准学习率。
- `distributed.backend`: 默认使用 `nccl`。
- `evaluation`: 定义评估时使用的指标和 sLORETA 参数。

### 6.2 物理约束模式
1. **动态导联场模式** (`physics.fingerprint_registry_path` 已配置):
   系统根据 batch 中样本的电极配置自动检索预计算的物理矩阵。
2. **静态导联场模式**:
   设置 `use_fixed_leadfield: true` 并指定 `leadfield_path`。

---

## 七、数据格式说明

### 7.1 样本文件 (.pt)
每个样本 Tensor 字段：`x (C, T)`, `pos (C, 6)`, `sensor_type (C,)`。

---

## 八、物理约束系统工作原理

1. **SourceSpace**: 定义了 72 个标准脑区作为神经活动的目标空间。
2. **LeadfieldManager**: 利用 MNE-Python 实时计算或从缓存加载导联场矩阵。
3. **缓存机制**: 采用内存 + 磁盘两级缓存，确保多卡训练时物理矩阵的高效分发。

---

## 九、多数据集训练

1. **通道数聚类**: 通过 `BucketBatchSampler` 将相同通道数（如 128 或 64 通道）的样本归入同一 batch。
2. **多源适配**: 不同数据集自动匹配各自的导联场，但在统一的 72 脑区源空间内建模动力学。

---

## 十、常见问题

- **CUDA OOM**: 减小 `training.batch_size`。在 DDP 模式下，显存占用随 `n_dim` 和通道数增加。
- **多卡梯度不同步**: 确保 `dist.init_process_group` 成功。PENCI 已处理模型中的 `register_buffer` 以适应 DDP。

---

## 十一、项目结构快速参考

- `penci/utils/metrics.py`: 传感器空间核心监控指标实现。
- `scripts/evaluate.py`: 独立三层级评估脚本。
- `scripts/train.py`: 支持 DDP 的训练执行脚本。
- `penci/models/penci_model.py`: PENCI 主模型架构。
