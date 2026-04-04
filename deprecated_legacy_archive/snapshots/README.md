# PENCI: 基于物理约束的端到端神经连接推断

PENCI (Physics-constrained End-to-end Neural Connectivity Inference) 是一个基于物理约束的深度学习框架，用于从 EEG/MEG 信号推断神经动力学。

## 架构设计

PENCI 采用"三明治"架构，包含三个核心组件：

### 1. 编码器 (Encoder)
- **来源**：移植自 BrainOmni 项目（移除量化层）
- **功能**：将传感器空间的 EEG/MEG 信号映射到隐式源空间
- **模块**：
  - `BrainSensorModule`: 传感器位置和类型嵌入
  - `SEANetEncoder`: 时域卷积编码器
  - `BackWardSolution`: 交叉注意力反向求解

### 2. 动力学核心 (Dynamics Core)
- **来源**：原创设计
- **功能**：建模源空间中神经活动的时间演化
- **实现**：
  - `DynamicsCore`: 基于 Transformer 的动力学模型
  - `DynamicsRNN`: 基于 LSTM 的轻量级替代方案

### 3. 物理解码器 (Physics Decoder)
- **功能**：将源空间活动映射回传感器空间
- **约束**：可使用固定的导联场矩阵，确保物理一致性
- **实现**：
  - 固定导联场模式：使用预计算的物理矩阵
  - 注意力模式：可学习的 ForwardSolution

## 安装

```bash
# 克隆仓库
git clone https://github.com/TanZunsheng/PENCI.git
cd PENCI

# 使用指定环境安装
/work/2024/tanzunsheng/anaconda3/envs/EEG/bin/pip install -e .

# 或仅安装依赖
/work/2024/tanzunsheng/anaconda3/envs/EEG/bin/pip install -r requirements.txt
```

## 项目结构

```
PENCI/
├── penci/
│   ├── encoders/           # 编码器模块（移植自 BrainOmni）
│   │   ├── sensor_embed.py      # 传感器嵌入
│   │   ├── backward_solution.py # 反向/前向求解
│   │   └── encoder.py           # 完整编码器
│   ├── modules/            # 基础模块
│   │   ├── attention.py         # 注意力、RMSNorm、FFN
│   │   ├── conv.py              # 卷积模块
│   │   ├── lstm.py              # LSTM 模块
│   │   └── seanet.py            # SEANet 编码器
│   ├── models/             # 模型定义
│   │   ├── dynamics.py          # 动力学核心
│   │   ├── physics_decoder.py   # 物理解码器
│   │   └── penci_model.py       # 完整 PENCI 模型
│   └── data/               # 数据模块
│       └── dataset.py           # 数据集和加载器
├── scripts/                # 脚本
│   └── train.py                 # 训练脚本
├── configs/                # 配置文件
│   └── default.yaml             # 默认配置
├── tests/                  # 测试
│   └── test_smoke.py            # Smoke test
├── requirements.txt        # 依赖列表
├── pyproject.toml          # 项目配置
└── README.md               # 本文件
```

## 数据格式

PENCI 使用与 BrainOmni 预处理兼容的数据格式：

```python
# 每个样本是一个 .pt 文件，包含：
{
    "x": Tensor (C, T),        # EEG/MEG 信号
                               # C = 通道数, T = 时间采样点
    "pos": Tensor (C, 6),      # 电极位置 + 方向
                               # (x, y, z, nx, ny, nz)
    "sensor_type": Tensor (C,) # 传感器类型
                               # 0=EEG, 1=MEG_grad, 2=MEG_mag
}
```

数据路径：`/work/2024/tanzunsheng/PENCIData/`

支持的数据集：
- HBN_EEG
- SEED-DV
- Broderick2018
- Grootswagers2019
- ThingsEEG

## 快速开始

### 基础用法

```python
import torch
from penci.models import PENCI

# 创建模型
model = PENCI(
    n_dim=256,           # 特征维度
    n_neuro=64,          # 源数量
    n_head=4,            # 注意力头数
    n_sensors=128,       # 传感器数量
    dynamics_type="transformer",
    dynamics_layers=4,
)

# 准备输入
batch_size = 4
x = torch.randn(batch_size, 128, 2560)  # 10秒 @ 256Hz
pos = torch.randn(batch_size, 128, 6)
sensor_type = torch.zeros(batch_size, 128, dtype=torch.long)

# 前向传播
output = model(x, pos, sensor_type, return_source=True)

print(f"重建信号: {output['reconstruction'].shape}")
print(f"源活动: {output['source_activity'].shape}")
```

### 计算损失

```python
# 计算训练损失
losses = model.compute_loss(x, pos, sensor_type)

print(f"总损失: {losses['loss'].item():.4f}")
print(f"重建损失: {losses['recon_loss'].item():.4f}")
print(f"动力学损失: {losses['dynamics_loss'].item():.4f}")
```

### 使用数据加载器

```python
from penci.data import get_train_val_loaders

train_loader, val_loader = get_train_val_loaders(
    data_root="/work/2024/tanzunsheng/PENCIData",
    dataset_name="HBN_EEG",
    batch_size=32,
    num_workers=4,
)

for batch in train_loader:
    x = batch["x"]           # (B, C, T)
    pos = batch["pos"]       # (B, C, 6)
    sensor_type = batch["sensor_type"]  # (B, C)
    
    output = model(x, pos, sensor_type)
    # ...
```

## 训练

### 运行训练

```bash
# 使用默认配置训练
/work/2024/tanzunsheng/anaconda3/envs/EEG/bin/python scripts/train.py \
    --config configs/default.yaml

# 覆盖配置参数
/work/2024/tanzunsheng/anaconda3/envs/EEG/bin/python scripts/train.py \
    --config configs/default.yaml \
    --output_dir outputs/experiment_1
```

### Smoke Test

验证模型是否可以正常工作：

```bash
/work/2024/tanzunsheng/anaconda3/envs/EEG/bin/python tests/test_smoke.py
```

## 配置说明

`configs/default.yaml` 包含以下主要配置：

```yaml
model:
  n_dim: 256              # 特征维度
  n_neuro: 64             # 源数量
  n_head: 4               # 注意力头数
  
  seanet:
    n_filters: 32         # SEANet 滤波器数
    ratios: [8, 4, 2]     # 下采样比例
    
  dynamics:
    type: "transformer"   # 动力学类型
    n_layers: 4           # 层数
    
  physics:
    use_fixed_leadfield: true  # 使用固定导联场

data:
  root_dir: "/work/2024/tanzunsheng/PENCIData"
  sample_rate: 256
  n_channels: 128

training:
  batch_size: 32
  learning_rate: 1e-4
  max_steps: 100000
```

## 技术细节

### 从 BrainOmni 移植的模块

以下模块直接从 BrainOmni 移植，保持核心实现不变：

- `penci/modules/attention.py` - RMSNorm, FeedForward, SelfAttention, RotaryEmbedding
- `penci/modules/conv.py` - SConv1d, SConvTranspose1d
- `penci/modules/lstm.py` - SLSTM
- `penci/modules/seanet.py` - SEANetEncoder, SEANetResnetBlock, Snake1d
- `penci/encoders/sensor_embed.py` - BrainSensorModule
- `penci/encoders/backward_solution.py` - BackWardSolution, ForwardSolution

### 原创模块

- `penci/models/dynamics.py` - DynamicsCore (Transformer), DynamicsRNN
- `penci/models/physics_decoder.py` - PhysicsDecoder
- `penci/models/penci_model.py` - PENCI, PENCILite

### 与 BrainOmni 的主要区别

1. **移除量化层**：BrainOmni 使用 RVQ 进行离散化，PENCI 直接使用连续表示
2. **添加动力学建模**：PENCI 引入 Transformer 动力学核心
3. **物理约束**：PENCI 可使用固定导联场矩阵

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.7+ (推荐)

Conda 环境：`/work/2024/tanzunsheng/anaconda3/envs/EEG`

## 许可证

MIT License

## 致谢

- 编码器架构改编自 [BrainOmni](https://github.com/xxx/BrainOmni) 项目
- 数据预处理流程遵循 BrainOmni 标准
