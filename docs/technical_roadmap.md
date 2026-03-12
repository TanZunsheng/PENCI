# PENCI 技术路线文档

> Physics-constrained End-to-end Neural Connectivity Inference
>
> 本文档整合了 PENCI 项目从初始设计到后续功能扩展的完整技术路线，详细说明每个模块要做什么、为什么这样做、以及当前的实现状态。

---

## 一、项目定位与核心思想

### 1.1 要解决的问题

EEG（脑电图）记录的是头皮表面电极采集到的电信号，但我们真正关心的是**大脑皮层内部神经源的活动**。从传感器信号推断源活动，本质上是一个**逆问题**（inverse problem）。传统方法（如 eLORETA、beamforming）依赖手工特征和先验假设，缺乏端到端学习的能力。

PENCI 的目标是：**用深度学习端到端地从原始 EEG 信号中推断神经源活动，同时利用物理约束（导联场矩阵）保证解的物理合理性。**

### 1.2 核心设计理念

PENCI 采用**"三明治"架构**，将物理约束嵌入到深度学习模型中：

```
原始 EEG 信号 (B, C, T)
        │
        ▼
┌─────────────────────────┐
│  ① 编码器 (Encoder)      │  传感器空间 → 源空间
│  数据驱动，学习逆映射     │  (B, C, T) → (B, 72, T', D)
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│  ② 动力学核心 (Dynamics)  │  源空间内时间演化
│  建模神经动力学规律       │  (B, 72, T', D) → (B, 72, T', D)
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│  ③ 物理解码器 (Decoder)   │  源空间 → 传感器空间
│  固定导联场矩阵，不可训练  │  (B, 72, T', D) → (B, C, T')
└─────────────────────────┘
        │
        ▼
重建的 EEG 信号 (B, C, T')
```

**关键约束**：

- 编码器是纯数据驱动的（可学习参数）
- 动力学核心是可学习的（Transformer 或 RNN）
- **解码器使用固定的物理导联场矩阵（不参与训练）**，这是物理约束的核心——迫使编码器和动力学核心学到的源空间表示必须是物理上有意义的
- 源空间维度固定为 **72**（对应 72 个脑区）
- 训练目标：输入 EEG → 编码到源空间 → 动力学演化 → 导联场投影回传感器空间 → 与原始 EEG 计算重建损失

### 1.3 为什么不用交叉注意力做前向映射

早期设计中考虑过用可学习的交叉注意力（ForwardSolution）来模拟源→传感器的映射。但这存在根本问题：如果前向映射也是可学习的，模型可以"作弊"——编码器和解码器可以协商出一套与真实物理无关的源空间表示，只要重建损失低即可。**固定物理导联场是唯一的物理锚点**，确保源空间表示具有真实的神经生理学意义。

---

## 二、数据规格

### 2.1 数据来源

数据路径：`/work/2024/tanzunsheng/PENCIData`

| 数据集 | 描述 |
|--------|------|
| HBN_EEG | Healthy Brain Network EEG 数据 |
| Brennan_Hale2019 | 自然语言理解 EEG 数据 |
| Grootswagers2019 | 视觉感知 EEG 数据 |
| THINGS-EEG | 物体识别 EEG 数据 |
| ThingsEEG | THINGS 数据的另一版本 |
| SEED-DV | 情感识别 EEG 数据 |
| Broderick2018_NaturalSpeech | Broderick 自然语言语音数据 |
| Broderick2018_NaturalSpeechReverse | Broderick 反向语音数据 |
| Broderick2018_SpeechInNoise | Broderick 噪声背景语音数据 |
| Broderick2018_CocktailParty | Broderick 鸡尾酒会效应数据 |

共 **10** 个数据集，样本总量约 **814.5 万** 条。

### 2.2 单个样本格式

每个样本是一个 `.pt` 文件，包含字典：

```python
{
    "x": tensor,          # (C, T) — EEG 信号，dtype=bfloat16
    "pos": tensor,        # (C, 6) — 电极位置 (x, y, z, nx, ny, nz)，dtype=bfloat16
    "sensor_type": tensor  # (C,)   — 传感器类型，dtype=int32，全部为 0（EEG）
}
```

其中：
- `C` = 通道数（电极数），不同样本可能不同
- `T` = 时间采样点数
- `pos` 的前 3 维是 3D 坐标 `(x, y, z)`，单位为**米**（已确认）
- `pos` 坐标范围约 `[-0.1, 0.1]`（米制环境下）

### 2.3 电极配置

项目实现了一套**电极指纹系统 (Electrode Fingerprint)**。虽然通道数（如 128ch）可能相同，但电极坐标的微小差异可能代表不同的布局。

**重要特性**：通过 `BucketBatchSampler` 确保同一 batch 内的所有样本共享相同的电极指纹，从而共享同一个导联场矩阵。

---

## 三、模块设计详解

### 3.1 编码器（Encoder）

#### 3.1.1 作用

将原始 EEG 传感器空间信号编码到 72 维隐式源空间表示。这对应 EEG 逆问题的求解。

#### 3.1.2 架构（移植自 BrainOmni，已移除量化层）

```
输入: (B, C, T)
  │
  ├─ 时间分段: (B, C, T) → (B, C, N, L)    # N=T/window_size 段，每段长度 L
  │
  ├─ SEANet 编码: (B*C*N, 1, L) → (B*C*N, D, T')  # 每通道每段独立编码
  │     └─ 1D 卷积 + 残差块 + LSTM + 下采样（64倍）
  │     └─ 输出: 每段从 L=320 个采样点压缩为 T'=5 个时间步
  │
  ├─ 传感器嵌入: (B, C, 6) + (B, C) → (B, C, D)
  │     └─ BrainSensorModule: 位置 MLP + 传感器类型 Embedding + FFN 融合
  │
  └─ 反向求解: (B*W, C, D) → (B*W, 72, D)    # W=N*T'
        └─ BackWardSolution: 交叉注意力
        └─ Query = 可学习的 72 个神经源基向量 (nn.Parameter)
        └─ Key = 传感器嵌入 + 编码信号
        └─ Value = 编码信号

输出: (B, 72, N, T', D)
```

**关键设计决策**：
- SEANet 对每个通道**独立编码**（不做通道间交互），通道间信息融合由反向求解模块完成
- 72 个源基向量 `self.neuros` 是**可学习参数**，代表隐式的神经源位置
- 反向求解使用交叉注意力，让源基向量（Query）从传感器信号（Key/Value）中提取信息

#### 3.1.3 涉及文件

| 文件 | 类 | 说明 |
|------|-----|------|
| `penci/encoders/encoder.py` | `PENCIEncoder` | 顶层编码器，处理输入格式适配和时间分段 |
| `penci/encoders/encoder.py` | `BrainTokenizerEncoder` | 核心编码器，组合 SEANet + 传感器嵌入 + 反向求解 |
| `penci/encoders/sensor_embed.py` | `BrainSensorModule` | 传感器嵌入：位置 + 类型 → 统一嵌入 |
| `penci/encoders/backward_solution.py` | `BackWardSolution` | 交叉注意力实现的逆问题求解 |
| `penci/modules/seanet.py` | `SEANetEncoder` | 1D 卷积编码器（移植自 Meta Encodec） |

### 3.2 动力学核心（Dynamics Core）

#### 3.2.1 作用

在源空间内建模时间演化。编码器输出的是离散时间步的源活动快照，动力学核心学习这些快照之间的时间关联。

#### 3.2.2 架构

**Transformer 模式**（默认）：

```
输入: (B, 72, T_total, D)
  │
  ├─ 重排列: (B*72, T_total, D)    # 每个源独立处理时间序列
  │
  ├─ N 层 SelfAttnBlock:
  │     └─ RMSNorm → 自注意力（带 RoPE） → 残差
  │     └─ RMSNorm → FFN（SELU 激活） → 残差
  │
  └─ 最终 RMSNorm

输出: (B, 72, T_total, D)
```

**关键设计决策**：
- 72 个源之间**不做跨源注意力**（每个源独立处理时间序列），跨源交互由导联场矩阵隐式完成
- 使用因果注意力（causal=True），建模时间的单向依赖
- 使用旋转位置编码（RoPE）编码时间步的相对位置

### 3.3 物理解码器（Physics Decoder）

#### 3.3.1 作用

将源空间的神经活动映射回传感器空间的 EEG 信号。这是项目的物理核心，通过真实的导联场矩阵实现。

#### 3.3.2 实现详情 ✅

```
输入: source_activity (B, 72, T, D)
  │
  ├─ 时间特征解码: (B, 72, T, D) → (B, 72, T)
  │     └─ Linear(D→D) → GELU → Linear(D→1) → squeeze
  │
  └─ 导联场投影: (B, 72, T) → (B, C, T)
        └─ einsum("cs,bst->bct", L, source_signal)
        └─ L 为 (C, 72) 的真实物理导联场矩阵

输出: (B, C, T) — 重建的传感器空间信号
```

**关键特性**：
- **物理真实性**：导联场矩阵由 `LeadfieldManager` 动态提供，基于 MNE-Python 计算。
- **单 Batch 共享**：利用 `BucketBatchSampler` 保证 batch 内电极配置一致，因此 `einsum` 使用 `"cs,bst->bct"` 形式即可，无需 per-sample batched einsum。

---

## 四、动态导联场系统 ✅

### 4.1 系统概述

该系统已完整实现，解决了早期模型使用随机矩阵和仅支持固定通道数的问题。

### 4.2 核心组件

1.  **LeadfieldManager**： ✅ 实现
    - **自动识别**：基于电极坐标生成唯一指纹 (Fingerprint)。
    - **MNE 计算**：首次遇到新指纹时，调用 MNE 基于 fsaverage 头模型计算 (C, 20484) 导联场。
    - **空间聚合**：利用 DK68 图谱 + 4 个皮层下区域将 20,484 个顶点聚合为 **72 个脑区**。
    - **两级缓存**：内存缓存 + 磁盘缓存（含文件锁，支持多进程安全）。

2.  **SourceSpace (72 源定义)**： ✅ 实现
    - 采用 Desikan-Killiany (DK68) 皮层分区（68个区）。
    - 增加 4 个关键皮层下区域：双侧海马 (Hippocampus) 和双侧杏仁核 (Amygdala)。
    - 聚合策略：对每个区域内的所有源点取平均（Mean aggregation）。

3.  **ElectrodeConfigRegistry**： ✅ 实现
    - 管理所有已知的数据集电极配置。
    - 提供离线预计算支持，避免训练时的重复扫描。

### 4.3 物理计算流

```python
# 获取导联场矩阵 L
L = leadfield_manager.get_leadfield(pos) # (C, 72)

# 执行物理映射
# source_signal: (B, 72, T)
# sensor_reconstruction = L @ source_signal
sensor_signal = torch.einsum("cs,bst->bct", L, source_signal)
```

---

## 五、实施计划 ✅

### 阶段 0：前置依赖 ✅
- **fsaverage 数据**：✅ 已就绪，存放在 `/work/2024/tanzunsheng/mne_data/`。
- **坐标系确认**：✅ 已确认使用 MNE 米制坐标系。
- **72 源定义**：✅ 已确定为 DK68 (68) + Subcortical (4)。

### 阶段 1：LeadfieldManager ✅
- 实现 `penci/physics/` 模块。
- 完成 `LeadfieldManager`、`SourceSpace` 和 `ElectrodeConfigRegistry`。

### 阶段 2：PhysicsDecoder 改造 ✅
- 接入动态导联场获取机制。
- 实现基于电极指纹的 batch 级别投影。

### 阶段 3：PENCI 模型适配 ✅
- 更新 `forward` 接口传递 `pos`。
- 实现 `setup_physics()` 初始化流程。

### 阶段 4：数据管线 ✅
- **BucketBatchSampler (方案B)**：✅ 已实现。按 (通道数, 指纹) 分桶。
- **DistributedBucketBatchSampler**：✅ 已实现。支持 DDP 环境下的分桶采样。

### 阶段 5：配置和测试 ✅
- 11 项单元测试与集成测试全部通过。

### 阶段 6：验证提交 ✅
- 完整训练链路已在 8 卡集群验证通过。

---

## 六、已决决策问题 ✅

| # | 问题 | 状态 | 最终决策 |
|---|------|------|---------|
| 1 | **72 个源空间如何定义？** | ✅ 已决定 | DK68 (68皮层) + 双侧海马/杏仁核 (4皮层下) = 72 |
| 2 | **pos 坐标单位是什么？** | ✅ 已确认 | 米制坐标 (MNE 标准) |
| 3 | **batch 内不同通道数如何处理？** | ✅ 已实现 | 方案 B：使用 `BucketBatchSampler` 分桶采样，保证 batch 内一致 |
| 4 | **缓存路径放哪？** | ✅ 已配置 | `/work/2024/tanzunsheng/leadfield_cache/` |
| 5 | **无 MNE fallback** | ✅ 已实现 | 在无 MNE 环境下自动回退到随机矩阵并发出警告 |

---

## 七、当前项目状态总览

### 已完成 ✅

| 模块 | 状态 | 说明 |
|------|------|------|
| **核心模型** | ✅ 完成 | 编码器、动力学核心、物理解码器全线贯通 |
| **物理系统** | ✅ 完成 | LeadfieldManager, SourceSpace (72 regions), 指纹系统 |
| **分布式支持** | ✅ 完成 | DDP, DistributedBucketBatchSampler, 线性 LR 缩放 |
| **评估体系** | ✅ 完成 | 三层评估架构 (传感器/源空间/仿真), `evaluate.py` |
| **数据管线** | ✅ 完成 | 支持 10 个数据集，分桶采样，AMP 兼容 |
| **工程优化** | ✅ 完成 | 导联场离线预计算, 混合精度 (AMP) 修复, Warmup+Cosine |
| **测试覆盖** | ✅ 通过 | 11 项测试全绿 |

---

## 八、关键约束清单

| 约束 | 原因 |
|------|------|
| 禁止 import BrainOmni | 所有代码必须在 PENCI 项目内自包含 |
| 导联场不可训练 | 物理约束的核心——必须是固定的物理映射 |
| n_neuro = 72 | 固定源空间维度，对应 DK68+4 标准分区 |
| 注释和文档使用中文 | 项目规范 |
| 电极指纹一致性 | 保证 batch 内导联场共享的前提 |

---

## 九、配置参数说明

```yaml
model:
  n_dim: 256                    # 全局特征维度
  n_neuro: 72                   # 源空间脑区数量 (DK68 + 4)
  n_head: 4                     # 编码器交叉注意力头数
  
  dynamics:
    type: "transformer"         # 动力学模型类型
    n_layers: 4                 # Transformer 层数
    n_heads: 8                  # 注意力头数

  physics:
    use_fixed_leadfield: true   # 核心物理约束
    leadfield_path: null        # 动态计算模式

data:
  datasets:                     # 10 个标准数据集
    - "HBN_EEG"
    - "Brennan_Hale2019"
    - ...
  use_bucket_sampler: true      # 必须开启以支持动态导联场

physics:
  subjects_dir: "/work/2024/tanzunsheng/mne_data/MNE-fsaverage-data"
  leadfield_cache_dir: "/work/2024/tanzunsheng/leadfield_cache"
  fingerprint_registry_path: "/work/2024/tanzunsheng/leadfield_cache/fingerprint_registry.pt"

training:
  batch_size: 32                # per-GPU batch size
  learning_rate: 1.0e-4         # 基准学习率 (DDP 下自动线性缩放)
  warmup_steps: 1000
  mixed_precision: true         # AMP 混合精度
```

---

## 十、多卡分布式训练 (DDP)

PENCI 完美支持基于 `torchrun` 的多卡分布式训练：

-   **启动方式**：使用 `torchrun --nproc_per_node=N scripts/train.py`。
-   **LR 缩放**：执行线性缩放规则：`effective_lr = base_lr × world_size`。
-   **采样器**：`DistributedBucketBatchSampler` 负责在 DDP 环境下对各桶数据进行均匀切分，确保各 rank 的 batch 数量对齐。
-   **AMP 集成**：使用 `torch.autocast` 和 `GradScaler` 加速训练并降低显存占用。
-   **指标同步**：使用 `reduce_metric` 在验证阶段聚合所有 GPU 的评估结果。
-   **检查点保存**：仅在 Rank-0 保存，且保存前自动剥离 `model.module` 封装。

---

## 十一、评估体系

PENCI 采用三层嵌套评估架构，全方位衡量模型性能：

1.  **层级 1：传感器空间 (Sensor Space)**
    - 计算重建 EEG 与原始 EEG 的 Pearson 相关系数、信噪比 (SNR_dB) 和 NRMSE。
2.  **层级 2：源空间对比 (Source Comparison)**
    - 与经典物理逆解方法 (如 sLORETA) 进行对比。
    - 将 sLORETA 的高维源空间 (20484) 投影到 72 区，与 PENCI 的 72 区输出计算 Pearson 相关性。
3.  **层级 3：仿真测试 (Simulation Test)**
    - 使用 `scripts/evaluate.py` 进行偶极子仿真。
    - 在已知位置放置仿真偶极子，比较 PENCI 和 sLORETA 的偶极子定位误差 (DLE)。

---

## 十二、依赖关系图 ✅

```
                    ┌─────────────────────┐
                    │  全部核心功能已实现    │
                    │  ✅ fsaverage 就绪    │
                    │  ✅ 72 源定义完成     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  LeadfieldManager    │
                    │  ✅ 物理模块构建完成  │
                    │  ✅ 两级缓存机制生效  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  PhysicsDecoder 改造  │
                    │  ✅ 动态导联场接入    │
                    │  ✅ Batch 级投影加速  │
                    └──────────┬──────────┘
                               │
                    ┌──────────┴──────────┐
                    │                     │
                    ▼                     ▼
          ┌──────────────────┐  ┌──────────────────┐
          │  PENCI 模型适配    │  │  数据管线调整      │
          │  ✅ 物理初始化完成  │  │  ✅ 分桶采样器实现  │
          │  ✅ 接口对齐        │  │  ✅ DDP 兼容支持   │
          └────────┬─────────┘  └────────┬─────────┘
                   │                     │
                   └──────────┬──────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │  验证与评估体系       │
                   │  ✅ 三层评估完成      │
                   │  ✅ DDP 训练通过      │
                   │  ✅ 11 项测试通过     │
                   └──────────────────────┘
```
