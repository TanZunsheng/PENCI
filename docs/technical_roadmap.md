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

共 **8,145,732** 条样本。

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
- `pos` 的前 3 维是 3D 坐标 `(x, y, z)`，后 3 维是法向量 `(nx, ny, nz)`，**法向量在当前数据中全为 0**
- `pos` 坐标范围约 `[-0.65, 0.65]`，均值接近 0。**单位待确认**（详见第六节待决策问题）

### 2.3 电极配置

全部数据只有 **5 种唯一的电极配置**：

| 通道数 | 出现的数据集 |
|--------|-------------|
| 128ch | 所有 6 个数据集 |
| 61ch  | HBN, SEED-DV, Brennan, Groot, THINGS |
| 60ch  | Brennan, Groot, THINGS, ThingsEEG |
| 63ch  | Groot, THINGS, ThingsEEG |
| 127ch | THINGS-EEG |

**重要特性**：同一电极配置下，所有样本的 `pos` 完全相同。即相同通道数的样本共享同一组电极位置。

### 2.4 元数据格式

每个数据集有 `train.json` 和 `val.json`，格式为 JSON 数组：

```json
[
    {
        "dataset": "HBN_EEG",
        "path": "/work/2024/tanzunsheng/PENCIData/HBN_EEG/derivatives/.../121_data.pt",
        "channels": 128,
        "is_eeg": true,
        "is_meg": false
    },
    ...
]
```

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
| `penci/modules/attention.py` | `RMSNorm`, `FeedForward`, etc. | 基础注意力组件 |
| `penci/modules/conv.py` | `SConv1d`, `SConvTranspose1d` | 带归一化的卷积模块 |
| `penci/modules/lstm.py` | `SLSTM` | 带跳跃连接的 LSTM |

### 3.2 动力学核心（Dynamics Core）

#### 3.2.1 作用

在源空间内建模时间演化。编码器输出的是离散时间步的源活动快照，动力学核心学习这些快照之间的时间关联，捕获神经动力学规律。

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

**RNN 模式**（轻量级替代）：

```
输入: (B, 72, T_total, D)
  │
  ├─ 线性投影: D → hidden_dim
  ├─ LSTM / GRU
  ├─ 线性投影: hidden_dim → D
  └─ RMSNorm

输出: (B, 72, T_total, D)
```

**关键设计决策**：
- 72 个源之间**不做跨源注意力**（每个源独立处理时间序列），跨源交互由导联场矩阵隐式完成
- 使用因果注意力（causal=True），建模时间的单向依赖
- 使用旋转位置编码（RoPE）编码时间步的相对位置

#### 3.2.3 涉及文件

| 文件 | 类 | 说明 |
|------|-----|------|
| `penci/models/dynamics.py` | `DynamicsCore` | Transformer 动力学核心 |
| `penci/models/dynamics.py` | `DynamicsRNN` | RNN 动力学核心（LSTM/GRU） |

### 3.3 物理解码器（Physics Decoder）

#### 3.3.1 作用

将源空间的神经活动映射回传感器空间的 EEG 信号。这对应 EEG 前向模型。

#### 3.3.2 当前实现

```
输入: source_activity (B, 72, T, D)
  │
  ├─ 时间特征解码: (B, 72, T, D) → (B, 72, T)
  │     └─ Linear(D→D) → GELU → Linear(D→1) → squeeze
  │     └─ 将 D 维特征向量压缩为标量振幅
  │
  └─ 导联场投影: (B, 72, T) → (B, C, T)
        └─ einsum("cs,bst->bct", L, source_signal)
        └─ L 是 (C, 72) 的导联场矩阵（register_buffer，不可训练）

输出: (B, C, T) — 重建的传感器空间信号
```

#### 3.3.3 当前的问题

**当前导联场矩阵是 `torch.randn(128, 72)` ——一个随机矩阵，没有任何物理意义。** 这是项目初期的占位符实现，必须替换为真实的物理导联场。

此外，当前实现假设所有样本的通道数都是 128，无法处理不同电极配置（60/61/63/127/128ch）。

#### 3.3.4 涉及文件

| 文件 | 类 | 说明 |
|------|-----|------|
| `penci/models/physics_decoder.py` | `PhysicsDecoder` | 核心物理解码器 |
| `penci/models/physics_decoder.py` | `SEANetPhysicsDecoder` | 基于 SEANet 的对比方案（保留备用） |
| `penci/encoders/backward_solution.py` | `ForwardSolution` | 交叉注意力前向映射（**已弃用，不再使用**） |

### 3.4 主模型（PENCI）

#### 3.4.1 前向传播流程

```python
def forward(self, x, pos, sensor_type, return_source=False):
    # 1. 编码：传感器空间 → 源空间
    source_encoded = self.encoder(x, pos, sensor_type)    # (B, 72, N, T_enc, D)

    # 2. 重排列
    source_flat = rearrange(source_encoded, "B N Ns T D -> B N (Ns T) D")

    # 3. 动力学演化
    source_evolved = self.dynamics(source_flat)             # (B, 72, T_total, D)

    # 4. 传感器嵌入（用于注意力模式，固定导联场模式下未使用）
    sensor_embedding = self.sensor_module(pos, sensor_type) # (B, C, D)

    # 5. 物理解码：源空间 → 传感器空间
    reconstruction = self.decoder(source_evolved, sensor_embedding)  # (B, C, T')

    return {"reconstruction": reconstruction}
```

#### 3.4.2 训练损失

```python
total_loss = λ_recon × MSE(reconstruction, target) + λ_dyn × dynamics_regularization
```

- **重建损失**：预测的传感器信号与真实信号的 MSE
- **动力学正则化**：源活动时间维度一阶差分的 L2 范数（鼓励时间平滑性）
- 默认权重：`λ_recon=1.0`, `λ_dyn=0.1`

#### 3.4.3 涉及文件

| 文件 | 类 | 说明 |
|------|-----|------|
| `penci/models/penci_model.py` | `PENCI` | 完整模型 |
| `penci/models/penci_model.py` | `PENCILite` | 轻量级版本（用于快速实验） |
| `penci/models/penci_model.py` | `build_penci_from_config` | 从配置文件构建模型 |

### 3.5 数据管线（Data Pipeline）

#### 3.5.1 当前实现

```
PENCIDataset:
  - 从 JSON 元数据文件加载样本路径
  - 按需加载 .pt 文件
  - 时间维度：超长截断（随机裁剪），不足填充 0
  - 通道维度：不足填充 0 到 target_channels=128，超出截断

PENCICollator:
  - 简单 torch.stack（要求所有样本形状相同）

数据增强：
  - RandomScaling：随机缩放信号幅度
  - RandomNoise：添加随机高斯噪声
```

#### 3.5.2 涉及文件

| 文件 | 函数/类 | 说明 |
|------|---------|------|
| `penci/data/dataset.py` | `PENCIDataset` | 数据集类 |
| `penci/data/dataset.py` | `PENCICollator` | 批处理器 |
| `penci/data/dataset.py` | `create_dataloader` | DataLoader 工厂函数 |
| `penci/data/dataset.py` | `get_train_val_loaders` | 训练/验证加载器 |

### 3.6 训练脚本

#### 3.6.1 训练流程

```
加载配置 (YAML)
  → 创建模型 (build_penci_from_config)
  → 创建数据加载器 (get_train_val_loaders)
  → 创建优化器 (AdamW) + 调度器 (CosineAnnealing)
  → 训练循环:
      每个 epoch:
        train_one_epoch: 前向 → 计算损失 → 反向 → 梯度裁剪 → 更新
        evaluate: 验证集评估
        保存检查点 (best + 定期)
  → TensorBoard 日志
```

#### 3.6.2 涉及文件

| 文件 | 说明 |
|------|------|
| `scripts/train.py` | 训练入口脚本 |
| `configs/default.yaml` | 默认配置文件 |

---

## 四、动态导联场系统（待实现的核心功能）

### 4.1 为什么需要动态导联场

当前实现有两个致命问题：

1. **导联场是随机矩阵**：`torch.randn(128, 72)` 没有任何物理意义，模型学到的源空间表示也没有物理意义
2. **只支持固定 128 通道**：数据中有 5 种不同的电极配置，通道数从 60 到 128 不等，当前的固定导联场无法处理

### 4.2 目标

实现一个 `LeadfieldManager` 类，能够：

1. **自动识别电极配置**：根据输入的 `pos` 张量自动识别当前样本使用的电极配置
2. **稳定 hash**：为每种电极配置生成确定性的 hash 标识（相同配置 → 相同 hash）
3. **自动计算导联场**：首次遇到新配置时，调用 MNE-Python 基于 fsaverage 头模型计算真实的物理导联场矩阵
4. **两级缓存**：内存缓存（字典）+ 磁盘缓存（`.pt` 文件），后续直接加载
5. **并发安全**：支持多进程/多卡 DataParallel/DDP 训练时的并发访问（文件锁）
6. **完全透明**：用户无需手动指定任何导联场文件

### 4.3 导联场的物理含义

导联场矩阵 $L$ 描述了 $N_s$ 个源到 $N_c$ 个传感器的线性映射：

$$\mathbf{v}(t) = L \cdot \mathbf{s}(t)$$

- $\mathbf{v}(t) \in \mathbb{R}^{N_c}$：传感器测量（EEG 电压）
- $\mathbf{s}(t) \in \mathbb{R}^{N_s}$：源活动（神经元电流密度）
- $L \in \mathbb{R}^{N_c \times N_s}$：导联场矩阵

$L$ 的每个元素 $L_{ij}$ 表示：**第 $j$ 个源以单位强度激活时，在第 $i$ 个传感器上产生的电压**。这完全由几何关系和导电介质的物理性质决定，与数据无关。

### 4.4 计算方法

使用 MNE-Python 的前向建模功能，基于 fsaverage 标准头模型：

```python
import mne

# 1. 从 pos 创建电极配置
montage = mne.channels.make_dig_montage(
    ch_pos={"ch_i": pos[i, :3] for i in range(n_channels)},
    coord_frame="head"
)

# 2. 创建 Info 对象
info = mne.create_info(ch_names, sfreq=256, ch_types="eeg")
info.set_montage(montage)

# 3. 加载 fsaverage BEM 和源空间
subjects_dir = "/home/2024/tanzunsheng/mne_data/MNE-fsaverage-data"
bem = mne.read_bem_solution(f"{subjects_dir}/fsaverage/bem/fsaverage-5120-5120-5120-bem-sol.fif")
src = mne.read_source_spaces(f"{subjects_dir}/fsaverage/bem/fsaverage-ico-5-src.fif")

# 4. 计算前向解
fwd = mne.make_forward_solution(info, trans="fsaverage", src=src, bem=bem)
fwd = mne.convert_forward_solution(fwd, force_fixed=True)

# 5. 提取导联场矩阵
leadfield = fwd["sol"]["data"]  # shape: (n_channels, n_sources)
```

注意：fsaverage ico-5 源空间有约 20,484 个源点（每个半球 10,242 个），需要降采样到 72 个。

### 4.5 LeadfieldManager 设计

```python
class LeadfieldManager:
    """
    动态导联场管理器

    职责：
    - 接收电极位置 (pos)，返回对应的导联场矩阵
    - 自动识别电极配置并缓存结果
    - 支持多进程并发安全
    """

    def __init__(
        self,
        n_sources: int = 72,
        cache_dir: str = "~/.cache/penci/leadfields",
        subjects_dir: str = None,
    ):
        self.n_sources = n_sources
        self.cache_dir = Path(cache_dir).expanduser()
        self.subjects_dir = subjects_dir
        self._memory_cache: Dict[str, Tensor] = {}  # hash → leadfield

    def get_leadfield(self, pos: Tensor) -> Tensor:
        """
        输入: pos (C, 6) — 单个样本的电极位置
        输出: leadfield (C, 72) — 对应的导联场矩阵
        """
        hash_key = self._compute_hash(pos)

        # 1. 查内存缓存
        if hash_key in self._memory_cache:
            return self._memory_cache[hash_key]

        # 2. 查磁盘缓存
        cached = self._load_cache(hash_key)
        if cached is not None:
            self._memory_cache[hash_key] = cached
            return cached

        # 3. 首次遇到 → 计算
        leadfield = self._compute_leadfield_mne(pos)
        self._save_cache(hash_key, leadfield)
        self._memory_cache[hash_key] = leadfield
        return leadfield

    def get_batch_leadfield(self, pos_batch: Tensor) -> Tensor:
        """
        输入: pos_batch (B, C, 6)
        输出: leadfield_batch (B, C, 72)

        注意：同一 batch 内可能有不同电极配置的样本
        """
        B = pos_batch.shape[0]
        leadfields = []
        for i in range(B):
            L = self.get_leadfield(pos_batch[i])
            leadfields.append(L)
        return torch.stack(leadfields)  # (B, C, 72)

    def _compute_hash(self, pos: Tensor) -> str:
        """基于通道数 + 坐标值的稳定 hash"""
        # 使用 float32 精度确保确定性
        pos_f32 = pos[:, :3].float()  # 只用 xyz，忽略法向量
        n_ch = pos_f32.shape[0]
        data = f"{n_ch}:".encode() + pos_f32.cpu().numpy().tobytes()
        return hashlib.sha256(data).hexdigest()[:16]

    def _compute_leadfield_mne(self, pos: Tensor) -> Tensor:
        """使用 MNE-Python 计算导联场"""
        # ... MNE 前向建模流程（见 4.4 节）
        # 返回: (C, 72)

    def _save_cache(self, hash_key: str, leadfield: Tensor):
        """保存到磁盘（带文件锁）"""
        # 使用 filelock 保证多进程安全

    def _load_cache(self, hash_key: str) -> Optional[Tensor]:
        """从磁盘加载"""
```

### 4.6 PhysicsDecoder 改造

**改动前**：

```python
# 单一固定导联场（随机矩阵）
self.register_buffer("leadfield", torch.randn(128, 72) / (72 ** 0.5))

# forward 中：
sensor_signal = torch.einsum("cs,bst->bct", self.leadfield, source_signal)
# 所有样本共享同一个 (128, 72) 矩阵
```

**改动后**：

```python
# 不再在 __init__ 中创建导联场 buffer
# 由 LeadfieldManager 动态提供

# forward 中：
# L_batch: (B, C, 72) — 每个样本有自己的导联场
sensor_signal = torch.einsum("bcs,bst->bct", L_batch, source_signal)
```

关键变化：
1. `einsum` 从 `"cs,bst->bct"` 变为 `"bcs,bst->bct"`，支持 per-sample 不同的导联场
2. `forward` 方法增加 `pos` 参数，用于从 manager 获取导联场
3. 导联场矩阵的通道维度 `C` 随样本变化（60/61/63/127/128）

### 4.7 数据管线适配

当前 `PENCICollator` 使用 `torch.stack`，要求所有样本的通道数相同。当 batch 内有不同通道数的样本时需要处理。

**可能的方案**：

| 方案 | 优点 | 缺点 |
|------|------|------|
| A: 保持 padding 到 128 | 实现最简单，当前行为不变 | 浪费计算，padding 位置的导联场需要特殊处理 |
| B: 按通道数分桶采样 | 无 padding 浪费 | Sampler 复杂度高，可能影响数据均衡性 |
| C: 动态 padding 到 batch 最大值 | 折中方案 | Collator 需要改造 |

**待用户决策。**

---

## 五、实施计划

### 阶段 0：前置依赖 ⏳

| 任务 | 状态 | 说明 |
|------|------|------|
| 手动下载 fsaverage BEM 数据 | ⏳ 等待用户操作 | root.zip + bem.zip 从 OSF 下载 |
| 确认 pos 坐标系和单位 | ⏳ 等待用户确认 | MNE 期望米制坐标 |
| 确认 72 个源空间如何定义 | ⏳ 等待用户决策 | atlas 脑区质心？均匀采样？ |

**fsaverage 下载信息**：

| 文件 | 下载地址 | 放置位置 |
|------|---------|---------|
| root.zip (179个文件) | `https://osf.io/3bxqt/download?version=2` | 解压到 `/home/2024/tanzunsheng/mne_data/MNE-fsaverage-data/` |
| bem.zip (12个文件) | `https://osf.io/7ve8g/download?version=4` | 解压到 `/home/2024/tanzunsheng/mne_data/MNE-fsaverage-data/fsaverage/` |

### 阶段 1：实现 LeadfieldManager

**新建文件**：
- `penci/physics/__init__.py`
- `penci/physics/leadfield_manager.py`

**核心内容**：
- `LeadfieldManager` 类（详见 4.5 节）
- hash 计算、MNE 计算、两级缓存、文件锁
- 源空间降采样策略（20,484 → 72）

### 阶段 2：改造 PhysicsDecoder

**修改文件**：`penci/models/physics_decoder.py`

**改动内容**：
- `__init__` 移除固定导联场 buffer，接收 `LeadfieldManager` 实例
- `forward` 增加 `pos` 参数
- `_forward_leadfield` 中的 einsum 改为支持 per-sample 导联场
- 处理不同通道数的导联场矩阵（padding 对齐）

### 阶段 3：适配 PENCI 主模型

**修改文件**：`penci/models/penci_model.py`

**改动内容**：
- `__init__` 中创建 `LeadfieldManager` 并传递给 `PhysicsDecoder`
- `forward` 中将 `pos` 传递给 decoder
- `compute_loss` 中同步调整
- `build_penci_from_config` 增加导联场相关配置的读取

### 阶段 4：调整数据管线

**可能修改文件**：`penci/data/dataset.py`

**改动内容**：
- 根据用户对 batch 内不同通道数的处理方案决定
- 如果选择方案 B（分桶采样），需要实现自定义 Sampler

### 阶段 5：更新配置和测试

**修改文件**：
- `configs/default.yaml`：添加导联场缓存路径、subjects_dir 等配置
- `tests/test_smoke.py`：更新解码器测试（传入 pos）、添加 LeadfieldManager 单元测试

### 阶段 6：验证和提交

- 运行全部 smoke test
- 验证 5 种电极配置的导联场计算
- Git commit + push

---

## 六、待决策问题

| # | 问题 | 为什么重要 | 建议方案 |
|---|------|-----------|---------|
| 1 | **72 个源空间如何定义？** | 直接决定导联场矩阵的物理意义。fsaverage ico-5 有 20,484 个源点，必须降采样到 72。 | 方案 A：Desikan-Killiany 68 区 + 4 个皮层下区域的质心；方案 B：使用 aparc 分区后对每个区域的源取平均；方案 C：自定义 72 个均匀分布位置 |
| 2 | **pos 坐标单位是什么？** | MNE 期望**米制**坐标（头模型约 [-0.1, 0.1] 米）。当前 pos 范围 [-0.65, 0.65]，如果不是米制需要缩放，否则导联场计算完全错误。 | 需要确认数据预处理流程中 pos 的坐标系和单位 |
| 3 | **batch 内不同通道数如何处理？** | 影响 DataLoader 和训练效率 | 建议保持当前 padding 到 128 方案，导联场在 padding 位置填 0 行 |
| 4 | **缓存路径放哪？** | 多用户/多项目场景下的隔离 | 建议 `~/.cache/penci/leadfields/` |
| 5 | **是否需要无 MNE 环境的 fallback？** | CI 和可移植性 | 建议有：测试环境用 mock 随机矩阵 |

---

## 七、当前项目状态总览

### 已完成 ✅

| 模块 | 状态 | 说明 |
|------|------|------|
| SEANet 编码器 | ✅ 完成 | 移植自 BrainOmni，已通过 smoke test |
| 传感器嵌入 | ✅ 完成 | BrainSensorModule，已通过 smoke test |
| 反向求解 | ✅ 完成 | BackWardSolution（交叉注意力），已通过 smoke test |
| PENCIEncoder | ✅ 完成 | 顶层编码器，已通过 smoke test |
| DynamicsCore | ✅ 完成 | Transformer 动力学核心，已通过 smoke test |
| DynamicsRNN | ✅ 完成 | RNN 动力学核心，已通过 smoke test |
| PhysicsDecoder（骨架） | ✅ 完成 | 架构完成，但导联场是随机矩阵占位符 |
| PENCI 主模型 | ✅ 完成 | 完整前向传播链路，已通过 smoke test |
| PENCILite | ✅ 完成 | 轻量级版本 |
| 数据管线 | ✅ 完成 | Dataset + Collator + DataLoader |
| 训练脚本 | ✅ 完成 | 完整训练循环 + TensorBoard |
| 配置系统 | ✅ 完成 | YAML 配置 + 中文注释 |
| Smoke Test | ✅ 全部 7 项通过 | 基础模块 / 编码器 / 动力学 / 解码器 / 完整模型 / 数据加载 / 集成测试 |

### 待完成 ⏳

| 模块 | 状态 | 阻塞项 |
|------|------|--------|
| LeadfieldManager | ⏳ 未开始 | 等待 fsaverage 下载 + 源空间定义决策 + pos 坐标确认 |
| PhysicsDecoder 改造（真实导联场） | ⏳ 未开始 | 依赖 LeadfieldManager |
| PENCI 模型适配（pos 传递） | ⏳ 未开始 | 依赖 PhysicsDecoder 改造 |
| 数据管线适配（多通道数） | ⏳ 未开始 | 等待用户决策方案 |
| 配置和测试更新 | ⏳ 未开始 | 依赖上述所有模块 |

### Git 状态

- 2 commits on main，已 push 到 origin
- Working tree clean
- GitHub: `https://github.com/TanZunsheng/PENCI`

---

## 八、关键约束清单

| 约束 | 原因 |
|------|------|
| 禁止 import BrainOmni | 所有代码必须在 PENCI 项目内自包含 |
| 导联场不可训练 | 物理约束的核心——必须是固定的 register_buffer |
| 不使用交叉注意力做前向映射 | 可学习的前向映射会破坏物理约束（详见 1.3 节） |
| n_neuro = 72 | 固定源空间维度，对应 72 个脑区 |
| 注释和文档使用中文 | 项目规范 |
| 不自动下载 fsaverage | 用户网络无法访问 OSF，需手动下载 |
| 不做简化/demo 版本 | 必须完整实现 |
| 不删除测试来通过构建 | 修复代码而非测试 |

---

## 九、配置参数说明

```yaml
model:
  n_dim: 256                    # 全局特征维度（编码器/动力学/解码器共享）
  n_neuro: 72                   # 源空间脑区数量
  n_head: 4                     # 编码器交叉注意力头数
  dropout: 0.0                  # 全局 Dropout 比率

  seanet:
    n_filters: 32               # SEANet 基础滤波器数（逐层翻倍）
    ratios: [8, 4, 2]           # 下采样比率（总计 8×4×2 = 64 倍）
    kernel_size: 7              # 主卷积核大小
    residual_kernel_size: 3     # 残差块卷积核大小
    last_kernel_size: 7         # 最后一层卷积核大小
    dilation_base: 2            # 膨胀卷积基数
    causal: false               # 非因果卷积（可看到未来帧）
    norm: "weight_norm"         # 权重归一化
    lstm: 2                     # SEANet 内部 LSTM 层数

  dynamics:
    type: "transformer"         # 动力学模型类型
    n_layers: 4                 # Transformer 层数
    n_heads: 8                  # 注意力头数
    dim_feedforward: 1024       # FFN 隐藏层维度
    dropout: 0.1                # Dropout 比率

  physics:
    use_fixed_leadfield: true   # 使用固定物理导联场（必须为 true）
    leadfield_path: null        # 导联场路径（将被 LeadfieldManager 替代）
    # 以下为待添加配置项：
    # leadfield_cache_dir: "~/.cache/penci/leadfields"
    # subjects_dir: "/home/2024/tanzunsheng/mne_data/MNE-fsaverage-data"
    # n_sources: 72

data:
  root_dir: "/work/2024/tanzunsheng/PENCIData"
  datasets: ["HBN_EEG"]
  sample_rate: 256              # 采样率 Hz
  window_length: 256            # 编码窗口长度（采样点）= 1 秒
  n_channels: 128               # 最大通道数
  time_window: 10               # 每样本时长 秒
  stride: 5                     # 滑动步长 秒

training:
  batch_size: 32
  num_workers: 4
  learning_rate: 1.0e-4
  weight_decay: 0.01
  warmup_steps: 1000
  max_steps: 100000
  gradient_clip: 1.0
  loss:
    reconstruction: 1.0
    dynamics: 0.1
  log_interval: 100
  save_interval: 5000
  eval_interval: 1000

hardware:
  device: "cuda"
  mixed_precision: true
  compile: false
```

---

## 十、依赖关系图

```
                    ┌─────────────────────┐
                    │  前置依赖（阶段 0）    │
                    │  • fsaverage 下载     │
                    │  • pos 坐标确认       │
                    │  • 源空间定义决策      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  LeadfieldManager    │
                    │  （阶段 1）           │
                    │  新建 physics/ 模块   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  PhysicsDecoder 改造  │
                    │  （阶段 2）           │
                    │  接入 manager         │
                    │  einsum 改为 per-sample│
                    └──────────┬──────────┘
                               │
                    ┌──────────┴──────────┐
                    │                     │
                    ▼                     ▼
         ┌──────────────────┐  ┌──────────────────┐
         │  PENCI 模型适配    │  │  数据管线调整      │
         │  （阶段 3）        │  │  （阶段 4）        │
         │  forward 传递 pos │  │  多通道数处理      │
         └────────┬─────────┘  └────────┬─────────┘
                  │                     │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │  配置 + 测试更新       │
                  │  （阶段 5）            │
                  │  default.yaml         │
                  │  test_smoke.py        │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │  验证 + 提交           │
                  │  （阶段 6）            │
                  │  git commit + push    │
                  └──────────────────────┘
```
