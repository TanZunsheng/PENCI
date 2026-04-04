# PENCI 动态导联场 — 现状与实施计划

## 一、你提出的需求（原文摘要）

> 请在 PENCI 项目中实现一个动态导联场（Lead Field）管理机制，用于支持不同电极配置的数据训练与推理。

核心要求：
1. 根据输入数据中的电极位置（pos）自动识别电极配置
2. 为每种配置生成稳定的 hash 标识
3. 首次遇到新配置时，用 MNE-Python 自动计算导联场矩阵
4. 缓存计算结果到本地文件，后续直接加载复用
5. 支持多进程/多卡训练下的并发安全
6. 全自动——用户无需手动指定导联场文件
7. 导联场不可训练，是固定物理约束
8. 所有导联场统一映射到 72 个固定源

---

## 二、目前已完成的操作

### 2.1 代码更改（已 commit 并 push）

**共 2 个 commit，均已推送到 origin/main：**

| Commit | 说明 |
|--------|------|
| `cb85a35` | v0.2.0: 完整 PENCI 实现（编码器、动力学、物理解码器） |
| `9e6fb67` | 配置更新：n_neuro 64→72，导联场默认使用固定物理矩阵 |

**第二个 commit 的具体改动：**

| 文件 | 改动内容 |
|------|----------|
| `configs/default.yaml` | 全部注释改为中文；`n_neuro: 64` → `72`；`use_fixed_leadfield` 确认为 `true` |
| `penci/encoders/encoder.py` | `PENCIEncoder` 默认 `n_neuro=64` → `72` |
| `penci/models/penci_model.py` | `PENCI` 默认 `n_neuro=64` → `72`；`use_fixed_leadfield` 默认 `False` → `True`；`build_penci_from_config` 中默认值同步 |
| `penci/models/physics_decoder.py` | `PhysicsDecoder` 默认 `n_sources=64` → `72`；`use_fixed_leadfield` 默认 `False` → `True` |
| `tests/test_smoke.py` | 所有测试中 `n_neuro=64` → `72` |

### 2.2 数据分析（已完成，未涉及代码更改）

对 `/work/2024/tanzunsheng/PENCIData` 进行了全面分析：

- 6 个数据集，共 **8,145,732** 条样本
- 发现全部数据只有 **5 种唯一电极配置**：

| 通道数 | 出现的数据集 |
|--------|-------------|
| 128ch | 所有 6 个数据集 |
| 61ch  | HBN, SEED-DV, Brennan, Groot, THINGS |
| 60ch  | Brennan, Groot, THINGS, ThingsEEG |
| 63ch  | Groot, THINGS, ThingsEEG |
| 127ch | THINGS-EEG |

- 数据格式：每个 `.pt` 文件包含 `{"x": (C,T), "pos": (C,6), "sensor_type": (C,)}`
- `pos` 格式为 `(x, y, z, nx, ny, nz)`，法向量全为 0
- pos 坐标范围约 `[-0.65, 0.65]`，均值接近 0，**可能是归一化坐标（非米制）**
- 同一电极配置下所有样本的 pos 完全相同

### 2.3 MNE fsaverage 模型下载（失败）

尝试下载 `mne.datasets.fetch_fsaverage()` 但因 OSF 网络不可达而失败。

**需要手动下载的文件：**

| 文件 | 下载地址 | 放置位置 |
|------|---------|---------|
| root.zip (179个文件) | https://osf.io/3bxqt/download?version=2 | 解压到 `/work/2024/tanzunsheng/mne_data/MNE-fsaverage-data/` |
| bem.zip (12个文件) | https://osf.io/7ve8g/download?version=4 | 解压到 `/work/2024/tanzunsheng/mne_data/MNE-fsaverage-data/fsaverage/` |

**计算导联场必须的关键文件（均在 bem.zip 中）：**
- `fsaverage-5120-5120-5120-bem-sol.fif` — BEM 解
- `fsaverage-ico-5-src.fif` — 源空间定义
- `fsaverage-trans.fif` — head→MRI 坐标变换
- `fsaverage-fiducials.fif` — 基准点

### 2.4 尚未进行任何动态导联场代码修改

**零代码变更。** 当前 `PhysicsDecoder` 的导联场仍然是 `torch.randn(n_sensors, n_sources)`（随机矩阵），没有真实物理意义。

---

## 三、当前代码结构（与导联场相关）

```
penci/
├── models/
│   ├── physics_decoder.py    ← PhysicsDecoder 类（当前：单一固定导联场 buffer）
│   ├── penci_model.py        ← PENCI 主模型（forward 调用 decoder）
│   └── dynamics.py           ← 动力学核心
├── encoders/
│   ├── encoder.py            ← PENCIEncoder
│   ├── sensor_embed.py       ← BrainSensorModule
│   └── backward_solution.py  ← BackWardSolution, ForwardSolution
├── data/
│   └── dataset.py            ← PENCIDataset, PENCICollator
├── modules/                  ← 基础模块（attention, conv, lstm, seanet）
configs/
└── default.yaml              ← 配置文件
tests/
└── test_smoke.py             ← 冒烟测试
```

**当前 PhysicsDecoder.forward 数据流：**
```
source_activity (B, 72, T, D)
    → temporal_decoder → source_signal (B, 72, T)
    → einsum("cs,bst->bct", self.leadfield, source_signal)
    → sensor_signal (B, 128, T)
```
其中 `self.leadfield` 是形状 `(128, 72)` 的**固定随机矩阵**（register_buffer）。

**问题：当前设计只支持单一固定通道数（128ch），无法处理不同电极配置（60/61/63/127/128ch）。**

---

## 四、实施计划

### 阶段 0：前置依赖

| 任务 | 状态 | 说明 |
|------|------|------|
| 手动下载 fsaverage BEM 数据 | **⏳ 等待你操作** | 需从 OSF 下载 root.zip + bem.zip |
| 验证 MNE 可加载 BEM | 待前置完成 | `mne.read_bem_solution(...)` |
| 确认 pos 坐标系 | **⚠️ 需讨论** | 当前 pos 范围 [-0.65, 0.65]，需确认是否为米制还是需要缩放 |

### 阶段 1：新建 LeadfieldManager（核心模块）

**新文件：`penci/physics/leadfield_manager.py`**

```python
class LeadfieldManager:
    """动态导联场管理器"""

    def __init__(self, cache_dir, n_sources=72, subjects_dir=None):
        ...

    def get_leadfield(self, pos: Tensor) -> Tensor:
        """
        输入: pos (C, 6) — 单个样本的电极位置
        输出: leadfield (C, 72) — 对应的导联场矩阵

        流程:
        1. 计算 pos 的 hash（通道数 + 坐标值）
        2. 查内存缓存 → 命中则直接返回
        3. 查磁盘缓存 → 命中则加载到内存并返回
        4. 首次遇到 → 用 MNE 计算 → 保存磁盘 → 加入内存缓存 → 返回
        """

    def _compute_hash(self, pos: Tensor) -> str:
        """稳定 hash: 通道数 + pos 的 float32 bytes"""

    def _compute_leadfield_mne(self, pos: Tensor) -> Tensor:
        """
        使用 MNE-Python 计算导联场:
        1. 从 pos 创建 montage + Info
        2. 加载 fsaverage BEM + 源空间
        3. make_forward_solution
        4. convert_forward_solution (fixed orientation)
        5. 提取 fwd['sol']['data'] → (n_sensors, n_sources)
        """

    def _save_cache(self, hash_key: str, leadfield: Tensor):
        """保存到磁盘（带文件锁，支持多进程）"""

    def _load_cache(self, hash_key: str) -> Optional[Tensor]:
        """从磁盘加载（带文件锁）"""
```

**关键设计决策（需要你确认）：**

1. **源空间定义**：72 个源如何定义？
   - 方案 A：使用 fsaverage ico-5 源空间（约 10242 个源/半球），然后降采样到 72
   - 方案 B：使用 Desikan-Killiany atlas 的 68 个脑区质心 + 4 个皮层下结构
   - 方案 C：自定义 72 个均匀分布的源位置
   - **需要你指定或讨论**

2. **pos 坐标转换**：
   - 当前 pos 范围 [-0.65, 0.65]，MNE 期望米制（头模型约 [-0.1, 0.1] 米）
   - 需确认：pos 是否已经是米制？还是需要乘以某个缩放因子？
   - **需要你确认数据预处理时 pos 的坐标系和单位**

3. **缓存路径**：
   - 默认 `~/.cache/penci/leadfields/`？还是放在项目目录下？

### 阶段 2：修改 PhysicsDecoder

**文件：`penci/models/physics_decoder.py`**

改动要点：
1. `__init__` 中不再创建单一固定导联场 buffer
2. 接收 `LeadfieldManager` 实例
3. `forward` 签名增加 `pos` 参数
4. 根据 batch 中每个样本的 pos，从 manager 获取对应导联场
5. einsum 从 `"cs,bst->bct"` 改为 `"bcs,bst->bct"`（支持 per-sample 不同导联场）

```python
# 改动前
sensor_signal = torch.einsum("cs,bst->bct", self.leadfield, source_signal)

# 改动后
# L_batch: (B, C, S) — 每个样本可能有不同的导联场
sensor_signal = torch.einsum("bcs,bst->bct", L_batch, source_signal)
```

### 阶段 3：修改 PENCI 主模型

**文件：`penci/models/penci_model.py`**

改动要点：
1. `__init__` 中创建 `LeadfieldManager`
2. `forward` 中将 `pos` 传递给 decoder
3. `compute_loss` 中同步调整

```python
# 改动前
reconstruction = self.decoder(source_evolved, sensor_embedding)

# 改动后
reconstruction = self.decoder(source_evolved, sensor_embedding, pos=pos)
```

### 阶段 4：修改数据管线

**文件：`penci/data/dataset.py`**

改动要点：
1. `PENCICollator` 需要支持不同通道数的 batch
   - **关键问题**：当前实现用 `torch.stack` 要求所有样本通道数相同
   - 如果 batch 内有不同通道数的样本，需要 padding 到最大通道数
   - 或者在 DataLoader 层面按通道数分组（sampler 策略）
2. `PENCIDataset` 当前已有 `target_channels` 参数做 padding/truncation，但可能需要调整

**方案选择（需讨论）：**
- 方案 A：保持 padding 到固定 target_channels=128，导联场也对应 padding（当前行为）
- 方案 B：按通道数分桶采样，同一 batch 内通道数相同
- 方案 C：动态 padding 到 batch 内最大通道数

### 阶段 5：更新配置和测试

**文件：`configs/default.yaml`**
```yaml
physics:
  use_fixed_leadfield: true
  # leadfield_path: null          ← 删除，改用动态管理
  leadfield_cache_dir: "~/.cache/penci/leadfields"
  n_sources: 72
  subjects_dir: "/work/2024/tanzunsheng/mne_data/MNE-fsaverage-data"
```

**文件：`tests/test_smoke.py`**
- 添加 LeadfieldManager 单元测试
- 更新 PhysicsDecoder 测试（传入 pos）
- 更新集成测试

### 阶段 6：Git commit 并 push

---

## 五、待讨论 / 待你决策的问题

| # | 问题 | 影响 | 我的建议 |
|---|------|------|---------|
| 1 | **72 个源空间如何定义？** 是使用 atlas 脑区质心？还是均匀采样？还是你有自定义的 72 个位置？ | 直接影响导联场矩阵的物理意义 | 使用 Desikan-Killiany 68 区 + 4 个皮层下区域的质心 |
| 2 | **pos 坐标单位是什么？** 范围 [-0.65, 0.65]，MNE 期望米制。数据预处理时做了什么变换？ | 不正确的坐标会导致导联场计算完全错误 | 需要你确认原始数据的坐标系 |
| 3 | **batch 内不同通道数如何处理？** padding 还是分桶？ | 影响数据管线和训练效率 | 保持当前 padding 到 128 的方案，导联场也做对应 padding（在 padded 位置填 0） |
| 4 | **缓存路径放哪？** 用户目录下还是项目目录下？ | 多用户/多项目场景 | `~/.cache/penci/leadfields/` |
| 5 | **是否需要在没有 MNE/fsaverage 的环境下也能运行？**（fallback 到随机矩阵？） | 影响 CI 和可移植性 | 是，测试环境用 mock/随机矩阵 fallback |

---

## 六、文件改动清单预览

| 操作 | 文件路径 | 说明 |
|------|---------|------|
| **新建** | `penci/physics/__init__.py` | 包初始化 |
| **新建** | `penci/physics/leadfield_manager.py` | 核心：LeadfieldManager 类 |
| **修改** | `penci/models/physics_decoder.py` | PhysicsDecoder 接入 manager，支持动态导联场 |
| **修改** | `penci/models/penci_model.py` | PENCI forward 传递 pos 到 decoder |
| **修改** | `configs/default.yaml` | 添加 leadfield 缓存配置 |
| **修改** | `tests/test_smoke.py` | 更新测试 |
| 可能修改 | `penci/data/dataset.py` | 如果需要调整 collator |

---

## 七、依赖关系图

```
阶段 0 (前置依赖)
  ├── 下载 fsaverage [你手动操作]
  └── 确认 pos 坐标系 [你确认]
       ↓
阶段 1 (LeadfieldManager)
       ↓
阶段 2 (PhysicsDecoder 改造)  ←  依赖阶段 1
       ↓
阶段 3 (PENCI 模型适配)      ←  依赖阶段 2
       ↓
阶段 4 (数据管线调整)        ←  可与 2/3 并行（如果方案确定）
       ↓
阶段 5 (配置 + 测试)         ←  依赖 1-4 全部完成
       ↓
阶段 6 (commit + push)
```
