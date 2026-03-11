# PENCI 项目完整会话总结

## 一、用户请求记录

1. **初始请求**：开发 PENCI (Physics-constrained End-to-end Neural Connectivity Inference) 项目
2. **配置更新请求**：config 中配置的每个参数的注释修改为中文，同时映射到脑区的源数量应该是 72（不是 64），导联场使用固定的物理导联场，不需要通过交叉注意力学习源空间到传感器空间的映射
3. **动态导联场请求**：在 PENCI 项目中实现一个动态导联场（Lead Field）管理机制，用于支持不同电极配置的数据训练与推理
4. **fsaverage 下载失败处理**：不需要下载模型，列出下载地址和放置路径即可（OSF 网络不可达）
5. **计划审查请求**：列出导联场需求的所有更改和操作计划，撰写到 md 文件中供审查

## 二、最终目标

实现一个**动态导联场管理系统**，当训练或推理时，系统能自动：

- 根据输入数据中的电极位置信息自动识别电极配置
- 为每种配置生成稳定 hash 标识
- 首次遇到配置时用 MNE-Python 计算导联场矩阵
- 缓存计算结果，后续直接加载
- 支持多进程/多卡训练并发安全
- 全自动，用户无需手动指定导联场文件

---

## 三、已完成工作

### 3.1 PENCI 项目核心实现（v0.2.0）

- **项目路径**：`/work/2024/tanzunsheng/Code/PENCI`
- **GitHub**：`https://github.com/TanZunsheng/PENCI`
- **Python 环境**：`/work/2024/tanzunsheng/anaconda3/envs/EEG/bin/python`

**Git 状态**：2 commits on main，clean working tree，已 push 到 origin：

| Commit | 说明 |
|--------|------|
| `cb85a35` | v0.2.0: Complete PENCI implementation |
| `9e6fb67` | 配置更新：n_neuro 改为 72，导联场默认使用固定物理矩阵 |

**"三明治"架构**：

1. **编码器** — 移植自 BrainOmni，移除量化层
2. **动力学核心** — Transformer/RNN 建模源空间演化
3. **物理解码器** — 固定导联场矩阵投影回传感器空间

**项目文件结构**：

```
penci/
├── __init__.py
├── modules/
│   ├── __init__.py
│   ├── attention.py          # RMSNorm, FeedForward, SelfAttention, SelfAttnBlock, RotaryEmbedding
│   ├── conv.py               # SConv1d, SConvTranspose1d
│   ├── lstm.py               # SLSTM
│   └── seanet.py             # SEANetEncoder, SEANetResnetBlock, Snake1d
├── encoders/
│   ├── __init__.py
│   ├── sensor_embed.py       # BrainSensorModule
│   ├── backward_solution.py  # BackWardSolution, ForwardSolution
│   └── encoder.py            # BrainTokenizerEncoder, PENCIEncoder
├── models/
│   ├── __init__.py
│   ├── dynamics.py           # DynamicsCore, DynamicsRNN
│   ├── physics_decoder.py    # PhysicsDecoder, SEANetPhysicsDecoder
│   └── penci_model.py        # PENCI, PENCILite, build_penci_from_config
├── data/
│   └── dataset.py            # PENCIDataset, PENCICollator, create_dataloader
scripts/
└── train.py
configs/
└── default.yaml
tests/
└── test_smoke.py
docs/
├── dynamic_leadfield_plan.md # 动态导联场实施计划
└── session_summary.md        # 本文件
```

### 3.2 配置更新（已完成并 push — commit 9e6fb67）

| 文件 | 改动内容 |
|------|----------|
| `configs/default.yaml` | 全部注释改为中文；`n_neuro: 64` → `72`；`use_fixed_leadfield` 确认为 `true` |
| `penci/encoders/encoder.py` | `PENCIEncoder` 默认 `n_neuro=64` → `72` |
| `penci/models/penci_model.py` | `PENCI` 默认 `n_neuro=64` → `72`；`use_fixed_leadfield` 默认 `False` → `True`；`build_penci_from_config` 中默认值同步 |
| `penci/models/physics_decoder.py` | `PhysicsDecoder` 默认 `n_sources=64` → `72`；`use_fixed_leadfield` 默认 `False` → `True` |
| `tests/test_smoke.py` | 所有测试中 `n_neuro=64` → `72` |

### 3.3 数据分析（已完成，无代码变更）

**数据路径**：`/work/2024/tanzunsheng/PENCIData`

- 6 个数据集：HBN_EEG、Brennan_Hale2019、Grootswagers2019、THINGS-EEG、ThingsEEG、SEED-DV
- 共 **8,145,732** 条样本

**关键发现 — 全部数据只有 5 种唯一电极配置**：

| 通道数 | 出现的数据集 |
|--------|-------------|
| 128ch | 所有 6 个数据集 |
| 61ch  | HBN, SEED-DV, Brennan, Groot, THINGS |
| 60ch  | Brennan, Groot, THINGS, ThingsEEG |
| 63ch  | Groot, THINGS, ThingsEEG |
| 127ch | THINGS-EEG |

**数据格式**：

- 每个样本是 `.pt` 文件：`{"x": (C, T), "pos": (C, 6), "sensor_type": (C,)}`
- `pos` 格式：`(x, y, z, nx, ny, nz)` — dtype=bfloat16
- **法向量 (nx, ny, nz) 全为 0** — 只有位置信息有实际值
- pos 坐标范围约 `[-0.65, 0.65]`，均值接近 0。**单位待确认（可能是归一化值，非米制）**
- 同一电极配置下所有样本 pos 完全相同
- `sensor_type` 全部为 0（EEG）
- 实际数据路径示例：`/work/2024/tanzunsheng/PENCIData/HBN_EEG/derivatives/preprocessing/sub-NDARVN232MNL/eeg/.../121_data.pt`

### 3.4 fsaverage 下载排查

**下载失败原因**：OSF (osf.io) 网络不可达

**需要手动下载 2 个文件**：

| 文件 | URL | MD5 | 解压到 |
|------|-----|-----|--------|
| root.zip | `https://osf.io/3bxqt/download?version=2` | `5133fe92b7b8f03ae19219d5f46e4177` | `/work/2024/tanzunsheng/mne_data/MNE-fsaverage-data/` |
| bem.zip | `https://osf.io/7ve8g/download?version=4` | `b31509cdcf7908af6a83dc5ee8f49fb1` | `/work/2024/tanzunsheng/mne_data/MNE-fsaverage-data/fsaverage/` |

**MNE 配置**：

- MNE 版本：1.8.0
- SUBJECTS_DIR：`/work/2024/tanzunsheng/mne_data/MNE-fsaverage-data`
- fsaverage 目录：`/work/2024/tanzunsheng/mne_data/MNE-fsaverage-data/fsaverage/`（✅ 已完整解压，BEM 文件就绪）
- 需要的关键 BEM 文件：`fsaverage-5120-5120-5120-bem-sol.fif`、`fsaverage-ico-5-src.fif`、`fsaverage-trans.fif`

### 3.5 动态导联场计划文档（已写入）

写入到 `/work/2024/tanzunsheng/Code/PENCI/docs/dynamic_leadfield_plan.md`，包含：

- 6 个实施阶段（前置依赖 → LeadfieldManager → PhysicsDecoder 改造 → PENCI 模型适配 → 数据管线 → 配置测试）
- 5 个待用户决策的问题
- 完整的文件改动清单和依赖关系图

**动态导联场代码实现尚未开始（零代码变更）。**

---

## 四、剩余任务

### 4.1 等待用户决策的 5 个问题

| # | 问题 | 影响 |
|---|------|------|
| 1 | **72 个源空间如何定义？** atlas 脑区质心？均匀采样？自定义位置？ | 直接影响导联场矩阵的物理意义 |
| 2 | **pos 坐标单位是什么？** 范围 [-0.65, 0.65]，MNE 期望米制 | 不正确的坐标会导致导联场计算完全错误 |
| 3 | **batch 内不同通道数如何处理？** padding 到 128？分桶？ | 影响数据管线和训练效率 |
| 4 | **缓存路径放哪？** `~/.cache/penci/leadfields/`？项目目录？ | 多用户场景 |
| 5 | **是否需要无 MNE 环境的 fallback？** | 影响 CI 和可移植性 |

### 4.2 前置依赖

- **用户手动下载 fsaverage**（root.zip + bem.zip）到指定路径
- **确认 pos 坐标系和单位**

### 4.3 代码实施（6 个阶段）

详见 `docs/dynamic_leadfield_plan.md`：

1. **阶段 1**：新建 `penci/physics/__init__.py` + `penci/physics/leadfield_manager.py`（LeadfieldManager 类）
2. **阶段 2**：修改 `penci/models/physics_decoder.py` — PhysicsDecoder 接入 manager，einsum 从 `"cs,bst->bct"` 改为 `"bcs,bst->bct"`
3. **阶段 3**：修改 `penci/models/penci_model.py` — forward 传递 pos 到 decoder
4. **阶段 4**：可能修改 `penci/data/dataset.py` — collator 处理不同通道数
5. **阶段 5**：更新 `configs/default.yaml` + `tests/test_smoke.py`
6. **阶段 6**：Git commit + push

---

## 五、关键代码上下文

### 5.1 PhysicsDecoder 当前 forward 路径

文件：`penci/models/physics_decoder.py`（L117-137）

```python
def _forward_leadfield(self, source_activity):
    source_signal = self.temporal_decoder(source_activity).squeeze(-1)  # (B, 72, T)
    sensor_signal = torch.einsum("cs,bst->bct", self.leadfield, source_signal)  # (B, 128, T)
    # self.leadfield 当前是 torch.randn(128, 72)（随机矩阵，无物理意义）
```

改动后需要：`torch.einsum("bcs,bst->bct", L_batch, source_signal)` 支持 per-sample 不同导联场。

### 5.2 PENCI forward

文件：`penci/models/penci_model.py`（L143-195）

```python
def forward(self, x, pos, sensor_type, return_source=False):
    source_encoded = self.encoder(x, pos, sensor_type)    # (B, 72, N, T_enc, D)
    source_flat = rearrange(source_encoded, "B N Ns T D -> B N (Ns T) D")
    source_evolved = self.dynamics(source_flat)             # (B, 72, T_total, D)
    sensor_embedding = self.sensor_module(pos, sensor_type) # (B, C, D)
    reconstruction = self.decoder(source_evolved, sensor_embedding)  # 需要加 pos 参数
```

---

## 六、环境与外部引用

### 6.1 环境配置

| 项目 | 值 |
|------|-----|
| 项目路径 | `/work/2024/tanzunsheng/Code/PENCI` |
| Python 环境 | `/work/2024/tanzunsheng/anaconda3/envs/EEG/bin/python` |
| 数据路径 | `/work/2024/tanzunsheng/PENCIData` |
| GitHub | `https://github.com/TanZunsheng/PENCI` |
| MNE 版本 | 1.8.0 |
| SUBJECTS_DIR | `/work/2024/tanzunsheng/mne_data/MNE-fsaverage-data` |

### 6.2 外部引用

- MNE-Python forward modeling：`mne.make_forward_solution()`、`mne.channels.make_dig_montage()`
- fsaverage BEM 路径（下载后）：`{subjects_dir}/fsaverage/bem/fsaverage-5120-5120-5120-bem-sol.fif`
- 导联场提取：`fwd['sol']['data']` → shape `(n_sensors, n_sources)`
- MNE manifest 文件位于：`/work/2024/tanzunsheng/anaconda3/envs/EEG/lib/python3.9/site-packages/mne/datasets/_fsaverage/`

---

## 七、关键约束（MUST NOT DO）

- **禁止 import BrainOmni** — 所有代码必须在 PENCI 内部
- **导联场不可训练** — 必须是固定物理约束，register_buffer
- **不使用交叉注意力学习源→传感器映射** — 用户明确要求用固定物理导联场
- **注释/文档使用中文**
- **n_neuro 固定为 72** — 不是 64
- **不做简化/demo 版本** — 必须完整实现
- **不删除测试来让构建通过** — 修复代码而非测试
- **不自动下载 fsaverage** — 用户网络访问不了 OSF，需要手动下载

---

## 八、当前状态

- **Git**：clean，2 commits on main，已 push 到 origin
- **fsaverage**：目录存在但为空，需要用户手动下载
- **Smoke tests**：之前全部 7 项通过
- **动态导联场**：计划文档已写入，代码实现零变更
- **等待**：用户审阅计划文档后提供对 5 个问题的决策
