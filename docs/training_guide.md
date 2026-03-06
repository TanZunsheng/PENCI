# PENCI 训练指南

本文档提供 PENCI (Physics-constrained End-to-end Neural Connectivity Inference) 项目的训练指导，涵盖项目现状、环境准备、训练流程、配置说明及常见问题。

---

## 一、项目现状总结

PENCI 项目已进入生产就绪状态，核心架构与训练流水线已完全打通。

- **架构实现**：已完整实现"三明治"架构：编码器（移植自 BrainOmni）→ 动力学核心（Transformer/RNN）→ 物理解码器（物理导联场矩阵）。
- **系统集成**：动态导联场（Dynamic Leadfield）系统已无缝集成至训练脚本，支持自动计算、缓存和分发。
- **验证情况**：所有 10 个核心测试用例全部通过。端到端 dry-run 验证成功，涵盖了从数据加载、导联场解析、模型前向传播、反向传播到参数更新的完整闭环。
- **模型规模**：总参数量约为 6,736,897。
- **验证数据集**：HBN_EEG（包含 1,568,305 个训练样本，48,504 个验证样本，128 通道配置）。

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
- **fsaverage 头模型**：`/work/2024/tanzunsheng/mne_data/MNE-fsaverage-data/`（物理计算基石）
- **导联场缓存**：`/work/2024/tanzunsheng/leadfield_cache/`（加速后续训练）
- **BIDS 元数据 (ProcessedData)**：`/work/2024/tanzunsheng/ProcessedData/`（包含电极配置文件 `electrodes.tsv`）
- **训练数据 (PENCIData)**：`/work/2024/tanzunsheng/PENCIData/`（.pt 格式样本）

---

## 三、训练方法

### 3.1 运行前验证
建议在正式训练前运行 smoke test 确保组件正常：
```bash
/work/2024/tanzunsheng/anaconda3/envs/EEG/bin/pytest tests/test_smoke.py -v
```

### 3.2 启动训练
使用默认配置文件启动训练：
```bash
/work/2024/tanzunsheng/anaconda3/envs/EEG/bin/python scripts/train.py --config configs/default.yaml
```

指定自定义输出目录：
```bash
/work/2024/tanzunsheng/anaconda3/envs/EEG/bin/python scripts/train.py \
    --config configs/default.yaml \
    --output_dir outputs/experiment_v1
```

### 3.3 训练脚本自动化流程
1. **物理组件初始化**：自动加载 SourceSpace 和 LeadfieldManager。
2. **导联场解析**：基于 `BucketBatchSampler`，为每个 batch 自动解析对应的物理矩阵。
3. **混合精度支持**：默认启用 FP16 混合精度训练以提高效率。

---

## 四、配置说明 (`default.yaml`)

### 4.1 关键配置项

| 配置项 | 默认值 | 说明 |
|------|--------|------|
| `model.n_dim` | 256 | 特征隐藏维度 |
| `model.n_neuro` | 72 | 源空间脑区数量 |
| `model.dynamics.type` | "transformer" | 动力学建模方式 (transformer/rnn) |
| `data.use_bucket_sampler` | true | **必须开启**，确保同 batch 内电极配置一致 |
| `data.datasets` | ["HBN_EEG"] | 指定训练使用的数据集列表 |
| `hardware.mixed_precision` | true | 是否开启混合精度训练 |

### 4.2 物理解码器模式 (`model.physics`)

1. **动态导联场模式**（当前默认）：
   - `use_fixed_leadfield: true`
   - `leadfield_path: null`
   - 系统根据 batch 中样本的电极配置动态计算导联场矩阵。

2. **静态导联场模式**：
   - `use_fixed_leadfield: true`
   - `leadfield_path: "path/to/matrix.pt"`
   - 所有样本强制使用同一个预定义的导联场矩阵。

3. **注意力模式**：
   - `use_fixed_leadfield: false`
   - 使用可学习的交叉注意力机制替代固定物理约束。

---

## 五、数据格式说明

### 5.1 样本文件 (.pt)
每个样本应包含以下 Tensor 字段：
- `x`: `(C, T)` - EEG/MEG 信号张量。
- `pos`: `(C, 6)` - 电极坐标及法线向量。
- `sensor_type`: `(C,)` - 传感器类型编码。

### 5.2 元数据与电极配置
- **元数据**：JSON 格式，描述样本所属数据集、被试 ID 等。
- **电极配置**：位于 `ProcessedData/{dataset}/bids/derivatives/` 目录下的 `electrodes.tsv`。该文件是计算物理导联场的核心依据，必须包含通道名称及对应的米制坐标。

---

## 六、物理约束系统工作原理

1. **SourceSpace**：定义了 72 个标准脑区（68 个皮层区 + 4 个皮层下结构），作为神经活动推断的目标空间。
2. **LeadfieldManager**：集成 MNE-Python，利用 BEM 模型计算从 72 个脑区到传感器位置的电磁传播矩阵。
3. **两级缓存机制**：
   - **内存缓存**：极速访问最近使用的配置。
   - **磁盘缓存**：在 `leadfield_cache_dir` 下以配置 hash 命名保存，确保重启训练后无需重新计算。
4. **解析流程**：在训练循环中，`resolve_leadfield_for_batch` 函数根据样本元数据查找对应的物理矩阵，并将其作为非训练参数注入模型。

---

## 七、多数据集训练

虽然当前验证主要针对 `HBN_EEG`，但 PENCI 架构天然支持多数据集联合训练：

1. **通道数差异处理**：通过 `BucketBatchSampler`，系统会自动将相同通道数的样本聚类到同一个 batch 中。
2. **物理一致性**：不同数据集的样本会动态匹配各自的导联场矩阵，但共用同一个 72 脑区的源空间表示。
3. **扩展方法**：只需在 `data.datasets` 列表中添加新数据集名称，并确保 `ProcessedData` 下存在对应的 `electrodes.tsv` 即可。

---

## 八、常见问题与注意事项

- **fsaverage 数据**：由于网络限制，`fsaverage` 数据集无法自动下载。如路径缺失，需手动从指定位置下载并解压至 `physics.subjects_dir`。
- **首次计算耗时**：当遇到全新的电极配置时，MNE 需要进行数分钟的前向建模计算。一旦计算完成并存入缓存，后续加载仅需毫秒级时间。
- **导联场梯度**：导联场矩阵在模型中通过 `register_buffer` 注册，**不参与参数更新**，它是模型必须遵循的物理先验。
- **内存/显存优化**：如果遇到 CUDA OOM，请尝试减小 `training.batch_size` 或 `model.n_dim`。

---

## 九、项目结构快速参考

- `penci/models/penci_model.py`：主模型入口及构建逻辑。
- `penci/physics/leadfield_manager.py`：导联场计算与缓存调度中心。
- `penci/data/dataset.py`：实现 Bucket 分桶的数据加载逻辑。
- `scripts/train.py`：训练执行入口脚本。
- `configs/default.yaml`：全量实验参数配置。
- `docs/dynamic_leadfield_system.md`：物理系统的深度技术文档。