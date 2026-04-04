# AGENTS.md — PENCI V1 Codebase Guide

## Project Overview

PENCI 当前的活动主线是 V1 两阶段建模框架：
1. `Stage1Model` 从 EEG/MEG 恢复显式脑区状态 `S_t`
2. `StaticConnectivityModel` 在冻结 `S_t` 上学习静态有效连接 `A_base`
3. 共享基础设施提供 DDP、HDF5、file scheduler、node-union page-cache prefetch 与动态导联场支持

旧版单阶段主线已迁移到 `deprecated_legacy_archive/`。除非用户明确要求查看归档内容，否则默认只面向当前 V1 主线工作。

**Language**: Python 3.9+  |  **Framework**: PyTorch 2.0+  |  **Conda env**: `/work/2024/tanzunsheng/anaconda3/envs/EEG`

---

## Build / Install / Run

```bash
pip install -e .
pip install -e ".[dev]"
pip install -r requirements.txt
```

## Test / Lint / Format

```bash
pytest
pytest tests/test_smoke.py::test_v1_stage1_model
pytest --cov=penci
python tests/test_smoke.py

black penci/ tests/ scripts/
isort penci/ tests/ scripts/
flake8 penci/ tests/ scripts/
mypy penci/
```

All tool configs live in `pyproject.toml`. Pytest uses `testpaths = ["tests"]` and `addopts = "-v --tb=short"`.

## Training

### Stage1 simulation pretrain

```bash
python scripts/v1/train_stage1.py --config configs/stage1_sim.yaml --mode sim_pretrain
```

### Stage1 real-data finetune

```bash
python scripts/v1/train_stage1.py --config configs/stage1_real.yaml --mode real_finetune
```

### Stage1 multi-GPU DDP

```bash
torchrun --nproc_per_node=5 scripts/v1/train_stage1.py --config configs/stage1_real.yaml --mode real_finetune
```

### Stage1 evaluation

```bash
python scripts/v1/evaluate_stage1.py --config configs/stage1_eval.yaml --checkpoint outputs/stage1_real/best_model.pt --dataset_mode real
```

### Stage2 train / eval

```bash
python scripts/v1/train_stage2.py --config configs/stage2_connectivity.yaml --stage1_checkpoint outputs/stage1_real/best_model.pt
python scripts/v1/evaluate_stage2.py --config configs/stage2_eval.yaml --stage1_checkpoint outputs/stage1_real/best_model.pt --stage2_checkpoint outputs/stage2/best_model.pt
```

**DDP notes**:
- `training.batch_size` is per-GPU in DDP mode
- `scripts/v1/train_stage1.py` reads `training.learning_rate` literally; if you want linear LR scaling, do it explicitly in config
- NCCL timeout / barrier handling / unwrap helpers live in `penci/training/distributed.py`

---

## Project Structure

```text
penci/
├── __init__.py              # Top-level V1 + shared re-exports
├── modules/                 # BrainOmni-ported low-level blocks — modify carefully
│   ├── attention.py
│   ├── conv.py
│   ├── lstm.py
│   └── seanet.py
├── encoders/                # BrainOmni-ported encoders — modify carefully
│   ├── sensor_embed.py
│   ├── backward_solution.py
│   └── encoder.py
├── v1/                      # Active two-stage mainline
│   ├── models/
│   │   ├── stage1_model.py      # Stage1Model, build_stage1_model_from_config
│   │   ├── connectivity.py      # StaticConnectivityModel, build_stage2_model_from_config
│   │   └── state_head.py        # StateHead
│   └── data/
│       └── simulation_dataset.py
├── shared/                  # Shared model components used by V1
│   └── models/
│       ├── dynamics.py          # DynamicsCore, DynamicsRNN
│       └── physics_decoder.py   # PhysicsDecoder, SEANetPhysicsDecoder
├── training/                # Shared training infrastructure
│   ├── distributed.py          # setup_distributed, unwrap_model, sync helpers
│   ├── prefetch.py             # page-cache prefetch and planning
│   └── physics.py              # setup_physics, resolve_leadfield_for_batch
├── data/
│   └── dataset.py           # PENCIDataset, collator, HDF5 / file scheduler loaders
├── physics/
│   ├── source_space.py
│   ├── electrode_utils.py
│   └── leadfield_manager.py
└── utils/
    ├── metrics.py
    └── state_metrics.py

scripts/
├── v1/
│   ├── train_stage1.py
│   ├── evaluate_stage1.py
│   ├── train_stage2.py
│   ├── evaluate_stage2.py
│   ├── generate_stage1_sim_data.py
│   └── generate_stage2_sim_data.py
├── convert_to_hdf5.py
├── convert_hbn_to_hdf5.py
├── convert_to_hdf5_by_fingerprint.py
└── precompute_all_leadfields.py

configs/
├── stage1_sim.yaml
├── stage1_real.yaml
├── stage1_eval.yaml
├── stage2_connectivity.yaml
└── stage2_eval.yaml

deprecated_legacy_archive/
├── moved/                   # Archived legacy code and docs
└── snapshots/               # Snapshots before rewrites
```

**Key rule**: `penci/modules/` and `penci/encoders/` are ported from BrainOmni. Modify carefully and preserve established behavior unless the user explicitly asks for a behavior change.

---

## Key Patterns

### Stage1 real-data I/O path

The active high-throughput path is `scripts/v1/train_stage1.py --mode real_finetune`.
It reuses the shared real-data loader stack in `penci/data/dataset.py`:
- HDF5-first sample loading with `.pt` fallback
- `BucketBatchSampler` / `DistributedBucketBatchSampler`
- `file_scheduler` to cluster batches by HDF5 file
- per-worker HDF5 handle caching

### Node-union page-cache prefetch

The prefetch planner/executor lives in `penci/training/prefetch.py`.
For DDP real-data finetune, rank schedules can be gathered and merged into a node-level warmup plan.
Relevant config lives under `data.io_prefetch` in `configs/stage1_real.yaml`.

### Dynamic leadfield resolution

Dynamic leadfield setup lives in `penci/training/physics.py` and `penci/physics/`.
In real-data Stage1 training:
- rank 0 prepares / warms leadfield resources when needed
- `resolve_leadfield_for_batch()` chooses per-batch leadfield tensors
- the resolved leadfield is passed into `Stage1Model.forward(..., leadfield=...)`

### DDP `unwrap_model()` pattern

DDP does not proxy custom methods reliably. Use the shared helper in `penci/training/distributed.py`:

```python
def unwrap_model(model):
    return model.module if hasattr(model, "module") else model

loss_dict = unwrap_model(model).compute_stage1_loss_real(...)
```

### Simulation vs. real-data split

- Simulation datasets live under `penci/v1/data/simulation_dataset.py`
- Real EEG/MEG loading lives under `penci/data/dataset.py`
- Do not mix the simulation loader path with the real-data HDF5 / scheduler path unless you are intentionally extending Stage2 to real data

---

## Code Style Guidelines

### Imports & Headers

Every `.py` file starts with `# -*- coding: utf-8 -*-` and a module docstring.
Use 3-group import ordering: stdlib → third-party → local. Prefer absolute imports over relative imports.

### Naming

| Category | Style | Examples |
|---|---|---|
| Classes | `PascalCase` | `Stage1Model`, `StaticConnectivityModel`, `PhysicsDecoder` |
| Functions/methods | `snake_case` | `build_stage1_model_from_config`, `resolve_leadfield_for_batch` |
| Private methods | `_leading_underscore` | `_build_sim_loader` |
| Tensor dims (einops) | Single uppercase | `B C T`, `B N T D` |

### Docstrings & Types

Chinese-primary docstrings are preferred. Public methods should include type annotations.
Document tensor shapes when they are not obvious.
Use `einops.rearrange` when reshaping semantics matter.

### Module Patterns

- Public subpackages should re-export their API with explicit `__all__`
- `penci/physics/__init__.py` may use lazy imports for heavy optional dependencies
- Use `RuntimeError` for missing required runtime arguments
- Prefer `logging.getLogger(__name__)` over `print()` in library code

### Configuration

Load YAML with `yaml.safe_load()`.
Access nested config with `.get()` defaults, e.g. `config.get("training", {}).get("batch_size", 8)`.
Current model builders are:
- `build_stage1_model_from_config(config)`
- `build_stage2_model_from_config(config)`

### Testing

Tests are standalone-runnable and pytest-compatible. Use `test_<component>` naming.
`tests/test_smoke.py` is the main regression suite and currently covers 20 smoke tests focused on the active V1 mainline.

---

## Archive Policy

If you need historical context, check `deprecated_legacy_archive/`.
Do not restore archived legacy files into the active tree unless the user explicitly asks for rollback or comparison work.
