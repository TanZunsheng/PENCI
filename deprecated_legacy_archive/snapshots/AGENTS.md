# AGENTS.md — PENCI Codebase Guide

## Project Overview

PENCI (Physics-constrained End-to-end Neural Connectivity Inference) is a PyTorch-based deep learning
framework for inferring neural dynamics from EEG/MEG signals. It uses a "sandwich" architecture:
Encoder (from BrainOmni) → Dynamics Core (Transformer/RNN) → Physics Decoder (leadfield matrix).

**Language**: Python 3.9+  |  **Framework**: PyTorch 2.0+  |  **Conda env**: `/work/2024/tanzunsheng/anaconda3/envs/EEG`

---

## Build / Install / Run

```bash
pip install -e .                 # Editable install
pip install -e ".[dev]"          # With dev deps (pytest, black, isort, flake8, mypy)
pip install -r requirements.txt  # Dependencies only
```

## Test / Lint / Format

```bash
pytest                                     # All tests (11 in test_smoke.py)
pytest tests/test_smoke.py::test_modules   # Single test function
pytest --cov=penci                         # With coverage
python tests/test_smoke.py                 # Standalone smoke test (no pytest)

black penci/ tests/ scripts/    # Format (line-length=100)
isort penci/ tests/ scripts/    # Sort imports (profile=black)
flake8 penci/ tests/ scripts/   # Lint
mypy penci/                     # Type check (ignore_missing_imports=true)
```

All tool configs live in `pyproject.toml`. Pytest: `testpaths = ["tests"]`, `addopts = "-v --tb=short"`.

## Training

```bash
# Single GPU
python scripts/train.py --config configs/default.yaml

# Multi-GPU DDP (e.g. 5 GPUs)
torchrun --nproc_per_node=5 scripts/train.py --config configs/default.yaml --output_dir outputs/exp1

# Evaluation
python scripts/evaluate.py --config configs/default.yaml --checkpoint outputs/exp1/best_model.pt
```

**DDP notes**: `batch_size` in config is **per-GPU**. Learning rate scales linearly:
`effective_lr = base_lr × world_size`. NCCL timeout is set to 60 minutes.

---

## Project Structure

```
penci/
├── __init__.py              # Top-level: PENCI, PENCILite, build_penci_from_config
├── modules/                 # Base building blocks (ported from BrainOmni — modify carefully)
│   ├── attention.py         # RMSNorm, FeedForward, SelfAttention, RotaryEmbedding
│   ├── conv.py              # SConv1d, SConvTranspose1d
│   ├── lstm.py              # SLSTM
│   └── seanet.py            # SEANetEncoder, SEANetResnetBlock, Snake1d
├── encoders/                # Encoder modules (ported from BrainOmni — modify carefully)
│   ├── sensor_embed.py      # BrainSensorModule
│   ├── backward_solution.py # BackWardSolution, ForwardSolution
│   └── encoder.py           # PENCIEncoder, BrainTokenizerEncoder
├── models/                  # Model definitions (original PENCI code)
│   ├── dynamics.py          # DynamicsCore (Transformer), DynamicsRNN
│   ├── physics_decoder.py   # PhysicsDecoder, SEANetPhysicsDecoder
│   └── penci_model.py       # PENCI, PENCILite, build_penci_from_config
├── physics/                 # Physics constraint utilities (lazy imports for heavy deps)
│   ├── source_space.py      # SourceSpace (72 brain regions)
│   ├── electrode_utils.py   # ElectrodeConfigRegistry, compute_fingerprint_from_pos
│   └── leadfield_manager.py # LeadfieldManager — MNE-based computation + caching
├── utils/
│   └── metrics.py           # pearson_correlation, snr_db, nrmse (evaluation metrics)
└── data/
    └── dataset.py           # PENCIDataset, PENCICollator, BucketBatchSampler,
                             # DistributedBucketBatchSampler, RandomScaling, RandomNoise

scripts/
├── train.py                     # Single/multi-GPU training (DDP-aware)
├── evaluate.py                  # Model evaluation with full metrics
├── precompute_all_leadfields.py # Batch leadfield precomputation
├── convert_to_hdf5.py          # Data format conversion
└── convert_hbn_to_hdf5.py      # HBN dataset conversion

configs/
├── default.yaml       # Full training config
└── smoke_test.yaml    # Quick validation config

tests/
├── test_smoke.py          # 11 tests (modules, encoders, models, data, config)
├── diagnose_system.py     # System diagnostic tool
└── analyze_training_log.py

docs/                      # Chinese documentation (8 files)
```

**Key rule**: `penci/modules/` and `penci/encoders/` are ported from BrainOmni — modify carefully
and preserve original behavior. `penci/models/` and `penci/physics/` are original PENCI code.

---

## Key Patterns

### Electrode Fingerprint System

Electrode configurations vary across datasets. `ElectrodeConfigRegistry` in `electrode_utils.py`
identifies configurations by a position-based fingerprint (`compute_fingerprint_from_pos`), enabling
automatic leadfield matrix lookup and caching per unique electrode layout.

### DDP `unwrap_model()` Pattern

DDP wraps models but does **not** proxy custom methods (`compute_loss`, `_prepare_target`).
Both `train.py` and `evaluate.py` define a local `unwrap_model()` helper:

```python
def unwrap_model(model):
    return model.module if hasattr(model, 'module') else model

# Rule: forward() goes through DDP wrapper; custom methods go through unwrap_model()
loss_dict = unwrap_model(model).compute_loss(x, pos, sensor_type, ...)
```

### DDP Setup Gotchas

- Call `torch.cuda.set_device(local_rank)` **before** `init_process_group` — never pass `device_id`
- Leadfield warmup: rank-0 only, then `dist.barrier()` (avoids file lock contention)
- Use `sync_ranks(world_size, local_rank)` with explicit `device_ids=[local_rank]`
- See `docs/ddp_hang_diagnosis.md` and `docs/ddp_attribute_error_fix.md` for past issues

### Multi-Dataset Data Loading

`PENCIDataset` supports multiple dataset directories. `PENCICollator` pads variable-length channels.
`DistributedBucketBatchSampler` groups similar-length samples for efficient DDP training with
minimal padding waste. Data augmentation: `RandomScaling`, `RandomNoise`, composable via `Compose`.

---

## Code Style Guidelines

### Imports & Headers

Every `.py` file starts with `# -*- coding: utf-8 -*-` and a module docstring.
Strict 3-group import ordering: stdlib → third-party → local (absolute imports only, never relative).

### Naming

| Category | Style | Examples |
|---|---|---|
| Classes | `PascalCase` | `PENCI`, `DynamicsCore`, `PhysicsDecoder` |
| Functions/methods | `snake_case` | `compute_loss`, `build_penci_from_config` |
| Private methods | `_leading_underscore` | `_prepare_target` |
| Tensor dims (einops) | Single uppercase | `B N T D`, `(B N) T D` |

### Docstrings & Types

Chinese-primary docstrings. Always document tensor shapes (`(B, C, T)`). Type annotations required
on all public method signatures. Use `einops.rearrange` for reshaping, not `.view()/.reshape()`.

### Module Patterns

- Each subpackage re-exports public API with explicit `__all__`
- `penci/physics/__init__.py` uses lazy `__getattr__` for heavy optional imports (MNE)
- `RuntimeError` for missing required args; `assert` only in tests
- Use `logging` module: `logger = logging.getLogger(__name__)`

### Configuration

YAML configs in `configs/` loaded via `yaml.safe_load()`. Nested access with defaults:
`config.get("training", {}).get("lr", 1e-4)`. Model construction: `build_penci_from_config(config)`.

### Testing

Tests are standalone-runnable and pytest-compatible. Name functions `test_<component>`.
Some tests skip gracefully when external data/MNE is unavailable.
