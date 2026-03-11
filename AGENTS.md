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

## Test Commands

```bash
pytest                                     # All tests
pytest tests/test_smoke.py                 # Single file
pytest tests/test_smoke.py::test_modules   # Single test function
pytest --cov=penci                         # With coverage
python tests/test_smoke.py                 # Standalone smoke test (no pytest)
```

**Pytest config** (in `pyproject.toml`): `testpaths = ["tests"]`, `addopts = "-v --tb=short"`

## Lint / Format / Type-Check

```bash
black penci/ tests/ scripts/    # Format (line-length=100)
isort penci/ tests/ scripts/    # Sort imports (profile=black)
flake8 penci/ tests/ scripts/   # Lint
mypy penci/                     # Type check (ignore_missing_imports=true)
```

All tool configs live in `pyproject.toml`.

## Training

```bash
python scripts/train.py --config configs/default.yaml
python scripts/train.py --config configs/default.yaml --output_dir outputs/experiment_1
```

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
├── physics/                 # Physics constraint utilities
│   ├── source_space.py      # SourceSpace (72 brain regions)
│   ├── electrode_utils.py   # Electrode coordinate reading/filtering
│   └── leadfield_manager.py # MNE-based leadfield computation + caching
└── data/
    └── dataset.py           # PENCIDataset, PENCICollator, BucketBatchSampler
```

**Key rule**: `penci/modules/` and `penci/encoders/` are ported from BrainOmni — modify carefully
and preserve original behavior. `penci/models/` and `penci/physics/` are original PENCI code.

---

## Code Style Guidelines

### File Header

Every `.py` file starts with:
```python
# -*- coding: utf-8 -*-
"""
Module description (Chinese or English).
Longer explanation of purpose, origin (if ported), and design decisions.
"""
```

### Import Order

Strict 3-group ordering with blank line separators:
```python
# 1. Standard library
import os
from typing import Any, Dict, Optional, Tuple

# 2. Third-party
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# 3. Local/project (absolute imports only — never relative)
from penci.modules.attention import RMSNorm, FeedForward, SelfAttnBlock
from penci.encoders.encoder import PENCIEncoder
```

### Naming Conventions

| Category | Style | Examples |
|---|---|---|
| Classes | `PascalCase` | `PENCI`, `DynamicsCore`, `PhysicsDecoder` |
| Functions/methods | `snake_case` | `compute_loss`, `build_penci_from_config` |
| Variables & hyperparams | `snake_case` | `n_dim`, `n_neuro`, `max_seq_len` |
| Private methods | `_leading_underscore` | `_prepare_target` |
| Module files | `snake_case` | `penci_model.py`, `physics_decoder.py` |
| Tensor dims (einops) | Single uppercase | `B N T D`, `(B N) T D` |

### Type Annotations

Required on all public method signatures. Use `typing` types for containers:
```python
def forward(self, x: torch.Tensor, leadfield: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
```

### Docstrings

Chinese-primary docstrings with structured parameter/return blocks. Always document tensor shapes:
```python
def compute_loss(self, x, pos, sensor_type, ...):
    """
    计算训练损失

    参数:
        x: 输入信号 (B, C, T)
        pos: 电极位置 (B, C, 6)
    返回:
        字典包含各项损失和总损失
    """
```

### nn.Module Pattern

- Store config values as `self.*` attributes
- Use `einops.rearrange` for complex reshaping (not `.view()/.reshape()`)
- Use `F.mse_loss`, `F.cross_entropy` from `torch.nn.functional`
- Each subpackage re-exports its public API with explicit `__all__`
- Use lazy `__getattr__` for heavy optional imports (see `penci/physics/__init__.py`)

### Error Handling

- `RuntimeError` for missing required arguments (e.g., missing leadfield matrix)
- `assert` for shape validation in **tests only** — not in production code
- Use `logging` module: `logger = logging.getLogger(__name__)`

### Configuration

- YAML configs in `configs/` loaded via `yaml.safe_load()`
- Nested access with defaults: `config.get("training", {}).get("lr", 1e-4)`
- Model construction: `build_penci_from_config(config)`

### Testing Pattern

Tests are standalone-runnable scripts that also work with pytest:
```python
def test_modules():
    """测试基础模块"""
    x = torch.randn(2, 10, 256)
    model = SomeModule(256)
    out = model(x)
    assert out.shape == x.shape, f"Shape error: {out.shape}"

if __name__ == "__main__":
    sys.exit(main())
```

Name test functions `test_<component>` (e.g., `test_modules`, `test_encoders`, `test_full_model`).
Some tests depend on external data/MNE and skip gracefully when unavailable.
