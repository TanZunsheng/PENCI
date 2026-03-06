# AGENTS.md — PENCI Codebase Guide

## Project Overview

PENCI (Physics-constrained End-to-end Neural Connectivity Inference) is a PyTorch-based deep learning
framework for inferring neural dynamics from EEG/MEG signals. It uses a "sandwich" architecture:
Encoder (from BrainOmni) → Dynamics Core (Transformer/RNN) → Physics Decoder (leadfield matrix).

**Language**: Python 3.9+  
**Framework**: PyTorch 2.0+  
**Conda env**: `/work/2024/tanzunsheng/anaconda3/envs/EEG`

---

## Build / Install / Run Commands

```bash
# Install (editable)
pip install -e .

# Install with dev dependencies (test, lint, format, typecheck)
pip install -e ".[dev]"

# Install dependencies only
pip install -r requirements.txt
```

## Test Commands

```bash
# Run all tests via pytest
pytest

# Run a single test file
pytest tests/test_smoke.py

# Run a single test function
pytest tests/test_smoke.py::test_modules

# Verbose with short traceback (default via pyproject.toml addopts)
pytest -v --tb=short

# With coverage
pytest --cov=penci

# Smoke test (standalone, no pytest required)
python tests/test_smoke.py
```

**Pytest config** (in `pyproject.toml`):
- `testpaths = ["tests"]`
- `python_files = ["test_*.py"]`
- `addopts = "-v --tb=short"`

## Lint / Format / Type-Check Commands

```bash
# Format code
black penci/ tests/ scripts/

# Sort imports
isort penci/ tests/ scripts/

# Lint
flake8 penci/ tests/ scripts/

# Type check
mypy penci/
```

**Tool configs** (all in `pyproject.toml`):
- **black**: `line-length = 100`, `target-version = ["py39", "py310", "py311"]`
- **isort**: `profile = "black"`, `line_length = 100`
- **mypy**: `python_version = "3.9"`, `ignore_missing_imports = true`

## Training

```bash
python scripts/train.py --config configs/default.yaml
python scripts/train.py --config configs/default.yaml --output_dir outputs/experiment_1
```

---

## Project Structure

```
penci/
├── __init__.py              # Top-level exports: PENCI, PENCILite, build_penci_from_config
├── modules/                 # Base building blocks (ported from BrainOmni)
│   ├── attention.py         # RMSNorm, FeedForward, SelfAttention, RotaryEmbedding
│   ├── conv.py              # SConv1d, SConvTranspose1d
│   ├── lstm.py              # SLSTM
│   └── seanet.py            # SEANetEncoder, SEANetResnetBlock, Snake1d
├── encoders/                # Encoder modules (ported from BrainOmni)
│   ├── sensor_embed.py      # BrainSensorModule
│   ├── backward_solution.py # BackWardSolution, ForwardSolution
│   └── encoder.py           # PENCIEncoder, BrainTokenizerEncoder
├── models/                  # Model definitions (original)
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

**Key distinction**: `penci/modules/` and `penci/encoders/` are ported from BrainOmni — modify
carefully. `penci/models/` and `penci/physics/` are original PENCI code.

---

## Code Style Guidelines

### File Header

Every `.py` file starts with:
```python
# -*- coding: utf-8 -*-
"""
Module description in Chinese/English.

Longer explanation of purpose, origin (if ported), and design decisions.
"""
```

### Import Order

Strict 3-group ordering separated by blank lines:
```python
# 1. Standard library
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# 2. Third-party
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# 3. Local/project
from penci.modules.attention import RMSNorm, FeedForward, SelfAttnBlock
from penci.encoders.encoder import PENCIEncoder
```

Use **absolute imports** from `penci.*` — never relative imports.

### Naming Conventions

- **Classes**: `PascalCase` — `PENCI`, `PENCILite`, `DynamicsCore`, `PhysicsDecoder`
- **Functions/methods**: `snake_case` — `compute_loss`, `build_penci_from_config`
- **Variables**: `snake_case` — `n_dim`, `n_neuro`, `n_head`, `source_activity`
- **Constants/hyperparams**: `snake_case` (not UPPER_CASE) — `n_dim`, `max_seq_len`
- **Private methods**: leading underscore — `_prepare_target`
- **Module files**: `snake_case` — `penci_model.py`, `physics_decoder.py`
- **Tensor dimensions**: single uppercase letters in einops — `B N T D`, `(B N) T D`

### Type Annotations

Used on all public method signatures. `typing` types for containers:
```python
def forward(
    self,
    x: torch.Tensor,
    pos: torch.Tensor,
    sensor_type: torch.Tensor,
    leadfield: Optional[torch.Tensor] = None,
    return_source: bool = False,
) -> Dict[str, torch.Tensor]:
```

### Docstrings

Chinese-primary docstrings describing parameters and return values in a structured block:
```python
def compute_loss(self, x, pos, sensor_type, ...):
    """
    计算训练损失

    参数:
        x: 输入信号 (B, C, T)
        pos: 电极位置 (B, C, 6)
        sensor_type: 传感器类型 (B, C)

    返回:
        字典包含各项损失和总损失
    """
```

Always document tensor shapes in comments: `# (B, C, T)`, `# (B, N_neuro, T', D)`.

### `__init__.py` Pattern

Each subpackage re-exports its public API with explicit `__all__`:
```python
from penci.models.penci_model import PENCI, PENCILite, build_penci_from_config
__all__ = ["PENCI", "PENCILite", "build_penci_from_config"]
```

Use lazy `__getattr__` for heavy optional imports (see `penci/physics/__init__.py`).

### nn.Module Pattern

```python
class MyModule(nn.Module):
    """Chinese docstring with design explanation."""

    def __init__(self, n_dim: int = 256, ...):
        super().__init__()
        self.n_dim = n_dim
        # Build submodules ...

    def forward(self, x: torch.Tensor, ...) -> torch.Tensor:
        """Docstring with input/output tensor shapes."""
        ...
```

- Store config values as `self.*` attributes for inspection
- Use `einops.rearrange` for tensor reshaping, not `.view()/.reshape()` for complex ops
- Use `F.mse_loss`, `F.cross_entropy` etc. from `torch.nn.functional`

### Error Handling

- Use `RuntimeError` for missing required arguments (e.g., missing leadfield matrix)
- Use `assert` for shape validation in tests only, not in production code
- Use Python `logging` module — `logger = logging.getLogger(__name__)`

### Configuration

- YAML-based configs in `configs/` loaded via `yaml.safe_load()`
- Access nested config via `.get()` with defaults: `config.get("training", {}).get("lr", 1e-4)`
- Model construction from config via `build_penci_from_config(config)`

### Testing Pattern

Tests are standalone-runnable scripts that also work with pytest:
```python
def test_modules():
    """Test description in Chinese."""
    # Setup
    x = torch.randn(2, 10, 256)
    model = SomeModule(256)
    # Forward
    out = model(x)
    # Assert shape
    assert out.shape == x.shape, f"Shape error: {out.shape}"

if __name__ == "__main__":
    # Standalone execution support
    sys.exit(main())
```

Test functions are named `test_<component>` (e.g., `test_modules`, `test_encoders`, `test_full_model`).
