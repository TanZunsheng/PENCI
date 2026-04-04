# -*- coding: utf-8 -*-
"""
PENCI: Physics-constrained End-to-end Neural Connectivity Inference
物理约束端到端神经连接推断框架

该框架实现了一个"三明治"结构的端到端深度学习模型：
1. 通用编码器 (Universal Encoder) - 改编自 BrainOmni
2. 动力学核心 (Dynamics Core) - 学习源空间的时序演化
3. 物理解码器 (Physics Decoder) - 使用导联场矩阵投影回传感器空间
"""

__version__ = "0.2.0"
__author__ = "PENCI Team"

from penci.legacy.models.penci_model import (
    PENCI,
    PENCILite,
    build_penci_from_config,
)
from penci.v1.models.connectivity import (
    StaticConnectivityModel,
    build_stage2_model_from_config,
)
from penci.v1.models.stage1_model import Stage1Model, build_stage1_model_from_config

# 别名，保持向后兼容
PENCIModel = PENCI

__all__ = [
    "PENCI",
    "PENCILite",
    "Stage1Model",
    "StaticConnectivityModel",
    "PENCIModel",
    "build_penci_from_config",
    "build_stage1_model_from_config",
    "build_stage2_model_from_config",
]
