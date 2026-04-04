# -*- coding: utf-8 -*-
"""
兼容入口：旧版单阶段 PENCI + V1 builder。
"""

from penci.legacy.models.penci_model import PENCI, PENCILite, build_penci_from_config
from penci.v1.models.connectivity import build_stage2_model_from_config
from penci.v1.models.stage1_model import build_stage1_model_from_config

__all__ = [
    "PENCI",
    "PENCILite",
    "build_penci_from_config",
    "build_stage1_model_from_config",
    "build_stage2_model_from_config",
]
