# -*- coding: utf-8 -*-
"""
兼容入口：PENCI V1 第一层模型。
"""

from penci.v1.models.stage1_model import Stage1Model, build_stage1_model_from_config

__all__ = [
    "Stage1Model",
    "build_stage1_model_from_config",
]
