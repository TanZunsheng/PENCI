# -*- coding: utf-8 -*-
"""
PENCI 旧版单阶段模型入口。
"""

from penci.legacy.models.penci_model import PENCI, PENCILite, build_penci_from_config

__all__ = [
    "PENCI",
    "PENCILite",
    "build_penci_from_config",
]
